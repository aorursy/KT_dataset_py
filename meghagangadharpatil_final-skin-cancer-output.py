# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

from sklearn.model_selection import train_test_split
import sklearn.metrics as skmetrics
from sklearn.metrics import classification_report

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler,WeightedRandomSampler
import torchvision.models as models

from PIL import Image
import io
import pathlib
skip_training = False
device = torch.device('cuda:0')
image_information = pd.read_csv("/kaggle/input/hammetdata/HAM10000_metadata1.csv")

image_information.head()
dx = image_information['dx'].value_counts()
dx
dx.plot.bar()
plt.xlabel('Frequency')
plt.ylabel('Label')
classes = ['akiec','bcc','bkl','df','mel','nv','vasc']
y = image_information.dx
X_train, X_val, y_train, y_test = train_test_split(y,y, test_size=0.2, stratify=y)

images_train_info = image_information.loc[y_train.index].reset_index(drop=True)
images_test_info = image_information.loc[y_test.index].reset_index(drop=True)

max_size = 2000
lst = [images_train_info]
for class_index, group in images_train_info.groupby('dx'):
    if len(group) < 2000:
        lst.append(group.sample(max_size-len(group), replace=True))
    
images_train_info = pd.concat(lst).reset_index(drop=True)

images_train_info['dx'].value_counts()
transform_train = transforms.Compose([
    transforms.Resize((299,299)), #inception-v3 accepts tensors of size Nx3x299x299
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1),
    transforms.ToTensor(),  
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
])

transform_test = transforms.Compose([
    transforms.Resize((299,299)), #inception-v3 accepts tensors of size Nx3x299x299
    transforms.ToTensor(),  
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
])
class HAMN10000DataSet(Dataset):

    def __init__(self, csv_path, img_dir, transform=None):
    
        df = csv_path
        self.img_dir = img_dir
        self.csv_path = csv_path
        self.img_names = df['image_id'] + '.jpg' #name image id
        self.y = torch.from_numpy(np.array(df['dx'].astype('category').cat.codes, dtype=np.uint8)).long()
        self.y_str = df['dx'].values 
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_dir,
                                      self.img_names[index]))
        
        if self.transform is not None:
            img = self.transform(img)
        
        label = self.y[index]
        return img, label
    
    def __visualizeitem__(self, index):
        img = Image.open(os.path.join(self.img_dir,
                                      self.img_names[index]))
        
        label = self.y_str[index]
        return img, label

    def __len__(self):
        return self.y.shape[0]
dataset_train = HAMN10000DataSet(csv_path=images_train_info,
                              img_dir='/kaggle/input/skin-lesion-dataset/HAM10000_images_cnn', transform=transform_train)

dataset_test = HAMN10000DataSet(csv_path=images_test_info,
                              img_dir='/kaggle/input/skin-lesion-dataset/HAM10000_images_cnn', transform=transform_test)
batchsize= 16

trainloader = torch.utils.data.DataLoader(dataset_train, batch_size=batchsize, shuffle=True)
testloader = torch.utils.data.DataLoader(dataset_test, batch_size=batchsize, shuffle=False)
#Visualize random items from the training set (pretransformation)
img1, label1 = dataset_train.__visualizeitem__(100)
img2, label2 = dataset_train.__visualizeitem__(200)
img3, label3 = dataset_train.__visualizeitem__(500)
img4, label4 = dataset_train.__visualizeitem__(400)

f = plt.figure(figsize=(10,10))

ax1 = f.add_subplot(221)
ax2 = f.add_subplot(222)
ax3 = f.add_subplot(223)
ax4 = f.add_subplot(224)

ax1.imshow(img1)
ax1.title.set_text(label1)

ax2.imshow(img2)
ax2.title.set_text(label2)

ax3.imshow(img3)
ax3.title.set_text(label3)

ax4.imshow(img4)
ax4.title.set_text(label4)
net = models.inception_v3(pretrained=True)
net.AuxLogits.fc = nn.Linear(768, 7)
net.fc = nn.Linear(2048, 7) #inception-v3 output layer num_classes change from default 1000 to 7
net.to(device)
def compute_accuracy(net, testloader):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

n_epochs=14
accuracies = []
for epoch in range(n_epochs):
    net.train() #set net to train mode again otherwise auxillary values fail
    running_loss = 0.0
    print_every = 100  # mini-batches
    for i, (inputs, labels) in enumerate(trainloader, 0):
        # Transfer to GPU
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs, aux_out = net(inputs) #inception model has auxillary output to handle vanishing gradient problem
        
        l1_loss = criterion(outputs, labels)
        l2_aux_loss = criterion(aux_out, labels)
        loss = l1_loss + 0.4* l2_aux_loss
        
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if (i % print_every) == (print_every-1):
            print('[%d, %5d] loss: %.3f' % (epoch+1, i+1, running_loss/print_every))
            running_loss = 0.0
        if skip_training:
            break

    accuracy = compute_accuracy(net, testloader)
    accuracies.append(accuracy)
    print('Accuracy of the network on the test images: %.3f' % accuracy)

print('Finished Training')
# Save the network to a file, submit this file together with your notebook
filename = 'inception_v3_skin_cancer.pth'
if not skip_training:
    try:
        do_save = input('Do you want to save the model (type yes to confirm)? ').lower()
        if do_save == 'yes':
            torch.save(net.state_dict(), filename)
            print('Model saved to %s' % filename)
        else:
            print('Model not saved')
    except:
        raise Exception('The notebook should be run or validated with skip_training=True.')
else:
    net = Net()
    net.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))
    net.to(device)
    print('Model loaded from %s' % filename)
def confusion_matrix(net, testloader):
    net.eval()
    true_labels = []
    predictions = []
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            predicted = torch.argmax(outputs.data, 1)
            true_labels.append(labels.cpu().numpy())
            predictions.append(predicted.cpu().numpy())
    true_labels = np.hstack(true_labels)
    predictions = np.hstack(predictions)
    
   

    return true_labels, predictions
true_labels, predictions = confusion_matrix(net, testloader)
c_matrix = skmetrics.confusion_matrix(true_labels, predictions)
plt.figure(figsize=(6, 6))
plt.title("Confusion matrix")
sns.heatmap(c_matrix, cmap='Blues', annot=True, xticklabels=classes, yticklabels=classes, fmt='g', cbar=False)
plt.xlabel('predictions')
plt.ylabel('true labels')
plt.plot(accuracies)
plt.xlabel('epoch')
plt.ylabel('Accuracy')
report = classification_report(true_labels, predictions, target_names=classes)
print(report)
# Calculate the number of positive predictions
num_values = len(true_labels)
num_pos_preds = accuracy * num_values
num_neg_preds = num_values - num_pos_preds
print("Number of Positive Predictions: {0} \n"
      "Number of Negative Predictions: {1}".format(num_pos_preds, num_neg_preds))

# Calculate the False Negatives
from sklearn.metrics import recall_score
recall= recall_score(predictions ,true_labels,average='weighted')
print("recall Score: {}".format(recall))
FN = num_pos_preds * (1 - recall)
print("FN: {0}".format(FN))

# Calculate the True Positives
TP = num_pos_preds - FN
print("TP: {0}".format(TP))

# Calculate the True Negatives
TN = num_pos_preds - TP
print("TN: {0}".format(TN))

# Calculate the False Positives
FP = num_neg_preds - TN
print("FP: {0}".format(FP))
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(predictions ,true_labels)
print("Accuracy Score: {}".format(accuracy))

from sklearn.metrics import precision_score
precision= precision_score(predictions ,true_labels,average='weighted')
print("precision Score: {}".format(precision))

plt.plot(predictions )
plt.xlabel('epoch')
plt.ylabel('Accuracy')
plt.plot( true_labels  )
plt.xlabel('epoch')
plt.ylabel('Accuracy')
print(predictions )
print(true_labels)