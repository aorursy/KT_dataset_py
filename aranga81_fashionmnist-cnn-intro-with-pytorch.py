import numpy as np

import pandas as pd

from tqdm import tqdm

import random

import matplotlib.pyplot as plt

import seaborn as sns

from PIL import Image



import sklearn.metrics as metrics



import torch

import torchvision

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

from torchvision import transforms

from torch.utils.data import DataLoader, Dataset



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



print("Running the model on Device : {}".format(device))
# import data

train = pd.read_csv('/kaggle/input/fashionmnist/fashion-mnist_train.csv')

test = pd.read_csv('/kaggle/input/fashionmnist/fashion-mnist_test.csv')



# Class Names

class_names = {0: "top", 

               1: "Trouser", 

               2: "pullover", 

               3: "Dress", 

               4: "Coat", 

               5: "Sandal",

               6: "Shirt",

               7: "Sneaker",

               8: "Bag",

               9: "Ankle boot"}
print("Length of train set: {}".format(len(train)), "\n"

     "Length of test set: {}".format(len(test)))

train.head()
test.head()
# drop Label and data from train

X_train, Y_train = train.drop('label', 1), train['label']

X_test, Y_test = test.drop('label', 1), test['label']



del train

del test
# Data visualization

plt.figure(figsize=(10, 10))

for i in range(16):

    idx = random.randint(0, len(X_train))

    img = np.array(X_train.iloc[idx]).reshape(28, 28)

    plt.subplot(4, 4, i+1)

    plt.xticks([])

    plt.yticks([])

    plt.grid(False)

    plt.title(class_names[Y_train[idx]])

    plt.imshow(img)

plt.show()
# Distribution

sns.countplot(Y_train)
def preprocess_data(input):

    X = np.array(input, dtype=np.float32)

    X = X.reshape(-1, 28, 28, 1)

    X /= 255.0

    return X



train_feat = preprocess_data(X_train)

test_feat = preprocess_data(X_test)



print(train_feat.shape, test_feat.shape)



plt.figure(figsize=(5,5))

plt.imshow(train_feat[800][:,:,0], cmap="gray")

plt.show()
# Hyperparameters

Batch = 100

LearningRate = 1e-3

Epochs = 20
# torchvision define train/val transform

# train_transform = transforms.Compose([

#     torchvision.transforms.RandomOrder([

#     torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5),

#     torchvision.transforms.RandomHorizontalFlip(p=0.2),

#     torchvision.transforms.RandomRotation(degrees=10)]),

#     transforms.ToTensor()

# ])

train_transform = transforms.Compose([

    transforms.ToTensor()

])

test_transform = transforms.Compose([transforms.ToTensor()])



class FashionMNIST(Dataset):

    def __init__(self, data, transform=None, target_transform=None):

        self.data = data

        self.transform = transform

        self.target_transform = target_transform

        self.images, self.labels = self.data

        

    def __getitem__(self, index):

        image, label = self.images[index], self.labels[index]

#         image = Image.fromarray(image)

        

        if self.transform is not None:

            image = self.transform(image)

        

        if self.target_transform is not None:

            label = self.target_transform(label)

        

        return image, label

            

    def __len__(self):

        return len(self.images)



trainset = FashionMNIST([train_feat, Y_train], transform=train_transform)

testset = FashionMNIST([test_feat, Y_test], transform=test_transform)



trainloader = DataLoader(trainset, batch_size=Batch, shuffle=True, drop_last=True)

testloader = DataLoader(testset, batch_size=Batch, shuffle=False, drop_last=True)
# Model definition

class Net(nn.Module):

    def __init__(self):

        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, 3)

        self.conv2 = nn.Conv2d(32, 64, 3)

        self.pool1 = nn.MaxPool2d(2, 2)

        self.dropout1 = nn.Dropout(0.25)

        self.conv3 = nn.Conv2d(64, 128, 3)

        self.conv4 = nn.Conv2d(128, 256, 3)

        self.pool2 = nn.MaxPool2d(2, 2)

        

        self.fc1 = nn.Linear(256*4*4, 512)

        self.fc2 = nn.Linear(512, 256)

        self.dropout2 = nn.Dropout(0.25)

        self.fc3 = nn.Linear(256, 10)

    

    def forward(self, x):

        x = F.relu(self.conv1(x))

        x = F.relu(self.conv2(x))

        x = self.pool1(x)

        x = self.dropout1(x)

        x = F.relu(self.conv3(x))

        x = F.relu(self.conv4(x))

        x = self.pool2(x)

        

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))

        x = F.relu(self.fc2(x))

        x = self.dropout2(x)

        x = self.fc3(x)

        return x



# call net and move to device

net = Net()

net.to(device)
# train / eval 

def train(trainloader, testloader, model, lossfn, hyperp):

    batchsize, n_epochs, lr = hyperp

    

    # optim

    optimizer = optim.Adam(model.parameters(), lr=lr)

    

    for epoch in range(1, n_epochs+1):

        for phase in ['train', 'eval']:

            if phase == 'train':

                model.train()

            elif phase == 'eval':

                model.eval()

            

            if phase == 'train':

                running_loss = 0.0

                count = 0

                for img, label in trainloader:

                    img = img.to(device)

                    label = label.to(device)

                    

                    outputs = model(img)

                    train_loss = lossfn(outputs, label)



                    optimizer.zero_grad()

                    train_loss.backward()

                    optimizer.step()

                    

                    count += 1

                    #stats

                    running_loss += train_loss

#                     if count*img.shape[0]%1000 == 0:

                print("Epoch: {} | Step: {} | TrainLoss: {}".format(

                    epoch, count, running_loss/count))

            else:

                val_loss = 0.0

                correct_vals, total = 0, 0

                with torch.no_grad():

                    for img, target in testloader:

                        img = img.to(device)

                        target = target.to(device)

                        

                        outputs = model(img)

                        loss = lossfn(outputs, target)

                        _, predictions = torch.max(outputs, 1)

                        predictions = predictions.to(device)

                        

                        val_loss += loss

                        correct_vals += (predictions == target).sum().item()

                        total += 1

                print("Val loss :{} | Accuracy: {}".format(

                    val_loss/total, 100*correct_vals/(total*target.size(0))))
train(trainloader, 

      testloader, 

      model=net, 

      lossfn = nn.CrossEntropyLoss(), 

      hyperp = [Batch, Epochs, LearningRate])
def metric_eval(testloader, model, n_classes=10):

    class_preds, class_probs, class_labels = [], [], []

    correct, total = 0, 0

    with torch.no_grad():

        for img, label in tqdm(testloader):

            img, label = img.to(device), label.to(device)

            outputs = model(img)

            _, predicted = torch.max(outputs.data, 1)

            predicted = predicted.to(device)

            class_probs_batch = [F.softmax(el, dim=0) for el in outputs]



            class_preds.append(predicted)

            class_probs.append(class_probs_batch)

            class_labels.append(label)



    test_probs = torch.cat([torch.stack(batch) for batch in class_probs])

    test_preds = torch.cat(class_preds) 

    test_labels = torch.cat(class_labels) 

    return test_probs, test_preds, test_labels
# Run Eval

test_probs, test_preds, test_labels = metric_eval(testloader, net)
print(test_preds)
# confusion matrix

preds, targets = test_preds.to('cpu').numpy(), test_labels.to('cpu').numpy()

cm = metrics.confusion_matrix(targets, preds)



plt.figure(figsize=(10,10))

sns.heatmap(cm,annot=True)
#classification report

print(metrics.classification_report(targets, preds))