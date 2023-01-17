import os
from glob import glob
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch.optim as optim

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
# check if machine has gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Running on",device)
# data path that training set is located
path = "../input/fruits/fruits-360/"
# this joins the path + folder and each files e.g. 'fruits/fruits-360/Training/Apple Braeburn/115_100.jpg'
files_training = glob(os.path.join(path,'Training', '*/*.jpg'))
num_images = len(files_training)
print('Number of images in Training file:', num_images)
# just to see how many images we have for each label, minimum one and average one, with nice printing style

min_images = 1000
im_cnt = []
class_names = []
print('{:18s}'.format('class'), end='')
print('Count:')
print('-' * 24)
for folder in os.listdir(os.path.join(path, 'Training')):
    folder_num = len(os.listdir(os.path.join(path,'Training',folder)))
    im_cnt.append(folder_num)
    class_names.append(folder)
    print('{:20s}'.format(folder), end=' ')
    print(folder_num)
        
num_classes = len(class_names)
print("\nMinumum images per category:", np.min(im_cnt), 'Category:', class_names[im_cnt.index(np.min(im_cnt))])    
print('Average number of Images per Category: {:.0f}'.format(np.array(im_cnt).mean()))
print('Total number of classes: {}'.format(num_classes))
fruit_data = pd.DataFrame(data = im_cnt,index = class_names,columns=["image_number"])
fruit_data.head()

top_ten = fruit_data.sort_values(by="image_number",ascending=False)[:10]
bottom_ten = fruit_data.sort_values(by="image_number",ascending=True)[:10]

frames = [top_ten, bottom_ten]
merged_tens = pd.concat(frames)

from sklearn.utils import shuffle
merged_tens = shuffle(merged_tens)

import seaborn as sns
plt.figure(figsize = (12,8))
chart = sns.barplot(x=merged_tens.index, y = merged_tens["image_number"],data=merged_tens, palette="Accent")
chart.set_xticklabels(chart.get_xticklabels(), rotation=45)
chart.set_ylabel("Number of Images")
plt.axhline(y=np.mean(im_cnt), color='r', linestyle='--',label = "Average Number of Images")
plt.legend()
plt.title("Number of Images for Top and Least 10 Fruits")
plt.show()
# Just to guess pop_mean and pop_std

tensor_transform = transforms.Compose([transforms.ToTensor()])

training_data = ImageFolder(os.path.join(path, 'Training'), tensor_transform)

data_loader = torch.utils.data.DataLoader(training_data, batch_size=512, shuffle=True)


# this part takes a bit long so I am using latest estimates
pop_mean = [0.684091,0.5786672,0.5038491]    # normally it was [] (empty)
pop_std = [0.30335397,0.35989153,0.3913597]

# for i, data in tqdm(enumerate(data_loader, 0)):
#     numpy_image = data[0].numpy()
    
#     batch_mean = np.mean(numpy_image, axis=(0,2,3))
#     batch_std = np.std(numpy_image, axis=(0,2,3))
    
#     pop_mean.append(batch_mean)
#     pop_std.append(batch_std)

# pop_mean = np.array(pop_mean).mean(axis=0)
# pop_std = np.array(pop_std).mean(axis=0)


print(pop_mean)
print(pop_std)
np.random.seed(123)
shuffle = np.random.permutation(num_images)

# split validation images

split_val = int(num_images * 0.2)
print('Total number of images:', num_images)
print('Number images in validation set:',len(shuffle[:split_val]))
print('Number images in train set:',len(shuffle[split_val:]))
class FruitTrainDataset(Dataset):
    def __init__(self, files, shuffle, split_val, class_names, transform=transforms.ToTensor()):
        self.shuffle = shuffle
        self.class_names = class_names
        self.split_val = split_val
        self.data = np.array([files[i] for i in shuffle[split_val:]])
        self.transform=transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img = Image.open(self.data[idx])
        name = self.data[idx].split('/')[-2]
        y = self.class_names.index(name)
        img = self.transform(img)
            
        return img, y

class FruitValidDataset(Dataset):
    def __init__(self, files, shuffle, split_val, class_names, transform=transforms.ToTensor()):
        self.shuffle = shuffle
        self.class_names = class_names
        self.split_val = split_val
        self.data = np.array([files[i] for i in shuffle[:split_val]])
        self.transform=transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img = Image.open(self.data[idx])
        name = self.data[idx].split('/')[-2]
        y = self.class_names.index(name)
        img = self.transform(img)
            
        return img, y
    
class FruitTestDataset(Dataset):
    def __init__(self, path, class_names, transform=transforms.ToTensor()):
        self.class_names = class_names
        self.data = np.array(glob(os.path.join(path, '*/*.jpg')))
        self.transform=transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img = Image.open(self.data[idx])
        name = self.data[idx].split('/')[-2]
        y = self.class_names.index(name)
        img = self.transform(img)
            
        return img, y
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(pop_mean, pop_std) # These were the mean and standard deviations that we calculated earlier.
    ]),
    'Test': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(pop_mean, pop_std) # These were the mean and standard deviations that we calculated earlier.
    ]),
    'valid': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(pop_mean, pop_std) # These were the mean and standard deviations that we calculated earlier.
    ])
}

train_dataset = FruitTrainDataset(files_training, shuffle, split_val, class_names, data_transforms['train'])
valid_dataset = FruitValidDataset(files_training, shuffle, split_val, class_names, data_transforms['valid'])
test_dataset = FruitTestDataset("../input/fruits/fruits-360//Test", class_names, transform=data_transforms['Test'])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)
dataloaders = {'train': train_loader,
              'valid': valid_loader,
              'Test': test_loader}
dataset_sizes = {
    'train': len(train_dataset),
    'valid': len(valid_dataset),
    'Test': len(test_dataset)
}
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    inp = pop_std * inp + pop_mean
    inp = np.clip(inp, 0, 1)
    plt.figure(figsize = (12,8))
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)
inputs, classes = next(iter(train_loader))
out = make_grid(inputs)

cats = ['' for x in range(len(classes))]
for i in range(len(classes)):
    cats[i] = class_names[classes[i].item()]
    
imshow(out)
print(cats)
import random
x = random.randint(0,32)

# plotted for each channel
fig, (ax1, ax2, ax3) = plt.subplots(1,3,figsize = (12,8))
plt.axis('off')
ax1.imshow(inputs[x][0])
ax1.set_title("Original Image")
ax1.axis('off')
ax2.imshow(inputs[x][1])
ax2.set_title("Horizontal Flipped Image")
ax2.axis('off')
ax3.imshow(inputs[x][2])
ax3.set_title("Vertical Flipped Image")
ax2.axis('off')
# just to start from the basic NN and to observe how does it perform on data

class Net(nn.Module):
    def __init__(self):
        super().__init__() # initialize the parent class methods
        self.fc1 = nn.Linear(3*100*100, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 131)
        
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        
        return F.log_softmax(x,dim=1)
    
net = Net()
print(net)
# move network to GPU
net = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.7)
# let's train the network
def train(net):
    with open("model.log", "a") as f:
        for epoch in tqdm(range(20)):
            print("epoch {}".format(epoch))
            running_loss = 0.0
            correct = 0
            total = 0

            val_loss = 0.0
            val_cor = 0
            val_tot = 0

            for i,data in enumerate(train_loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].to(device), data[1].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs.view(-1,3*100*100))

                # in sample accuracy calculation
                _, predicted = torch.max(outputs, 1) 
                a = predicted == labels
                correct += np.count_nonzero(a.cpu())
                total += len(a)
                
                # validaion accuracy calculation
                val_data = next(iter(valid_loader))
                val_inputs, val_labels = val_data[0].to(device), val_data[1].to(device)
                val_outputs = net(val_inputs.view(-1,3*100*100))
                _, val_predicted = torch.max(val_outputs, 1) 
                b = val_predicted == val_labels
                val_cor += np.count_nonzero(b.cpu())
                val_tot += len(b)
                
                #print("Validation accuracy",val_cor/val_tot)

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
                val_loss = criterion(val_outputs, val_labels)
                val_loss += val_loss.item()

                if i % 20 == 19 :    # print every 20 mini-batches                  

                    #print('[%d, %5d] loss: %.3f, in sample accuracy: %.3f, val_loss: %.3f, val accuracy : %.3f' %(epoch, i + 1, running_loss / 5, correct/total,
                    #                                                       val_loss / 5, val_cor/val_tot))
                    
                    f.write(f"{float(epoch)},{float(correct/total)},{float(running_loss / 20)},{float(val_cor/val_tot)},{float(val_loss / 20)}\n")
                    
                    running_loss = 0.0
                    correct = 0
                    total = 0

                    val_loss = 0.0
                    val_cor = 0
                    val_tot = 0                   

        print('Finished Training')
    
train(net)
# to save the trained model
PATH = "fnn_net.pth"
torch.save(net.state_dict(),PATH)
model_data = pd.read_csv("model.log",names = ["epochs","accuracy","loss","validation_accuracy","validation_loss"])
model_data.head()
fig, ax = plt.subplots(figsize = (16,8))
ax.set_xlabel('Epochs')
ax.set_ylabel('Loss')
ax.plot(model_data["loss"], label = "In Sample Loss")
ax.plot(model_data["validation_loss"], label = "Validation Loss")
leg = ax.legend()
fig, ax = plt.subplots(figsize = (16,8))
ax.set_xlabel('Epochs')
ax.set_ylabel('Accuracy')
ax.plot(model_data["accuracy"], label = "In Sample Accuracy")
ax.plot(model_data["validation_accuracy"], label = "Validation Accuracy")
leg = ax.legend()
# just to show how saved models can be load

PATH = "fnn_net.pth"
net = Net().to(device)
net.load_state_dict(torch.load(PATH))
# Overal Accuracy
def test(net):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in tqdm(test_loader):
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images.view(-1,3*100*100))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test images: %d %%' % (
        100 * correct / total))

test(net)
# class wise accuracy
class_correct = list(0. for i in range(131))
class_total = list(0. for i in range(131))
with torch.no_grad():
    for data in tqdm(test_loader):
        images, labels = data[0].to(device), data[1].to(device)
        outputs = net(images.view(-1,3*100*100))
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        
        for i in range(len(labels)):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

for i in range(131):
    print('Accuracy of %5s : %2d %%' % (
    class_names[i], 100 * class_correct[i] / class_total[i]))
height = [(100 * class_correct[i] / class_total[i]) for i in range(131)]
count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
labels = ["0% - 10%", "10% - 20%", "20% - 30%", "30% - 40%", "40% - 50%", "50% - 60%", "60% - 70%", "70% - 80%", "80% - 90%", "90% - 100%", "100%"]

for i in range(131):
    group = height[i] /10
    count[int(group)] += 1

    y_pos = np.arange(len(labels))

plt.figure(figsize=(12,8))
plt.barh(y_pos, count, color = "skyblue")
plt.title('Number of Right Classified Fruits in Accuracy Range')
plt.xlabel('Number of Classes')
plt.ylabel('Accuracy Percentage Ranges')
plt.yticks(y_pos, labels)
plt.show()