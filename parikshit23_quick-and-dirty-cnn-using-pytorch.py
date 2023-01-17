# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import torch
import numpy as np
import pandas as pd
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from skimage import io
from torchvision import transforms
%matplotlib inline
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import os
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import time
from tqdm.notebook import tqdm
from scipy.special import softmax
train_info = pd.read_csv("/kaggle/input/plant-pathology-2020-fgvc7/train.csv")
test_info = pd.read_csv("/kaggle/input/plant-pathology-2020-fgvc7/test.csv")

train_info['dep'] = np.where(train_info['healthy']== 1, 0,
                            np.where(train_info['multiple_diseases']== 1, 1,
                                    np.where(train_info['rust']== 1, 2,
                                            np.where(train_info['scab']== 1, 3,0))))
train_info.head()
train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('CUDA is not available. Training on CPU ...')
else:
    print('CUDA is available. Training on GPU ...')
# Data Augmentation
transform = transforms.Compose(
                   [transforms.Resize((1024,1024)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomRotation(degrees=30),
                    transforms.CenterCrop(512),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Function to load train data
class load_data(Dataset):
    def __init__(self,train_info,root_dir,transform=transform):
        self.train_info=train_info
        self.root_dir=root_dir
        self.transform=transform
    def __len__(self):
        return len(train_info)
    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_name = os.path.join(self.root_dir,self.train_info.iloc[idx,0])
        img_name = img_name+'.jpg'
        #print(img_name)
        image = Image.open(img_name)
        target = self.train_info['dep'].iloc[idx]
        #target = np.array([target])
        #target = target.astype('float').reshape(-1,4)
        if self.transform:
            image = self.transform(image)
        return image, target

# Function to load test data
class load_data_test(Dataset):
    def __init__(self,train_info,root_dir,transform=transform):
        self.test_info=test_info
        self.root_dir=root_dir
        self.transform=transform
    def __len__(self):
        return len(test_info)
    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_name = os.path.join(self.root_dir,self.test_info.iloc[idx,0])
        img_name = img_name+'.jpg'
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
        return image      


    
num_workers = 0

dset_train = load_data(train_info,'/kaggle/input/plant-pathology-2020-fgvc7/images/')
dset_test = load_data_test(test_info,'/kaggle/input/plant-pathology-2020-fgvc7/images/')
num_train = len(dset_train)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(0.2 * num_train))
train_idx, valid_idx = indices[split:], indices[:split]

# define samplers for obtaining training and validation batches
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

# prepare data loaders (combine dataset and sampler)
train_loader = DataLoader(dset_train,batch_size=15,sampler = train_sampler,num_workers=num_workers)
valid_loader = DataLoader(dset_train,batch_size=15,sampler = valid_sampler,num_workers=num_workers,drop_last=True)
test_loader = DataLoader(dset_test,batch_size=15,num_workers=num_workers)
def imshow(image, ax=None, title=None, normalize=True):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    image = image.numpy().transpose((1, 2, 0))

    if normalize:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)

    ax.imshow(image)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', length=0)
    ax.set_xticklabels('')
    ax.set_yticklabels('')

    return ax

dataiter=iter(test_loader)
images=dataiter.next()

fig, axes = plt.subplots(figsize=(10,4),ncols=5)
for i in range(5):
    ax = axes[i]
    imshow(images[i],ax=ax, normalize=False)

dataiter=iter(train_loader)
images,labels=dataiter.next()
print(type(images))
print(images.shape)
print(labels.shape)
# define the CNN architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # convolutional layer (sees 512x512x3 image tensor)
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        # convolutional layer (sees 256x256x16 tensor)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        # convolutional layer (sees 128x128x32 tensor)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        # max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        # linear layer (64 * 64 * 64 -> 1024)
        self.fc1 = nn.Linear(64*64*64, 1024)
        # linear layer (1024 -> 512)
        self.fc2 = nn.Linear(1024, 512)
        # linear layer (500 -> 4)
        self.fc3 = nn.Linear(512, 4)
        # dropout layer (p=0.25)
        self.dropout = nn.Dropout(0.40)
       

    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        # flatten image input
        x = x.view(x.shape[0],-1)
        # add dropout layer
        x = self.dropout(x)
        # add 1st hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        # add dropout layer
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        # add 2nd hidden layer, with relu activation function
        x = self.fc3(x)
        return x

# create a complete CNN
model = Net()
print(model)
# move tensors to GPU if CUDA is available
if train_on_gpu:
    model.cuda()
# specify loss function (categorical cross-entropy)
criterion = nn.CrossEntropyLoss()
# specify optimizer
optimizer = optim.SGD(model.parameters(), lr=0.05)
# number of epochs to train the model
n_epochs = 60

valid_loss_min = np.Inf # track change in validation loss

for epoch in range(1, n_epochs+1):
    t1=time.time()

    # keep track of training and validation loss
    train_loss = 0.0
    valid_loss = 0.0
    
    ###################
    # train the model #
    ###################
    model.train()
    for data, target in train_loader:
        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the batch loss
        loss = criterion(output, target)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update training loss
        train_loss += loss.item()*data.size(0)
        
    ######################    
    # validate the model #
    ######################
    model.eval()
    for data, target in valid_loader:
        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the batch loss
        loss = criterion(output, target)
        # update average validation loss 
        valid_loss += loss.item()*data.size(0)
    
    # calculate average losses
    train_loss = train_loss/len(train_loader.sampler)
    valid_loss = valid_loss/len(valid_loader.sampler)
        
    # print training/validation statistics 
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} \tTime Taken: {:.6f}'.format(
        epoch, train_loss, valid_loss,(time.time() - t1)))
    
    # save model if validation loss has decreased
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
        valid_loss_min,
        valid_loss))
        torch.save(model.state_dict(), 'plant_path.pt')
        valid_loss_min = valid_loss

classes = ['healthy', 'multiple_diseases', 'rust', 'scab']
# track test loss
batch_size = 15
valid_loss = 0.0
class_correct = list(0. for i in range(4))
class_total = list(0. for i in range(4))

model.eval()
# iterate over test data
for data, target in valid_loader:
    # move tensors to GPU if CUDA is available
    if train_on_gpu:
        data, target = data.cuda(), target.cuda()
    # forward pass: compute predicted outputs by passing inputs to the model
    
    output = model(data)
    # calculate the batch loss
    loss = criterion(output, target)
    # update test loss 
    valid_loss += loss.item()*data.size(0)
    # convert output probabilities to predicted class
    _, pred = torch.max(output, 1)    
    # compare predictions to true label
    correct_tensor = pred.eq(target.data.view_as(pred))
    correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
    
    # calculate test accuracy for each object class
    for i in range(batch_size):
        label = target.data[i]
        class_correct[label] += correct[i].item()
        class_total[label] += 1

# average test loss
valid_loss = valid_loss/len(valid_loader.dataset)
print('Test Loss: {:.6f}\n'.format(valid_loss))

for i in range(4):
    if class_total[i] > 0:
        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
            classes[i], 100 * class_correct[i] / class_total[i],
            np.sum(class_correct[i]), np.sum(class_total[i])))
    else:
        print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
    100. * np.sum(class_correct) / np.sum(class_total),
    np.sum(class_correct), np.sum(class_total)))
def test_fn(net, loader):
    preds_for_output = np.zeros((1,4))
    with torch.no_grad():
        pbar = tqdm(total = len(loader))
        for _, images in enumerate(loader):
            images = images.to('cuda')
            net.eval()
            predictions = net(images)
            preds_for_output = np.concatenate((preds_for_output, predictions.cpu().detach().numpy()), 0)
            pbar.update()
    pbar.close()
    return preds_for_output
out = test_fn(model, test_loader)
output = pd.DataFrame(softmax(out,1), columns = ['healthy','multiple_diseases','rust','scab']) #the submission expects probability scores for each class
output.drop(0, inplace = True)
output.reset_index(drop=True,inplace=True)
output['image_id'] = test_info.image_id
output = output[['image_id','healthy','multiple_diseases','rust','scab']]
output.head()