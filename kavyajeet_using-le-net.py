# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, mean_squared_error, log_loss

from sklearn.preprocessing import OneHotEncoder

from PIL import Image



import torch

import torch.nn as nn

import torch.nn.functional as F

from torch import optim

from torch.utils.data import Dataset, DataLoader

import torchvision

import torchvision.transforms as transforms



import copy

import warnings

warnings.filterwarnings('ignore')



from tqdm import tqdm_notebook

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/train.csv')

data.head()
X_data = data.iloc[:,1:].values

Y_data = data.iloc[:,0].values

def normalize_image_data(X_data):

    maximum = np.max(X_data)

    minimum = np.min(X_data)

    images = []

    for i in X_data:

        i = i.reshape(28,28)

        i = (i-minimum)/(maximum-minimum)

        images.append(i)

    return np.array(images)

normalized_images = normalize_image_data(X_data)



index = np.random.randint(0,len(normalized_images))

print(len(normalized_images))

fig, (ax1, ax2) = plt.subplots(1, 2)

fig.set_size_inches(15,5)

ax1.hist(normalized_images[index].reshape(-1,1))

ax2.imshow(normalized_images[index])

ax2.set_title(Y_data[index])

plt.show()
torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

device
class Image_Dataset(Dataset):

  

    def __init__(self, X,Y, transform=None):

        self.x_data = torch.from_numpy(X).float()

        self.x_data = self.x_data.view(-1,1,28,28)

        self.y_data = torch.from_numpy(Y).long()

        self.len = self.x_data.shape[0]

    

    def __getitem__(self,item):

        return self.x_data[item], self.y_data[item]



    def __len__(self):

        return self.len

    

image_train_dataset = Image_Dataset(X_data,Y_data)

print(type(image_train_dataset.x_data))

print('X input shape:',image_train_dataset.x_data.size())

print('Y labels shape:',image_train_dataset.y_data.size())
batch_size = 10500

train_loader = DataLoader(image_train_dataset, batch_size=batch_size,shuffle=True)

dataiter = iter(train_loader)

images, labels = dataiter.next()



def show_image(img):

    img = img.numpy()

    img = np.transpose(img,(1,2,0))

    plt.imshow(img)

    plt.show()

  

show_image(torchvision.utils.make_grid(images))

print(labels)
class LeNet(nn.Module):

    

    def __init__(self):

        super(LeNet,self).__init__()

        

        self.conv = nn.Sequential(

            nn.Conv2d(1,6,5),

            nn.BatchNorm2d(6),

            nn.LeakyReLU(),

            nn.AvgPool2d(2,stride=2),

            nn.Conv2d(6,16,5),

            nn.BatchNorm2d(16),

            nn.LeakyReLU(),

            nn.AvgPool2d(2,stride=2)

        )

        

        self.fc = nn.Sequential(

            nn.Linear(256,120),

            nn.Dropout(0.5),

            nn.LeakyReLU(),

            nn.Linear(120,84),

            nn.Dropout(0.5),

            nn.LeakyReLU(),

            nn.Linear(84,10)

        )

        

    def forward(self,X):

        x = self.conv(X)

        x = x.view(x.size(0),-1)

        x = self.fc(x)

        return x
def return_accuracy(dataloader,model):

    total = 0

    correct = 0

    

    for images, labels in dataloader:

        images,labels = images.to(device),labels.to(device)

        y_pred = model(images)

        total += batch_size

        correct += torch.sum((torch.argmax(y_pred,dim=1) == labels)).item()

    return correct/total*100
def fit(dataloader_train, model, opt, loss_fn, epochs=15):

    

    # checkpointing

    best_accuracy = 0   

    iter_no = 0

    for epoch in tqdm_notebook(range(epochs),unit=' Epoch', total=epochs):

        best_model = model.state_dict()

        correct = 0

        total = 0

        

        for images,labels in dataloader_train:

            images,labels = images.float().to(device), labels.long().to(device)

            train_pred = model(images)

            

            ## compute the accuracy

            total += batch_size

            correct += torch.sum(torch.argmax(train_pred,dim=1) == labels).item()

            accuracy = correct/total*100

            

            loss = loss_fn(train_pred,labels)

            loss.backward()

            opt.step()

            opt.zero_grad()



            del images,labels,train_pred

            torch.cuda.empty_cache()

            iter_no += batch_size

            

            if best_accuracy < accuracy:

                best_accuracy = accuracy

                best_model = copy.deepcopy(model.state_dict())

                

                if best_accuracy % 100 == 0:

                    print('Accuracy:',accuracy,'at iteration:',iter_no,' | Loss:',loss.item())

                    break

        

    model.load_state_dict(best_model)

    return model
%%time



model = LeNet()

model.to(device)

opt = optim.Adam(model.parameters(),lr=0.001,betas=(0.9,0.99))

loss_fn = F.cross_entropy

model = fit(train_loader,model,opt,loss_fn,epochs=100)
return_accuracy(train_loader,model)
data_test = pd.read_csv('../input/test.csv')

data_test.head()
X_test = data_test.iloc[:,:].values

X_test = normalize_image_data(X_test)

X_test = torch.tensor(X_test).float().view(-1,1,28,28).to(device)

X_test.shape
Y_pred_test = torch.argmax(model(X_test),dim=1)



submission = {}

submission['ImageId'] = data_test.index.values+1

submission['Label'] = Y_pred_test.cpu().numpy()

submission = pd.DataFrame(submission)

submission.to_csv("submisision.csv", index=False)