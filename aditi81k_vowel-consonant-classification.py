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

from PIL import Image

import matplotlib.pyplot as plt



import torch

import torchvision

from torch.utils.data import DataLoader, Dataset, random_split

import torchvision.transforms as transforms



#For converting the dataset to torchvision dataset format

class VowelConsonantDataset(Dataset):

    def __init__(self, file_path,train=True,transform=None):

        self.transform = transform

        self.file_path=file_path

        self.train=train

        self.file_names=[file for _,_,files in os.walk(self.file_path) for file in files]

        self.len = len(self.file_names)

        if self.train:

            self.classes_mapping=self.get_classes()

    def __len__(self):

        return len(self.file_names)

    

    def __getitem__(self, index):

        file_name=self.file_names[index]

        image_data=self.pil_loader(self.file_path+"/"+file_name)

        if self.transform:

            image_data = self.transform(image_data)

        if self.train:

          file_name_splitted=file_name.split("_")

          Y1 = self.classes_mapping[file_name_splitted[0]]

          Y2 = self.classes_mapping[file_name_splitted[1]]

          z1,z2=torch.zeros(10),torch.zeros(10)

          z1[Y1-10],z2[Y2]=1,1

          label=torch.stack([z1,z2])



          return image_data, label



        else:

          return image_data, file_name

          

    def pil_loader(self,path):

      with open(path, 'rb') as f:

        img = Image.open(f)

        return img.convert('RGB')



      

    def get_classes(self):

        classes=[]

        for name in self.file_names:

            name_splitted=name.split("_")

            classes.extend([name_splitted[0],name_splitted[1]])

        classes=list(set(classes))

        classes_mapping={}

        for i,cl in enumerate(sorted(classes)):

            classes_mapping[cl]=i

        return classes_mapping
# check if CUDA is available

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

train_on_gpu = torch.cuda.is_available()



if not train_on_gpu:

    print('CUDA is not available.  Training on CPU ...')

else:

    print('CUDA is available!  Training on GPU ...')
!unzip ../input/padhai-hindi-vowel-consonant-classification/train.zip

!unzip ../input/padhai-hindi-vowel-consonant-classification/test.zip
# define training and test data directories

data_dir = '../input/output'

train_dir = os.path.join(data_dir, 'train/')

test_dir = os.path.join(data_dir, 'test/')
data_transform = transforms.Compose([

    transforms.Resize(224),

    transforms.ToTensor(),

    transforms.Normalize(mean=[0.485, 0.456, 0.406],

                                 std=[0.229, 0.224, 0.225]),

    ])
# load and transform data

data = VowelConsonantDataset('train',train=True,transform=data_transform)

train_size = int(0.85 * len(data))

test_size = len(data) - train_size 

train_data, validation_data = random_split(data, [train_size, test_size])

test_data=VowelConsonantDataset('test',train=False,transform=data_transform)



# print out some data stats

print('Num training images: ', train_size)

print('Num validation images: ', test_size)

print('Num test images: ', len(test_data))
# define dataloader parameters

batch_size = 20

num_workers=1



# prepare data loaders

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, 

                                           num_workers=num_workers, shuffle=True)

validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=batch_size, 

                                          num_workers=num_workers, shuffle=True)

test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, 

                                          num_workers=num_workers, shuffle=True)
# Visualize some sample data



# obtain one batch of training images

dataiter = iter(train_loader)

images, labels = dataiter.next()

images = images.numpy() # convert images to numpy for display



# plot the images in the batch, along with the corresponding labels

fig = plt.figure(figsize=(25, 4))

for idx in np.arange(20):

    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])

    plt.imshow(np.transpose(images[idx], (1, 2, 0)))

    # ax.set_title([labels[idx]])
from torchvision import models

import torch.nn as nn

import torch.nn.functional as F

import copy

class MyModel(nn.Module):

    def __init__(self, num_classes1, num_classes2):

        super(MyModel, self).__init__()

        self.model_resnet50 = models.resnet50(pretrained=True)

        final_in_features = self.model_resnet50.fc.in_features

        self.model_resnet50.fc = nn.Sequential()

        self.fc1 = nn.Sequential(

              nn.BatchNorm1d(final_in_features),

              nn.Dropout(0.3),

              nn.Linear(in_features=final_in_features, out_features=1000,bias=True),

              nn.ReLU(),

              nn.BatchNorm1d(1000, eps=1e-05, momentum=0.3),

              nn.Dropout(0.3),

              nn.Linear(in_features=1000,out_features=num_classes1,bias=True))

        self.fc2 = nn.Sequential(

              nn.BatchNorm1d(final_in_features), 

              nn.Dropout(0.3),

              nn.Linear(in_features=final_in_features,out_features=1000,bias=True),

              nn.ReLU(),

              nn.BatchNorm1d(1000, eps=1e-05, momentum=0.3),

              nn.Dropout(0.3),

              nn.Linear(in_features=1000, out_features=num_classes2,bias=True))



    def forward(self, x):

        x = self.model_resnet50(x)

        out1 = self.fc1(x)

        out2 = self.fc2(x)

        return out1, out2
net  = MyModel(10,10)

net = net.to(device)
import torch.optim as optim



# specify loss function (categorical cross-entropy)

criterion = nn.CrossEntropyLoss()



# specify optimizer (stochastic gradient descent) and learning rate = 0.001

optimizer = optim.SGD(net.parameters(), lr=0.01,momentum=0.9)
def evaluation(dataloader,model):

    total,correct=0,0

    for data in dataloader:

        inputs,labels=data

        inputs,labels=inputs.to(device),labels.to(device)

        out1,out2=model(inputs)

        _,pred1=torch.max(out1.data,1)

        _,pred2=torch.max(out2.data,1)

        _,labels1=torch.max(labels[:,0,:].data,1)

        _,labels2=torch.max(labels[:,1,:].data,1)

        total+=labels.size(0)

        fin1=(pred1==labels1)

        fin2=(pred2==labels2)

        

        correct+=(fin1==fin2).sum().item()

    return 100*correct/total
loss_epoch_arr = []

loss_arr = []

max_epochs = 6

min_loss = 1000

batch_size = 32

n_iters = np.ceil(9000/batch_size)

for epoch in range(max_epochs):

    for i, data in enumerate(train_loader, 0):

        net.train()

        images, labels = data

#         print(images.shape)

        images = images.to(device)

        targetnp=labels[:,0,:].cpu().numpy()

        targetnp1 = labels[:,1,:].cpu().numpy()

        # Convert predictions classes from one hot vectors to labels: [0 0 1 0 0 ...] --> 2

        with torch.no_grad():

            new_targets1 = np.argmax(targetnp,axis=1)

            new_targets2 = np.argmax(targetnp1,axis=1)

        new_targets1=torch.LongTensor(new_targets1)

        new_targets2=torch.LongTensor(new_targets2)

        new_targets1 = new_targets1.to(device)

        new_targets2 = new_targets2.to(device)

        optimizer.zero_grad()

        out = net.forward(images)

        loss_fc1 = criterion(out[0], new_targets1)

        loss_fc2 = criterion(out[1],new_targets2)

        loss = torch.add(loss_fc1,loss_fc2)

        loss.backward()

        optimizer.step()   

        if min_loss > loss.item():

            min_loss = loss.item()

            best_model = copy.deepcopy(net.state_dict())

            print('Min loss %0.2f' % min_loss)

#         if min_loss < 0.8:

#             opt = optim.SGD(my_model.parameters(),lr=0.01,momentum=0.99,nesterov=True)

        if i % 100 == 0:

            print('Iteration: %d/%d, Loss: %0.2f' % (i, n_iters, loss.item()))

        del images, labels, out

        torch.cuda.empty_cache()

        loss_arr.append(loss.item())

    print("Epoch number :",epoch)

    print("Train Accuracy :",evaluation(train_loader,net))

    print("Test Accuracy :"  ,evaluation(validation_loader,net))

    loss_epoch_arr.append(loss.item())

#     my_model.load_state_dict(best_model)

plt.plot(loss_arr)

plt.show()
print(evaluation(validation_loader,net))
net.eval()

plist=[]

final_list=[]

for inputs_test, fn in test_loader:

    inputs_test=inputs_test.to(device)

    out1,out2=net.forward(inputs_test)

    _,pred1=torch.max(out1,1)

    pred1=pred1.tolist()

    _,pred2=torch.max(out2,1)

    pred2=pred2.tolist()

    for x,y,z in zip(pred1,pred2,fn):

        p="V"+str(x)+"_"+"C"+str(y)

        plist.append(p)

        final_list.append(z)
submission = pd.DataFrame({"ImageId":final_list, "Class":plist})

submission.to_csv('submission.csv', index=False)