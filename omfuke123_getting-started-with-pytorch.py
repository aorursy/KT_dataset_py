import numpy as np 

import pandas as pd 

from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms

import matplotlib.pyplot as plt

import torch



import warnings

warnings.filterwarnings("ignore")

pd.set_option("display.max_columns",None)



import torch.nn as nn

PATH = "../input/digit-recognizer/"
train = pd.read_csv(PATH+"train.csv")

test = pd.read_csv(PATH+"test.csv")

ss = pd.read_csv(PATH+"sample_submission.csv")
train.head()
test.head()
plt.imshow(np.array(train.iloc[4,1:]).reshape(-1,28),cmap="gray")
input_size = 28*28

learning_rate = 0.001

hidden_size = 100

num_classes = 10

num_epochs = 20

batch_size = 100
class TRAIN_DATASET(Dataset):

    

        

    def __init__(self,csv_file,transform = None):

        self.data = pd.read_csv(csv_file)

        self.transform = transform

        

    def __len__(self):

        return len(self.data)

        

    def __getitem__(self,index):

        

        ##reshaping

        img = np.array(self.data.iloc[index,1:]).reshape((1,28*28))

        label = np.array(self.data.iloc[index,0])

        img = torch.from_numpy(img).float()

        label = torch.from_numpy(label).type(torch.LongTensor)

        

        

        sample = {"image":img,"label":label}

        

        return sample

    
class ToTensor(object):

    """Convert ndarrays in sample to Tensors."""



    def __call__(self, sample):

        image, label = sample['image'], sample['label']



        # swap color axis because

        # numpy image: H x W x C

        # torch image: C X H X W

        return {'image': torch.from_numpy(image),

                'label': torch.from_numpy(label)}
train_dataset = TRAIN_DATASET(PATH+"train.csv",transform = None)

class TEST_DATASET(Dataset):

    

        

    def __init__(self,csv_file,transform = None):

        self.data = pd.read_csv(csv_file)

        self.transform = transform

        

    def __len__(self):

        return len(self.data)

        

    def __getitem__(self,index):

        

        ##reshaping

        img = np.array(self.data.iloc[index,:]).reshape((1,28*28))

        img = torch.from_numpy(img).float()

                       

        return img

    
test_dataset = TEST_DATASET(PATH+"test.csv",transform = None)
train_loader = DataLoader(train_dataset, batch_size=100,

                        shuffle=True, num_workers=4)

                        

test_loader = DataLoader(test_dataset,batch_size = 100,shuffle=False,num_workers = 4) 
example = next(iter(train_loader))

image = example['image'].view(-1,28*28)

class Net(nn.Module):

    def __init__(self,input_size,hidden_size, num_classes):

        super(Net, self).__init__()

        self.l1 = nn.Linear(input_size,hidden_size)

        self.relu = nn.ReLU()

        self.l2 = nn.Linear(hidden_size,num_classes)

        

    def forward(self, x):

        out  = self.l1(x)

        out = self.relu(out)

        out = self.l2(out)

        

        return out

model = Net(input_size,hidden_size,num_classes)



criterion  = nn.CrossEntropyLoss()



optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
n_total_steps = int(len(train_loader)/100)

count = 0

loss_list = []

iteration_list = []

accuracy_list = []



for i in range(num_epochs):

    for j,sample in enumerate(train_loader):

        images = sample['image']

     

        labels = sample['label']

        

        images = images.view(-1,28*28)

        

        outputs = model(images)

        loss = criterion(outputs,labels)

        

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        

        

        print(f'epoch {i+1} / {num_epochs} , step{j+1}/ {n_total_steps},loss = {loss.item():.4f}')

    

    if i%4==0:

        

        torch.save(model.state_dict(),f'model{i}.pth')