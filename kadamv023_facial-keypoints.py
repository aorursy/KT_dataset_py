from torchvision import transforms

import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.autograd import Variable

import numpy as np

import cv2

import matplotlib.pyplot as plt

from torch.utils.data import Dataset,DataLoader

import pandas as pd

import matplotlib.image as mpimage

import os
class FacialDetectionDataset(Dataset):

    def __init__(self,csv_file,root_dir,transform=None):

        self.key_pts_frame = pd.read_csv(csv_file)

        self.root_dir = root_dir

        self.transform = transform

        

    def __len__(self):

        return len(self.key_pts_frame)

    

    def __getitem__(self,idx):

        image_name = self.key_pts_frame.iloc[idx,0]

        image = mpimage.imread(os.path.join(self.root_dir,image_name))

        

        if image.shape[2] == 4:

            image = image[:,:,0:3]

            

        key_pts = self.key_pts_frame.iloc[idx,1:].as_matrix()

        key_pts = key_pts.astype('float').reshape(-1,2)

        

        sample = {'image':image,'key_pts':key_pts}

        

        if self.transform is not None:

            sample = self.transform(sample)

            

        return sample
class Normalise(object):

    def __call__(self,sample):

        image,key_pts = sample['image'],sample['key_pts']

        

        image_copy = np.copy(image)

        key_pts_copy = np.copy(key_pts)

        

        image_copy = cv2.cvtColor(image_copy,cv2.COLOR_RGB2GRAY)

        

        image_copy = image_copy/255.0

        image_copy = image_copy.reshape(1,image_copy.shape[0],image_copy.shape[1])

        

        key_pts_copy = (key_pts_copy - 100)/50

        

        return {'image':image_copy,'key_pts':key_pts_copy}
class Rescale(object):

    def __init__(self,output_size):

        assert isinstance(output_size,(int,tuple))

        self.output_size = output_size

        

    def __call__(self,sample):

        image , key_pts = sample['image'],sample['key_pts']

        

        h,w = image.shape[:2]

        

        if isinstance(self.output_size,int):

            if h>w:

                new_h , new_w = self.output_size * h/w , self.output_size

            else:

                new_h , new_w = self.output_size , self.output_size * w/h

        else:

            new_h , new_w = self.output_size

            

        new_h , new_w = int(new_h) , int(new_w)

        

        img = cv2.resize(image,(new_h,new_w))

        

        key_pts = key_pts * [new_w/w , new_h/h]

        

        return {'image':img,'key_pts':key_pts}

    
class RandomCrop(object):

    def __init__(self,output_size):

        assert isinstance(output_size,(int,tuple))

        if isinstance(output_size , int):

            self.output_size = (output_size,output_size)

        else:

            assert len(output_size)==2

            self.output_size = output_size

            

    def __call__(self,sample):

        image , key_pts = sample['image'],sample['key_pts']

        

        h , w = image.shape[:2]

        

        new_h , new_w = self.output_size

        

        top = np.random.randint(0,h - new_h)

        left = np.random.randint(0,w - new_w)

        

        image = image[top:top+new_h , left:left+new_w]

        key_pts = key_pts - [left,top]

        

        return {'image':image,'key_pts':key_pts}


class ToTensor(object):

    def __call__(self,sample):

        image , key_pts = sample['image'],sample['key_pts']

        

        if image.shape == 2:

            image = image.reshape(image.shape[0],image.shape[1],1)

            

        return {'image':torch.from_numpy(image),'key_pts':torch.from_numpy(key_pts)}
data_transform = transforms.Compose([Rescale((96,96)),Normalise(),ToTensor()])
dataset = FacialDetectionDataset(csv_file='../input/data/data/training_frames_keypoints.csv',root_dir='../input/data/data/training/',

                                transform=data_transform)
kwargs = {'num_workers' : 4} if torch.cuda.is_available() else {}
len(dataset)
batch_size = 128



train_loader = DataLoader(dataset,batch_size = batch_size , shuffle = True , **kwargs)
len(train_loader)
test_dataset = FacialDetectionDataset(csv_file = '../input/data/data/test_frames_keypoints.csv',root_dir = '../input/data/data/test/',

                                      transform= data_transform)
test_loader = DataLoader(test_dataset , batch_size= batch_size  , **kwargs)
class Network(nn.Module):

    def __init__(self):

        super(Network,self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1 ,out_channels= 32 , kernel_size=4 ,stride=1)

        self.conv2 = nn.Conv2d(in_channels=32 ,out_channels= 46 , kernel_size =3 ,stride=1)

        self.conv3 = nn.Conv2d(in_channels=46 ,out_channels = 128 , kernel_size = 2 ,stride =1)

        self.conv4 = nn.Conv2d(in_channels=128 ,out_channels = 256,kernel_size =1 ,stride =1)

        

        self.pool = nn.MaxPool2d(kernel_size=2 , stride =2)

        

        #self.fc1 = nn.Linear(128*6*6,6400)

        self.fc2 = nn.Linear(6400,1000)

        self.fc3 = nn.Linear(1000,500)

        self.fc4 = nn.Linear(500,136)

        

        #self.d1 = nn.Dropout(p = 0.1)

#         self.d2 = nn.Dropout(p = 0.1)

#         self.d3 = nn.Dropout(p = 0.1)

#         self.d4 = nn.Dropout(p = 0.1)

        self.d5 = nn.Dropout(p = 0.1)

        self.d6 = nn.Dropout(p = 0.1)

        #self.d7 = nn.Dropout(p = 0.2)

        

    def forward(self,x):

        out = self.conv1(x)

        out = F.elu(out)

        out = self.pool(out)

        #out = self.d1(out)

        

        out = self.conv2(out)

        out = F.elu(out)

        out = self.pool(out)

        #out = self.d2(out)

        

        out = self.conv3(out)

        out = F.elu(out)

        out = self.pool(out)

        #out = self.d3(out)

        

        out = self.conv4(out)

        out = F.elu(out)

        out = self.pool(out)

        #out = self.d4(out)

        

        out = out.view(out.size(0),-1)

        

#         out = self.fc1(out)

#         out = F.elu(out)

#         out = self.d5(out)

        

        out = self.fc2(out)

        out = F.elu(out)

        out = self.d5(out)

        

        out = self.fc3(out)

        out = F.elu(out)

        out = self.d6(out)

        

        out = self.fc4(out)

        return out
net = Network()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = net.to(device)
n_epochs = 20

criterion = nn.MSELoss()
learning_rate = 0.001

optimizer = torch.optim.Adam(net.parameters(),lr = learning_rate)
len(train_loader)
sample = dataset[500]

image , keypoints = sample['image'],sample['key_pts']

# print(image)

# print(keypoints)

image = image*255

keypoints = keypoints*50 + 100

# print(image)

# print(keypoints)

image= image.reshape(96,96)

plt.imshow(image,cmap='gray')

plt.scatter(keypoints[:, 0], keypoints[:, 1], s=20, marker='.', c='r')

plt.show()
iter = 0

# vis = Visdom()

# vis_window = vis.line(np.array([0]),np.array([0]))

for epoch in range(n_epochs):

    learning_loss =0.0

    count=0

    for i,sample in enumerate(train_loader):

        image , keypoints = sample['image'],sample['key_pts']

        image = Variable(image)

        image = image.type(torch.cuda.FloatTensor)

        keypoints = Variable(keypoints).type(torch.cuda.FloatTensor)

#         print(type(image))

#         print(type(keypoints))

        optimizer.zero_grad()

        

        outputs = net(image)

        outputs = outputs.reshape(outputs.shape[0],-1,2)

        loss = criterion(outputs,keypoints)

        loss.backward()

        

        optimizer.step()

        learning_loss +=(loss.data.item())

        iter += 1

        count +=1

        

#         vis.line(np.array([loss.item()]),np.array([iter]),win = vis_window ,update = 'append',opts =dict(xlabel='iterations',

#                                                                                                          ylabel='MSE error',

#                                                                                                          title='epoch loss and accuracy') )

        if i%10 == 0:

            print('Epoch: {}, Batch: {}, Avg. Loss: {}'.format(epoch + 1, i+1, learning_loss/1000))

            learning_loss = 0.0

            print(count)
sample = test_dataset[200]
image , keypoints = sample['image'] ,sample['key_pts']
# image = image*255

keypoints = keypoints*50 + 100
image = image.reshape(1,1,96,96)
image = image.type(torch.cuda.FloatTensor)
output = net.forward(image)
for param in net.parameters():

    param.requires_grad = False
image = image.cpu().numpy()

keypoints = keypoints.numpy()
image= image.reshape(96,96)
image = image*255

output = output*50 + 100
output = output.detach()
output = output.reshape(68,2)

output = output.cpu()
#print(output)
plt.imshow(image,cmap='gray')

plt.scatter(keypoints[:, 0], keypoints[:, 1], s=20, marker='.', c='r')

plt.scatter(output[:,0],output[:,1],s=20,marker='.',c='b')