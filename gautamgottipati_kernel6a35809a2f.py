import os

import torch

import pandas as pd

import numpy as np

from torch.utils.data import Dataset, random_split, DataLoader

from PIL import Image

import torchvision.models as models

import matplotlib.pyplot as plt

import torchvision.transforms as transforms

from sklearn.metrics import f1_score

import torch.nn.functional as F

import torch.nn as nn

from tqdm.notebook import tqdm

from torchvision.utils import make_grid

%matplotlib inline



print("Imported libraries...")

class dr(Dataset):

    def __init__(self,csv_file,root_dir,transform=None,test_set=False):

        self.df = pd.read_csv(csv_file)

        self.transform = transform

        self.root_dir = root_dir

        self.test_set = test_set

        

    def __len__(self):

        return len(self.df)

    

    def __getitem__(self,idx):

#         print("Came here",idx,type(idx))

        row = self.df.loc[int(idx)]

        if self.test_set == True:

                img_id, img_label = row['id_code'], 0

        else:

            img_id, img_label = row['id_code'], row['diagnosis']

        img_fname = self.root_dir + "/" + str(img_id) + ".png"

        img = Image.open(img_fname)

        if self.transform:

            img = self.transform(img)

        return img,img_label

    

class DRNet(nn.Module):

    def __init__(self,input_size,hidden_size,output_size):

        super(DRNet,self).__init__()

        self.layer1 = nn.Linear(input_size,hidden_size)

        self.layer2 = nn.Linear(hidden_size,hidden_size)

        self.layer3 = nn.Linear(hidden_size,output_size)

        

    def init_weights(self):

        nn.init.kaiming_normal_(self.layer1.weights)

        nn.init.kaiming_normal_(self.layer2.weights)

        

    def forward(self,x):

        out = self.layer1(x)

        out = F.relu(out)

        out = self.layer2(out)

        out = F.relu(out)

        out = self.layer3(out)

        return out

        

def single_image(image):

    showImage = image

    if CUDA:

        image =image.cuda()

    image = image.view(-1,input_size)

    output = net(image)

    print(output)

    _,predicted = torch.max(output.data,1)

    print("Prediction = {}".format(predicted[0]))

    return int(predicted[0])

#     show_sample(showImage,predicted,invert=False)

            



# Initialising data paths

print("Initialising paths...")

DATA_DIR = '../input/aptos2019-blindness-detection'



TRAIN_DIR = '../input/aptos2019-blindness-detection/train_images'

TEST_DIR = '../input/aptos2019-blindness-detection/test_images'



TRAIN_CSV = '../input/aptos2019-blindness-detection/train.csv'

TEST_CSV = '../input/aptos2019-blindness-detection/test.csv'



# labels

labels = {

    0: 'No DR',

    1: 'Mild',

    2: 'Moderate',

    3: 'Severe',

    4: 'Proliferative DR'

}



print("Transforming and spliting the dataset...")

transform = transforms.Compose([transforms.Resize((32,32)),transforms.ToTensor()])

dataset = dr(TRAIN_CSV,TRAIN_DIR,transform=transform)

test_dataset = dr(TEST_CSV,TEST_DIR,transform=transform,test_set=True)



torch.manual_seed(10)



val_percentage = 0.1

val_size = int(val_percentage * len(dataset))

train_size = len(dataset) - val_size



print("Validation set size = {0} , Training set size = {1}".format(val_size,train_size))



train_ds, val_ds = random_split(dataset, [train_size, val_size])



print("Size of training dataset = {0} and Validation dataset = {1} after spliting".format(len(train_ds),len(val_ds)))



batch_size = 64



train_dl = DataLoader(train_ds,batch_size,shuffle=True,num_workers=2,pin_memory=True)

val_dl = DataLoader(val_ds,batch_size*2 , num_workers = 4,pin_memory=True)



input_size = 3*32*32

output_size = 5

hidden_size = 1500



print("Creating object for the network")

# Create an object that represents your network

net = DRNet(input_size,hidden_size,output_size)

CUDA = torch.cuda.is_available()

if CUDA:

    net = net.cuda()

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)



print("Training network...")



epochs = 10



for epoch in range(epochs):

    correct_train = 0

    running_loss = 0

    for i,(images,labels) in tqdm(enumerate(train_dl)):

        images = images.view(-1,input_size)

        if CUDA:

            images = images.cuda()

            labels = labels.cuda()

        

        outputs = net(images)

#         print(outputs.data)

        _,predicted = torch.max(outputs.data,1)

#         print(predicted)

#         print(labels)

        correct_train += (predicted==labels).sum()

        loss = criterion(outputs,labels)

        running_loss += loss.item()

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        

    print('Epoch [{}/{}], Training Loss: {:.3f}, Training Accuracy: {:.3f}%'.format

          (epoch+1, epochs, running_loss/len(train_dl), (100*correct_train.double()/len(train_ds))))

print("Training Done Successfully")

    

print("Validation network...")

with torch.no_grad():

    correct = 0

    for images,labels in val_dl:

        if CUDA:

            images = images.cuda()

            labels = labels.cuda()

        images = images.view(-1,input_size)

        outputs = net(images)

        _,predicted = torch.max(outputs.data,1)

        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the {0} validation images: {1} %'.format(len(val_ds),100 * correct / len(val_ds)))

    

print("Testing dataset...")

mypreds = []

for i in range(len(test_dataset)):

    img, target = test_dataset[i]

    pred = single_image(img)

    print(pred)

    mypreds.append(pred)

    

submission_df = pd.read_csv(TEST_CSV)

submission_df['diagnosis'] = mypreds

sub_file = 'submission.csv'

submission_df.to_csv(sub_file,index=False)






