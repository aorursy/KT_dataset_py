# Import all the necessary Library 

import torchvision

import torch.utils.data as utils

from torchvision import datasets

import torchvision.transforms as transforms

from torch.utils.data import DataLoader,Dataset

from torch.autograd import Variable

import matplotlib.pyplot as plt

import torchvision.utils

import numpy as np

import time

import copy

from torch.optim import lr_scheduler

import os

from PIL import Image

import torch

from torch.autograd import Variable

import PIL.ImageOps    

import torch.nn as nn

from torch import optim

import torch.nn.functional as F

import pandas as pd 

def imshow(img,text=None,should_save=False):

    npimg = img.numpy()

    plt.axis("off")

    if text:

        plt.text(75, 8, text, style='italic',fontweight='bold',

            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})

    plt.imshow(np.transpose(npimg, (1, 2, 0)))

    plt.show()    



def show_plot(iteration,loss):

    plt.plot(iteration,loss)

    plt.show()
# First, look at everything.

from subprocess import check_output

print(check_output(["ls", "../input/sign_data/sign_data"]).decode("utf8"))



class Config():

    training_dir = "../input/sign_data/sign_data/train"

    testing_dir = "../input/sign_data/sign_data/test"

    train_batch_size = 32

    train_number_epochs = 20
training_dir="../input/sign_data/sign_data/train"

training_csv="../input/sign_data/sign_data/train_data.csv"

testing_csv="../input/sign_data/sign_data/test_data.csv"

testing_dir="../input/sign_data/sign_data/test"
class SiameseNetworkDataset():

    

    def __init__(self,training_csv=None,training_dir=None,transform=None):

        # used to prepare the labels and images path

        self.training_df=pd.read_csv(training_csv)

        self.training_df.columns =["image1","image2","label"]

        self.training_dir = training_dir    

        self.transform = transform



    def __getitem__(self,index):

        

        # getting the image path

        image1_path=os.path.join(self.training_dir,self.training_df.iat[index,0])

        image2_path=os.path.join(self.training_dir,self.training_df.iat[index,1])

        

        

        # Loading the image

        img0 = Image.open(image1_path)

        img1 = Image.open(image2_path)

        img0 = img0.convert("L")

        img1 = img1.convert("L")

        

        # Apply image transformations

        if self.transform is not None:

            img0 = self.transform(img0)

            img1 = self.transform(img1)

        

        return img0, img1 , torch.from_numpy(np.array([int(self.training_df.iat[index,2])],dtype=np.float32))

    

    def __len__(self):

        return len(self.training_df)
# Load the the dataset from raw image folders

siamese_dataset = SiameseNetworkDataset(training_csv,training_dir,

                                        transform=transforms.Compose([transforms.Resize((105,105)),

                                                                      transforms.ToTensor()

                                                                      ])

                                       )
# Viewing the sample of images and to check whether its loading properly

vis_dataloader = DataLoader(siamese_dataset,

                        shuffle=True,

                        batch_size=8)

dataiter = iter(vis_dataloader)





example_batch = next(dataiter)

concatenated = torch.cat((example_batch[0],example_batch[1]),0)

imshow(torchvision.utils.make_grid(concatenated))

print(example_batch[2].numpy())
class SiameseNetwork(nn.Module):

    def __init__(self):

        super(SiameseNetwork, self).__init__()

        

        # Setting up the Sequential of CNN Layers

        self.cnn1 = nn.Sequential(

            

            nn.Conv2d(1, 96, kernel_size=11,stride=1),

            nn.ReLU(inplace=True),

            nn.LocalResponseNorm(5,alpha=0.0001,beta=0.75,k=2),

            nn.MaxPool2d(3, stride=2),

            

            nn.Conv2d(96, 256, kernel_size=5,stride=1,padding=2),

            nn.ReLU(inplace=True),

            nn.LocalResponseNorm(5,alpha=0.0001,beta=0.75,k=2),

            nn.MaxPool2d(3, stride=2),

            nn.Dropout2d(p=0.3),



            nn.Conv2d(256,384 , kernel_size=3,stride=1,padding=1),

            nn.ReLU(inplace=True),

            nn.Conv2d(384,256 , kernel_size=3,stride=1,padding=1),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(3, stride=2),

            nn.Dropout2d(p=0.3),



        )

        

        # Defining the fully connected layers

        self.fc1 = nn.Sequential(

            nn.Linear(30976, 1024),

            nn.ReLU(inplace=True),

            nn.Dropout2d(p=0.5),

            

            nn.Linear(1024, 128),

            nn.ReLU(inplace=True),

            

            nn.Linear(128,2))

        

  

  

    def forward_once(self, x):

        # Forward pass 

        output = self.cnn1(x)

        output = output.view(output.size()[0], -1)

        output = self.fc1(output)

        return output



    def forward(self, input1, input2):

        # forward pass of input 1

        output1 = self.forward_once(input1)

        # forward pass of input 2

        output2 = self.forward_once(input2)

        return output1, output2

class ContrastiveLoss(torch.nn.Module):

    """

    Contrastive loss function.

    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf

    """



    def __init__(self, margin=2.0):

        super(ContrastiveLoss, self).__init__()

        self.margin = margin



    def forward(self, output1, output2, label):

        euclidean_distance = F.pairwise_distance(output1, output2)

        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +

                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))





        return loss_contrastive
# Load the dataset as pytorch tensors using dataloader

train_dataloader = DataLoader(siamese_dataset,

                        shuffle=True,

                        num_workers=8,

                        batch_size=Config.train_batch_size)
# Check whether you have GPU is loaded or not

if torch.cuda.is_available():

    print('Yes')
# Declare Siamese Network

net = SiameseNetwork().cuda()

# Decalre Loss Function

criterion = ContrastiveLoss()

# Declare Optimizer

optimizer = optim.RMSprop(net.parameters(), lr=1e-4, alpha=0.99, eps=1e-8, weight_decay=0.0005, momentum=0.9)
def train():

    counter = []

    loss_history = [] 

    iteration_number= 0

    

    for epoch in range(0,Config.train_number_epochs):

        for i, data in enumerate(train_dataloader,0):

            img0, img1 , label = data

            img0, img1 , label = img0.cuda(), img1.cuda() , label.cuda()

            optimizer.zero_grad()

            output1,output2 = net(img0,img1)

            loss_contrastive = criterion(output1,output2,label)

            loss_contrastive.backward()

            optimizer.step()

            if i %50 == 0 :

                print("Epoch number {}\n Current loss {}\n".format(epoch,loss_contrastive.item()))

                iteration_number +=10

                counter.append(iteration_number)

                loss_history.append(loss_contrastive.item())

    return net
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Train the model

model = train()

torch.save(model.state_dict(), "model.pt")

print("Model Saved Successfully")
# Load the saved model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = SiameseNetwork().to(device)

model.load_state_dict(torch.load("model.pt"))
# Load the test dataset

test_dataset = SiameseNetworkDataset(training_csv=testing_csv,training_dir=testing_dir,

                                        transform=transforms.Compose([transforms.Resize((105,105)),

                                                                      transforms.ToTensor()

                                                                      ])

                                       )



test_dataloader = DataLoader(test_dataset,num_workers=6,batch_size=1,shuffle=True)
# Print the sample outputs to view its dissimilarity

counter=0

list_0 = torch.FloatTensor([[0]])

list_1 = torch.FloatTensor([[1]])

for i, data in enumerate(test_dataloader,0): 

  x0, x1 , label = data

  concatenated = torch.cat((x0,x1),0)

  output1,output2 = model(x0.to(device),x1.to(device))

  eucledian_distance = F.pairwise_distance(output1, output2)

  if label==list_0:

    label="Orginial"

  else:

    label="Forged"

  imshow(torchvision.utils.make_grid(concatenated),'Dissimilarity: {:.2f} Label: {}'.format(eucledian_distance.item(),label))

  counter=counter+1

  if counter ==20:

     break