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

    

    
import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim



import torchvision

import matplotlib.pyplot as plt

from torchvision import datasets



import torchvision.transforms as transforms



import numpy as np

import pandas as pd



train_on_gpu = torch.cuda.is_available()
transform = transforms.Compose([

    transforms.ToTensor()])
full_data = VowelConsonantDataset("../input/train/train",train=True,transform=transform)

train_size = int(0.9 * len(full_data))

test_size = len(full_data) - train_size



train_data, validation_data = random_split(full_data, [train_size, test_size])



# train_loader = torch.utils.data.DataLoader(train_data, batch_size=60, shuffle=True) ##Worked well before

# validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=60, shuffle=True)



train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)

validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=128, shuffle=True)
test_data = VowelConsonantDataset("../input/test/test",train=False,transform=transform)

# test_loader = torch.utils.data.DataLoader(test_data, batch_size=60,shuffle=False)#Before



test_loader = torch.utils.data.DataLoader(test_data, batch_size=128,shuffle=False)
#trainset1=torchvision.datasets.ImageFolder(root="../input/test/test",transform=transforms.ToTensor())

testloader1=torch.utils.data.DataLoader(test_data,batch_size=1,shuffle=False)

dataiter = iter(testloader1)

images, labels = dataiter.next()



print(images.shape)

print(images[0].shape)

for u in range(len(labels)):

    print(labels[u])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)
from torchvision import models

resnet1 = models.resnet18(pretrained=False)

resnet2 = models.resnet18(pretrained=False)

for param in resnet1.parameters():

    param.requires_grad = True

    

for param in resnet2.parameters():

    param.requires_grad = True

in_features = resnet1.fc.in_features



in_features = resnet2.fc.in_features

resnet1.fc = nn.Linear(in_features, 10)



resnet2.fc = nn.Linear(in_features, 10)
def weights_init(m):

    classname = m.__class__.__name__

    if classname.find('Conv') != -1:

        

#         xavier(m.weight.data)

#         xavier(m.bias.data)

        nn.init.xavier_normal_(m.weight.data)

        #nn.init.xavier_normal_(m.bias.data)
resnet1 = resnet1.to(device)

resnet1.apply(weights_init)

resnet2 = resnet2.to(device)

resnet2.apply(weights_init)

loss_fn=nn.CrossEntropyLoss()





#opt1=optim.Adam(resnet1.parameters(),lr=0.0001)

#opt2=optim.Adam(resnet2.parameters(),lr=0.001)



# opt1=optim.Adam(resnet1.parameters(),lr=0.01,betas=(0.8, 0.998))#Better for 500 epochs 83% acc

# opt2=optim.Adam(resnet2.parameters(),lr=0.01,betas=(0.8, 0.998))



opt1=optim.Adam(resnet1.parameters(),lr=0.05,betas=(0.8, 0.998))

opt2=optim.Adam(resnet2.parameters(),lr=0.05,betas=(0.8, 0.998))



loss_arr1=[]

loss_epoch_arr1=[]

max_epochs=700



for epoch in range(max_epochs):

    for i,data in enumerate(train_loader,0):

        labels2=[]

        inputs,labels=data

        inputs, labels = inputs.to(device), labels.to(device)

        lar1=[]

        lar2=[]

        for l in labels:

            o=0

            for j in l:

                o=o+1

                if o==1:

                    lar1.append(j)

                else:

                    lar2.append(j)

                    

        lar1=torch.stack(lar1)

        lar2=torch.stack(lar2)

        

     

        

          

        opt1.zero_grad()

        

        outvow=resnet1(inputs)

        outvow=outvow.to(device)

        #outvow=outvow.type(torch.LongTensor)

        #print(outvow)

        

        #print(outcon)

        #print(" \n")

       # print(labels[-1][-1])

        #print(labels[1])

        

        lar1 = lar1.type(torch.LongTensor)

        lar1= lar1.to(device)

        #print(lar1)

       # print(lar1.shape)

        #print(outvow.shape)

        #loss = criterion(outputs, torch.max(labels, 1)[1])

        #lar11,_=torch.max(lar1, 1)[1]

        loss1=loss_fn(outvow.float(),torch.max(lar1, 1)[1])

        loss1.backward()#retain_graph=True

      

       

        

        

       

        

       

       

        

        opt1.step()

      

            

        del inputs, labels

        torch.cuda.empty_cache()

        

        loss_arr1.append(loss1.item())

    loss_epoch_arr1.append(loss1.item())

    print('Epoch: %d/%d' % (epoch, max_epochs))

    

plt.plot(loss_epoch_arr1)

plt.show()
loss_arr2=[]

loss_epoch_arr2=[]

max_epochs=700



for epoch in range(max_epochs):

    for i,data in enumerate(train_loader,0):

        labels2=[]

       

        inputs,labels=data

        inputs, labels = inputs.to(device), labels.to(device)

        lar1=[]

        lar2=[]

        for l in labels:

            o=0

            for j in l:

                o=o+1

                if o==1:

                    lar1.append(j)

                else:

                    lar2.append(j)

                    

        lar1=torch.stack(lar1)

        lar2=torch.stack(lar2)

        

     

        

          

        opt2.zero_grad()

        

        outcon=resnet2(inputs)

        outcon=outcon.to(device)

        lar2 = lar2.type(torch.LongTensor)

        lar2 = lar2.to(device)

        loss2=loss_fn(outcon.float(),torch.max(lar2, 1)[1])

        loss2.backward()

        

      

       

        

        

       

        

       

       

        

        opt2.step()

      

            

        del inputs, labels

        torch.cuda.empty_cache()

        

        loss_arr2.append(loss2.item())

    loss_epoch_arr2.append(loss2.item())

    print('Epoch: %d/%d' % (epoch, max_epochs))

    

plt.plot(loss_epoch_arr2)

plt.show()

                         

    
import csv

def evaluation1(dataloader):

    #

    s1=[]

    lab=[]

    #col={'ImageId', 'Class'}

    #df = pd.DataFrame(col) 

    total,correct=0,0

    for data in dataloader:

        

        resnet1.eval()

        resnet2.eval()

        inputs,labels=data

        inputs= inputs.to(device)

        #labels = labels.to(device)

        out11=resnet1(inputs)

        out22=resnet2(inputs)

        i=0

        _,pred1=torch.max(out11,1)

        _,pred2=torch.max(out22,1)

        #	V0_C0

        for g,h in zip(pred1,pred2):

            

            s="V"+str(g.item())+"_"+"C"+str(h.item())

            s1.append(s)

            lab.append(labels[i])

            i=i+1

        i=0

            

           # print(labels[i],s)

            

    #print("\n \n")

    #print(lab)

   

    #print(s1)

    

    submission = pd.DataFrame({'ImageId':lab, 'Class':s1})

    submission = submission[['ImageId', 'Class']]

    

    print(submission)

    

    print("\n \n \n \n \n")

    print(submission.head)  

    

    filename = 'submissionnew.csv'



    submission.to_csv(filename,index=False,quoting=csv.QUOTE_NONNUMERIC)



    print('Saved file: ' + filename)



    

       

        

       

       

      

        



evaluation1(test_loader)