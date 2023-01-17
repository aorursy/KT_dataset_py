import os 

import random 

import pandas as pd 

import glob

import pydicom

import torch 

import numpy as np 

import torchvision.transforms as transforms 

import matplotlib.pyplot as plt

import torch.nn.functional as fun

import torch.optim as optim

import torch.nn as nn

import json



from collections import OrderedDict

from torchvision import datasets ,models

from torch.utils.data.sampler import SubsetRandomSampler 

from torch.utils.data import Dataset

gpu=torch.cuda.is_available()

if gpu : print("gpu")

else : print("cpu")
import os 

os.chdir("../input/")

os.chdir('chest-xray-pneumonia')

os.chdir('chest_xray')

os.chdir('chest_xray')

test_dir='train'

train_dir='/kaggle/input'

#os.chdir('rsna-pneumonia-detection-challenge')

print(os.listdir(os.getcwd()))
detailed_labels=pd.read_csv("/kaggle/input/rsna-pneumonia-detection-challenge/stage_2_detailed_class_info.csv")

# train_labels=pd.read_csv("stage_2_train_labels.csv")

train_dir=os.path.join('/kaggle/input/rsna-pneumonia-detection-challenge',"stage_2_train_images")

# test_dir=os.path.join(os.getcwd(),'stage_2_test_images')

print(train_dir)

print(test_dir)
def get_dicom_fps(dicom_dir):

    dicom_fps = glob.glob(dicom_dir+'/'+'*.dcm')

    return list(set(dicom_fps))
def parse_dataset(dicom_dir, anns): 

    string=""

    image_fps = get_dicom_fps(dicom_dir)

    image_annotations = {fp: string for fp in image_fps}

    for index, row in anns.iterrows(): 

        if row['class']=="No Lung Opacity / Not Normal":

            fp = os.path.join(dicom_dir, row['patientId']+'.dcm')

            image_annotations.pop(fp)

        else:    

            fp = os.path.join(dicom_dir, row['patientId']+'.dcm')

            if row['class']=="Normal":

                image_annotations[fp]=(0)#row['Target']

            else :

                 image_annotations[fp]=(1)#row['Target']

    return image_fps, image_annotations 
image_fps,image_annotations=parse_dataset(train_dir,detailed_labels)

print(len(image_annotations))

# print(image_annotations)
keys=image_annotations.keys()

print(list(keys)[0])
class RsnaDataset(Dataset):

    

    def __init__(self, data, transform=None):

        self.data = data

        self.transform = transform

        self.list_keys=list(data.keys())

        

    def __len__(self):

        return len(self.data)

    def transform(self,image):

        return self.tansform(image)

    

    

    def __getitem__(self, index):

        # load image as ndarray type (Height * Width * Channels)

        # be carefull for converting dtype to np.uint8 [Unsigned integer (0 to 255)]

        # in this example, i don't use ToTensor() method of torchvision.transforms

        # so you can convert numpy ndarray shape to tensor in PyTorch (H, W, C) --> (C, H, W)

        image = pydicom.read_file(self.list_keys[index]).pixel_array

        #image=np.stack((image,)*3,axis=-1)

#         print("inside dataset",image.shape)

       # image = self.data.iloc[index, 1:].values.astype(np.uint8).reshape((1, 28, 28))

        label = torch.tensor(self.data[self.list_keys[index]])

        

        if self.transform is not None:

            image = self.transform(image)

#         print("inside dataset",image.shape)    

        return image, label
transform=transforms.Compose([transforms.ToPILImage(),                           

                              transforms.Resize(480),

                              transforms.CenterCrop(240),

                              transforms.ToTensor(),

                              transforms.Normalize((0.5,),(0.5,)),

#                               transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),   

                             ])

test_transform=transforms.Compose([transforms.Grayscale(),                          

                              transforms.Resize(480),

                              transforms.CenterCrop(240),

                              transforms.ToTensor(),

                              transforms.Normalize((0.5,),(0.5,)),

#                               transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),   

                             ])



train_data=RsnaDataset(image_annotations,transform=transform)

test_data=datasets.ImageFolder(test_dir,transform=test_transform)
batch_size=16

valid_size=0.2

# test_size=0.05
num_trained=len(train_data)

indices=list(range(num_trained))

np.random.shuffle(indices)

valid_split=int(np.floor(num_trained*valid_size))

# test_split=int(np.floor(num_trained*test_size))

# test_id=indices[:test_split]

valid_id=indices[:valid_split]

train_id=indices[valid_split:]
train_sampler=SubsetRandomSampler(train_id)

valid_sampler=SubsetRandomSampler(valid_id)

# test_sampler=SubsetRandomSampler(test_id)



train_Loader=torch.utils.data.DataLoader(train_data,sampler=train_sampler,

                                       batch_size=batch_size,num_workers=0)

valid_Loader=torch.utils.data.DataLoader(train_data,sampler=valid_sampler,

                                       batch_size=20,num_workers=0)

test_Loader=torch.utils.data.DataLoader(test_data,

                                       batch_size=batch_size,num_workers=0)



classes = ['Normal','Pnemunia']
# class Model(nn.Module):

    

#     def __init__(self):

#         super(Model, self).__init__()

#         self.conv1=nn.Conv2d(3, 22, kernel_size=(7,7),stride=(2, 2), padding=(3, 3), bias=False)

        

#         self.conv2=nn.Conv2d(22, 128, kernel_size=(7,7),stride=(2, 2), padding=(3, 3), bias=False)

        

#         #self.pool1=nn.MaxPool2d(2, 2)

#         #

#         self.conv3 = nn.Conv2d(128,128,kernel_size=(7,7),stride=(2,2),padding=(3,3),dilation=1,groups=128,bias=False)

#         self.pointwise1 = nn.Conv2d(128,128,1,1,0,1,1,bias=False)

        

#         self.conv4 = nn.Conv2d(128,128,kernel_size=(7,7),stride=(2,2),padding=(3,3),dilation=1,groups=128,bias=False)

#         self.pointwise2 = nn.Conv2d(128,256,1,1,0,1,1,bias=False)

#         #

#         #self.pool2=nn.MaxPool2d(2, 2)

        

#         self.batch1=nn.BatchNorm2d(num_features=256)

        

#         self.conv5 = nn.Conv2d(256,256,kernel_size=(7,7),stride=(2,2),padding=(3,3),dilation=1,groups=128,bias=False)

#         self.pointwise3 = nn.Conv2d(256,256,1,1,0,1,1,bias=False)

#         #

#         self.batch2=nn.BatchNorm2d(num_features=256)

        

#         self.conv6 = nn.Conv2d(256,256,kernel_size=(7,7),stride=(2,2),padding=(3,3),dilation=1,groups=128,bias=False)

#         self.pointwise4 = nn.Conv2d(256,512,1,1,0,1,1,bias=False)

#         #

#         #self.pool3=nn.MaxPool2d(2, 2)

#         self.conv7 = nn.Conv2d(512,512,kernel_size=(7,7),stride=(2,2),padding=(3,3),dilation=1,groups=128,bias=False)

#         self.pointwise5 = nn.Conv2d(512,512,1,1,0,1,1,bias=False)

#         self.batch3=nn.BatchNorm2d(num_features=512)

        

#         self.conv8 = nn.Conv2d(512,512,kernel_size=(7,7),stride=(2,2),padding=(3,3),dilation=1,groups=128,bias=False)

#         self.pointwise5 = nn.Conv2d(512,512,1,1,0,1,1,bias=False)

        

#         self.batch4=nn.BatchNorm2d(num_features=512)

#         self.conv9 = nn.Conv2d(512,512,kernel_size=(7,7),stride=(2,2),padding=(3,3),dilation=1,groups=128,bias=False)

        

#         self.pointwise6 = nn.Conv2d(512,512,1,1,0,1,1,bias=False)

#         self.batch4=nn.BatchNorm2d(num_features=512,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        

#         #self.pool4=nn.MaxPool2d(2, 2)

        

#         self.fc1=nn.Linear(1,2048)

#         self.dropout=nn.Dropout(p=0.25)

#         self.fc2=nn.Linear(2048,2)

#         self.fc3=nn.Linear(2,2)

#         self.softmax=nn.LogSoftmax(dim=1)

        

        

    

#     def forward(self,image):

#         #first layer 

#         #print(image.size())

#         image=self.conv1(image)

#         image=f.relu(image)

#         #second layer

#         image=self.conv2(image)

#         image=f.relu(image)

#         #third layer

#         #image=self.pool1(image)

#         #fourth layer

#         image = self.conv3(image)

#         image = self.pointwise1(image)

#         image=f.relu(image)

#         #fifth_layer

#         image = self.conv4(image)

#         image = self.pointwise2(image)

#         image=f.relu(image)

        

#         print("sixth layer")

#         #sixth layer

#        # image=self.pool2(image)

        

#         image=self.batch1(image)

        

#         image = self.conv5(image)

#         image = self.pointwise3(image)

#         image=f.relu(image)

        

#         image=self.batch2(image)

        

#         image = self.conv6(image)

#         image = self.pointwise4(image)

#         image=f.relu(image)

        

#         #image=self.pool3(image)

        

#         image = self.conv7(image)

#         image = self.pointwise5(image)

#         image=f.relu(image)

#         image=self.batch3(image)

        

#         image = self.conv8(image)

#         image = self.pointwise5(image)

#         image=f.relu(image)

#         #image=self.batch4(image)

#         #image=self.pool4(image)

        

#         image = self.conv9(image)

#         image = self.pointwise6(image)

#         image=f.relu(image)

#         image=self.batch4(image)

#         print(image.size())

#         print("linear layers")

#         image=image.view(-1,1,512,16)

#         image=self.fc1(image)

#         print(image.size())

#         print("after linear layer 1")

#         image=f.relu(image)

#         print(image.size())

#         image=self.dropout(image)

#         print(image.size())

#         image=self.fc2(image)

#         print(image.size())

#         image=self.fc3(image)

#         print(image.size())

#         print("after linear network",image.size())

#         image=f.relu(image)

#         print(image.size())

#         image=self.softmax(image)

#         print(image.size())

#         #print("before return ")

#         return image

      

        

# model= Model()

# model  
class Net(nn.Module):

    def __init__(self):

        super(Net,self).__init__()

        self.conv1=nn.Conv2d(1,32,3,padding=1)

        #self.batch1=nn.BatchNorm2d(num_features=)

        self.conv2=nn.Conv2d(32,64,3,padding=1)

        self.conv3=nn.Conv2d(64,128,3,padding=1)

        self.conv4=nn.Conv2d(128,256,3,padding=1)

        self.conv5=nn.Conv2d(256,512,3,padding=1)

        self.conv6=nn.Conv2d(512,1024,3,padding=1)

        self.conv7=nn.Conv2d(1024,512,3,padding=1)

        self.conv8=nn.Conv2d(512,256,3,padding=1)

#         self.conv9=nn.Conv2d(2048,4096,3,padding=1)

        self.pool=nn.MaxPool2d(2,2)

        self.fc1=nn.Linear(256*30*30,128)

        self.fc2=nn.Linear(128,2)

        self.dropout=nn.Dropout(0.25)

        self.softmax=nn.LogSoftmax(dim=1)

    def forward (self,x):

        x= fun.relu(self.conv1(x))

#         print(x.size())

        x= self.pool(fun.relu(self.conv2(x)))

#         print(x.size())

        x= fun.relu(self.conv3(x))

#         print(x.size())

        x= self.pool(fun.relu(self.conv4(x)))

#         print(x.size())

        x= fun.relu(self.conv5(x))

#         print(x.size())

        x= self.pool(fun.relu(self.conv6(x)))

#         print(x.size())

        x= fun.relu(self.conv7(x))

#         print(x.size())

        x= fun.relu(self.conv8(x))

#         print(x.size())

#         x= fun.relu(self.conv9(x))

#         print(x.size())

        x=x.view(-1,256*30*30)

#         print(x.size())

        x=self.dropout(x)

#         print(x.size())

        x=fun.relu(self.fc1(x))

#         print(x.size())

        x=self.dropout(x)

#         print(x.size())

        x=self.fc2(x)

        x=self.softmax(x)

        return x

    

 





model=Net()

print(model)

if gpu:

    model.cuda()
# classifier = nn.Sequential(OrderedDict([

#                           ('fc1', nn.Linear(1024, 512)),

#                           ('relu', nn.ReLU()),

#                           ('dropout1', nn.Dropout(p=0.25)),

#                           ('fc2', nn.Linear(512, 2)),

#                           ('output', nn.LogSoftmax(dim=1))

#                           ]))

# model.classifier=classifier
#criterion=nn.NLLLoss()

#criterion=nn.BCELoss()

criterion=nn.CrossEntropyLoss()

# optimizer=optim.SGD(model.parameters(),lr=0.0001,momentum=0.9)

optimizer = optim.Adam(model.parameters(),betas=(0.9, 0.999), lr=0.0001)
import copy

train_data

epochs=5

valid_min_loss=np.Inf



for epoch in range(1,epochs+1):

    train_loss=0.0

    valid_loss=0.0

    

    # train model 

    

    model.train()

    

    for data,target in train_Loader:

#         

        if gpu: 

            

            data=data.cuda()

#             data=data.unsqueeze(0)

           

            target=target.cuda()

#             print(target1.size())

#             target=target.squeeze(0)

            optimizer.zero_grad()

            output=model(data)

            #print(output.size())

            #print(target[0])

            loss=criterion(output,target)

            loss.backward()

            optimizer.step()

            train_loss=loss.item()*data.size(0)

            

    

    

    #validate model 

    

    model.eval()

    for data,target in valid_Loader:

        if gpu:

            data=data.cuda()

            target=target.cuda()

        output=model(data)

        loss=criterion(output,target)

        valid_loss+=loss.item()*data.size(0)

        

    train_loss=train_loss/len(train_Loader.dataset)

    valid_loss=valid_loss/len(valid_Loader.dataset)

    

    

    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(

        epoch, train_loss, valid_loss))

    

    # save model if validation loss has decreased

    if valid_loss <= valid_min_loss:

        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(

        valid_min_loss,

        valid_loss))

        torch.save(model.state_dict(), '/kaggle/working/new chest model1.pth')

        valid_min_loss = valid_loss



#torch.save(model.state_dict(), 'model_flower.pth')

print('end')
state_dict=torch.load("/kaggle/working/new chest model1.pth",map_location='cpu')

model.load_state_dict(state_dict)
test_loss=0

class_correct=list(0. for i in range (2))

class_total=list(0. for i in range(2))



model.eval



for data ,target in test_Loader:

    if gpu:

        data,target=data.cuda(),target.cuda()

    #print(list(target.size())[0],target.size())

    output=model(data)

    loss=criterion(output,target)

    test_loss +=loss.item()*data.size(0)

    _,pred=torch.max(output,1)

    correct_tensor=pred.eq(target.data.view_as(pred))

    correct=np.squeeze(correct_tensor.numpy()) if not gpu else np.squeeze(correct_tensor.cpu().numpy())

    

    batch=16

    if list(target.size())[0]==7: batch=6

    

    for i in range (batch):

        #print(i)

        label=target.data[i]

        

        class_correct[label]+=correct[i].item()

        class_total[label]+=1

        

test_loss = test_loss/len(test_Loader.dataset)

print('Test Loss: {:.6f}\n'.format(test_loss))



for i in range(2):

    if class_total[i] > 0:

        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (

            classes[i], 100 * class_correct[i] / class_total[i],

            np.sum(class_correct[i]), np.sum(class_total[i])))

    else:

        print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))



print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (

    100. * np.sum(class_correct) / np.sum(class_total),

    np.sum(class_correct), np.sum(class_total)))
test_loss=0

class_correct=list(0. for i in range (2))

class_total=list(0. for i in range(2))

model.eval



for data ,target in train_Loader:

    if gpu:

        data,target=data.cuda(),target.cuda()

    #print(list(target.size())[0],target.size())

    output=model(data)

    loss=criterion(output,target)

    test_loss +=loss.item()*data.size(0)

    _,pred=torch.max(output,1)

    correct_tensor=pred.eq(target.data.view_as(pred))

    correct=np.squeeze(correct_tensor.numpy()) if not gpu else np.squeeze(correct_tensor.cpu().numpy())

    

    batch=16

    if list(target.size())[0]==7: batch=6

    

    for i in range (batch):

        if correct.size<16 :

            continue

        label=target.data[i]

      #  print("batch",batch,"label",label,"class_correct",class_correct,"target_size",target.size(),"correct",correct)

        class_correct[label]+=correct[i].item()

        class_total[label]+=1

        

test_loss = test_loss/len(test_Loader.dataset)

print('Test Loss: {:.6f}\n'.format(test_loss))



for i in range(2):

    if class_total[i] > 0:

        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (

            classes[i], 100 * class_correct[i] / class_total[i],

            np.sum(class_correct[i]), np.sum(class_total[i])))

    else:

        print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))



print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (

    100. * np.sum(class_correct) / np.sum(class_total),

    np.sum(class_correct), np.sum(class_total)))
# dataiter=iter(test_Loader)



# for i in range(1):



#     images,labels=dataiter.next()

#     images.numpy()



#     if gpu :

#         images=images.cuda()



#     output=model(images)

#     _,preds_tensor=torch.max(output,1)

#     preds=np.squeeze(preds_tensor.numpy()) if not gpu else np.squeeze(preds_tensor.cpu().numpy())



#     fig=plt.figure(figsize=(50,50))



#     for id in np.arange(1,16):

#         photo=fig.add_subplot(2,20/2,id+1,xticks=[],yticks=[])

#         images=images.cpu()

#         imshow(images[id])



#         photo.set_title("{} ({})".format(classes[preds[id]], classes[labels[id].numpy()]),

#                      color=("green" if preds[id]==labels[id].item() else "red"))