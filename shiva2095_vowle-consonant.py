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



import copy



train_on_gpu = torch.cuda.is_available()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

device
os.mkdir('../Inputs')



import zipfile



# Will unzip the files so that you can see them..

with zipfile.ZipFile("../input/padhai-hindi-vowel-consonant-classification/train.zip","r") as z:

    z.extractall("../Inputs/")

with zipfile.ZipFile("../input/padhai-hindi-vowel-consonant-classification/test.zip","r") as z:

    z.extractall("../Inputs/")
transform = transforms.Compose([

    transforms.ColorJitter(),

    transforms.Resize(256),

    transforms.CenterCrop(224),

    transforms.ToTensor(), ])
batch_size = 60

full_data=VowelConsonantDataset("../Inputs/train",train=True,transform=transform)

train_size = int(0.9 * len(full_data))

test_size = len(full_data) - train_size



train_data, validation_data = random_split(full_data, [train_size, test_size])



train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=batch_size, shuffle=True)



test_data=VowelConsonantDataset("../Inputs/test",train=False,transform=transform)

test_loader = torch.utils.data.DataLoader(test_data, batch_size=60,shuffle=False)
from torchvision import models
class MyModel(nn.Module):

    def __init__(self, num_classes1, num_classes2):

        super(MyModel, self).__init__()

        self.model_snet = models.mobilenet_v2(pretrained=True)

        final_in_features = self.model_snet.classifier[1].in_features

        mod_classifier = list(self.model_snet.classifier.children())[:-1]

        self.model_snet.classifier = nn.Sequential(*mod_classifier)

#         for param in self.model_snet.parameters():

#             param.requires_grad = False

        self.fc1 = nn.Linear(final_in_features, num_classes1,bias=True)

        torch.nn.init.xavier_uniform_(self.fc1.weight)

        torch.nn.init.zeros_(self.fc1.bias)

        self.fc2 = nn.Linear(final_in_features, num_classes2,bias=True)

        torch.nn.init.xavier_uniform_(self.fc2.weight)

        torch.nn.init.zeros_(self.fc2.bias)



    def forward(self, x):

        x = self.model_snet(x)

        out1 = self.fc1(x)

        out2 = self.fc2(x)

        return out1, out2
net  = MyModel(10,10)
net = net.to(device)
#to Compute accuracy

def evaluation(dataloader):

    

    total, correct = 0, 0

    for data in dataloader:

        inputs, labels = data

        inputs, labels = inputs.to(device), labels.to(device)

        # Extracting Actual Labels

        _, actual_v = torch.max(labels[:,0,:].data, 1)

        _, actual_c = torch.max(labels[:,1,:].data, 1)

        

        outputs_v,outputs_c = net(inputs)

        _, pred_v = torch.max(outputs_v.data, 1)

        _, pred_c = torch.max(outputs_c.data, 1)

        

        total += labels.size(0)

        correct_v = (pred_v == actual_v)*1

        correct_c = (pred_c == actual_c)*1

        correct_v[correct_v == 0] = 2

        correct_c[correct_c == 0] = 3

        correct += ((correct_v==correct_c)).sum().item()

    return 100 * correct / total
import torch.optim as optim



loss_fn = nn.CrossEntropyLoss()



plist = [

        {'params': net.fc1.parameters(), 'lr': 5e-3},

        {'params': net.fc2.parameters(), 'lr': 5e-3}

        ]

lr=0.01

opt = optim.SGD(net.parameters(),lr=0.01,momentum=0.9,nesterov=True)
%%time

loss_arr = []

loss_epoch_arr = []

max_epochs = 10

min_loss = 1000

best_model = None

for epoch in range(max_epochs):



    for i, data in enumerate(train_loader, 0):

        

        inputs, labels = data

        inputs, labels = inputs.to(device), labels.to(device)

        labels_v = labels[:,0,:]

        labels_c = labels[:,1,:]

        _, actual_v = torch.max(labels_v.data, 1)

        _, actual_c = torch.max(labels_c.data, 1)

        opt.zero_grad()

        

        outputs_v, outputs_c = net(inputs)

        loss_v = loss_fn(outputs_v, actual_v)

        loss_c = loss_fn(outputs_c, actual_c)

        loss = torch.add(loss_v,loss_c)

        loss.backward()

        opt.step()

        

        if min_loss > loss.item():

            min_loss = loss.item()

            best_model = copy.deepcopy(net.state_dict())

        

        loss_arr.append(loss.item())

        

        del inputs, labels, outputs_v, outputs_c

        torch.cuda.empty_cache()

        

    loss_epoch_arr.append(loss.item())

        

    print('Epoch: %d/%d, Test acc: %0.2f, Train acc: %0.2f' % (epoch, max_epochs, evaluation(validation_loader), evaluation(train_loader)))

    

net.load_state_dict(best_model)

plt.plot(loss_epoch_arr)

plt.plot(loss_arr)

plt.show()
net.eval()

plist=[]

fn_list=[]

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

        fn_list.append(z)
submission = pd.DataFrame({"ImageId":fn_list, "Class":plist})

submission.head()
submission.to_csv('submission.csv', index=False)