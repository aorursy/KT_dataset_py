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

    
import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

from torch.optim import lr_scheduler

import torchvision.models as models

import torchvision

import matplotlib.pyplot as plt

from torchvision import datasets

import copy

import pdb

import torchvision.transforms as transforms

from torch.autograd import Variable



import numpy as np

import pandas as pd



train_on_gpu = torch.cuda.is_available()
transform_train = transforms.Compose([

    transforms.ColorJitter(),

    transforms.Resize(256),

    transforms.CenterCrop(224),

    transforms.ToTensor(),

#     transforms.Normalize(mean=[0.46390891, 0.4635186 , 0.46358757], std=[0.32574076, 0.32559644, 0.32563856]),

     

#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

    ])
full_data = VowelConsonantDataset("../input/padhai-hindivowelconsonantdataset/train/train",train=True,transform=transform_train)

# we are splitting total 10000 images into 9000 images for training and remaining 1000 for validation 

train_size = int(0.9 * len(full_data))

test_size = len(full_data) - train_size



train_data, validation_data = random_split(full_data, [train_size, test_size])



train_loader = torch.utils.data.DataLoader(train_data, batch_size=50, shuffle=True)

validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=50, shuffle=True)

test_data = VowelConsonantDataset("../input/padhai-hindivowelconsonantdataset/test/test",train=False,transform=transform_train)

test_loader = torch.utils.data.DataLoader(test_data, batch_size=20,shuffle=False)

print(len(test_data))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)
#         self.model_resnet = self.model_resnet.to(device)

class MyModel(nn.Module):

    def __init__(self, num_classes1, num_classes2):

        super(MyModel, self).__init__()

        self.model_snet = models.mobilenet_v2(pretrained=True)

#         num_ftrs = self.model_snet.fc.in_features

        self.model_snet.classifier = nn.Sequential(nn.Dropout(p=0.2))

#         x.view(x.size(0), self.num_classes)

        self.fc1 = nn.Linear(1280, num_classes1,bias=True)

        torch.nn.init.xavier_uniform_(self.fc1.weight)

        torch.nn.init.zeros_(self.fc1.bias)

        self.fc2 = nn.Linear(1280, num_classes2,bias=True)

        torch.nn.init.xavier_uniform_(self.fc2.weight)

        torch.nn.init.zeros_(self.fc2.bias)



    def forward(self, x):

        x = self.model_snet(x)

#         x= x.view(x.size(0), :)

#         print(x.parameters().weight)

        out1 = self.fc1(x)

        out2 = self.fc2(x)

        return out1, out2
my_model = MyModel(10,10)
my_model = my_model.to(device)
def evaluation(dataloader,model):

    total,correct=0,0

    for data in dataloader:

        inputs,labels=data

        inputs,labels=inputs.to(device),labels.to(device)

        out1,out2=my_model(inputs)

        _,pred1=torch.max(out1.data,1)

        _,pred2=torch.max(out2.data,1)

        _,labels1=torch.max(labels[:,0,:].data,1)

        _,labels2=torch.max(labels[:,1,:].data,1)

        total+=labels.size(0)

        fin1=(pred1==labels1)

        fin2=(pred2==labels2)

        

        correct+=(fin1==fin2).sum().item()

    return 100*correct/total

loss_fn = nn.CrossEntropyLoss()



plist = [

        {'params': my_model.fc1.parameters(), 'lr': 5e-3},

        {'params': my_model.fc2.parameters(), 'lr': 5e-3}

        ]

lr=0.01

opt = optim.SGD(my_model.parameters(),lr=0.01,momentum=0.9,nesterov=True)

# lr_sch = lr_scheduler.StepLR(opt, step_size=10, gamma=1.6)

# step_size: at how many multiples of epoch you decay

# step_size = 1, after every 1 epoch, new_lr = lr*gamma 

# step_size = 2, after every 2 epoch, new_lr = lr*gamma 

# opt = optim.Adam(my_model.parameters(),lr=0.01)
loss_epoch_arr = []

loss_arr = []

max_epochs = 2

min_loss = 1000

batch_size = 50

n_iters = np.ceil(9000/batch_size)

for epoch in range(max_epochs):

    for i, data in enumerate(train_loader, 0):

        my_model.train()

        images, labels = data

        images = images.to(device)

        targetnp=labels[:,0,:].cpu().numpy()

        targetnp1 = labels[:,1,:].cpu().numpy()

        # Convert predictions classes from one hot vectors to labels: [0 0 1 0 0 ...] --> 2

        new_targets1 = np.argmax(targetnp,axis=1)

        new_targets2 = np.argmax(targetnp1,axis=1)

        new_targets1=torch.LongTensor(new_targets1)

        new_targets2=torch.LongTensor(new_targets2)

        new_targets1 = new_targets1.to(device)

        new_targets2 = new_targets2.to(device)

        opt.zero_grad()

        out = my_model.forward(images)

        loss_fc1 = loss_fn(out[0], new_targets1)

        loss_fc2 = loss_fn(out[1],new_targets2)

        loss = torch.add(loss_fc1,loss_fc2)

        loss.backward()

        opt.step()   

        if min_loss > loss.item():

            min_loss = loss.item()

            best_model = copy.deepcopy(my_model.state_dict())

            print('Min loss %0.2f' % min_loss)

        if i % 100 == 0:

            print('Iteration: %d/%d, Loss: %0.2f' % (i, n_iters, loss.item()))

        del images, labels, out

        torch.cuda.empty_cache()

        loss_arr.append(loss.item())

    print("Epoch number :",epoch)

    print("Train Accuracy :",evaluation(train_loader,my_model))

    print("Test Accuracy :"  ,evaluation(validation_loader,my_model))

    print('Learning Rate :',lr)

    loss_epoch_arr.append(loss.item())

#     my_model.load_state_dict(best_model)

plt.plot(loss_arr)

plt.show()
plt.plot(loss_epoch_arr)

plt.show()
my_model.load_state_dict(best_model)
evaluation(validation_loader,my_model)
my_model.eval()

plist=[]

fn_list=[]

for inputs_test, fn in test_loader:

    inputs_test=inputs_test.to(device)

    out1,out2=my_model.forward(inputs_test)

    _,pred1=torch.max(out1,1)

    pred1=pred1.tolist()

    _,pred2=torch.max(out2,1)

    pred2=pred2.tolist()

    for x,y,z in zip(pred1,pred2,fn):

        p="V"+str(x)+"_"+"C"+str(y)

        plist.append(p)

        fn_list.append(z)
submission = pd.DataFrame({"ImageId":fn_list, "Class":plist})

submission.to_csv('submission.csv', index=False)
submission.head()