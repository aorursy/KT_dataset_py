# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
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

import torchvision.models as models

import torch.utils.model_zoo as model

import torchvision

import copy

import matplotlib.pyplot as plt

from torchvision import datasets



import torchvision.transforms as transforms



import numpy as np

import pandas as pd



train_on_gpu = torch.cuda.is_available()
transform_train = transforms.Compose([

    transforms.Resize(224),

    transforms.ToTensor(),

    transforms.Normalize(mean=[0.485, 0.456, 0.406],

                                 std=[0.229, 0.224, 0.225]),

    ])
full_data = VowelConsonantDataset("../input/train/train",train=True,transform=transform_train)

train_size = int(0.95 * len(full_data))

test_size = len(full_data) - train_size



train_data, validation_data = random_split(full_data, [train_size, test_size])



train_loader = torch.utils.data.DataLoader(train_data, batch_size=20, shuffle=True)

validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=20, shuffle=True)
test_data = VowelConsonantDataset("../input/test/test",train=False,transform=transform_train)

test_loader = torch.utils.data.DataLoader(test_data, batch_size=20,shuffle=False)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)
class MyModel(nn.Module):

    def __init__(self, num_classes1, num_classes2):

        super(MyModel, self).__init__()

        self.model_resnet = models.resnet50(pretrained=True)

        num_ftrs = self.model_resnet.fc.in_features

        self.model_resnet.fc = nn.Sequential()

#         self.dropout = nn.Dropout(0.5)

        self.fc1 = nn.Sequential(

              nn.BatchNorm1d(num_ftrs),

              nn.Dropout(0.3),

              nn.Linear(in_features=num_ftrs, out_features=1000,bias=True),

              nn.ReLU(),

              nn.BatchNorm1d(1000, eps=1e-05, momentum=0.3),

              nn.Dropout(0.3),

              nn.Linear(in_features=1000, out_features=10,bias=True))

        self.fc2 = nn.Sequential(

              nn.BatchNorm1d(num_ftrs), 

              nn.Dropout(0.3),

              nn.Linear(in_features=num_ftrs,out_features=1000,bias=True),

              nn.ReLU(),

              nn.BatchNorm1d(1000, eps=1e-05, momentum=0.3),

              nn.Dropout(0.3),

              nn.Linear(in_features=1000, out_features=10,bias=True))

#         torch.nn.init.xavier_uniform_(self.fc1.weight)

#         torch.nn.init.zeros_(self.fc1.bias)

#         torch.nn.init.xavier_uniform_(self.fc2.weight)

#         torch.nn.init.zeros_(self.fc2.bias)



    def forward(self, x):

        x = self.model_resnet(x)

#         print(x.shape)

#         x=x.view(x.size(0),1)

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

opt = optim.SGD(my_model.parameters(),lr=0.01,momentum=0.9)#,nesterov=True)
loss_epoch_arr = []

loss_arr = []

max_epochs = 6

min_loss = 1000

batch_size = 32

n_iters = np.ceil(9000/batch_size)

for epoch in range(max_epochs):

    for i, data in enumerate(train_loader, 0):

        my_model.train()

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

#         if min_loss < 0.8:

#             opt = optim.SGD(my_model.parameters(),lr=0.01,momentum=0.99,nesterov=True)

        if i % 100 == 0:

            print('Iteration: %d/%d, Loss: %0.2f' % (i, n_iters, loss.item()))

        del images, labels, out

        torch.cuda.empty_cache()

        loss_arr.append(loss.item())

    print("Epoch number :",epoch)

    print("Train Accuracy :",evaluation(train_loader,my_model))

    print("Test Accuracy :"  ,evaluation(validation_loader,my_model))

    loss_epoch_arr.append(loss.item())

#     my_model.load_state_dict(best_model)

plt.plot(loss_arr)

plt.show()
print(evaluation(validation_loader,my_model))
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