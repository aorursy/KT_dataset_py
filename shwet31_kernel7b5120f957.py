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



train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=60, shuffle=True)
len(train_loader)
test_data = VowelConsonantDataset("../input/test/test",train=False,transform=transform)

test_loader = torch.utils.data.DataLoader(test_data, batch_size=60,shuffle=False)
for data in train_loader:

    img,lab = data

    im = np.transpose(img[0].numpy(),(1,2,0))

    

    print(im.shape)

    im = np.squeeze(im)

    print(im.shape)

    plt.imshow(im,)

   # print(img.shape,lab[0])

    break
full_data.get_classes()
import torch.nn as nn

import torch.optim as optim

import torchvision.models as models

from tqdm import tqdm_notebook
model_v = models.resnet18()

model_c = models.resnet18()
print(model_c)


model_c.fc = nn.Linear(512,10,bias=True)

model_v.fc = nn.Linear(512,10,bias=True)

print(model_c)
opt_v = optim.Adam(model_v.parameters())

opt_c = optim.Adam(model_c.parameters())

loss_fn_v = nn.CrossEntropyLoss()

loss_fn_c =  nn.CrossEntropyLoss()
print(lab[:,0,:],torch.max(lab[:,0,:],1))
max_epochs = 45

loss_arr= []

st='cuda:0'

model_v.to(st)

model_c.to(st)



for i in tqdm_notebook(range(max_epochs),total=max_epochs,unit='epochs'):

    for data in tqdm_notebook(train_loader,total=len(train_loader),unit='batch'):

        img,lab = data

        img,lab = img.to(st),lab.to(st)

        out_v = model_v(img)

        out_c = model_c(img)

        

        opt_v.zero_grad()

        opt_c.zero_grad()

        val,ind = torch.max(lab[:,0,:],1)

        val,ind1 = torch.max(lab[:,1,:],1)

        lab_v = ind

        lab_c = ind1

        loss = loss_fn_v(out_v,lab_v)+loss_fn_c(out_c,lab_c)

        

        loss.backward()

        opt_v.step()

        opt_c.step()

        del img,lab

        

    print(loss)

    loss_arr.append(loss)        

        
def evaluation(dataloader,m1,m2):

    total=0

    v=0

    c=0

    for data in tqdm_notebook(dataloader,total=len(dataloader),unit='batch'):

        img,lab = data

        img,lab = img.to(st),lab.to(st)

        _,out_v = torch.max(m1(img),1)

        _,out_c = torch.max(m2(img),1)

        _,lab1 = torch.max(lab[:,0,:],1)

        _,lab2 = torch.max(lab[:,1,:],1)

        total += 64

        v += (out_v==lab1).sum().item()

        c += (out_c==lab2).sum().item()

    print('total images:',total)

    print('correct vowels predictions:',v)

    print('correct consonants predictions:',c)

    print('Vowel Accuracy: ',(v/total)*100, '%')

    print('Consonants Accuracy: ',(c/total)*100,'%')

        

        

evaluation(train_loader,model_v,model_c)

    
keys = full_data.classes_mapping.keys()

keys = list(keys)

png = []

arr = []

model_v.to('cuda:0')

model_c.to('cuda:0')

for data in tqdm_notebook(test_loader,total=len(test_loader),unit='batch'):

    img,lab = data

    lab = list(lab)

    png.extend(lab)

    img = img.to('cuda:0')

    out_v = model_v(img)

    out_c = model_c(img)

    _,ind1 = torch.max(out_v,1)

    _,ind2 = torch.max(out_c,1)

    for i,j in zip(ind2,ind1):

        arr.append(keys[j+10]+'_'+keys[i])

    

    

    

    

    
df = pd.DataFrame([png,arr])

df = df.transpose()

df.columns = ['ImageId','Class']
df.head()
df.to_csv('out.csv',index=False)