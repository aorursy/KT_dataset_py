%matplotlib inline

%config InlineBackend.figure_format = 'retina'



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import os

from collections import OrderedDict

import torch

from torch import nn

from torch import optim

import torch.nn.functional as F

from torchvision import datasets, transforms, models
# pretrain된 모델을 다운받는다

import torchvision.models as models

model = models.resnet18(pretrained = False)

model

# gpu를 사용할 수 있으면 device를 바꾼다

device = torch.device('cuda:0' if torch.cuda.is_available()else 'cpu')
!unzip ../input/dogs-vs-cats/train.zip -d train

!unzip ../input/dogs-vs-cats/test1.zip -d test

train_dir = '../working/train/train'

test_dir = '../working/test/test1'

train_files = os.listdir(train_dir)

test_files = os.listdir(test_dir)
from torch.utils.data import Dataset, DataLoader, ConcatDataset

from PIL import Image

import torchvision

class CatDogDataset(Dataset):

    def __init__(self, file_list, dir, mode = 'train', transform =None):

        self.file_list = file_list

        self.dir = dir

        self.mode = mode

        self.transform =transform

        #self.mode가 train으로 되어있고, file_list의 첫번쨰 행이 dog로 되어있으면 label 컬럼에 1을, 아니면 0

        if self.mode =='train':

            if 'dog' in self.file_list[0]:

                self.label = 1

            else :

                self.label = 0

    # file_list 데이터의 크기를 받음

    def __len__(self):

        return len(self.file_list)

    #

    def __getitem__(self, idx):

        img = Image.open(os.path.join(self.dir, self.file_list[idx])) 

        # 한개이상의 경로 요소를 이어붙임 ex) os.path.join('c:', foo) -> (c:foo)

        if self.transform:

            img = self.transform(img)

        if self.mode =='train':

            img = img.numpy()

            return img.astype('float32'), self.label

        else :

            img = img.numpy()

            return img.astype('float32'), self.file_list[idx]

        

data_transform = transforms.Compose([

    transforms.Resize(256), #PIL 이미지를 256으로 변경

    transforms.ColorJitter(),# brightness, contrast, saturation를 랜덤하게 변경 

    transforms.RandomCrop(224), # PIT이미지를 무작위 위치에서 crop, output크기는 224,224

    transforms.RandomHorizontalFlip(), # 주어진 확률만큼 horizontally filp시킴

    transforms.Resize(128),

    transforms.ToTensor()

])

    

cat_files = [tf for tf in train_files if 'cat' in tf]# (train_file의 각 행)tf에서 cat이 있으면 tf를 cat_files(리스트)에 추가

dog_files = [tf for tf in train_files if 'dog' in tf]



#data_transform에서 제시된 인자대로 cat_file의 자료들을 변형

cats =CatDogDataset(cat_files, train_dir, transform = data_transform)

dogs =CatDogDataset(dog_files, train_dir, transform = data_transform)

#합침

catdogs = ConcatDataset([cats,dogs])
dataloader = DataLoader(catdogs, batch_size = 64, shuffle=True, num_workers=4)
samples, labels = iter(dataloader).next()

plt.figure(figsize=(16,24))

grid_imgs = torchvision.utils.make_grid(samples[:24])

np_grid_imgs = grid_imgs.numpy()

# tensor에서 이미지는 (batch, width, height),이다. 따라서 이를 보이기 위해서 이미지를 (w,h,b)로 전치시켜야한다.

plt.imshow(np.transpose(np_grid_imgs,(1,2,0)))
# parameter들을 고정

for param in model.parameters():

    param.requires_grad = False
#resnet18의 위에 Classifier 아키텍터를 올림

fc = nn.Sequential(OrderedDict([

    ('fc1', nn.Linear(512,100)),

    ('relu',nn.ReLU()),

    ('fc2', nn.Linear(100,2)),

    ('output',nn.LogSoftmax(dim=1))

]))



model.fc = fc
# 모델을 gpu로 변환

model.to(device)

model
# 모델을 훈련시키기 위한 function

def train(model, trainloader, criterion, optimizer, epochs = 5):

    train_loss = []

    for e in range(epochs):

        running_loss =0

        for images, labels in trainloader:

            inputs, labels = images.to(device), labels.to(device)

            

            optimizer.zero_grad() # torch.Tensor에 있는 gradient를 정리함

            img = model(inputs)

            

            loss = criterion(img, labels)

            running_loss+=loss

            loss.backward()

            optimizer.step() # backward의 gradient를 출력해줌

        print('Epoch : {}/{}..'.format(e+1,epochs),

             'Training Loss : {:.6f}'.format(running_loss/len(trainloader)))

        train_loss.append(running_loss) # running_loss들을 저장

    plt.plot(train_loss,label = 'Training Loss')

    plt.show()

    



epochs =3

model.train() # train 모델들의 값

optimizer = optim.Adam(model.fc.parameters(),lr=0.001)

criterion = nn.NLLLoss()

train(model,dataloader, criterion, optimizer, epochs)

#모델 저장

filename_pth = 'ckpt_resnet18_catdog.pth'

torch.save(model.state_dict(),filename_pth)



# 테스트 데이터셋 변환

test_transform = transforms.Compose([

    transforms.Resize((128,128)),

    transforms.ToTensor()

])



testset = CatDogDataset(test_files, test_dir, mode='test', transform =test_transform)

testloader = DataLoader(testset, batch_size = 64, shuffle=False, num_workers=4)
model.eval()

fn_list = []

pred_list = []

for x, fn in testloader:

    with torch.no_grad():

        x = x.to(device)

        output = model(x)

        pred = torch.argmax(output, dim=1)

        fn_list += [n[:-4] for n in fn] # fn

        pred_list += [p.item() for p in pred]



submission = pd.DataFrame({"id":fn_list, "label":pred_list})

submission.to_csv('preds_resnet18.csv', index=False)        
samples, _ = iter(testloader).next()

samples = samples.to(device)

fig = plt.figure(figsize=(24, 16))

fig.tight_layout()

output = model(samples[:24])

pred = torch.argmax(output, dim=1)

pred = [p.item() for p in pred]

ad = {0:'cat', 1:'dog'}

for num, sample in enumerate(samples[:24]):

    plt.subplot(4,6,num+1)

    plt.title(ad[pred[num]])

    plt.axis('off')

    sample = sample.cpu().numpy()

    plt.imshow(np.transpose(sample, (1,2,0)))
# factor들을 줄일 수 있게 해줌

with open('hello.txt', 'w') as f: # f라는 이름으로 파일을 열고

    f.write('hello, world!') # f에 문장을 입력하고 닫음

    

# with 없이 사용

f = open('hello.txt','w')

try :

    f.write('hello, world')

finally:

    f.close()
# Contest Manager

class ManagedFile:

    def __init__(self,name): # 파일의 이름저장

        self.name = name

        

    def __enter__(self):# enter이 불리면 열것임

        self.file = open(self.name,'w')

        return self.file

    

    def __exit__(self, exc_type, exc_val, exc_tb): #exc : 예외처리

        if self.file:

            self.file.close()


with ManagedFile('hello.txt') as f:

    f.write('hello, world')



mf = ManagedFile('hello.txt')

mf
with mf as the_file: #enter에 들어가서 결과값을 thefile로 받고

    the_file.write('hello.txt') # the_file에 들어간 후 마지막에 __exit__ 에 있는 거승ㄹ 실행함
from cntextlib import contextmanager

@contextmanager

def managed_file(name):

    try :

        f = open(name,'w')

        yield f

    finally:

        f.close()

        

with manged_file('hello.txt') as f:

    f.write('hello, world')

    f.write('bye')
import pandas as pd

sampleSubmission = pd.read_csv("../input/dogs-vs-cats/sampleSubmission.csv")