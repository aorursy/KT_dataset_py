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
import torchvision

from torchvision import transforms

import torch

from torch import nn,optim

from torch.utils.data.sampler import SubsetRandomSampler

train_path = '../input/car_data/car_data/train'

transform = transforms.Compose([

    	transforms.RandomHorizontalFlip(),

    	transforms.CenterCrop(224),

    	transforms.ToTensor(),

    	transforms.Normalize([0.485, 0.456, 0.406],

                         	[0.229, 0.224, 0.225])])

traindata = torchvision.datasets.ImageFolder(root=train_path, transform=transform)

dataset_size = len(traindata)

indices = list(range(dataset_size))

split = int(np.floor(0.2 * dataset_size))

np.random.seed(7)

np.random.shuffle(indices)

train_indices, val_indices = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_indices)

valid_sampler = SubsetRandomSampler(val_indices)



train_loader = torch.utils.data.DataLoader(traindata, batch_size=8, 

                                           sampler=train_sampler,num_workers=4)

val_loader = torch.utils.data.DataLoader(traindata, batch_size=8,

                                                sampler=valid_sampler,num_workers=4)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.cuda.empty_cache()

model = torchvision.models.resnext101_32x8d(pretrained=True)

for name,child in model.named_children():

    if name == 'layer4':

        for param in child.parameters():

            param.requires_grad = True

    else:

        for param in child.parameters():

            param.requires_grad = False

classifier = nn.Sequential(nn.BatchNorm1d(2048),

                            nn.Linear(2048,1000),

                          nn.BatchNorm1d(1000),

                          nn.ReLU(),

                          nn.Linear(1000,196),

                          nn.LogSoftmax(dim=1))

model.fc = classifier





model.to(device)
torch.cuda.empty_cache()

optimizer = optim.SGD([{'params':model.layer4.parameters()},

                       {'params':model.fc.parameters()}],lr=0.003)

scheduler=optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min', verbose=True,patience=1)

acc = 0

vacc=0

tacc=0



criterion = nn.CrossEntropyLoss()

epochs = 30

valid_loss_min = np.Inf

for ep in range(epochs):

    train_loss = 0

    valid_loss = 0

    model.train()

    for data, target in train_loader:

        data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()

        output = model(data)

        preds = torch.exp(output)

        _,pred = torch.max(preds,1)

        t_acc = torch.sum(pred==target.data)

        acc+=t_acc

        loss = criterion(output, target)

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

    model.eval()

    for data, target in val_loader:

        data, target = data.cuda(), target.cuda()

        output = model(data)

        preds = torch.exp(output)

        _,pred = torch.max(preds,1)

        v_acc = torch.sum(pred==target.data)

        vacc+=v_acc

        loss = criterion(output, target)

        valid_loss += loss.item()

    

    train_loss = train_loss/len(train_loader.sampler)

    valid_loss = valid_loss/len(val_loader.sampler)

    scheduler.step(valid_loss)

        

    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(

        ep, train_loss, valid_loss))

    if valid_loss <= valid_loss_min:

        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(

        valid_loss_min,

        valid_loss))

        torch.save(model.state_dict(), 'model_hackathon.pt')

        valid_loss_min = valid_loss
model.load_state_dict(torch.load('model_hackathon.pt'))
mod = model

p=[]

mod.eval()

torch.cuda.empty_cache()

mod.to(device)

testset = torchvision.datasets.ImageFolder(root='../input/car_data/car_data/test',transform=transform)

testloader = torch.utils.data.DataLoader(testset,batch_size=1,num_workers=4)

t_loss=0

for data, target in testloader:

        data, target = data.cuda(), target.cuda()

        output = mod(data)

        preds = torch.exp(output)

        top_p,top_class = preds.topk(1,dim=1)

        _,pred = torch.max(preds,1)

        te_acc = torch.sum(pred==target.data)

        tacc+=te_acc

        p.append(int(pred))

        loss = criterion(output,target)

        t_loss+=loss.item()

print(t_loss/len(testloader))

torch.cuda.empty_cache()

# mod.cpu()

# valid = torch.utils.data.DataLoader(traindata,batch_size=1,sampler=valid_sampler,num_workers=4)

# for data, target in testloader:

#         #orch.cuda.ipc_collect()

#         #data, target = data.cuda(), target.cuda()

#         output = mod(data)

#         preds = torch.exp(output)

#         top_p,top_class = preds.topk(1,dim=1)

#         y.append(target.numpy())

#         y_score.append(top_p)
# from sklearn.metrics import roc_auc_score

# from sklearn.preprocessing import LabelBinarizer



# lb = LabelBinarizer()

# lb.fit(p)

# p = lb.transform(y)

# y_score = lb.transform(y_score)



# roc_auc_score(p,y_scor

classes = os.listdir('../input/car_data/car_data/test')

names = []

pr=[]

for cl in classes:

    for f in os.listdir(f'../input/car_data/car_data/test/{cl}'):

        f = f.split('.')[0]

        names.append(f)

for i, pre in enumerate(p):

    p[i]+=1
import pandas

ddf = pandas.DataFrame(data={'id':names,'Predicted':p})

ddf.to_csv('./submission.csv',sep=',',index=False)

print(f"Train Loss:{train_loss}")

print(f'Test Loss:{t_loss}')

print(f'Valid Loss:{valid_loss}')

print(f'Train Acc:{acc/len(train_loader)}')

print(f'Valid Acc:{vacc/len(valid_loader)}')

print(f'Test Acc:{tacc/len(testloader)}')