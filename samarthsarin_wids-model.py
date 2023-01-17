# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the file/s in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import torch

import torchvision

import torch.nn as nn

from torchvision.models import resnet50

import torch.optim as optim

from torch.utils.data import DataLoader,Dataset

from torch.autograd import Variable

from PIL import Image

from torch.utils.data.sampler import SubsetRandomSampler

import torchvision.transforms as transforms

import cv2

import matplotlib.pyplot as plt

%matplotlib inline
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device
df = pd.read_csv('../input/widsdatathon2019/traininglabels.csv')

df.head()
class OilDataset(Dataset):

    def __init__(self,csv,transform):

        self.data = pd.read_csv(csv)

        self.transform = transform

        self.label = torch.eye(2)[self.data['has_oilpalm']]

    

    def __len__(self):

        return len(self.data)

    

    def __getitem__(self,idx):

        image_path = os.path.join('../input/widsdatathon2019/train_images/train_images/'+self.data.loc[idx,'image_id'])

        image = Image.open(image_path)

        image = self.transform(image)

        label = torch.tensor(self.data.loc[idx,'has_oilpalm'])

        return {'image':image,'labels':label}
k = Image.open('../input/widsdatathon2019/train_images/train_images/img_000012017.jpg')

k.show()
simple_transform = transforms.Compose([transforms.Resize((224,224)),transforms.RandomHorizontalFlip(),transforms.RandomRotation(45),

                                       transforms.ToTensor()

                                      ,transforms.Normalize([0.486,0.456,0.406],[0.229,0.225,0.224])])
train_dataset = OilDataset('../input/widsdatathon2019/traininglabels.csv',simple_transform)
data_size = len(train_dataset)

indics = list(range(data_size))
np.random.shuffle(indics)
validation_split = 0.1

split = int(np.round(validation_split*data_size))

train_indices,validation_indices = indics[split:],indics[:split]
train_sampler = SubsetRandomSampler(train_indices)

valid_sampler = SubsetRandomSampler(validation_indices)
train_loader = DataLoader(train_dataset,batch_size=32,sampler=train_sampler)

valid_loader = DataLoader(train_dataset,batch_size=32,sampler = valid_sampler)
model = resnet50(pretrained=False)

model.load_state_dict(torch.load('../input/resnet50/resnet50.pth'))

for params in model.parameters():

    params.require_grad = False



model.fc = nn.Sequential(

    nn.Linear(2048,1024),

    nn.ReLU(),

    nn.Linear(1024,5)

)

fc_parameter = model.fc.parameters()

for params in fc_parameter:

    params.require_grad = True

model = model.to(device)
model = model.cuda()
criteria = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(),lr=0.001)
def fit(epochs,model,optimizer,criteria):

    

    for epoch in range(epochs+1):

        training_loss = 0.0

        validation_loss = 0.0

        correct = 0

        total = 0

        

        print('{}/{} Epoch'.format(epoch+1,epochs))

        model.train()

        for batch_idx,d in enumerate(train_loader):

            data = d['image'].cuda()

            target = d['labels'].cuda()

            optimizer.zero_grad()

            output = model(data)

            loss = criteria(output,target)

            

            loss.backward()

            optimizer.step()

            

            training_loss = training_loss + ((1/(batch_idx+1)*(loss.data-training_loss)))

            

            if batch_idx%10==0:

                print('Loss is of {}'.format(training_loss))

            

            pred = output.data.max(1,keepdim = True)[1]

            correct+=np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())

            total +=data.size(0)

            if batch_idx%40==0:

                print('Accuracy for batch {} on Training {}'.format(batch_idx,(100*(correct/total))))

            

        model.eval()

        for batch_idx,d in enumerate(valid_loader):

            data = d['image'].cuda()

            target = d['labels'].cuda()

            

            output = model(data)

            loss = criteria(output,target)

            

            validation_loss = validation_loss+(((1/(batch_idx+1))*(loss.data-validation_loss)))

        

            pred = output.data.max(1,keepdim =True)[1]

            correct+=np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())

            total += data.size(0)

            if batch_idx%40==0:

                print('Validation Accuracy {}'.format((100*correct/total)))

            

    return model
fit(12,model,optimizer,criteria)
class prediction(Dataset):

    def __init__(self,csv,transform):

        self.test_data = pd.read_csv(csv)

        self.transform = transform

        

    def __len__(self):

        return len(self.test_data)

    

    def __getitem__(self,idx):

        try:

            image_path = os.path.join('../input/widsdatathon2019/leaderboard_test_data/leaderboard_test_data/',self.test_data.loc[idx]['image_id'])

            image = Image.open(image_path)

        except:

            image_path = os.path.join('../input/widsdatathon2019/leaderboard_holdout_data/leaderboard_holdout_data/',self.test_data.loc[idx]['image_id'])

            image = Image.open(image_path)

        image = self.transform(image)

        name = self.test_data.loc[idx]['image_id']

        return {'images':image,'names':name}
predict_dataset = prediction('../input/widsdatathon2019/SampleSubmission.csv',simple_transform)
prediction_loader = DataLoader(predict_dataset)
len(prediction_loader)
import gc

gc.collect()
predict = []

model.eval()

for i, d in enumerate(prediction_loader):

    data = d['images'].cuda()

    output = model(data)  

    output1 = output.cpu().detach().numpy()    

    predict.append(output1[0])
len(predict)
submission = pd.read_csv('../input/widsdatathon2019/SampleSubmission.csv')

submission['has_oilpalm'] = np.argmax(predict,1)
submission.to_csv('sample_submission.csv',index = False)