import pandas as pd

import numpy as np

import torch

import torchvision

import torch.nn as nn

from torchvision.models import densenet121,inception_v3,densenet201,resnet152,resnet18

import torchvision.transforms as transforms

import torch.optim as optim

from PIL import Image

from torch.utils.data import DataLoader,Dataset

from torch.utils.data.sampler import SubsetRandomSampler

import cv2

import seaborn as sns

from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt

import glob

%matplotlib inline

from tqdm import tqdm

tqdm.pandas()



df = pd.read_csv('../input/plant-pathology-2020-fgvc7/train.csv')

df.head()
columns = ['healthy','multiple_diseases','rust','scab']

target = []

for i in tqdm(range(len(df))):

    target.append(columns[np.argmax(df.iloc[i].values[1:])])
df['Target'] = target

le = LabelEncoder()

df['Target'] = le.fit_transform(df['Target'])

df.head()
images = glob.glob('../input/plant-pathology-2020-fgvc7/images/*.jpg')

for i in range(5):

    image = cv2.imread(images[i])

    plt.figure(figsize=(12,5))

    plt.subplot(2,4,i+1)

    plt.xticks([])

    plt.yticks([])

    plt.imshow(image)
print(df['Target'].value_counts())

sns.countplot(df['Target'])
class PlantDataset(Dataset):

    def __init__(self,csv,transformer):

        self.data = csv

        self.transform = transformer

        self.labels = torch.eye(4)[self.data['Target']]

        

    def __len__(self):

        return len(self.data)

    

    def __getitem__(self,idx):

        image = Image.open('../input/plant-pathology-2020-fgvc7/images/'+self.data.loc[idx]['image_id']+'.jpg')

        image = self.transform(image)

        labels = torch.tensor(self.data.loc[idx]['Target'])

        return {'images':image,'labels':labels}
simple_transform = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize([0.496,0.456,0.406],[0.229,0.224,0.225])])

train_dataset = PlantDataset(df,simple_transform)
indices = range(len(train_dataset))

split = int(0.1*len(train_dataset))

train_indices = indices[split:]

test_indices = indices[:split]
train_sampler = SubsetRandomSampler(train_indices)

valid_sampler = SubsetRandomSampler(test_indices)



train_loader = DataLoader(train_dataset,sampler=train_sampler,batch_size=32)

valid_loader = DataLoader(train_dataset,sampler=valid_sampler,batch_size=32)
model = resnet18(pretrained = False)

model.load_state_dict(torch.load('../input/resnet18/resnet18.pth'))

for param in model.parameters():

    param.trainable = False



model.fc = nn.Linear(512,4)

fc_parameters = model.fc.parameters()



for param in fc_parameters:

    param.trainable = True

    

model = model.cuda()
criteria = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(),lr = 0.001)
def fit(epochs,model,criteria,optimizer):

    for epoch in range(epochs+1):

        training_loss = 0.0

        validation_loss = 0.0

        correct = 0.0

        total = 0.0

        

        print('{}/{} Epochs'.format(epoch+1,epochs))

        

        model.train()

        for batch_idx,d in enumerate(train_loader):

            data = d['images'].cuda()

            target = d['labels'].cuda()

            

            optimizer.zero_grad()

            output = model(data)

            loss = criteria(output,target)

            loss.backward()

            optimizer.step()

            

            

            pred = output.data.max(1,keepdim = True)[1]

            

            training_loss = training_loss + (1+(batch_idx+1))*(loss.data-training_loss)

            

            if batch_idx % 20 ==0:

                print('Batch Id {} is having training loss of {}'.format(batch_idx,training_loss))

                

            correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())

            

            total += data.size(0)

            print('Batch Id {} is having training accuracy of {}'.format(batch_idx,(correct*100/total)))

            

        model.eval()

        for batch_idx,d in enumerate(valid_loader):

            data = d['images'].cuda()

            target = d['labels'].cuda()

            

            output = model(data)

            loss = criteria(output,target)

            

            validation_loss = validation_loss + (1/(batch_idx+1))*(loss.data-validation_loss)

            pred = output.data.max(1,keepdim = True)[1]

            if batch_idx % 20 == 0:

                print('Batch Id {} is having validation loss of {}'.format(batch_idx,validation_loss))

                

            correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())

            total += data.size(0)

            

            print('Batch Id {} is having validation accuracy of {}'.format(batch_idx,correct*100/total))  
fit(20,model,criteria,optimizer)
class PredictData(Dataset):

    def __init__(self,csv,transform):

        self.data = csv

        self.transform = transform

        

    def __len__(self):

        return len(self.data)

    

    def __getitem__(self,idx):

        image = Image.open('../input/plant-pathology-2020-fgvc7/images/'+self.data.loc[idx]['image_id']+'.jpg')

        image = self.transform(image)

        return {'images':image}
test_df = pd.read_csv('../input/plant-pathology-2020-fgvc7/test.csv')

test_df.head()
predictionData = PredictData(test_df,simple_transform)
predictloader = DataLoader(predictionData)
predict = []

for batch_idx, d in tqdm(enumerate(predictloader)):

    data = d['images'].cuda()

    

    output = model(data)

    output = output.cpu().detach().numpy()

    #output = np.argmax(output)

    predict.append(output)

    #print(output)

    
sample_submission = pd.read_csv('../input/plant-pathology-2020-fgvc7/sample_submission.csv')

sample_submission.head()
healthy = []

multiple_disease = []

rust = []

scab = []

for i in tqdm(range(len(predict))):

    healthy.append(predict[i][0][0])

    multiple_disease.append(predict[i][0][1])

    rust.append(predict[i][0][2])

    scab.append(predict[i][0][3])
sample_submission['healthy'] = healthy

sample_submission['multiple_diseases'] = multiple_disease

sample_submission['rust'] = rust

sample_submission['scab'] = scab
sample_submission.head()
sample_submission.to_csv('sample_submission.csv',index = False)