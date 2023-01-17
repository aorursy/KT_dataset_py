import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import torch 

import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader,Dataset , SubsetRandomSampler 

import torch.nn.functional as F

import torchvision

import torchvision.transforms as transforms

import torch.optim as optim

from torch.optim import lr_scheduler

import cv2

import albumentations

from albumentations import torch as AT

from torch.optim import lr_scheduler



from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold,StratifiedKFold

from sklearn.metrics import roc_auc_score

from sklearn.metrics import cohen_kappa_score

import os

print(os.listdir("../input"))
base_dr = "../input/aptos2019-blindness-detection/"
# Reading the CSVs

train = pd.read_csv(base_dr+'train.csv')

test = pd.read_csv(base_dr+'test.csv')
def cohen_k_score(y_true , y_pred):

    skl = cohen_kappa_score(np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1), weights='quadratic')

    return skl
train.head() 
test.head()
#### Label Distribution:

import seaborn as sns

sns.countplot(x='diagnosis' , data=train);
print('No of images in the train set:', format(train.shape[0]))

print('No of images in the test set:', format(test.shape[0]))
diagnosis = {'No DR': 0, 

'Mild': 1, 

'Moderate': 2, 

'Severe': 3, 

'Proliferative DR': 4}
### preparing some useful stuff for easy vizualizations later on

Rev = dict((v,k) for k,v in diagnosis.items())  ### need to reverse

dia = pd.DataFrame(train['diagnosis'].map(Rev))

dia.rename(columns={'diagnosis': "diagnosisText"},inplace=True)

dia = pd.concat([train,pd.DataFrame(dia)],1)

dia.head()
from random import sample

import cv2

import matplotlib.image as mpimg



def plotClass(category,N):

    # credit : https://www.kaggle.com/pheadrus/purepytorchmodels?scriptVersionId=15756053

    categoryIdx = dia[dia['diagnosisText']==category].index[:30]

    randIdx = sample(list(categoryIdx),N)

    jpegName = dia.iloc[randIdx,:]['id_code'].values

    fig = plt.figure(figsize=(18,14))

    for i , jpeg in enumerate(list(jpegName)):

        plt.subplot(1,N ,i+1)

        imgFile = mpimg.imread('{}/train_images/{}.png'.format(base_dr,jpeg))

        plt.imshow(imgFile)
plotClass('No DR',4)
plotClass('Proliferative DR',4)
plotClass('Severe',4)
plotClass('Moderate',4)
from sklearn.preprocessing import OneHotEncoder,LabelEncoder



def prepare_labels(y):

    values = np.array(y)

    label_encoder = LabelEncoder()

    integer_encoded = label_encoder.fit_transform(values)



    onehot_encoder = OneHotEncoder(sparse=False)

    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)

    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)



    y = onehot_encoded

    return y, label_encoder
y, le_full = prepare_labels(train['diagnosis'])

y.shape
y[0:5]
train.iloc[0:5]['diagnosis']
class BlindnessDataset(Dataset):

    def __init__(self, df, datatype='train', transform = transforms.Compose([transforms.CenterCrop(32),transforms.ToTensor()]), y = None):

        self.df = df

        self.datatype = datatype

        self.image_files_list = [f'../input/aptos2019-blindness-detection/{self.datatype}_images/{i}.png' for i in df['id_code'].values]

        if self.datatype == 'train':

            self.labels = y

        else:

            self.labels = np.zeros((df.shape[0], 5))

        self.transform = transform



    def __len__(self):

        return len(self.image_files_list)



    def __getitem__(self, idx):

        img_name = self.image_files_list[idx]

        img = cv2.imread(img_name)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        image = self.transform(image=img)

        image = image['image']



        img_name_short = self.image_files_list[idx].split('.')[0]



        label = self.labels[idx]

        if self.datatype == 'test':

            return image, label, img_name

        else:

            return image, label
#data_transforms = albumentations.Compose([

#    albumentations.Resize(224, 224),

#    albumentations.HorizontalFlip(),

#    albumentations.RandomBrightness(),

#    albumentations.ShiftScaleRotate(rotate_limit=15, scale_limit=0.10),

#    albumentations.JpegCompression(80),

#    albumentations.HueSaturationValue(),

#    albumentations.Normalize(),

#    AT.ToTensor()

#    ])

data_transforms = albumentations.Compose([

    albumentations.Resize(224, 224),

    albumentations.HorizontalFlip(p=0.5),

    albumentations.Transpose(p=0.5),

    albumentations.Flip(p=0.0),

    albumentations.OneOf([

        albumentations.CLAHE(clip_limit=2), albumentations.IAASharpen(), albumentations.IAAEmboss(), 

        albumentations.RandomBrightness(), albumentations.RandomContrast(),

        albumentations.JpegCompression(), albumentations.Blur(), albumentations.GaussNoise()], p=0.5), 

    albumentations.HueSaturationValue(p=0.5), 

    albumentations.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, rotate_limit=45, p=0.5),

    albumentations.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),

    AT.ToTensor()

   ])

data_transforms_test = albumentations.Compose([

    albumentations.Resize(224, 224),

    albumentations.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),

    #albumentations.Random

    AT.ToTensor()

    ])
batch_size = 64

num_workers = 8



## 90 - 10 train & validation split 

tr, val = train_test_split(train.diagnosis, stratify=train.diagnosis, test_size=0.1)



# SubsetSampler for train & validation

train_sampler = SubsetRandomSampler(list(tr.index))

valid_sampler = SubsetRandomSampler(list(val.index))



# Train Dataset

dataset = BlindnessDataset(df=train, datatype='train', transform=data_transforms, y=y)



#Test Dataset

test_set = BlindnessDataset(df=test, datatype='test', transform=data_transforms_test)



# prepare data loaders (combine dataset and sampler)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 

                                           sampler=train_sampler, num_workers=num_workers)

valid_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 

                                           sampler=valid_sampler, num_workers=num_workers)

test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, 

                                          num_workers=num_workers)
images_batch, labels_batch = iter(train_loader).next()

print(images_batch.shape)

print(labels_batch.shape)
model = torchvision.models.resnet50()

model.load_state_dict(torch.load("../input/resnet50/resnet50.pth"))
#model
# Freeze model weights

for param in model.parameters():

    param.requires_grad = False
def count_parameters(model):

    '''

    Count of trainable weights in a model

    '''

    return sum(p.numel() for p in model.parameters() if p.requires_grad)



count_parameters(model)
for name, child in model.named_children():

    print(name)
for name, child in model.named_children():

    if name in ['layer3', 'layer4']:

        print(name + ' is unfrozen')

        for param in child.parameters():

            param.requires_grad = True

    else:

        print(name + ' is frozen')

        for param in child.parameters():

            param.requires_grad = False
num_ftrs = model.fc.in_features

print(num_ftrs)

#model.fc = nn.Sequential(

#                          nn.BatchNorm1d(num_ftrs, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),

#                          nn.Dropout(p=0.5),

#                          nn.Linear(in_features=2048, out_features=1024, bias=True),

#                          nn.SELU(),

#                          #nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),

#                          #nn.Dropout(p=0.5),

#                          #nn.Linear(in_features=2048, out_features=1024, bias=True),

#                          nn.BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),

#                          nn.SELU(),

#                          nn.Dropout(p=0.4),

#                          nn.Linear(in_features=1024, out_features=100, bias=True),

#                          nn.SELU(),

#                          nn.Dropout(p=0.3),

#                          nn.Linear(in_features=100, out_features=5, bias=True),

#                         )



model.fc =  model.last_linear = nn.Sequential(

                          nn.BatchNorm1d(num_ftrs, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),

                          nn.Dropout(p=0.25),

                          nn.Linear(in_features=2048, out_features=2048, bias=True),

                          nn.ReLU(),

                          nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),

                          nn.Dropout(p=0.5),

                          nn.Linear(in_features=2048, out_features=5, bias=True),

                         )
model
model.cuda()

criterion = nn.BCEWithLogitsLoss()



optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr= 1e-4)

#scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

#scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)

scheduler = lr_scheduler.StepLR(optimizer, 5, gamma=0.2)

scheduler_cosineAL = lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader), eta_min=1e-6)
useCosine = True

valid_ck_min = 0

patience = 10

# current number of epochs, where validation loss didn't increase

p = 0

# whether training should be stopped

stop = False



# number of epochs to train the model

n_epochs = 70

for epoch in range(1, n_epochs+1):

    print('Epoch:', format(epoch))



    train_loss = []

    train_ck_score = []



    for batch_i, (data, target) in enumerate(train_loader):

        

        model.train()



        data, target = data.cuda(), target.cuda()

        

        #print('target:',format(target.shape))



        optimizer.zero_grad()

        output = model(data)

        

        #print('output:',format(output.shape))

        

        loss = criterion(output, target.float())

        train_loss.append(loss.item())

        

        a = target.data.cpu().numpy()

        #b = output[:,-1].detach().cpu().numpy()

        b = output.detach().cpu().numpy()

        train_ck_score.append(cohen_k_score(a, b))

        

        loss.backward()

        optimizer.step()

        

    model.eval()

    val_loss = []

    val_ck_score = []

    for batch_i, (data, target) in enumerate(valid_loader):

        data, target = data.cuda(), target.cuda()

        output = model(data)



        loss = criterion(output, target.float())



        val_loss.append(loss.item()) 

        a = target.data.cpu().numpy()

        b = output.detach().cpu().numpy()

        val_ck_score.append(cohen_k_score(a, b))



    # print(f'Epoch {epoch}, train loss: {np.mean(train_loss):.4f}, valid loss: {np.mean(val_loss):.4f}, train auc: {np.mean(train_auc):.4f}, valid auc: {np.mean(val_auc):.4f}')

    print(f'Epoch {epoch}, train loss: {np.mean(train_loss):.4f}, valid loss: {np.mean(val_loss):.4f}.')

    print(f'Epoch {epoch}, train cohen: {np.mean(train_ck_score):.4f}, valid cohen: {np.mean(val_ck_score):.4f}.')

    

    #valid_loss = np.mean(val_loss)

    val_ck_score = np.mean(val_ck_score)

    

    if useCosine:

        scheduler_cosineAL.step()

    else:

        scheduler.step()

    

    #scheduler.step(val_ck_score)

    if val_ck_score > valid_ck_min:

        print('Validation CK score increased ({:.6f} --> {:.6f}).  Saving model ...'.format(

        valid_ck_min,

        val_ck_score))

        #torch.save(model_conv.state_dict(), 'model.pt')

        valid_ck_min = val_ck_score

        p = 0



    # check if validation loss didn't improve

    if val_ck_score < valid_ck_min:

        p += 1

        print(f'{p} epochs of decreasing val ck score')

        if p > patience:

            print('Stopping training : Early Stopping')

            stop = True

            break        

            

    if stop:

        break
sub = pd.read_csv('../input/aptos2019-blindness-detection/sample_submission.csv')



model.eval()

for (data, target, name) in test_loader:

    data = data.cuda()

    output = model(data)

    output = output.cpu().detach().numpy()

    for i, (e, n) in enumerate(list(zip(output, name))):

        sub.loc[sub['id_code'] == n.split('/')[-1].split('.')[0], 'diagnosis'] = le_full.inverse_transform([np.argmax(e)])

        

sub.to_csv('submission.csv', index=False)
sub.head()
sub.diagnosis.value_counts(normalize = True)