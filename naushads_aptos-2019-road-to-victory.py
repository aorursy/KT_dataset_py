# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import os

import matplotlib.pyplot as plt

%matplotlib inline

from PIL import Image



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.metrics import confusion_matrix



import albumentations

from albumentations import torch as AT



import warnings

warnings.filterwarnings('ignore')



class_desc = {

    0:"0 - No DR",

    1:"1 - Mild",

    2:"2 - Moderate",

    3:"3 - Severe",

    4:"4 - Proliferative DR"

}
# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/aptos2019-blindness-detection/train.csv')

test = pd.read_csv('../input/aptos2019-blindness-detection/test.csv')

submission = pd.read_csv('../input/aptos2019-blindness-detection/sample_submission.csv')
print(train.shape)

print(test.shape)
train.head()
print(train['diagnosis'].value_counts())

train['diagnosis'].value_counts().plot(kind='bar',title='Class Counts');
fig = plt.figure(figsize=(25,16))

#display 10 images from each class

for class_id in sorted(train['diagnosis'].unique()):

    for i, (idx,row) in enumerate(train.loc[train['diagnosis'] == class_id].sample(10).iterrows()):

#         print(f"class_id {class_id} i {i} idx {idx} row {row['id_code']}")

        ax = fig.add_subplot(5,10,class_id * 10 + i + 1,xticks=[],yticks=[])

        im = Image.open(f"../input/aptos2019-blindness-detection/train_images/{row['id_code']}.png")

        plt.imshow(im)

        ax.set_title(f'Label: {class_id}')
for class_id in sorted(train.diagnosis.unique()):

    for i, (index,rows) in enumerate(train.loc[train.diagnosis == class_id].sample(3).iterrows()):

        plt.figure(figsize=(15,15))

        im = Image.open(f"../input/aptos2019-blindness-detection/train_images/{rows['id_code']}.png")

        plt.xticks([]);plt.xticks([]);

        plt.title(class_desc[class_id])

        plt.imshow(im)

        plt.show()
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def prepare_labels_2darray(y):

    '''

    Input : labels =>   idx  class

                        0    2

                        1    4

                        2    1

    Output : 2d array => array([[0., 0., 1., 0., 0.],

                                [0., 0., 0., 0., 1.],

                                [0., 1., 0., 0., 0.],...])

    '''

    y = np.array(y)

    # y = array([2, 4, 1, ..., 2, 0, 2])

    le = LabelEncoder()

    int_enc = le.fit_transform(y)

    # int_enc = array([2, 4, 1, ..., 2, 0, 2])

    # LE is not required here, since label values start from 0 and ends with 4 for 5 classes

    # useful if class values are arbitrary eg. 2,4,6 -> transformed = 0,1,2

    int_enc = int_enc.reshape(len(int_enc),1)

    ohe = OneHotEncoder(sparse=False)

    ohe_enc = ohe.fit_transform(int_enc)

    # default is sparse=True, if that's the case,

    #ohe_enc = (0, 2)	1.0

    #          (1, 4)	1.0 , etc

    # if sparse=False,

    #ohe_enc = array([[0., 0., 1., 0., 0.],

    #                [0., 0., 0., 0., 1.],...])

    y = ohe_enc

    return y, le



y,le = prepare_labels_2darray(train["diagnosis"]);
import torchvision

import torchvision.transforms as transforms

import cv2
class GlassDataset():

    def __init__(self, df, datatype="train", transform=transforms.Compose([transforms.CenterCrop(32),transforms.ToTensor()]),y=None):

        self.df = df

        self.datatype = datatype

        self.image_files_list = [f'../input/aptos2019-blindness-detection/{self.datatype}_images/{i}.png' for i in df['id_code'].values]

        if self.datatype == 'train':

            self.labels = y

        else:

            self.labels = np.zeros((df.shape[0],5))

        self.transform = transform

        

    def __len__(self):

        return len(self.image_files_list)

    

    def __getitem__(self,idx):

        img_name = self.image_files_list[idx]

        img = cv2.imread(img_name)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        image = self.transform(image=img)

        image = image['image']

        

        #img_name_short = self.image_files_list[idx].split('.')[0]

        

        label = self.labels[idx]

        if self.datatype == 'test':

            return image, label, img_name

        else:

            return image, label
data_transformations = albumentations.Compose([

    albumentations.Resize(224,224),

    albumentations.HorizontalFlip(),

    albumentations.RandomBrightnessContrast(),

    albumentations.ShiftScaleRotate(rotate_limit=15,scale_limit=0.10),

    albumentations.JpegCompression(80),

    albumentations.HueSaturationValue(),

    albumentations.Normalize(),

    AT.ToTensor()

])

data_transformations_test = albumentations.Compose([

    albumentations.Resize(224,224),

    albumentations.Normalize(),

    AT.ToTensor()

])

dataset = GlassDataset(df=train, datatype='train', transform=data_transformations, y=y)

test_set = GlassDataset(df=test, datatype='test', transform=data_transformations_test)
from sklearn.model_selection import train_test_split

trn, validation = train_test_split(train.diagnosis, stratify=train.diagnosis, test_size=0.1)

"""

trn - type = pandas.series, content = train.diagnosis values with arbitrarily ordered idx's



print(trn[:3]) ##similar for validation

           id_code  diagnosis

2009  8d4ff745a409          0

494   233d948e2544          0

50    03e25101e8e8          1



print(trn.index[:3])

Int64Index([2009, 494, 50], dtype='int64')



print(trn.values[:3])

[0 0 1]

""";

from torch.utils.data.sampler import SubsetRandomSampler

train_sampler = SubsetRandomSampler(list(trn.index))

validation_sampler = SubsetRandomSampler(list(validation.index))

"""

print(trn.index[:3])

Int64Index([527, 879, 1959], dtype='int64')



print(list(trn.index[:3]))

[527, 879, 1959]



type(train_sampler)

torch.utils.data.sampler.SubsetRandomSampler

""";
import torch

batch_size = 64

num_workers = 0



train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)

valid_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=validation_sampler, num_workers=num_workers)

test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, num_workers=num_workers)
model_conv = torchvision.models.resnet50()

model_conv.load_state_dict(torch.load('../input/pytorch-pretrained-image-models/resnet50.pth'))

num_features = model_conv.fc.in_features # 2048

from torch import nn

model_conv.fc = nn.Linear(num_features,5)
torch.cuda.current_device() #0

torch.cuda.is_available() #True

torch.cuda.get_device_name(0) #Tesla P100-PCIE-16GB
model_conv.cuda()

criterion = nn.BCEWithLogitsLoss()

from torch import optim

optimizer = optim.SGD(model_conv.fc.parameters(), lr=0.01, momentum=0.99)

from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR

# scheduler = StepLR(optimizer, step_size=3, gamma=0.1)

scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=2)
import time
valid_loss_min = np.Inf; #inf, type-float

patience = 5

p = 0





stop = False



#number of epochs to train the model

n_epoch = 20

for epoch in range(1, n_epoch+1):

    print(time.ctime(), 'Epoch:',epoch)

    

    train_loss = []

    train_auc = []

    

    for batch_idx, (data,target) in enumerate(train_loader):

        #print(batch_idx, "data",data.shape, "target", target.shape) #data torch.Size([64, 3, 224, 224]) target torch.Size([64, 5])

        

        data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()

        output = model_conv(data)

        loss = criterion(output, target.float())

        train_loss.append(loss.item())

        

        a = target.data.cpu().numpy()

        b = output[:,-1].detach().cpu().numpy()

        loss.backward()

        optimizer.step()

    model_conv.eval()

    val_loss = []

    val_auc = []

    for batch_i, (data,target) in enumerate(valid_loader):

        data,target = data.cuda(), target.cuda()

        output = model_conv(data)

        loss = criterion(output, target.float())

        val_loss.append(loss.item())

        a = target.data.cpu().numpy()

        b = output[:,-1].detach().cpu().numpy()

    print(f"train_loss = {np.mean(train_loss):.4f} validation_loss = {np.mean(val_loss):.4f}")

    

    valid_loss = np.mean(val_loss)

    scheduler.step(valid_loss)

    if valid_loss <= valid_loss_min:

        print(f'Validation loss decreased ({valid_loss_min:.6f} --> {valid_loss:.6f})')

        torch.save(model_conv.state_dict(),'model.pth')

        valid_loss_min = valid_loss

        p = 0

        

    # check if validation loss didn't improve

    if valid_loss > valid_loss_min:

        p += 1

        print(f"{p} epochs of increasing val loss")

        if p > patience:

            print("Stopping Training")

            stop = True

            break

    

    if stop:

        break
sub = pd.read_csv('../input/aptos2019-blindness-detection/sample_submission.csv')



model_conv.eval()

for (data, target, name) in test_loader:

    data = data.cuda()

    output = model_conv(data)

    output = output.cpu().detach().numpy()

    for i, (e, n) in enumerate(list(zip(output, name))):

        sub.loc[sub['id_code'] == n.split('/')[-1].split('.')[0], 'diagnosis'] = le.inverse_transform([np.argmax(e)])

        

sub.to_csv('submission.csv', index=False)
sub.head()
sub['diagnosis'].value_counts()
actuals = np.array([4, 4, 3, 4, 4, 4, 1, 1, 2, 1])

preds   = np.array([0, 2, 1, 0, 0, 0, 1, 1, 2, 1])

O = confusion_matrix(actuals, preds); O
w = np.zeros((5,5)); w
N=5

for i in range(len(w)):

    for j in range(len(w)):

        w[i][j] = float(((i-j)**2)/(N-1)**2) #as per formula, for this competition, N=5

w
act_hist=np.zeros([N])

for item in actuals: 

    act_hist[item]+=1

    

pred_hist=np.zeros([N])

for item in preds: 

    pred_hist[item]+=1



print(f'Actuals value counts:   {act_hist} \nPrediction value counts:{pred_hist}')
E = np.outer(act_hist, pred_hist); E
print(E.sum());E = E/E.sum(); print(E.sum())
print(O.sum()); O = O/O.sum(); print(O.sum())
E
O
num=0

den=0

for i in range(len(w)):

    for j in range(len(w)):

        num+=w[i][j]*O[i][j]

        den+=w[i][j]*E[i][j]

 

weighted_kappa = (1 - (num/den)); weighted_kappa
def quadratic_kappa(actuals, preds, N=5):

    """This function calculates the Quadratic Kappa Metric used for Evaluation in the PetFinder competition

    at Kaggle. It returns the Quadratic Weighted Kappa metric score between the actual and the predicted values 

    of adoption rating."""

    O = confusion_matrix(actuals, preds)

    w = np.zeros((N,N))

    for i in range(len(w)): 

        for j in range(len(w)):

            w[i][j] = float(((i-j)**2)/(N-1)**2)

    

    act_hist=np.zeros([N])

    for item in actuals: 

        act_hist[item]+=1

    

    pred_hist=np.zeros([N])

    for item in preds: 

        pred_hist[item]+=1

                         

    E = np.outer(act_hist, pred_hist);

    E = E/E.sum();

    O = O/O.sum();

    num=0

    den=0

    for i in range(len(w)):

        for j in range(len(w)):

            num+=w[i][j]*O[i][j]

            den+=w[i][j]*E[i][j]

    return (1 - (num/den))
actuals = np.array([4, 4, 3, 4, 4, 4, 1, 1, 2, 0])

preds   = np.array([4, 4, 3, 4, 4, 4, 1, 1, 2, 0])

quadratic_kappa(actuals, preds)