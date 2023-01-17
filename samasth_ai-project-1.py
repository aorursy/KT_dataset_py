# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/career-con-2019/X_train.csv')

target = pd.read_csv('../input/career-con-2019/y_train.csv')
from seaborn import countplot

plt.figure(figsize=(23,5)) 

sns.set(style="white")

countplot(x="group_id", data=target, order = target['group_id'].value_counts().index)

plt.show()
from sklearn import preprocessing

%matplotlib inline

le = preprocessing.LabelEncoder()



target['surface'] = le.fit_transform(target['surface'])
X_tra = pd.DataFrame(data[:485120][:])

X_tes = pd.DataFrame(data[485120:][:])

Y_tra = pd.DataFrame(target[:3790]['surface'])

Y_tes = pd.DataFrame(target[3790:]['surface'])
X_tra.shape
X_tra.describe()
def quaternion_to_euler(x, y, z, w):

    import math

    t0 = +2.0 * (w * x + y * z)

    t1 = +1.0 - 2.0 * (x * x + y * y)

    X = math.atan2(t0, t1)



    t2 = +2.0 * (w * y - z * x)

    t2 = +1.0 if t2 > +1.0 else t2

    t2 = -1.0 if t2 < -1.0 else t2

    Y = math.asin(t2)



    t3 = +2.0 * (w * z + x * y)

    t4 = +1.0 - 2.0 * (y * y + z * z)

    Z = math.atan2(t3, t4)



    return X, Y, Z
def fe_step0 (actual):

    

    actual['norm_quat'] = (actual['orientation_X']**2 + actual['orientation_Y']**2 + actual['orientation_Z']**2 + actual['orientation_W']**2)

    actual['mod_quat'] = (actual['norm_quat'])**0.5

    actual['norm_X'] = actual['orientation_X'] / actual['mod_quat']

    actual['norm_Y'] = actual['orientation_Y'] / actual['mod_quat']

    actual['norm_Z'] = actual['orientation_Z'] / actual['mod_quat']

    actual['norm_W'] = actual['orientation_W'] / actual['mod_quat']

    

    return actual
X_tra = fe_step0(X_tra)

X_tes = fe_step0(X_tes)

print(X_tra.shape)

X_tra.head()
import matplotlib.pyplot as plt

import seaborn as sns

fig, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4, figsize=(18, 5))



ax1.set_title('quaternion X')

sns.kdeplot(X_tra['norm_X'], ax=ax1, label="train")

sns.kdeplot(X_tes['norm_X'], ax=ax1, label="test")



ax2.set_title('quaternion Y')

sns.kdeplot(X_tra['norm_Y'], ax=ax2, label="train")

sns.kdeplot(X_tes['norm_Y'], ax=ax2, label="test")



ax3.set_title('quaternion Z')

sns.kdeplot(X_tra['norm_Z'], ax=ax3, label="train")

sns.kdeplot(X_tes['norm_Z'], ax=ax3, label="test")



ax4.set_title('quaternion W')

sns.kdeplot(X_tra['norm_W'], ax=ax4, label="train")

sns.kdeplot(X_tes['norm_W'], ax=ax4, label="test")



plt.show()
def fe_step1 (actual):

    """Quaternions to Euler Angles"""

    

    x, y, z, w = actual['norm_X'].tolist(), actual['norm_Y'].tolist(), actual['norm_Z'].tolist(), actual['norm_W'].tolist()

    nx, ny, nz = [], [], []

    for i in range(len(x)):

        xx, yy, zz = quaternion_to_euler(x[i], y[i], z[i], w[i])

        nx.append(xx)

        ny.append(yy)

        nz.append(zz)

    

    actual['euler_x'] = nx

    actual['euler_y'] = ny

    actual['euler_z'] = nz

    return actual
X_tra = fe_step1(X_tra)

X_tes = fe_step1(X_tes)

print (X_tra.shape)

X_tra.head()
fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(15, 5))



ax1.set_title('Roll')

sns.kdeplot(X_tra['euler_x'], ax=ax1, label="train")

sns.kdeplot(X_tes['euler_x'], ax=ax1, label="test")



ax2.set_title('Pitch')

sns.kdeplot(X_tra['euler_y'], ax=ax2, label="train")

sns.kdeplot(X_tes['euler_y'], ax=ax2, label="test")



ax3.set_title('Yaw')

sns.kdeplot(X_tra['euler_z'], ax=ax3, label="train")

sns.kdeplot(X_tes['euler_z'], ax=ax3, label="test")



plt.show()
X_tra.head()
def feat_eng(data):

    

    df = pd.DataFrame()

    data['totl_anglr_vel'] = (data['angular_velocity_X']**2 + data['angular_velocity_Y']**2 + data['angular_velocity_Z']**2)** 0.5

    data['totl_linr_acc'] = (data['linear_acceleration_X']**2 + data['linear_acceleration_Y']**2 + data['linear_acceleration_Z']**2)**0.5

    data['totl_xyz'] = (data['orientation_X']**2 + data['orientation_Y']**2 + data['orientation_Z']**2)**0.5

    data['acc_vs_vel'] = data['totl_linr_acc'] / data['totl_anglr_vel']

    

    def mean_change_of_abs_change(x):

        return np.mean(np.diff(np.abs(np.diff(x))))

    

    for col in data.columns:

        if col in ['row_id','series_id','measurement_number']:

            continue

        df[col + '_mean'] = data.groupby(['series_id'])[col].mean()

        df[col + '_median'] = data.groupby(['series_id'])[col].median()

        df[col + '_max'] = data.groupby(['series_id'])[col].max()

        df[col + '_min'] = data.groupby(['series_id'])[col].min()

        df[col + '_std'] = data.groupby(['series_id'])[col].std()

        df[col + '_range'] = df[col + '_max'] - df[col + '_min']

        df[col + '_maxtoMin'] = df[col + '_max'] / df[col + '_min']

        df[col + '_mean_abs_chg'] = data.groupby(['series_id'])[col].apply(lambda x: np.mean(np.abs(np.diff(x))))

        df[col + '_mean_change_of_abs_change'] = data.groupby('series_id')[col].apply(mean_change_of_abs_change)

        df[col + '_abs_max'] = data.groupby(['series_id'])[col].apply(lambda x: np.max(np.abs(x)))

        df[col + '_abs_min'] = data.groupby(['series_id'])[col].apply(lambda x: np.min(np.abs(x)))

        df[col + '_abs_avg'] = (df[col + '_abs_min'] + df[col + '_abs_max'])/2

    return df
%%time

X_tra = feat_eng(X_tra)

X_tes = feat_eng(X_tes)

#print ("New features: ",data.shape)
X_t = X_tra[:3600]

X_v = X_tra[3600:]

Y_t = Y_tra[:3600]

Y_v = Y_tra[3600:]
from sklearn.model_selection import KFold, StratifiedKFold

folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=59)
predicted = np.zeros((X_tes.shape[0],9))

measured= np.zeros((X_tra.shape[0]))

score = 0


from sklearn.ensemble import RandomForestClassifier

import gc



for times, (trn_idx, val_idx) in enumerate(folds.split(X_tra.values,Y_tra['surface'].values)):

    model = RandomForestClassifier(n_estimators=500, n_jobs = -1)

    #print(trn_idx)

    #print(val_idx)

    #print(times)

    #model = RandomForestClassifier(n_estimators=500, max_depth=10, min_samples_split=5, n_jobs=-1)

    model.fit(X_tra.iloc[trn_idx],Y_tra['surface'][trn_idx])

    measured[val_idx] = model.predict(X_tra.iloc[val_idx])

    predicted += model.predict_proba(X_tes)/folds.n_splits

    #print(predicted.shape)

    score += model.score(X_tra.iloc[val_idx],Y_tra['surface'][val_idx])

    print("Fold: {} score: {}".format(times,model.score(X_tra.iloc[val_idx],Y_tra['surface'][val_idx])))



    importances = model.feature_importances_

    #print(len(importances))

    indices = np.argsort(importances)

    #print(indices)

    features = X_tra.columns

    #print(features)

    #print(len(features))

    

    if model.score(X_tra.iloc[val_idx],Y_tra['surface'][val_idx]) > 0.92000:

        hm = 30

        plt.figure(figsize=(7, 10))

        plt.title('Feature Importances')

        plt.barh(range(len(indices[:hm])), importances[indices][:hm], color='b', align='center')

        plt.yticks(range(len(indices[:hm])), [features[i] for i in indices])

        plt.xlabel('Relative Importance')

        plt.show()



    gc.collect()
sub = pd.read_csv('../input/career-con-2019/sample_submission.csv')

sub = sub[3796:]
sub['surface'] = predicted.argmax(axis=1)

sub = pd.DataFrame(sub['surface'])

sub.head()
from sklearn.metrics import classification_report

di = classification_report(Y_tes, sub, output_dict=True)

print(di['accuracy'])
import torch

from torchvision import datasets

import torchvision.transforms as transforms
X = torch.from_numpy(X_t.values)

y = torch.from_numpy(Y_t.values)

X_va = torch.from_numpy(X_v.values)

Y_va = torch.from_numpy(Y_v.values)

X_te = torch.from_numpy(X_tes.values)

y_te = torch.from_numpy(Y_tes.values)
import torch

import torch.nn as nn

import torch.nn.functional as F

import matplotlib.pyplot as plt

import sklearn.datasets
from sklearn.preprocessing import StandardScaler

sc=StandardScaler() # normalization, x-mean/std

X=sc.fit_transform(X) # apply to data

X=torch.tensor(X) # convert numpy to torch tensor

#y=torch.tensor(y).unsqueeze(1) # add extra dim (768,) to (768,1)

y=torch.tensor(y) # add extra dim (768,) to (768,1)



X_va = sc.fit_transform(X_va) # apply to data

X_va=torch.tensor(X_va) # convert numpy to torch tensor

#y=torch.tensor(y).unsqueeze(1) # add extra dim (768,) to (768,1)

Y_va=torch.tensor(Y_va) # add extra dim (768,) to (768,1)



X_te = sc.fit_transform(X_te)

X_te = torch.tensor(X_te) # convert numpy to torch tensor

#y_te = torch.tensor(y_te).unsqueeze(1) # add extra dim (768,) to (768,1)

y_te = torch.tensor(y_te) # add extra dim (768,) to (768,1)
print(X.shape,X.dtype)

print(y.shape,y.dtype)

print(X_va.shape,X_va.dtype)

print(Y_va.shape,Y_va.dtype)

print(X_te.shape,X_te.dtype)

print(y_te.shape,y_te.dtype)
from torch.autograd import Variable

X=Variable(X) #Variable initialization

y=Variable(y)



X_va = Variable(X_va)

Y_va = Variable(Y_va)



X_te = Variable(X_te)

y_te = Variable(y_te)
from torch.utils.data import Dataset

class Dataset(Dataset):

  def __init__(self,x,y):

    self.x=x

    self.y=y

  

  def __getitem__(self,index):

    return self.x[index], self.y[index]

  

  def __len__(self):

    return len(self.x)
dataset=Dataset(X,y)

print(dataset.x.shape)

print(dataset.y.shape)
train_loader= torch.utils.data.DataLoader(dataset=dataset, batch_size=50, shuffle=True) # load data

#test_loader= torch.utils.data.DataLoader(dataset=dataset_test, shuffle=True) # load data
class Model(nn.Module):

  def __init__(self):

    super(Model,self).__init__()

    self.fc1 =torch.nn.Linear(X.shape[1],300)

    self.relu=nn.ReLU() 

    self.batchnorm1 = nn.BatchNorm1d(300)

    self.sigmoid=torch.nn.Sigmoid()

    self.fc2 =torch.nn.Linear(300,150) 

    self.batchnorm2 = nn.BatchNorm1d(150)

    self.fc3 =torch.nn.Linear(150,9)

    #self.batchnorm3 = nn.BatchNorm1d(100)

    #self.fc4 =torch.nn.Linear(100,9)

    #self.sigmoid=torch.nn.Sigmoid()

    self.softmax = nn.Softmax()



    

  def forward(self,x):

    out =self.fc1(x)

    out = self.batchnorm1(out)

    out =self.sigmoid(out)

    #out = self.relu(out)

    out =self.fc2(out)

    out = self.batchnorm2(out)

    #out =self.sigmoid(out)

    out = self.relu(out)

    out =self.fc3(out)

    #out = self.batchnorm3(out)

    #out = self.relu(out)

    #out =self.sigmoid(out)

    #out =self.fc4(out)

    #out= self.sigmoid(out)

    out = self.softmax(out)

    return out
net = Model()

#criterion =torch.nn.MSELoss(size_average=True)

criterion = torch.nn.CrossEntropyLoss(size_average=True)

optimizer =torch.optim.Adam(net.parameters(), lr=0.01)
from sklearn.metrics import classification_report

num_epochs = 200

for epoch in range(num_epochs):

  data = 0

  for inputs, labels in train_loader:

    inputs = Variable(inputs.float())

    labels= Variable(labels.float())

    output = net(inputs)

    #print(torch.round(output))

    #print(output.argmax(1))

    optimizer.zero_grad()

    #print(output.argmax(1))

    #print(labels.squeeze())

    loss = criterion(output,labels.squeeze().long())

    #output = (output>0.5).float()

    #data += (output == labels).float().sum()

    data += classification_report(labels, output.argmax(1),output_dict=True)['accuracy']

    #data += (output.argmax(1) == labels.argmax(1)).float().sum()

    # loss = -(labels * torch.log(output)+ (1-labels) * torch.log(1-output)).mean()

    loss.backward()

    optimizer.step()

    #print(output - labels)

  print(data/76)

  #output = (output>0.5).float()

  #print(output)

  #print(labels)

  #print(classification_report(labels, output))

  #data += (output == labels).float().sum()

  #print(correct)

  #print("Epoch {}/{}, loss:{:.3f}, Accuracy: {:.3f}".format(epoch+1,num_epochs,loss.item(),data/2800*9))
inputs = Variable(X_va.float())

labels= Variable(Y_va.float())

output = net(inputs)

pre_d = output.argmax(1).unsqueeze(1)

y_tru = labels
type(output)

di = classification_report(y_tru, pre_d, output_dict=True)

print(di['accuracy'])
inputs = Variable(X_te.float())

labels= Variable(y_te.float())

output = net(inputs)

pre_d = output.argmax(1).unsqueeze(1)

y_tru = labels

type(output)

di = classification_report(y_tru, pre_d, output_dict=True)

print(di['accuracy'])