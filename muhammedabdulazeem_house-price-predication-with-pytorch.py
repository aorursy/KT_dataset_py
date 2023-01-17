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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import torch

import torch.nn as nn

import torch.nn.functional as F

pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', None)
data = pd.read_csv('/kaggle/input/housesalesprediction/kc_house_data.csv')

data.head()
data.drop(['id','date'],axis=1,inplace=True)
data.shape
data.drop('zipcode',axis=1,inplace=True)
data.head()
sns.distplot(data['price'])
from statsmodels.graphics.gofplots import qqplot
fig = qqplot(data['price'],line='s')

plt.show()
data['log_price'] = np.log1p(data['price'])
fig = qqplot(data['log_price'],line='s')

plt.show()
data['log_price_2'] = np.log(data['price'])
fig = qqplot(data['log_price_2'],line='s')

plt.show()
data.drop('log_price_2',axis=1,inplace=True)
data.drop('price',axis=1,inplace=True)
sns.distplot(data['log_price'])
from sklearn.model_selection import train_test_split
sns.countplot(data['bedrooms'])
data.drop(data[data['bedrooms'] == 33].index,inplace=True)
bed_map = {

    1:1,

    2:2,

    3:3,

    4:4,

    5:5,

    6:6,

    0:7,

    7:7,

    8:7,

    9:7,

    10:7,

    11:7

}
data['bedrooms'] = data['bedrooms'].map(bed_map)
sns.countplot(data['bedrooms'])
data['bathrooms'] = data['bathrooms'].values.round()
sns.countplot(data['bathrooms'])
bath_map = {

    1:1,

    2:2,

    3:3,

    4:4,

    5:5,

    6:5,

    0:5,

    7:5,

    8:5

}
data['bathrooms'] = data['bathrooms'].map(bath_map)
sns.countplot(data['bathrooms'])
fig = qqplot(data['sqft_living'],line='s')

plt.show()
fig = qqplot(data['sqft_lot'],line='s')

plt.show()
fig = qqplot(data['sqft_above'],line='s')

plt.show()
fig = qqplot(data['sqft_living15'],line='s')

plt.show()
fig = qqplot(data['sqft_lot15'],line='s')

plt.show()
data['sqft_living'] = np.log1p(data['sqft_living'])

data['sqft_lot'] = np.log1p(data['sqft_lot'])

data['sqft_above'] = np.log1p(data['sqft_above'])

data['sqft_living15'] = np.log1p(data['sqft_living15'])

data['sqft_lot15'] = np.log1p(data['sqft_lot15'])
fig = qqplot(data['sqft_living'],line='s')

plt.show()
fig = qqplot(data['sqft_lot'],line='s')

plt.show()
fig = qqplot(data['sqft_above'],line='s')

plt.show()
fig = qqplot(data['sqft_living15'],line='s')

plt.show()
fig = qqplot(data['sqft_lot15'],line='s')

plt.show()
data.head()
sns.countplot(data['floors'])
data['floors'] = data['floors'].values.round()
sns.countplot(data['floors'])
data.drop(data[data['floors'] == 4].index,inplace=True)
data['waterfront'].value_counts()
data['view'].value_counts()
sns.countplot(data['condition'])
condition_map = {

    1:1,

    2:1,

    3:2,

    4:3,

    5:4

}
data['condition'] = data['condition'].map(condition_map)
sns.countplot(data['grade'])
grade_map = {

    1:1,

    2:1,

    3:1,

    4:1,

    5:1,

    6:2,

    7:3,

    8:4,

    9:5,

    10:6,

    11:7,

    12:7,

    13:7

}
data['grade'] = data['grade'].map(grade_map)
sns.distplot(data['sqft_basement'])
plt.figure(figsize=(20,15))

sns.countplot(y = 'yr_built',data=data)
data[data['yr_renovated'] == 0].shape
actually_renovated = data[data['yr_renovated'] != 0]

actually_renovated.head()
plt.figure(figsize=(20,15))

sns.countplot(y = 'yr_renovated',data=actually_renovated)
actually_renovated.loc[actually_renovated['yr_renovated'] <= 1960,'yr_renovated'] = 1960
actually_renovated.loc[(actually_renovated['yr_renovated'] >= 1961) & (actually_renovated['yr_renovated'] <= 1970),'yr_renovated'] = 1970
plt.figure(figsize=(20,15))

sns.countplot(y = 'yr_renovated',data=actually_renovated)
data.loc[data['yr_renovated'] <= 1960,'yr_renovated'] = 1960
data.loc[(data['yr_renovated'] >= 1961) & (data['yr_renovated'] <= 1970),'yr_renovated'] = 1970
data.loc[(data['yr_renovated'] >= 1971) & (data['yr_renovated'] <= 1980),'yr_renovated'] = 1980
data.columns
data.drop(['lat','long'],axis=1,inplace=True)
features = data.drop('log_price',axis=1)

labels = data['log_price']
X_train, X_val, y_train, y_val = train_test_split(features.values, labels.values, test_size=0.15, random_state=42)
X_train = torch.FloatTensor(X_train)

X_test = torch.FloatTensor(X_val)

y_train = torch.LongTensor(y_train)

y_test = torch.LongTensor(y_val)
class ANN_Model(nn.Module):

    def __init__(self,input_features=15,hidden1=32,hidden2=64,hidden3=128,hidden4=256,out_features=1):

        super().__init__()

        self.f_connected1=nn.Linear(input_features,hidden1)

        self.f_connected2=nn.Linear(hidden1,hidden2)

        self.f_connected3=nn.Linear(hidden2,hidden3)

        self.f_connected4=nn.Linear(hidden3,hidden4)

        self.out=nn.Linear(hidden4,out_features)

    def forward(self,x):

        x=F.relu(self.f_connected1(x))

        x=F.relu(self.f_connected2(x))

        x=F.relu(self.f_connected3(x))

        x=F.relu(self.f_connected4(x))

        x=self.out(x)

        return x

        
torch.manual_seed(42)
model = ANN_Model()

model.parameters
loss_function = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
epochs = 100

final_losses = []



for epoch in range(epochs):

    epoch = epoch + 1

    ypred = model.forward(X_train)

    loss = torch.sqrt(loss_function(ypred,y_train))

    final_losses.append(loss)



    

    print('{} and loss is {}'.format(epoch,loss.item()))



    optimizer.zero_grad()

    loss.backward()

    optimizer.step()  
khkbkb
import math

epochs = 100

train_losses = []

test_losses = []



for epoch in range(epochs):

    model.train()

    train_loss = 0

    

    for batch in range(len(train_batch)):

        optimizer.zero_grad()

        output = model(train_batch[i])

        loss = torch.sqrt(loss_function(torch.log(output),torch.log(label_batch[i])))

        loss.backward()

        optimizer.step()

        

        train_loss += loss.item()

        

    else:

        test_loss = 0

        accuracy = 0

        

        with torch.no_grad():

            model.eval()

            predications = model(X_test)

            print(predications.shape,X_test.shape)

            test_loss += torch.sqrt(loss_function(predications,y_test))

            

        train_losses.append(train_loss/len(train_batch))

        test_losses.append(test_loss)

        print("Epoch: {}/{}.. ".format(epoch+1, epochs),

              "Training Loss: {:.3f}.. ".format(train_loss/len(train_batch)),

              "Test Loss: {:.3f}.. ".format(test_loss))