import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

sns.set_palette('muted')

import torch

from torch.utils.data import Dataset

! mkdir models
df = pd.read_csv('../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')

df.head()
try:

    df.drop(columns=['id','name','host_id','host_name','last_review'],inplace=True)

except:

    pass

df.rename(columns={'neighbourhood_group':'borough'},inplace=True)

df=df[df['availability_365']>0]

df['reviews_per_month'].fillna(0,inplace=True)

df.head()
df.shape
temp_df = df['room_type'].value_counts(ascending=False)

print(temp_df)

plt.figure(figsize=[10,3])

sns.countplot(y='room_type',data=df,order=temp_df.index).set(title='Listings per Room Type',

                                                            xlabel='Listings',

                                                            ylabel='Room Type')
temp_column = 'borough'

temp_df = df[temp_column].value_counts(ascending=False)

print(temp_df)

plt.figure(figsize=[10,5])

sns.countplot(y=temp_column,data=df,order=temp_df.index).set(title='Listing per Borough',

                                                            xlabel='Listings',

                                                            ylabel='Borough')
df[['price']].describe()
plt.figure(figsize=(15,3))

sns.distplot(df[df['price']<(111+254)]['price'])
df[['minimum_nights']].describe()
plt.figure(figsize=(15,3))

sns.distplot(df[df['minimum_nights']<(5+30)]['minimum_nights'])
df[['number_of_reviews']].describe()
plt.figure(figsize=(15,3))

sns.distplot(df[df['number_of_reviews']<(24+44)]['number_of_reviews'])
df[['reviews_per_month']].fillna(0).describe()
plt.figure(figsize=(15,3))

sns.distplot(df[df['reviews_per_month']<(1.58+1.6)][['reviews_per_month']].fillna(0))
df[['calculated_host_listings_count']].describe()
plt.figure(figsize=(15,3))

sns.distplot(df[df['calculated_host_listings_count']<3+40]['calculated_host_listings_count'])
df[['availability_365']].describe()
plt.figure(figsize=(15,3))

sns.distplot(df[df['availability_365']<(305+126)]['availability_365'])
plt.figure(figsize=(15,5))

sns.scatterplot('minimum_nights','price',data=df).set(title='Minimum Nights vs Price per Night',

                                                     xlabel='Minimum Nights',

                                                     ylabel='Price per Night')
temp_df = df[['borough','price']].groupby('borough').describe().sort_values(by=('price','50%'),ascending=False)

temp_df
plt.figure(figsize=(15,7))

sns.boxplot('price','borough',data=df[df['price']<(220+291)],order=temp_df.index).set(title='Borough vs Price per Night',

                                                     xlabel='Price per Night',

                                                     ylabel='Borough')
temp_df = df[['room_type','price']].groupby('room_type').describe().sort_values(by=('price','50%'),ascending=False)

temp_df
plt.figure(figsize=(15,5))

sns.boxplot('price','room_type',data=df[df['price']<(229+284)],order=temp_df.index).set(title='Room Type vs Price per Night',

                                                     xlabel='Room Type',

                                                     ylabel='Price per Night')
plt.figure(figsize=(15,5))

sns.scatterplot('number_of_reviews','price',data=df).set(title='Number of Reviews vs Price per Night',

                                                        xlabel='Number of Reviews',

                                                        ylabel='Price per Night')
plt.figure(figsize=(15,5))

sns.scatterplot('reviews_per_month','price',data=df).set(title='Reviews Per Month vs Price per Night',

                                                        xlabel='Reviews Per Month',

                                                        ylabel='Price per Night')
plt.figure(figsize=(15,5))

sns.regplot('reviews_per_month','number_of_reviews',data=df)
sns.pairplot(data=df,y_vars=['availability_365'],x_vars=['price','number_of_reviews','reviews_per_month','calculated_host_listings_count','minimum_nights'])
sns.scatterplot('availability_365','room_type',data=df)
df.drop(columns=['borough','neighbourhood'], inplace=True)
df.head()
from sklearn.preprocessing import MinMaxScaler

import joblib
for column in df.drop(columns=['price','reviews_per_month']).columns:

    if df[column].dtype=='object':

        continue

    normr = MinMaxScaler()

    normr.fit(df[column].values.reshape(-1,1))

    joblib.dump(normr,'models/'+column+'_normr.pth')
df_2=pd.get_dummies(df)

df_2.head()
df_2
from sklearn.model_selection import train_test_split
df_train,df_test = train_test_split(df_2,train_size=0.75)
from torch.utils.data import DataLoader

from torch.utils.data import Dataset

torch.set_default_dtype(torch.float64)
class data_set(Dataset):

    def __init__(self, dataframe):

        for column in dataframe.columns:

            try:

                normr = joblib.load('models/'+column+'_normr.pth')

                dataframe[column]=normr.transform(dataframe[column].values.reshape(-1,1))

            except:

                pass

        self.x = torch.Tensor(dataframe.drop(columns=['price','reviews_per_month']).copy().values).double()

        self.y = torch.Tensor(dataframe[['price','reviews_per_month']].copy().values).double()

        self.shape = [self.x.shape,self.y.shape]

    def __len__(self):

        return len(self.x)

    def __getitem__(self,idx):

        return self.x[0].double(), self.y[0].double()
train_set = data_set(df_train)

train_loader = DataLoader(train_set,

                         batch_size=43,

                         shuffle=True)

test_set = data_set(df_test)

test_loader = DataLoader(test_set)
import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

import numpy as np

torch.set_default_dtype(torch.float64)
class Net(nn.Module):

    def __init__(self):

        super(Net,self).__init__()

        self.fc1 = nn.Linear(9, 242)

        self.fc2 = nn.Linear(242, 242)

        self.fc3 = nn.Linear(242, 242)

        self.fc4 = nn.Linear(242, 2)

    def forward(self, x):

        x = self.fc1(x)

        x = self.fc2(x)

        x = self.fc3(x)

        return self.fc4(x)
model = Net()

optimizer = optim.Adam(model.parameters(),lr=0.01)

criterion = nn.MSELoss()
for epoch in range(10):

    for data in train_loader:

        inputs, labels = data

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()
with torch.no_grad():

        for data in test_loader:

            inputs, labels = data

            accuracy = model(inputs)/labels

            print('Price Accuracy: {}\tReviews per Month: {}'.format(accuracy[0][0],accuracy[0][1]))