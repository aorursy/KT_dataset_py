import warnings

warnings.filterwarnings("ignore")

import numpy as np

import pandas as pd

import sklearn

from scipy import stats

from matplotlib import pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set() # for plot styling

from sklearn.cluster import KMeans

from sklearn.metrics import pairwise_distances_argmin # You can use this to find clostest center
data = pd.read_csv('/kaggle/input/eval-lab-3-f464/train.csv')

data_te = pd.read_csv('/kaggle/input/eval-lab-3-f464/test.csv')
data.head()
data.nunique()
data.isnull().any().sum()
data.dtypes
data['TotalCharges'].dtype

data_te[data_te['TotalCharges'].isnull()]
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors = 'coerce')
data_te['TotalCharges'] = pd.to_numeric(data_te['TotalCharges'], errors = 'coerce')
data_te['TotalCharges'].dtype
data[data['TotalCharges'].isnull()]
orig = pd.read_csv('/kaggle/input/eval-lab-3-f464/train.csv')
orig['TotalCharges'].value_counts()
data_wrangled = data.dropna(axis=0, subset = ['TotalCharges'])

data_wrangled = data_wrangled.reset_index(drop=True)

data_te['TotalCharges'].fillna(data_wrangled['TotalCharges'].mean(), inplace=True)
data_te['TotalCharges'].isnull().sum()
data_te['TotalCharges'].dtypes
data_wrangled['TotalCharges'].isnull().sum()
data_wrangled.head()
sns.scatterplot(x = 'MonthlyCharges', y = 'TotalCharges', data = data_wrangled)
data_wrangled.nunique()
data_wrangled.drop('Internet', axis=1, inplace = True)

data_te.drop('Internet', axis=1, inplace = True)
gender_map = { 'Male': 0, 'Female': 1 }

data_wrangled['gender'] = data_wrangled['gender'].map(gender_map)

data_te['gender'] = data_te['gender'].map(gender_map)

#data_wrangled.drop('gender', axis=1, inplace=True)

#data_te.drop('gender', axis=1, inplace=True)
data_wrangled['Children'].value_counts()

data_te['Children'].value_counts()
data_wrangled['Married'].value_counts()
data_wrangled['AddedServices'].value_counts()
yes_no_map = { 'Yes': 1, 'No': 0}

data_wrangled['Married'] = data_wrangled['Married'].map(yes_no_map)

data_te['Married'] = data_te['Married'].map(yes_no_map)
data_wrangled['Children'] = data_wrangled['Children'].map(yes_no_map)

data_te['Children'] = data_te['Children'].map(yes_no_map)
data_wrangled['AddedServices'] = data_wrangled['AddedServices'].map(yes_no_map)

data_te['AddedServices'] = data_te['AddedServices'].map(yes_no_map)
data_wrangled = pd.get_dummies(data= data_wrangled, columns = ['TVConnection', 'PaymentMethod'])

data_te = pd.get_dummies(data= data_te, columns = ['TVConnection', 'PaymentMethod'])
data_wrangled.head()
channel_map = {'Yes' : 2, 'No' : 1, 'No tv connection' : 0}

data_wrangled[['Channel1', 'Channel2', 'Channel3', 'Channel4', 'Channel5', 'Channel6']] = data_wrangled[['Channel1', 'Channel2', 'Channel3', 'Channel4', 'Channel5', 'Channel6']].replace({'Yes' : 2, 'No' : 1, 'No tv connection' : 0})

data_te[['Channel1', 'Channel2', 'Channel3', 'Channel4', 'Channel5', 'Channel6']] = data_te[['Channel1', 'Channel2', 'Channel3', 'Channel4', 'Channel5', 'Channel6']].replace({'Yes' : 2, 'No' : 1, 'No tv connection' : 0})

#data_wrangled.drop(['Channel1', 'Channel2', 'Channel3', 'Channel4', 'Channel5', 'Channel6'], axis=1, inplace=True)

#data_te.drop(['Channel1', 'Channel2', 'Channel3', 'Channel4', 'Channel5', 'Channel6'], axis=1, inplace=True)
data_wrangled.head()
data_wrangled.isnull().any().sum()
data_wrangled = data_wrangled.drop('custId', axis=1)

custId_te = data_te['custId']

data_te = data_te.drop('custId', axis=1)
y = data_wrangled['Satisfied']

data_wrangled.columns
data_wrangled['Subscription'].value_counts()
data_wrangled['Subscription'].replace({'Monthly' : 1, 'Biannually' : 6, 'Annually': 12}, inplace=True)

data_te['Subscription'].replace({'Monthly' : 1, 'Biannually' : 6, 'Annually': 12}, inplace=True)
data_wrangled['HighSpeed'].replace({ 'Yes':2, 'No':1, 'No internet':0}, inplace=True)

data_te['HighSpeed'].replace({ 'Yes':2, 'No':1, 'No internet':0}, inplace=True)

data_wrangled['HighSpeed'].head()

#data_wrangled.drop('HighSpeed', axis=1, inplace=True)

#data_te.drop('HighSpeed', axis=1, inplace=True)
data_wrangled['Subscription'].value_counts()
data_wrangled.head()
y.shape
data_wrangled.shape
data_wrangled.columns.values
data_te.dtypes
from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler()

cols = ['TotalCharges', 'tenure', 'MonthlyCharges']

data_wrangled[cols] = scaler.fit_transform(data_wrangled[cols])

data_te[cols] = scaler.transform(data_te[cols])
data_wrangled[cols].head()
data_wrangled.corr()['Satisfied']
data_wrangled = data_wrangled.drop('Satisfied', axis=1)
from sklearn.metrics import roc_auc_score

from sklearn.cluster import KMeans

nc = 10

kmeanss = KMeans(n_clusters = nc, n_init=100, random_state=4).fit(data_wrangled)

y_pred = kmeanss.labels_



ones = [0 for i in range(nc)]

zeros = [0 for i in range(nc)]



n = len(y)



for i in range(n):

    if (y[i] == 0):

        zeros[y_pred[i]] = zeros[y_pred[i]] + 1

    else:

        ones[y_pred[i]] = ones[y_pred[i]] + 1

        

mp = {}



for i in range(nc):

    if (ones[i] > 0.75*(ones[i] + zeros[i])):

        mp[i]=1

    else:

        mp[i]=0



y_pred = pd.Series(y_pred).map(mp)

acc = roc_auc_score(y,y_pred)



print(acc)
data_te.isnull().any().sum()
y_test = kmeanss.predict(data_te)

y_test.shape

df = pd.DataFrame(custId_te)

y_test = pd.Series(y_test).map(mp)



df['Satisfied'] = y_test

df.head()
df.to_csv('ev3.csv', index = False)