import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from scipy import stats

%matplotlib inline
train = pd.read_csv("../input/train.csv")

train.head()
train.describe() # Describe is only work for numerical data
train.info() #This will tell us about the missing values or can say present values with their 

#datatype for each feature.
corr = train.corr()

corr
f,ax = plt.subplots(figsize=(15,15))

sns.heatmap(corr,square=True) #train.corr get the correlation matrix of the 

#numerical features
# get the top 10 numeric features having highest correlation wuth saleprice

cols = train.corr().nlargest(10,'SalePrice').index

dtrain = train[cols]

f,ax = plt.subplots(figsize=(10,10))

colormap = plt.cm.cubehelix_r

sns.heatmap(dtrain.corr(),square=True,annot=True,cmap=colormap)
cols_ = cols.drop(['1stFlrSF','TotRmsAbvGrd','GarageArea'])

sns.pairplot(train[cols_])
categ_feats = train.dtypes[train.dtypes == "object"].index

categ_feats
itrain = pd.get_dummies(train[categ_feats])

itrain_ = pd.concat([itrain,train['SalePrice']],axis=1)

itrain_.shape
f, ax = plt.subplots(figsize=(15,15))

cols = itrain_.corr().nlargest(10,'SalePrice').index

it = itrain_[cols]

sns.heatmap(it.corr(),annot=True)
traindata = pd.concat([itrain,np.log1p(train.drop(categ_feats,axis=1))],axis=1)
train_data = traindata.fillna(traindata.mean())