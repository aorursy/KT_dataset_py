import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test=pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
train.head()
test.head()
train.columns
train.info()
plt.figure(figsize=(20,20))

sns.heatmap(train.isnull(),yticklabels=False, cbar=False, cmap='viridis')
parameter_na=[i for i in train.columns if train[i].isnull().sum()>1]



for i in parameter_na:

    print(i, np.round(train[i].isnull().mean(),2),' % missing values')
numerical_features = [feature for feature in train.columns if train[feature].dtypes != 'O']



print('Number of numerical datatype parameters:',len(numerical_features))
train[numerical_features].head()
year_feature = [f for f in numerical_features if 'Yr' in f or 'Year' in f]

year_feature
train.groupby('YrSold')['SalePrice'].median().plot()

plt.xlabel('Year the House was Sold')

plt.ylabel('Median House Price')

plt.title("House Price vs Year House was Sold")
# Let's check out other type sof variables like Numerical variables.

# As, all know it is further subdivided into two types: 1) Continous, and 2) Discrete variables



discrete_feature = [p for p in numerical_features if len(train[p].unique())<25 and p not in year_feature+['Id']]

print("Discrete Variables Count: {}".format(len(discrete_feature)))
# Names of columns in train datsets with discrete variables

print(discrete_feature)
# Let's check out continuous varables:

cont_feature=[f for f in numerical_features if f not in discrete_feature+year_feature+['Id']]

print("Continuous feature paramters are Count {}".format(len(cont_feature)))
print(cont_feature)
corrmat = train.corr()

top=corrmat.index

plt.figure(figsize=(30,30))

graph = sns.heatmap(train[top].corr(),annot=True,cmap='RdYlGn')