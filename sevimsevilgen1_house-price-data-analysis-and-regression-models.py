



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")

train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

test = pd.read_csv ("../input/house-prices-advanced-regression-techniques/test.csv")
train.head()
train.shape,test.shape
train.describe()
numeric_features = train.select_dtypes(include=[np.number])

numeric_features.columns
numeric_features.head()
numeric_features.shape
g = sns.jointplot(numeric_features.YearBuilt,numeric_features.SalePrice,kind='kde',size=7)



g
# list of variables that contain year information

year_feature = [feature for feature in numeric_features if 'Yr' in feature or 'Year' in feature]



year_feature
for feature in year_feature:

    print(feature, train[feature].unique())
for feature in year_feature:

    if feature!='YrSold':

        data=train.copy()

        ## We will capture the difference between year variable and year the house was sold for

        data[feature]=data['YrSold']-data[feature]

        plt.bar(data[feature],data['SalePrice'])

        plt.xlabel(feature)

        plt.ylabel('SalePrice')

        plt.show()
sns.set()

columns = ['SalePrice','OverallQual','TotalBsmtSF','GrLivArea','GarageArea','FullBath','YearBuilt','YearRemodAdd']

sns.pairplot(train[columns],size = 2 ,kind ='scatter',diag_kind='kde')

plt.show()
numeric_features.head()
dataset = data.loc[:,["YearBuilt","YearRemodAdd","GarageYrBlt","YrSold","SalePrice"]]

dataset
correlation = numeric_features.corr()

print(correlation['SalePrice'].sort_values(ascending = False),'\n')
f , ax = plt.subplots(figsize = (14,12))

plt.title('Correlation of Numeric Features with Sale Price',y=1,size=16)

sns.heatmap(correlation,square = True,  vmax=0.8, annot=True)
numeric_features.head()
sns.lmplot(x='1stFlrSF',y='SalePrice',data=numeric_features)
plt.scatter(x= 'GrLivArea', y='SalePrice', data = numeric_features)
plt.figure(figsize=(16,8))

sns.boxplot(x='GarageCars',y='SalePrice',data=numeric_features)

plt.show()
sns.lmplot(x='OverallQual',y='SalePrice',data=numeric_features)
sns.jointplot(numeric_features.LotArea,numeric_features.SalePrice,kind='scatter',size=7)

plt.figure(figsize=(16,8))

sns.barplot(x='FullBath',y = 'SalePrice',data=numeric_features)

plt.show()