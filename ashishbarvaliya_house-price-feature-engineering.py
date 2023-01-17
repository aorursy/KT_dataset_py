# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

train.shape, test.shape
train.head()
train.isna().sum().sort_values()
for col in ['Alley','FireplaceQu','Fence','MiscFeature','PoolQC']:

    train[col].fillna('NA', inplace=True)

    test[col].fillna('NA', inplace=True)

    

train['LotFrontage'].fillna(train["LotFrontage"].value_counts().to_frame().index[0], inplace=True)

test['LotFrontage'].fillna(test["LotFrontage"].value_counts().to_frame().index[0], inplace=True)



train[['GarageQual','GarageFinish','GarageYrBlt','GarageType','GarageCond']].isna().head(7)

for col in ['GarageQual','GarageFinish','GarageYrBlt','GarageType','GarageCond']:

    train[col].fillna('NA',inplace=True)

    test[col].fillna('NA',inplace=True)



for col in ['BsmtQual','BsmtCond','BsmtFinType1','BsmtFinType2','BsmtExposure']:

    train[col].fillna('NA',inplace=True)

    test[col].fillna('NA',inplace=True)



train['Electrical'].fillna('SBrkr',inplace=True)



missings = ['GarageCars','GarageArea','KitchenQual','Exterior1st','SaleType','TotalBsmtSF','BsmtUnfSF','Exterior2nd',

            'BsmtFinSF1','BsmtFinSF2','BsmtFullBath','Functional','Utilities','BsmtHalfBath','MSZoning']



numerical=['GarageCars','GarageArea','TotalBsmtSF','BsmtUnfSF','BsmtFinSF1','BsmtFinSF2','BsmtFullBath','BsmtHalfBath']

categorical = ['KitchenQual','Exterior1st','SaleType','Exterior2nd','Functional','Utilities','MSZoning']



# using Imputer class of sklearn libs.

from sklearn.preprocessing import Imputer

imputer = Imputer(strategy='median',axis=0)

imputer.fit(test[numerical] + train[numerical])

test[numerical] = imputer.transform(test[numerical])

train[numerical] = imputer.transform(train[numerical])



for i in categorical:

    train[i].fillna(train[i].value_counts().to_frame().index[0], inplace=True)

    test[i].fillna(test[i].value_counts().to_frame().index[0], inplace=True)    



train[train['MasVnrType'].isna()][['SalePrice','MasVnrType','MasVnrArea']]



train[train['MasVnrType']=='None']['SalePrice'].median()

train[train['MasVnrType']=='BrkFace']['SalePrice'].median()

train[train['MasVnrType']=='Stone']['SalePrice'].median()

train[train['MasVnrType']=='BrkCmn']['SalePrice'].median()



train['MasVnrArea'].fillna(181000,inplace=True)

test['MasVnrArea'].fillna(181000,inplace=True)



train['MasVnrType'].fillna('NA',inplace=True)

test['MasVnrType'].fillna('NA',inplace=True)



print(train.isna().sum().sort_values()[-2:-1])

print(test.isna().sum().sort_values()[-2:-1])
int64 =[]

objects = []

for col in train.columns.tolist():

    if np.dtype(train[col]) == 'int64' or np.dtype(train[col]) == 'float64':

        int64.append(col)

    else:

        objects.append(col)                      #here datatype is 'object'

len(int64), len(objects)        
train[int64].head()
continues_int64_cols = ['LotArea', 'LotFrontage', 'MasVnrArea','BsmtFinSF2','BsmtFinSF1','BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF',

                  'GrLivArea','GarageArea','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','MiscVal']

categorical_int64_cols=[]

for i in int64:

    if i not in continues_int64_cols:

        categorical_int64_cols.append(i)



print("continues int64 columns",len(continues_int64_cols))

print("categorical int64 columns",len(categorical_int64_cols)) 

continues_int64_cols, categorical_int64_cols
def barplot(X,Y):

    plt.figure(figsize=(7,7))

    sns.barplot(x=X, y=Y)

    plt.show()

def scatter(X,Y):

    plt.figure(figsize=(7,7))

    sns.scatterplot(alpha=0.4,x=X, y=Y)

    plt.show()

def hist(X):

    plt.figure(figsize=(7,7))

    sns.distplot(X, bins=40, kde=True)

    plt.show()

def box(X):

    plt.figure(figsize=(3,7))

    sns.boxplot(y=X)

    plt.show() 

def line(X,Y):

    plt.figure(figsize=(7,7))    

    sns.lineplot(x=X, y=Y,color="coral")

    plt.show() 
pd.plotting.scatter_matrix(train[continues_int64_cols[:5]],diagonal='kde', figsize=(10,10))

plt.show()
# used log to see all small values

hist(np.log(train['MasVnrArea']+1))
hist(np.log(train['BsmtFinSF2']+1))
print(train['MasVnrArea'].value_counts())

print(train['BsmtFinSF2'].value_counts())
train['MasVnrArea'] = train['MasVnrArea'].apply(lambda row: 1.0 if row>0.0 else 0.0)

train['BsmtFinSF2'] = train['BsmtFinSF2'].apply(lambda row: 1.0 if row>0.0 else 0.0)
binary_cate_int64_cols = []

binary_cate_int64_cols.append('MasVnrArea')

binary_cate_int64_cols.append('BsmtFinSF2')
pd.plotting.scatter_matrix(train[continues_int64_cols[5:11]],diagonal='kde', figsize=(10,10))

plt.show()
train['LowQualFinSF'].value_counts()
train['LowQualFinSF'] = train['LowQualFinSF'].apply(lambda row: 1.0 if row>0.0 else 0.0)
binary_cate_int64_cols.append('LowQualFinSF')
pd.plotting.scatter_matrix(train[continues_int64_cols[11:14]],diagonal='kde', figsize=(8,8))

plt.show()
pd.plotting.scatter_matrix(train[continues_int64_cols[14:]],diagonal='kde', figsize=(11,11))

plt.show()
for i in continues_int64_cols[14:]:

    train[i] = train[i].apply(lambda row: 1.0 if row>0.0 else 0.0)

    binary_cate_int64_cols.append(i)



for j in binary_cate_int64_cols:

    if j in continues_int64_cols:

        continues_int64_cols.remove(j)        #these special columns removing from the continues_int64_cols

        

print(len(continues_int64_cols))   

print(len(binary_cate_int64_cols))        

continues_int64_cols, binary_cate_int64_cols    
fig, axes = plt.subplots(3, 3, figsize=(20,11))

m=0

for i in range(3):

    for j in range(3):

        if m !=8:              # subplots are 9 and columns we have is 8 so ignoring last box, thats why,i apllied this condition

            sns.barplot(train[binary_cate_int64_cols[m]], train['SalePrice'],ax=axes[i,j])

            m+=1

plt.show()
# we changed values of train only, here for test set

for i in binary_cate_int64_cols:

    test[i] = test[i].apply(lambda row: 1.0 if row>0.0 else 0.0)
test[binary_cate_int64_cols].head(6)
train[categorical_int64_cols].head()
plt.figure(figsize=(15,7))

test.groupby('YearBuilt')['YearBuilt'].count().plot()

train.groupby('YearBuilt')['YearBuilt'].count().plot()

plt.legend(['test','train'])
plt.figure(figsize=(15,7))

test.groupby('YrSold')['YrSold'].count().plot()

train.groupby('YrSold')['YrSold'].count().plot()

plt.legend(['test','train'])
plt.figure(figsize=(15,7))

test.groupby('YearRemodAdd')['YearRemodAdd'].count().plot()

train.groupby('YearRemodAdd')['YearRemodAdd'].count().plot()

plt.legend(['test','train'])
fig, axes = plt.subplots(4, 3, figsize=(20,15))

sns.barplot(train[categorical_int64_cols[1]], train['SalePrice'],ax=axes[0,0])

sns.barplot(train[categorical_int64_cols[2]], train['SalePrice'],ax=axes[0,1])

sns.barplot(train[categorical_int64_cols[3]], train['SalePrice'],ax=axes[0,2])

sns.barplot(train[categorical_int64_cols[6]], train['SalePrice'],ax=axes[1,0])

sns.barplot(train[categorical_int64_cols[7]], train['SalePrice'],ax=axes[1,1])

sns.barplot(train[categorical_int64_cols[8]], train['SalePrice'],ax=axes[1,2])

sns.barplot(train[categorical_int64_cols[9]], train['SalePrice'],ax=axes[2,0])

sns.barplot(train[categorical_int64_cols[10]], train['SalePrice'],ax=axes[2,1])

sns.barplot(train[categorical_int64_cols[11]], train['SalePrice'],ax=axes[2,2])

sns.barplot(train[categorical_int64_cols[12]], train['SalePrice'],ax=axes[3,0])

sns.barplot(train[categorical_int64_cols[13]], train['SalePrice'],ax=axes[3,1])

sns.barplot(train[categorical_int64_cols[14]], train['SalePrice'],ax=axes[3,2])

plt.show()
barplot(train[categorical_int64_cols[15]], train['SalePrice'])
train[objects].head()
fig, axes = plt.subplots(4, 4, figsize=(20,15))

m=0

for i in range(4):

    for j in range(4):

        sns.barplot(train[objects[m]], train['SalePrice'], ax=axes[i,j])

        m+=1

plt.show()        
fig, axes = plt.subplots(4, 4, figsize=(20,15))

m=16

for i in range(4):

    for j in range(4):

        sns.barplot(train[objects[m]], train['SalePrice'], ax=axes[i,j])

        m+=1

plt.show()        
ordinal_categorical_cols =[]

ordinal_categorical_cols.extend(['ExterQual','ExterCond','BsmtQual','BsmtCond','BsmtExposure','HeatingQC','KitchenQual'])
fig, axes = plt.subplots(3, 4, figsize=(20,15))

m=32

for i in range(3):

    for j in range(4):

        sns.barplot(train[objects[m]], train['SalePrice'], ax=axes[i,j])

        m+=1

plt.show()        
ordinal_categorical_cols.extend(['FireplaceQu', 'GarageQual','GarageCond','PoolQC'])
plt.figure(figsize=(15,7))

test.groupby('GarageYrBlt')['GarageYrBlt'].count().plot()

train.groupby('GarageYrBlt')['GarageYrBlt'].count().plot()

plt.legend(['test','train'])
for i in ordinal_categorical_cols:

    if i in objects:

        objects.remove(i)            # removing ordinal features from the objects

len(objects), len(ordinal_categorical_cols)        
print('ordinal categorical cols ',len(ordinal_categorical_cols))

print('continues int64 cols ',len(continues_int64_cols))             

print('numeric categorical int64 cols ',len(categorical_int64_cols))

print('objects(text) categorical ',len(objects) ) 

print('binary int64 categorical ',len(binary_cate_int64_cols) )                   
# removinf 'Id' and 'SalePrice'

categorical_int64_cols.remove('Id')

categorical_int64_cols.remove('SalePrice')
len(categorical_int64_cols + objects)
train_objs_num = len(train)

dataset = pd.concat(objs=[train[categorical_int64_cols + objects], test[categorical_int64_cols+ objects]], axis=0)

dataset_preprocessed = pd.get_dummies(dataset.astype(str), drop_first=True)

train_nominal_onehot = dataset_preprocessed[:train_objs_num]

test_nominal_onehot= dataset_preprocessed[train_objs_num:]

train_nominal_onehot.shape, test_nominal_onehot.shape
train_nominal_onehot.head()
test_nominal_onehot.head()
# train[ordinal_categorical_cols].head()

for i in ordinal_categorical_cols:

    print(train[i].value_counts())
train['BsmtExposure'] = train['BsmtExposure'].map({'Gd':4, 'Av':3, 'Mn':2, 'No':1,'NA':0})

test['BsmtExposure'] = test['BsmtExposure'].map({'Gd':4, 'Av':3, 'Mn':2, 'No':1,'NA':0})



order = {'Ex':5,

        'Gd':4, 

        'TA':3, 

        'Fa':2, 

        'Po':1,

        'NA':0 }

for i in ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC']:

    train[i] = train[i].map(order)

    test[i] = test[i].map(order)

test[ordinal_categorical_cols].head()         
train[ordinal_categorical_cols].head()         
X = pd.concat([train[ordinal_categorical_cols], train[continues_int64_cols], train[binary_cate_int64_cols], train_nominal_onehot], axis=1)

y = train['SalePrice']

test_final = pd.concat([test[ordinal_categorical_cols], test[continues_int64_cols], test[binary_cate_int64_cols], test_nominal_onehot], axis=1)
X.shape, y.shape, test_final.shape
X.to_csv('new_train.csv',index=False)

test_final.to_csv('new_test.csv',index=False)