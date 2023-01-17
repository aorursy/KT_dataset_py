# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import scipy.stats as stats

import sklearn.linear_model as linear_model

import seaborn as sns



%matplotlib inline



import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

warnings.filterwarnings("ignore", category=DeprecationWarning)

#warnings.filterwarnings("ignore")





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train=pd.read_csv(r'../input/train.csv')

test=pd.read_csv(r'../input/test.csv')
train.head()
test.head()
print('The train data has' , train.shape[0], 'sales values and' ,train.shape[1], 'features')

print('The test data has' , test.shape[0], 'sales values and' ,test.shape[1], 'features')
train.info()
test.info()
sns.set()

sns.distplot(train['SalePrice'])
print('Sales price Skewness', train['SalePrice'].skew())

print('Sales price Kurtosis', train['SalePrice'].kurt())
train['SalesPrice_Log'] = np.log(train['SalePrice'])

sns.distplot(train['SalesPrice_Log'])
print('Sales price Skewness', train['SalesPrice_Log'].skew())

print('Sales price Kurtosis', train['SalesPrice_Log'].kurt())
train.drop('SalePrice', axis=1, inplace=True)
numerical = train.select_dtypes(include='number')

categorical = train.select_dtypes(include='object')
numerical.isna().any()
total = train.isna().sum().sort_values(ascending=False)

percentage = ((train.isna().sum())/(train.isna().count())).sort_values(ascending=False)

missing_data = pd.concat([total, percentage], axis=1, keys=['Total', 'Percent'])
missing_data.head(19).index
train.isna().any()        
clist=['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 

       'GarageQual', 'GarageCond', 'GarageFinish', 'GarageType', 

       'BsmtExposure', 'BsmtFinType2', 'BsmtQual', 'BsmtCond', 'BsmtFinType1',

       'MasVnrType', 'Electrical']



nlist=['LotFrontage','GarageYrBlt']



for i in clist:

    train[i].fillna('None',inplace=True)

    test[i].fillna('None',inplace=True)

    
train['MasVnrArea'].fillna(0,inplace=True)

test['MasVnrArea'].fillna(0,inplace=True)



for i in nlist:

    train[i].fillna(train[i].median(),inplace=True)

    test[i].fillna(test[i].median(),inplace=True)
test.fillna('None',inplace=True)
plt.hist(train['LotFrontage'],bins=30)
plt.hist(train['GarageYrBlt'],bins=30)
train['GarageYrBlt'].mode()
plt.hist(train['MasVnrArea'],bins=30)
total2 = train.isna().sum().sort_values(ascending=False)

percentage2 = ((train.isna().sum())/(train.isna().count())).sort_values(ascending=False)

missing_data2 = pd.concat([total2, percentage2], axis=1, keys=['Total', 'Percent'])

missing_data2
total2 = test.isna().sum().sort_values(ascending=False)

percentage2 = ((test.isna().sum())/(test.isna().count())).sort_values(ascending=False)

missing_data2 = pd.concat([total2, percentage2], axis=1, keys=['Total', 'Percent'])

missing_data2.head()
train.isna().sum().sum()
test.isna().sum().sum()
for i in train[numerical.columns]:

    print('{:15}'.format(i),'Skewness:', '{:05.2f}'.format(round(train[i].skew(),3)),'    ','Kurtosis:','{:06.2f}'.format(round(train[i].kurt(),3)))
k=np.log(train['LotArea'])

train['SalesPrice_Log'].corr(k)
categorical.columns
k=train.corr()['SalesPrice_Log'].abs().sort_values(ascending=False)

to_drop=k[k<0.45].index

to_keep=k[k>0.45]
numerical = train.select_dtypes(include='number')
k
for df in [train,test]:

    df.drop(to_drop,axis=1,inplace=True)
fig = plt.figure(figsize = (10,10))

sns.heatmap(train.corr(),annot = True)

plt.show()
train.drop(['GarageArea','1stFlrSF','TotRmsAbvGrd','GarageYrBlt'],axis = 1, inplace = True)

test.drop(['GarageArea','1stFlrSF','TotRmsAbvGrd','GarageYrBlt'],axis = 1, inplace = True)
to_keep
to_keep.drop(['GarageArea','1stFlrSF','TotRmsAbvGrd','GarageYrBlt'],axis=0, inplace = True)

to_keep.index
fig, axs = plt.subplots(3, 3, figsize=(10,10))



for i in range(0,3):

    for j in range(0,3):

        num= i*3 + j

        if num < len(to_keep):

            sns.scatterplot(x=train[to_keep.index[num]], y=train['SalesPrice_Log'], ax = axs[i][j])

            

plt.tight_layout()    

plt.show()  

test.columns
train = train[train['GrLivArea']<4500]

train = train[train['TotalBsmtSF']<4000]

for i in train[to_keep.index]:

    print('{:15}'.format(i),'Skewness:', '{:05.2f}'.format(round(train[i].skew(),3)),'    ','Kurtosis:','{:06.2f}'.format(round(train[i].kurt(),3)))
categorical.columns
from sklearn.preprocessing import LabelEncoder



labelencoder = LabelEncoder()

for i in categorical.columns:

    labelencoder.fit(list(categorical[i].values))

    categorical[i] = labelencoder.transform(list(categorical[i].values))



categorical['SalesPrice_Log']=train['SalesPrice_Log']
corr = categorical.corr().abs()

corr.sort_values(["SalesPrice_Log"], ascending = False,inplace=True)

print(corr.SalesPrice_Log)
keep= corr.SalesPrice_Log[corr.SalesPrice_Log>0.45]

keep=keep.drop('SalesPrice_Log')

catnames = keep.index
fig = plt.figure(figsize = (8,8))

sns.heatmap(categorical[catnames].corr(),annot = True)

plt.show()
train.drop(['GarageType'],axis=1,inplace=True)

test.drop(['GarageType'],axis=1,inplace=True)
catdrop= corr.SalesPrice_Log[corr.SalesPrice_Log<0.45]

catdrop= catdrop.index
for df in [train,test]:

    df.drop(catdrop,axis=1,inplace=True)

        
train.head()
train.columns
subsection=train[['YearBuilt', 'YearRemodAdd','TotalBsmtSF', 'GrLivArea']]
subsection2=test[['YearBuilt', 'YearRemodAdd','TotalBsmtSF', 'GrLivArea']]
subsection2
from sklearn.preprocessing import StandardScaler



sc = StandardScaler()

subsection = sc.fit_transform(subsection)

subsection2[subsection2['TotalBsmtSF'] == "None"]
subsection2[subsection2['TotalBsmtSF'] == "None"]
subsection2.at[660, 'TotalBsmtSF'] = 0

subsection2.at[660, 'TotalBsmtSF'] = subsection2['TotalBsmtSF'].mean()
sc2=StandardScaler()

subsection2 = sc2.fit_transform(subsection2)

train.columns
subsection=pd.DataFrame(subsection, columns=['YearBuilt1', 'YearRemodAdd1','TotalBsmtSF1', 'GrLivArea1'])

subsection2=pd.DataFrame(subsection2, columns=['YearBuilt2', 'YearRemodAdd2','TotalBsmtSF2', 'GrLivArea2'])
train= pd.concat([train,subsection],axis=1)
train.head()
train.drop(['YearBuilt', 'YearRemodAdd','TotalBsmtSF', 'GrLivArea'],axis=1,inplace=True)
test= pd.concat([test,subsection2],axis=1)
test.drop(['YearBuilt', 'YearRemodAdd','TotalBsmtSF', 'GrLivArea'],axis=1,inplace=True)
# A function to label encode our ordinal data



def label_encoding(df):

    df = df.replace({"BsmtQual" : {"None" : 0, "Po" : 1, "Fa" : 2, "TA": 3, "Gd" : 4, "Ex" : 5},

    "ExterQual" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5},

    "FireplaceQu" : {"None" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},

    "GarageFinish" :{"None" : 0, "Unf" : 1, "RFn" : 2, "Fin" : 3},

    "KitchenQual" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5}})

    return df

train = label_encoding(train)

test = label_encoding(test)
fig = plt.figure(figsize = (10,10))

sns.heatmap(train.corr(),annot = True)

plt.show()
train.columns
train=pd.get_dummies(train, drop_first=True,columns=['OverallQual', 'ExterQual', 'BsmtQual', 'FullBath', 'KitchenQual', 'Fireplaces','FireplaceQu', 'GarageFinish', 'GarageCars'])
train.columns
test=pd.get_dummies(test, drop_first=True,columns=['OverallQual', 'ExterQual', 'BsmtQual', 'FullBath', 'KitchenQual', 'Fireplaces','FireplaceQu', 'GarageFinish', 'GarageCars'])
test.columns
test.columns
test.drop(['FullBath_4','KitchenQual_None','Fireplaces_4','GarageCars_5.0','GarageCars_None'],axis=1,inplace=True)
train.shape
train = train.apply(pd.to_numeric, errors='coerce')

train.fillna(0, inplace=True)
X = train.drop('SalesPrice_Log', axis=1).values

y = train['SalesPrice_Log'].values

test = test.apply(pd.to_numeric, errors='coerce')

test.fillna(0, inplace=True)



testing_X = test.values
from sklearn.model_selection import  train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)



np.any(np.isnan(X_train))
from sklearn.linear_model import LinearRegression

lm = LinearRegression()



#We train our model using 80% of the training data and predict

lm.fit(X_train,y_train)

y_pred_reg = lm.predict(X_test)
import math

def rmsle(y, y_pred):

    assert len(y) == len(y_pred)

    terms_to_sum = [(math.log(y_pred[i] + 1) - math.log(y[i] + 1)) ** 2.0 for i,pred in enumerate(y_pred)]

    return (sum(terms_to_sum) * (1.0/len(y))) ** 0.5
rmsle(np.exp(y_test), np.exp(y_pred_reg))

from sklearn.linear_model import Lasso

lasso_model = Lasso()

lasso_model.fit(X_train,y_train)

y_pred_lasso = lasso_model.predict(X_test)

rmsle(np.exp(y_test), np.exp(y_pred_lasso))

from sklearn.linear_model import LassoCV

lcv = LassoCV()

lcv.fit(X_train,y_train)

y_pred_lassocv = lcv.predict(X_test)

rmsle(np.exp(y_test),np.exp(y_pred_lassocv))

from sklearn.ensemble import RandomForestRegressor



RFR = RandomForestRegressor(n_estimators=500)

RFR.fit(X_train,y_train)

y_pred_random = RFR.predict(X_test)

rmsle(np.exp(y_test),np.exp(y_pred_random))

from sklearn.linear_model import  Ridge,ElasticNet
ridge = Ridge(alpha = 10)

ridge.fit(X_train, y_train)

#Predicting the Test set results

ridge_test_lm = ridge.predict(X_test)
rmsle(np.exp(y_test),np.exp(ridge_test_lm))

rmsle(np.exp(y_test),np.exp(y_pred))

model = LinearRegression()

model.fit(X,y)

y_pred = model.predict(testing_X)

predictions = np.exp(y_pred)
model = RandomForestRegressor(n_estimators=500)

model.fit(X,y)

y_pred_random = model.predict(testing_X)

predictions4 = np.exp(y_pred_random)

test['Id']=np.arange(1461,2920)
model = Lasso(lcv.alpha_)

model.fit(X,y)

y_pred = model.predict(testing_X)

predictions2 = np.exp(y_pred)
model = Ridge(alpha = 10)

model.fit(X,y)

#Predicting the Test set results

y_pred = ridge.predict(testing_X)

predictions3 = np.exp(y_pred)
result3=pd.DataFrame({'Id':test.Id, 'SalePrice':predictions4})

result3.to_csv('submission2.csv', index=False)