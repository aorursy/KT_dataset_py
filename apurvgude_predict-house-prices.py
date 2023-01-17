# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
sns.set(style='dark')

pd.set_option('display.max_columns', 100)

from sklearn.preprocessing import StandardScaler

from sklearn.compose import ColumnTransformer

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import RandomizedSearchCV,cross_val_score

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from sklearn.ensemble import GradientBoostingRegressor

from xgboost import XGBRegressor

from catboost import CatBoostRegressor,Pool

from lightgbm import LGBMRegressor



train = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")

describe = open("/kaggle/input/house-prices-advanced-regression-techniques/data_description.txt","r")

sam = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv")

print(describe.read())
n = pd.DataFrame(train.isna().sum())

m = n.loc[n[0]!=0]

m['% Missing'] = m[0]*100.00/1460.00

m
n = pd.DataFrame(test.isna().sum())

m = n.loc[n[0]!=0]

m['% Missing'] = m[0]*100.00/1460.00

m.sort_values(by='% Missing',ascending=False)
plt.figure(figsize=(7,5))

sns.boxplot(x = 'MSSubClass',y  = 'SalePrice',data = train)
a = pd.DataFrame(train.groupby('MSSubClass')['SalePrice'].mean()).sort_values(by='SalePrice',ascending=False)

b = np.array(a.index)

c = np.arange(len(a),0,-1)

for i in range(len(c)):

    train.loc[train['MSSubClass']==b[i],'MSSubClass']=c[i]

for j in range(len(c)):

    test.loc[test['MSSubClass']==b[j],'MSSubClass']=c[j]

    test.loc[test['MSSubClass']==150,'MSSubClass']= 1

plt.figure(figsize=(7,5))

sns.boxplot(x = 'MSZoning',y  = 'SalePrice',data = train)
a = pd.DataFrame(train.groupby('MSZoning')['SalePrice'].mean()).sort_values('SalePrice')

b = np.array(a.index)

c = np.arange(1,len(a)+1)

for i in range(len(c)):

    train.loc[train['MSZoning']==b[i],'MSZoning']=c[i]

    

test['MSZoning']=test['MSZoning'].fillna('RL')

for i in range(len(c)):

    test.loc[test['MSZoning']==b[i],'MSZoning']=c[i]    
a = train.loc[train['LotFrontage'].isna()==False,'LotFrontage'].mean()

train['LotFrontage'] = train['LotFrontage'].fillna(a)

test['LotFrontage'] = test['LotFrontage'].fillna(train['LotFrontage'].mean())
sns.scatterplot(x ='LotFrontage',y= 'SalePrice',data=train)
plt.figure(figsize=(7,5))

sns.boxplot(x = 'Street',y  = 'SalePrice',data = train)
a = pd.DataFrame(train.groupby('Street')['SalePrice'].mean()).sort_values('SalePrice')

b = np.array(a.index)

c = np.arange(1,len(a)+1)

for i in range(len(c)):

    train.loc[train['Street']==b[i],'Street']=c[i]

for i in range(len(c)):

    test.loc[test['Street']==b[i],'Street']=c[i]    
train = train.drop(['Alley'],axis=1)

test=  test.drop(['Alley'],axis=1)
plt.figure(figsize=(7,5))

sns.boxplot(x = 'LotShape',y  = 'SalePrice',data = train)

plt.show()



a = pd.DataFrame(train.groupby('LotShape')['SalePrice'].mean()).sort_values('SalePrice')

b = np.array(a.index)

c = np.arange(1,len(a)+1)

for i in range(len(c)):

    train.loc[train['LotShape']==b[i],'LotShape']=c[i]

for i in range(len(c)):

    test.loc[test['LotShape']==b[i],'LotShape']=c[i]    
train = train.drop(['FireplaceQu','Fence','PoolQC','MiscFeature'],axis=1)

test = test.drop(['FireplaceQu','Fence','PoolQC','MiscFeature'],axis=1)

b = ['MasVnrType','GarageType','GarageFinish','Electrical','BsmtFinType1','BsmtFinType2']



#MasVnrType we'll change all Nan values to None

train['MasVnrType'] = train['MasVnrType'].fillna('None')

test['MasVnrType'] = test['MasVnrType'].fillna('None')



#GarageType Nan values become None meaning no garages. Same for Garage Finish

train['GarageType'] = train['GarageType'].fillna('None')

train['GarageFinish'] = train['GarageFinish'].fillna('None')

test['GarageType'] = test['GarageType'].fillna('None')

test['GarageFinish'] = test['GarageFinish'].fillna('None')





train['Electrical'] = train['Electrical'].fillna('SBrkr')

train['BsmtFinType1'] = train['BsmtFinType1'].fillna('None')

train['BsmtFinType2'] = train['BsmtFinType2'].fillna('None')

test['BsmtFinType1'] = test['BsmtFinType1'].fillna('None')

test['BsmtFinType2'] = test['BsmtFinType2'].fillna('None')

test['Exterior1st'] = test['Exterior1st'].fillna('VinylSd')

test['Exterior2nd'] = test['Exterior2nd'].fillna('VinylSd')

test['SaleType'] = test['SaleType'].fillna('WD')

test['Functional'] = test['Functional'].fillna('Typ')

test['Utilities'] = test['Utilities'].fillna('AllPub')

l = ['LandContour','Utilities','LotConfig','LandSlope','Neighborhood','Condition1','Condition2',

     'BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','Foundation','Heating',

     'HeatingQC','CentralAir','Functional','PavedDrive','SaleType','MasVnrType','GarageType','GarageFinish','Electrical','BsmtFinType1','BsmtFinType2','SaleCondition']



for i in enumerate(l):

    a = pd.DataFrame(train.groupby(i[1])['SalePrice'].mean()).sort_values('SalePrice')

    b = np.array(a.index)

    c = np.arange(1,len(a)+1)

    for j in range(len(c)):

        train.loc[train[i[1]]==b[j],i[1]]=c[j]

    for j in range(len(c)):

        test.loc[test[i[1]]==b[j],i[1]]=c[j]    

        

        
b = ['ExterQual','ExterCond','GarageQual','GarageCond','KitchenQual','BsmtCond','BsmtExposure','BsmtQual']



train['GarageQual'] = train['GarageQual'].fillna('Po')

train['GarageCond'] = train['GarageCond'].fillna('Po')

train['BsmtCond'] = train['BsmtCond'].fillna('Fa')

train['BsmtExposure'] = train['BsmtExposure'].fillna('No')

train['BsmtQual'] = train['BsmtQual'].fillna('Fa')

test['GarageQual'] = test['GarageQual'].fillna('Po')

test['GarageCond'] = test['GarageCond'].fillna('Po')

test['BsmtCond'] = test['BsmtCond'].fillna('Fa')

test['BsmtExposure'] = test['BsmtExposure'].fillna('No')

test['BsmtQual'] = test['BsmtQual'].fillna('Fa')

test['KitchenQual'] = test['KitchenQual'].fillna('TA')







for i in enumerate(['ExterQual','KitchenQual','BsmtQual']):

                   train.loc[train[i[1]]=='Fa',i[1]] = 1

                   train.loc[train[i[1]]=='TA',i[1]] = 2

                   train.loc[train[i[1]]=='Gd',i[1]] = 3

                   train.loc[train[i[1]]=='Ex',i[1]] = 4

                   test.loc[test[i[1]]=='Fa',i[1]] = 1

                   test.loc[test[i[1]]=='TA',i[1]] = 2

                   test.loc[test[i[1]]=='Gd',i[1]] = 3

                   test.loc[test[i[1]]=='Ex',i[1]] = 4 

for i in enumerate(['ExterCond','GarageQual','GarageCond','BsmtCond']):

                   train.loc[train[i[1]]=='Po',i[1]] = 1

                   train.loc[train[i[1]]=='Fa',i[1]] = 2

                   train.loc[train[i[1]]=='TA',i[1]] = 3

                   train.loc[train[i[1]]=='Gd',i[1]] = 4

                   train.loc[train[i[1]]=='Ex',i[1]] = 5

                   test.loc[test[i[1]]=='Po',i[1]] = 1

                   test.loc[test[i[1]]=='Fa',i[1]] = 2

                   test.loc[test[i[1]]=='TA',i[1]] = 3

                   test.loc[test[i[1]]=='Gd',i[1]] = 4

                   test.loc[test[i[1]]=='Ex',i[1]] = 5 

                   

for i in enumerate(['BsmtExposure']):

                   train.loc[train[i[1]]=='No',i[1]] = 1

                   train.loc[train[i[1]]=='Mn',i[1]] = 2

                   train.loc[train[i[1]]=='Av',i[1]] = 3

                   train.loc[train[i[1]]=='Gd',i[1]] = 4

                   test.loc[test[i[1]]=='No',i[1]] = 1

                   test.loc[test[i[1]]=='Mn',i[1]] = 2

                   test.loc[test[i[1]]=='Av',i[1]] = 3

                   test.loc[test[i[1]]=='Gd',i[1]] = 4 
train['MasVnrArea'] = train['MasVnrArea'].fillna(train['MasVnrArea'].mean())

test['MasVnrArea'] = test['MasVnrArea'].fillna(test['MasVnrArea'].mean())



train.loc[train['GarageYrBlt'].isna()==True,'GarageYrBlt'] = np.array(train.loc[train['GarageYrBlt'].isna()==True,'YearBuilt'])

test.loc[test['GarageYrBlt'].isna()==True,'GarageYrBlt'] = np.array(test.loc[test['GarageYrBlt'].isna()==True,'YearBuilt'])

test['BsmtFullBath'] = test['BsmtFullBath'].fillna(0.0)

test['BsmtFinSF1'] = test['BsmtFinSF1'].fillna(0.0)

test['BsmtUnfSF'] = test['BsmtUnfSF'].fillna(0.0)

test['TotalBsmtSF'] = test['TotalBsmtSF'].fillna(test['TotalBsmtSF'].mean())

test['GarageCars'] = test['GarageCars'].fillna(2.0)



train = train.drop(['Id','MiscVal','PoolArea','ScreenPorch','3SsnPorch','EnclosedPorch','LowQualFinSF'],axis=1)

test = test.drop(['Id','MiscVal','PoolArea','ScreenPorch','3SsnPorch','EnclosedPorch','LowQualFinSF'],axis=1)
train = train.apply(pd.to_numeric)

test = test.apply(pd.to_numeric)
fig = plt.figure(figsize = (30,30))

sns.heatmap(train.corr())
a = pd.DataFrame(train.corr())

b = np.array(a.columns)

value=0.75

c=[]

for i in enumerate(b):

    for j in enumerate(b):

        if i<j:

            if a.loc[i[1],j[1]]>value:

                print(i[1] + ' and ' + j[1] + ' : '+ str(a.loc[i[1],j[1]]))
train  = train.drop(['GarageYrBlt','Exterior2nd','1stFlrSF','TotRmsAbvGrd','GarageArea','GarageCond','Utilities','LandSlope','Condition2','ExterCond','BsmtFinSF2','BsmtHalfBath'],axis=1)

test  = test.drop(['GarageYrBlt','Exterior2nd','1stFlrSF','TotRmsAbvGrd','GarageArea','GarageCond','Utilities','LandSlope','Condition2','ExterCond','BsmtFinSF2','BsmtHalfBath'],axis=1)
x = train.drop(['SalePrice'],axis=1)

y = train['SalePrice']



a = ['LotFrontage','LotArea','YearBuilt','YearRemodAdd','MasVnrArea','BsmtFinSF1','BsmtUnfSF','TotalBsmtSF','2ndFlrSF','GrLivArea','WoodDeckSF','YrSold']



ct = ColumnTransformer([('name',StandardScaler(),a)],remainder='passthrough')



x_norm = pd.DataFrame(ct.fit_transform(x))

test_norm = pd.DataFrame(ct.transform(test))
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

x1_train,x1_test,y1_train,y1_test = train_test_split(x_norm,y,test_size=0.2,random_state=42)
test_Id = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")['Id']

#XGBoost

model = XGBRegressor(random_state=42,booster='gbtree',eta=0,max_depth=3,learning_rate=0.09,n_estimators=600,reg_alpha=0.01,reg_lambda = 0.1)

model.fit(x_train,y_train)



a = model.predict(test)
model = CatBoostRegressor(random_state=42,iterations=10000,l2_leaf_reg=50,rsm=0.99,depth=5,random_strength =0.1)

eval_pool = Pool(x_test,y_test)

model.fit(x_train, y_train, eval_set=eval_pool, early_stopping_rounds=10)



b= model.predict(test)

model = LGBMRegressor(random_state=42,objective='regression',learning_rate=0.1,n_estimators=443,num_leaves=32,min_child_samples=5,verbose=5,reg_alpha=0.01,reg_lambda=0.001)



model.fit(x_train,y_train)



c = model.predict(test)


sub = (a+b+c)/3

my_submission = pd.DataFrame({'Id': test_Id, 'SalePrice': sub})

my_submission.to_csv('mean_submission.csv', index=False)




