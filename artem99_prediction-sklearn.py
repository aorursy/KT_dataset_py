import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import os

%matplotlib inline

import seaborn as sns

import tensorflow as tf

train='/kaggle/input/house-prices-advanced-regression-techniques/train.csv'

test='/kaggle/input/house-prices-advanced-regression-techniques/test.csv'

submission='/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv'

data='../input/housing/AmesHousing.csv'
train=pd.read_csv(train)

test=pd.read_csv(test)

submission=pd.read_csv(submission)

data=pd.read_csv(data)

display(train.head())

display(test.head())

display(submission.head())

display(data.head())
data=data.drop(['PID'],axis=1)

data.columns=train.columns

train=data

display(train.head())

train.shape

print('train_size' '{}'.format(train.shape))

print('test_size' '{}'.format(test.shape))

print('submission_size' '{}'.format(submission.shape))
train.drop('Id',axis=1,inplace=True)

test.drop('Id',axis=1,inplace=True)

sns.set_style('darkgrid')

train.SalePrice.plot(subplots=True, figsize=(12, 10))



sns.set_style('darkgrid')

for i in train.columns:

    if train[i].dtypes==int or train[i].dtypes==float:

      

        sns.jointplot(train[i],train.SalePrice)

        plt.show()

        


saleprice_reg=train[train.LotShape=='Reg'].SalePrice

reg=train[train.LotShape=='Reg'].GrLivArea



saleprice_ir1=train[train.LotShape=='IR1'].SalePrice

ir1=train[train.LotShape=='IR1'].GrLivArea



saleprice_ir2=train[train.LotShape=='IR2'].SalePrice

ir2=train[train.LotShape=='IR2'].GrLivArea



saleprice_ir3=train[train.LotShape=='IR3'].SalePrice

ir3=train[train.LotShape=='IR3'].GrLivArea
plt.figure(figsize=(10,10))

sns.boxplot(train.LotShape,train.SalePrice)
fig,az=plt.subplots(nrows=4,figsize=(12,15))

az[0].scatter(reg,saleprice_reg,color='red')

az[0].plot([2900,2900],[0,600000])

az[1].scatter(ir1,saleprice_ir1,color='green')

az[1].plot([4600,4600],[0,700000])

az[2].scatter(ir2,saleprice_ir2,color='blue')

az[2].plot([3500,3500],[0,500000])

az[3].scatter(ir3,saleprice_ir3,color='black')

az[3].plot([3000,3000],[0,350000])

plt.show() 
def price_area(x,_sum_x,lotshape):

    

    list_area=[]

  

    for area in x:

        if area > _sum_x:

            list_area.append(area)

    print('lotshape=====',lotshape)

    print(len(list_area))

    return list_area

            

  

  

            



list_reg=price_area(reg,2900,'REG')

list_ir1=price_area(ir1,4600,'IR1')

list_ir2=price_area(ir2,3500,'IR2')

list_ir3=price_area(ir3,3000,'IR3')
list_list=[list_reg,list_ir1,list_ir2,list_ir3]



list_area=[]

for i in range(len(list_list)):

    for j in list_list[i]:

        list_area.append(j)

print('len',len(list_area))
list_index=[]

for k in list_area:

    for i ,j in enumerate(train['GrLivArea']):

        

        if j==k:

          

            list_index.append(i)

    

for i in list_index:

    

    train.drop(i,inplace=True)
train.shape
saleprice_reg=train[train.LotShape=='Reg'].SalePrice

reg=train[train.LotShape=='Reg'].GrLivArea



saleprice_ir1=train[train.LotShape=='IR1'].SalePrice

ir1=train[train.LotShape=='IR1'].GrLivArea



saleprice_ir2=train[train.LotShape=='IR2'].SalePrice

ir2=train[train.LotShape=='IR2'].GrLivArea



saleprice_ir3=train[train.LotShape=='IR3'].SalePrice

ir3=train[train.LotShape=='IR3'].GrLivArea
fig,az=plt.subplots(nrows=4,figsize=(12,15))

az[0].scatter(reg,saleprice_reg,color='red')

az[1].scatter(ir1,saleprice_ir1,color='green')

az[2].scatter(ir2,saleprice_ir2,color='blue')

az[3].scatter(ir3,saleprice_ir3,color='black')

plt.show()
train.shape
from sklearn.preprocessing import  LabelEncoder


for i in train.columns:

    

    print(i,' sum NaN = ',train[i].notnull().sum())
yr_10=train[train.YrSold==2010].YrSold.value_counts()

yr_09=train[train.YrSold==2009].YrSold.value_counts()

yr_08=train[train.YrSold==2008].YrSold.value_counts()

yr_07=train[train.YrSold==2007].YrSold.value_counts()

yr_06=train[train.YrSold==2006].YrSold.value_counts()



labels=[yr_06.index.values,yr_07.index.values,yr_08.index.values,yr_09.index.values,yr_10.index.values]

plt.figure(figsize=(10,10))

plt.pie([yr_06.values,yr_07.values,yr_08.values,yr_09.values,yr_10.values],labels=labels)

plt.show()







plt.figure(figsize=(10,10))

sns.barplot(train.YrSold,train.SalePrice,palette='Blues')
plt.figure(figsize=(10,10))

sns.boxplot(train.SalePrice)
train.SalePrice.describe()
train['MasVnrType']=train['MasVnrType'].fillna('Stone')

train['BsmtQual']=train['BsmtQual'].fillna('Gd')

train['MasVnrArea']=train['MasVnrArea'].fillna(train['MasVnrArea'].mean())

train['BsmtCond']=train['BsmtCond'].fillna('TA')

train['BsmtExposure']=train['BsmtExposure'].fillna('Mn')

train['BsmtFinType1']=train['BsmtFinType1'].fillna('ALQ')

train['BsmtFinType2']=train['BsmtFinType2'].fillna('Unf')

train['Electrical']=train['Electrical'].fillna('SBrkr')

train['GarageType']=train['GarageType'].fillna('Attchd')

train['GarageYrBlt']=train['GarageYrBlt'].fillna(train['GarageYrBlt'].mean())

train['GarageFinish']=train['GarageFinish'].fillna('RFn')

train['GarageQual']=train['GarageQual'].fillna('TA')

train['GarageCond']=train['GarageCond'].fillna('TA')

train['LotFrontage']=train['LotFrontage'].fillna(train['LotFrontage'].mean())







colum=['Alley','Fence','MiscFeature','PoolQC','FireplaceQu','YrSold']

train=train.drop(colum,axis=1)
train.shape
label_en=LabelEncoder()

for i in train.columns:

    if train[i].dtypes==str:

        label_en.fit(train[i])

        train[i]=label_en.fit_transform(train[i])

    if train[i].dtypes==object:

        train[i]=train[i].astype(str)

        

        label_en.fit(train[i])

        train[i]=label_en.fit_transform(train[i])

        

    if type(train[i][0])==str:

        

        label_en.fit(train[i])

        train[i]=label_en.fit_transform(train[i])

    print(i, ':', train[i].dtype)
train['GarageCars']=train['GarageCars'].fillna(train.GarageCars).mean()

train.BsmtFinSF1=train.BsmtFinSF1.fillna(train.BsmtFinSF1).mean()

train.BsmtFinSF2=train.BsmtFinSF2.fillna(train.BsmtFinSF2).mean()

train.BsmtUnfSF=train.BsmtUnfSF.fillna(train.BsmtUnfSF).mean()

train.TotalBsmtSF=train.TotalBsmtSF.fillna(train.TotalBsmtSF).mean()

train.BsmtFullBath=train.BsmtFullBath.fillna(train.BsmtFullBath).mean()

train.BsmtHalfBath=train.BsmtHalfBath.fillna(train.BsmtHalfBath).mean()

train.BsmtFinSF1=train.BsmtFinSF1.fillna(train.BsmtFinSF1).mean()

train['GarageArea']=train['GarageArea'].fillna(train.GarageArea).mean()

saleprice=train.SalePrice

train=train.drop('SalePrice',axis=1)

plt.figure(figsize=(20,15))

train.corrwith(saleprice).plot(kind='bar')



print('MULTICORRELATION')

for column in train.columns:

    

   

    

    corr=train.corrwith(train[column])

    

    for i,j in zip(corr,train.columns):

        if i >=0.7 and i <0.99:

            print('correlation witch',column)

            print(j,'=',i)

            

    

    

        
columns=['MSSubClass','BldgType','Exterior2nd',

         'Exterior1st','TotalBsmtSF','1stFlrSF','TotRmsAbvGrd',

         'GrLivArea','GarageCars','GarageArea','GarageYrBlt','YearBuilt']

for column in columns:

    corr=train[column].corr(saleprice)

    print(column,corr)

train=train.drop(['BldgType','Exterior2nd','1stFlrSF','TotRmsAbvGrd','GarageArea','GarageYrBlt'],axis=1)
train.shape


for i in train.columns:

    

    print(i,' sum = ',train[i].notnull().sum())
from sklearn.preprocessing import MinMaxScaler
minmax_price=MinMaxScaler()

minmax=MinMaxScaler()

minmax.fit(train)

train=minmax.transform(train)





saleprice=np.array(saleprice)

saleprice=saleprice.reshape(2914,1)

minmax_price.fit(saleprice)

saleprice=minmax_price.transform(saleprice)
plt.figure(figsize=(10,10))

sns.distplot(saleprice)

print(saleprice.shape)

print(train.shape)
import tensorflow as tf

from sklearn.model_selection import train_test_split



x_tr,x_val,y_tr,y_val=train_test_split(train,saleprice,train_size=0.8,random_state=False)
from sklearn.metrics import mean_squared_error

from sklearn.linear_model import SGDRegressor

ff_reg=SGDRegressor(penalty="l1")

ff_reg.fit(x_tr,y_tr)

pred_a=ff_reg.predict(x_val)

mean_squared_error(y_val,pred_a)
from sklearn.linear_model import Ridge

f_regr=Ridge(alpha=1,solver='sag')

f_regr.fit(x_tr,y_tr)

pred_b=f_regr.predict(x_val)

mean_squared_error(y_val,pred_b)
from sklearn.ensemble import RandomForestRegressor



f_reg=RandomForestRegressor(max_depth=20,n_estimators=18,verbose=1,random_state=0)

f_reg.fit(x_tr,y_tr)

pred_c=f_reg.predict(x_val)

mean_squared_error(y_val,pred_c)


for i in test.columns:

    

    print(i,' sum NaN = ',test[i].notnull().sum())
test['BsmtFinSF1']=test['BsmtFinSF1'].fillna(test['BsmtFinSF1'].mean())

test['BsmtFinSF2']=test['BsmtFinSF2'].fillna(test['BsmtFinSF2'].mean())

test['BsmtUnfSF']=test['BsmtUnfSF'].fillna(test['BsmtUnfSF'].mean())

test['TotalBsmtSF']=test['TotalBsmtSF'].fillna(test['TotalBsmtSF'].mean())

test['BsmtFullBath']=test['BsmtFullBath'].fillna(test['BsmtFullBath'].mean())

test['BsmtHalfBath']=test['BsmtHalfBath'].fillna(test['BsmtHalfBath'].mean())

test['GarageCars']=test['GarageCars'].fillna(test['GarageCars'].mean())

test['GarageArea']=test['GarageArea'].fillna(test['GarageArea'].mean())

test['LotFrontage']=test['LotFrontage'].fillna(test['LotFrontage'].mean())


test['MasVnrType']=test['MasVnrType'].fillna('Stone')

test['BsmtQual']=test['BsmtQual'].fillna('Gd')

test['MasVnrArea']=test['MasVnrArea'].fillna(test['MasVnrArea'].mean())

test['BsmtCond']=test['BsmtCond'].fillna('TA')

test['BsmtExposure']=test['BsmtExposure'].fillna('Mn')

test['BsmtFinType1']=test['BsmtFinType1'].fillna('ALQ')

test['BsmtFinType2']=test['BsmtFinType2'].fillna('Unf')

test['Electrical']=test['Electrical'].fillna('SBrkr')

test['GarageType']=test['GarageType'].fillna('Attchd')

test['GarageYrBlt']=test['GarageYrBlt'].fillna(test['GarageYrBlt'].mean())

test['GarageFinish']=test['GarageFinish'].fillna('RFn')

test['GarageQual']=test['GarageQual'].fillna('TA')

test['GarageCond']=test['GarageCond'].fillna('TA')

test['Utilities']=test['Utilities'].fillna('AllPub')

test['Functional']=test['Functional'].fillna('Typ')

test['LotFrontage']=test['LotFrontage'].fillna(test['LotFrontage'].mean())

test['MSZoning']=test['MSZoning'].fillna('RL')
colum=['Alley','Fence','MiscFeature','PoolQC','FireplaceQu','YrSold']

test=test.drop(colum,axis=1)

test.shape
for i in test.columns:

    

    print(i,' sum NaN = ',test[i].notnull().sum())
label_en=LabelEncoder()

for i in test.columns:

    if test[i].dtypes==str:

        label_en.fit(test[i])

        test[i]=label_en.fit_transform(test[i])

    if test[i].dtypes==object:

        test[i]=test[i].astype(str)

        

        label_en.fit(test[i])

        test[i]=label_en.fit_transform(test[i])

        

    if type(test[i][0])==str:

        

        label_en.fit(test[i])

        test[i]=label_en.fit_transform(test[i])

    print(i, ':', test[i].dtype)
test=test.drop(['BldgType','Exterior2nd','1stFlrSF','TotRmsAbvGrd','GarageArea','GarageYrBlt'],axis=1)
test.shape


minmax.fit(test)

test=minmax.transform(test)
saleprice=minmax_price.inverse_transform(saleprice)
saleprice.shape
pred_test=f_reg.predict(test)

pred_test=np.reshape(pred_test,(1459,1))



pred_trans=minmax_price.inverse_transform(pred_test)
plt.figure(figsize=(20,10))

plt.plot(range(len(saleprice)),saleprice,color='red')

plt.plot(range(2914,2914+len(pred_trans)),pred_trans,color='blue')
submission.SalePrice=pred_trans

submission
submission.to_csv ('submission.csv', index = None, header = True)