import pandas as pd

import numpy as np

import warnings

warnings.filterwarnings('ignore')
data_train = pd.read_csv('../input/train.csv')

data_test = pd.read_csv('../input/test.csv')
data_train.info()
data_train.head()
data_list=[data_train,data_test]

data_combine = pd.concat(data_list,axis=0, join='outer',join_axes=None,ignore_index=False,levels=None,names=None,keys=['data_train','data_test'],

                         verify_integrity=False)
list_columns = list(data_combine.columns)



column_int          = []

column_int_nan      = []

column_int_notnan   = []

column_float        = []

column_float_nan    = []

column_float_natnan = []

column_object       = []



for i in list_columns:

    if data_combine[i].dtypes==int:

        column_int.append(i)

        if data_combine[i].isnull().sum()==0:

            column_int_notnan.append(i)

        else:

            column_int_nan.append(i)

    elif data_combine[i].dtypes==float:

        column_float.append(i)

        if data_combine[i].isnull().sum()==0:

            column_float_notnan.append(i)

        else:

            column_float_nan.append(i)

    else :#object

        column_object.append(i)
column_nan_list=[column_int_nan,column_float_nan]

for column in column_nan_list:

    print(data_combine[column].isnull().sum())

    print('-'*30)

print(data_combine.shape)
print('1%:',len(data_combine)*0.01)

print('5%:',len(data_combine)*0.05)

print('10%:',len(data_combine)*0.1)
column_float_nan_001=[]

column_float_nan_005=[]

column_float_nan_01 =[]



for i in list(column_float_nan):

        if data_combine[i].isnull().sum() <= 29:

            column_float_nan_001.append(i)

        elif data_combine[i].isnull().sum() > 29 and data_combine[i].isnull().sum() <= 292:

            column_float_nan_005.append(i)

        else:

            column_float_nan_01.append(i)
for i in column_float_nan_001:

    print(i)

    print(data_combine[i][:3])
for column in ['BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','GarageArea','MasVnrArea','TotalBsmtSF']:

    data_combine[column].fillna(int(data_combine[column].mean()),inplace=True)

for column in ['BsmtFullBath','BsmtHalfBath','GarageCars']:

    data_combine[column].fillna(int(data_combine[column].mode()),inplace=True)



data_combine[column_float_nan_001].isnull().sum()
for i in column_float_nan_005:

    print(i)

    print(data_combine[i][:3])
list_GarageYrBlt_was_missing = list(data_combine['GarageYrBlt'].isnull())



for i in range(len(list_GarageYrBlt_was_missing)):

    if list_GarageYrBlt_was_missing[i]==False:

        list_GarageYrBlt_was_missing[i]=1

    else:

        list_GarageYrBlt_was_missing[i]=0



data_combine['GarageYrBlt_was_missing']=list_GarageYrBlt_was_missing

data_combine['GarageYrBlt'].fillna(int(data_combine['GarageYrBlt'].mean()),inplace=True)



data_combine[column_float_nan_005].isnull().sum()
for i in column_float_nan_01:

    print(i)

    print(data_combine[i][:3])

#SalePrice不處理
list_LotFrontage_was_missing = list(data_combine['LotFrontage'].isnull())



for i in range(len(data_combine)):

    if list_LotFrontage_was_missing[i]==False:

        list_LotFrontage_was_missing[i]=1

    else:

        list_LotFrontage_was_missing[i]=0



data_combine['LotFrontage_was_missing']=list_LotFrontage_was_missing

data_combine['LotFrontage'].fillna(int(data_combine['LotFrontage'].mean()),inplace=True)



data_combine[column_float_nan_01].isnull().sum()
for column in column_object:

    print(column)

    print ('n_unique: ',data_combine[column].nunique())

    print(data_combine[column].unique(),'\n')
pd.get_dummies(data_combine['Alley']).head()
data_combine=data_combine.join(pd.get_dummies(data_combine[column_object]).astype(int))

for columns in column_object:

    data_combine.drop([columns],axis = 1,inplace=True)
column_float.remove('SalePrice')

for columns in column_float:

    data_combine[columns]=data_combine[columns].astype(int)
data_train=data_combine.head(1460)

data_train['SalePrice']=data_train['SalePrice'].astype(int)

data_test=data_combine.tail(1459)

data_test=data_test.drop(['SalePrice'],axis=1)

data_train.to_csv('train_data.csv')

data_test.to_csv('test_data.csv')
data_train.info()
data_train.head()