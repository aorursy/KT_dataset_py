# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
###Reading the data

train=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv',parse_dates=['YrSold'])

train=train.set_index("Id") ###이거 잊어먹지 말 것, 차원은 항상 맞춰야 한다. 

train.head()

###list(train.select_dtypes(['object']).columns)

####train.shape (1460,80)
###preprocessing for the train_set

train['MSZoning'].value_counts()

train['LotShape'].value_counts() #(925,484,41,10)

train['LandContour'].value_counts() #(1311,63,50,30)

train['Utilities'].value_counts() #utilities 뺀다

train['PoolArea'].value_counts() #PoolArea 뺀다

#### sorting useful obj columns

for object_columns in list(train.select_dtypes(['object']).columns):

    print(train[object_columns].value_counts())

too_many_categories=['Neighborhood','HouseStyle','RoofMatl','Exterior1st','Exterior2nd','GarageType','SaleType']

drop_obj_columns=['Street','Alley','LandContour','Condition1','MiscFeature','Condition2','RoofStyle','BsmtQual','BsmtFinType1','BsmtFinType2','Functional','PavedDrive','PoolQC','Fence','SaleCondition','Neighborhood','HouseStyle','RoofMatl','Exterior1st','Exterior2nd','GarageType','SaleType']

train_2=train.drop(drop_obj_columns,axis=1)

train_3=train_2.drop('YrSold',axis=1)

train_3.head()







### getting distributions of useful object columns

for object_columns in list(train_3.select_dtypes(['object']).columns):

    print(train_3.groupby(object_columns)['SalePrice'].mean())

###columns with Quality indices = ['ExterQual','ExterCond','BsmtCond','HeatingQC','KitchenQual','FireplaceQu','GarageQual','GarageCond']

###columns with 2~3 indices = ['CentralAir','GarageFinish']

columns_quality=['ExterQual','ExterCond','BsmtCond','HeatingQC','KitchenQual','FireplaceQu','GarageQual','GarageCond']

train_4=train_3[columns_quality].replace({'Ex':5,'Fa':4,'Gd':3, 'Po':2, 'TA':1})

train_5=train_3['CentralAir'].replace({'N':0,'Y':1})

train_6=train_3['GarageFinish'].replace({'Fin':3,'RFn':2,'Unf':1})

train_7=train_3.select_dtypes(exclude='object')

train_object_processed=train_7.merge(train_6,on='Id').merge(train_5,on='Id').merge(train_4,on='Id')

train_object_processed.head()

train_object_processed['YearBuilt'].value_counts()

train_object_processed['YearRemodAdd'].value_counts()

train_object_final=train_object_processed.drop(['YearBuilt','YearRemodAdd'],axis=1)

train_object_final.head()

train_8=train_object_final

###imputation

from sklearn.impute import SimpleImputer

my_imputer = SimpleImputer()

imputed_train = pd.DataFrame(my_imputer.fit_transform(train_8))

imputed_train.columns=train_8.columns

imputed_train.head()

train_final=imputed_train.drop('SalePrice',axis=1)

y=imputed_train.SalePrice

train_final.head()

train_4

###reading the test data

test=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

test.head()

test_2=test.drop(drop_obj_columns,axis=1)

test_2=test_2.set_index("Id")

test_2.head()

len(list(test_2.columns))

test_3=test_2.drop('YrSold', axis=1)

test_3.head()

####preprocessing object columns for the test set

columns_quality=['ExterQual','ExterCond','BsmtCond','HeatingQC','KitchenQual','FireplaceQu','GarageQual','GarageCond']

test_4=test_3[columns_quality].replace({'Ex':5,'Fa':4,'Gd':3, 'Po':2, 'TA':1})

test_5=test_3['CentralAir'].replace({'N':0,'Y':1})

test_6=test_3['GarageFinish'].replace({'Fin':3,'RFn':2,'Unf':1})

test_7=test_3.select_dtypes(exclude='object')

test_object_processed=test_7.merge(test_6,on='Id').merge(test_5,on='Id').merge(test_4,on='Id')

test_object_processed.head()

test_object_final=test_object_processed.drop(['YearBuilt','YearRemodAdd'],axis=1)

test_object_final.head()

test_8=test_object_final

from sklearn.impute import SimpleImputer

my_imputer = SimpleImputer()

imputed_test = pd.DataFrame(my_imputer.fit_transform(test_8))

imputed_test.columns=test_8.columns

test_final=imputed_test

test_final.head()

test_final





from sklearn.ensemble import RandomForestRegressor

house_model=RandomForestRegressor()

house_model.fit(train_final,y)

model_predict=house_model.predict(test_final)

sub=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')

sub['SalePrice']=model_predict

sub.head()

sub.to_csv('house_prices_1.csv',index=False)








