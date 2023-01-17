import numpy as np # linear algebra

import pandas as pd # Data preprocessing



# Visualization

import seaborn as sns 

import matplotlib.pyplot as plt

import missingno as mn



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

test_df = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
train.head()
mn.matrix(train, figsize = (30,10))
train.shape
train.info()
train['LotFrontage'] = train['LotFrontage'].fillna(train['LotFrontage'].mean())
train.drop(['Alley'], axis = 1, inplace = True)
train['BsmtCond'] = train['BsmtCond'].fillna(train['BsmtCond'].mode()[0])

train['BsmtQual'] = train['BsmtQual'].fillna(train['BsmtQual'].mode()[0])
train['FireplaceQu'] = train['FireplaceQu'].fillna(train['FireplaceQu'].mode()[0])

train['GarageType'] = train['GarageType'].fillna(train['GarageType'].mode()[0])
train.drop(['GarageYrBlt'], axis = 1, inplace = True)
train['GarageFinish'] = train['GarageFinish'].fillna(train['GarageFinish'].mode()[0])

train['GarageQual'] = train['GarageQual'].fillna(train['GarageQual'].mode()[0])

train['GarageCond'] = train['GarageCond'].fillna(train['GarageCond'].mode()[0])
train.drop(['PoolQC','Fence','MiscFeature', 'Id'], axis = 1, inplace = True)
train.shape
train.isnull().sum().sum()
train['MasVnrType'] = train['MasVnrType'].fillna(train['MasVnrType'].mode()[0])

train['MasVnrArea'] = train['MasVnrArea'].fillna(train['MasVnrArea'].mode()[0])
mn.matrix(train, figsize = (30, 10))
train['BsmtExposure'] = train['BsmtExposure'].fillna(train['BsmtExposure'].mode()[0])

train['BsmtFinType2'] = train['BsmtFinType2'].fillna(train['BsmtFinType2'].mode()[0])
train.dropna(inplace = True)
train.shape
train.head()
#Categorical columns:

columns=['MSZoning','Street','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood',

         'Condition2','BldgType','Condition1','HouseStyle','SaleType',

        'SaleCondition','ExterCond',

         'ExterQual','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',

        'RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','Heating','HeatingQC',

         'CentralAir',

         'Electrical','KitchenQual','Functional',

         'FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive']
len(columns)
def category_onehot_multcols(multcolumns):

    df_final = final_df

    i = 0

    for fields in multcolumns:

        

        print(fields)

        df1 = pd.get_dummies(final_df[fields], drop_first = True)

        

        final_df.drop([fields], axis=1, inplace=True)

        if i == 0:

            df_final = df1.copy()

        else:

            

            df_final = pd.concat([df_final, df1], axis=1)

        i = i + 1 

       

        

    df_final = pd.concat([final_df, df_final], axis=1)

        

    return df_final
main = train.copy()
# Handle test data
test_df.shape
test_df.head()
test_df.isnull().sum()
test_df['LotFrontage']=test_df['LotFrontage'].fillna(test_df['LotFrontage'].mean())

test_df['MSZoning']=test_df['MSZoning'].fillna(test_df['MSZoning'].mode()[0])

test_df['BsmtCond']=test_df['BsmtCond'].fillna(test_df['BsmtCond'].mode()[0])

test_df['BsmtQual']=test_df['BsmtQual'].fillna(test_df['BsmtQual'].mode()[0])

test_df['FireplaceQu']=test_df['FireplaceQu'].fillna(test_df['FireplaceQu'].mode()[0])

test_df['GarageType']=test_df['GarageType'].fillna(test_df['GarageType'].mode()[0])

test_df['GarageFinish']=test_df['GarageFinish'].fillna(test_df['GarageFinish'].mode()[0])

test_df['GarageQual']=test_df['GarageQual'].fillna(test_df['GarageQual'].mode()[0])

test_df['GarageCond']=test_df['GarageCond'].fillna(test_df['GarageCond'].mode()[0])
test_df.drop(['PoolQC','Fence','MiscFeature', 'Alley', 'GarageYrBlt', 'Id'],axis=1,inplace=True)
test_df['MasVnrType']=test_df['MasVnrType'].fillna(test_df['MasVnrType'].mode()[0])

test_df['MasVnrArea']=test_df['MasVnrArea'].fillna(test_df['MasVnrArea'].mode()[0])
test_df.shape
mn.matrix(test_df, figsize = (30, 10))
test_df['BsmtExposure'] = test_df['BsmtExposure'].fillna(test_df['BsmtExposure'].mode()[0])

test_df['BsmtFinType2'] = test_df['BsmtFinType2'].fillna(test_df['BsmtFinType2'].mode()[0])
mn.matrix(test_df, figsize = (30, 10))
test_df.loc[:, test_df.isnull().any()].head()
test_df['Utilities']=test_df['Utilities'].fillna(test_df['Utilities'].mode()[0])

test_df['Exterior1st']=test_df['Exterior1st'].fillna(test_df['Exterior1st'].mode()[0])

test_df['Exterior2nd']=test_df['Exterior2nd'].fillna(test_df['Exterior2nd'].mode()[0])

test_df['BsmtFinType1']=test_df['BsmtFinType1'].fillna(test_df['BsmtFinType1'].mode()[0])

test_df['BsmtFinSF1']=test_df['BsmtFinSF1'].fillna(test_df['BsmtFinSF1'].mean())

test_df['BsmtFinSF2']=test_df['BsmtFinSF2'].fillna(test_df['BsmtFinSF2'].mean())

test_df['BsmtUnfSF']=test_df['BsmtUnfSF'].fillna(test_df['BsmtUnfSF'].mean())

test_df['TotalBsmtSF']=test_df['TotalBsmtSF'].fillna(test_df['TotalBsmtSF'].mean())

test_df['BsmtFullBath']=test_df['BsmtFullBath'].fillna(test_df['BsmtFullBath'].mode()[0])

test_df['BsmtHalfBath']=test_df['BsmtHalfBath'].fillna(test_df['BsmtHalfBath'].mode()[0])

test_df['KitchenQual']=test_df['KitchenQual'].fillna(test_df['KitchenQual'].mode()[0])

test_df['Functional']=test_df['Functional'].fillna(test_df['Functional'].mode()[0])

test_df['GarageCars']=test_df['GarageCars'].fillna(test_df['GarageCars'].mean())

test_df['GarageArea']=test_df['GarageArea'].fillna(test_df['GarageArea'].mean())

test_df['SaleType']=test_df['SaleType'].fillna(test_df['SaleType'].mode()[0])
test_df.shape
test_df.head()
final_df = pd.concat([train, test_df], axis = 0)
final_df['SalePrice']
final_df.shape
final_df = category_onehot_multcols(columns)
final_df.shape
final_df = final_df.loc[:, ~final_df.columns.duplicated()]
final_df
df_train = final_df.iloc[:1422, :]

df_test = final_df.iloc[1422:, :]
df_train.head()
df_test.head()
df_train.shape
df_test.drop(['SalePrice'], axis = 1, inplace = True)
Xtrain = df_train.drop(['SalePrice'], axis = 1)

ytrain = df_train['SalePrice']
from keras import backend as K

def root_mean_squared_error(y_true, y_pred):

        return K.sqrt(K.mean(K.square(y_pred - y_true)))
import keras

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LeakyReLU,PReLU,ELU



# Initialising the ANN

classifier = Sequential()



# Adding the input layer and the first hidden layer

classifier.add(Dense(units = 50, kernel_initializer = 'he_uniform',activation='relu',input_dim = 174))



# Adding the second hidden layer

classifier.add(Dense(units = 25, kernel_initializer = 'he_uniform',activation='relu'))



# Adding the third hidden layer

classifier.add(Dense(units = 50, kernel_initializer = 'he_uniform',activation='relu'))



# Adding the output layer

classifier.add(Dense(units = 1, kernel_initializer = 'he_uniform'))



# Compiling the ANN

classifier.compile(loss=root_mean_squared_error, optimizer='Adamax')



# Fitting the ANN in the training set

model_history=classifier.fit(Xtrain.values, ytrain.values,validation_split=0.20, batch_size = 10, epochs = 3000)
ann_pred=classifier.predict(df_test.values)
sub = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv")
submission = pd.DataFrame()

submission['Id'] = sub['Id']

submission['SalePrice'] = ann_pred

submission.head()
sub.head()
sub.shape == submission.shape
submission.to_csv("my_submission.csv", index = False)

print("Send")