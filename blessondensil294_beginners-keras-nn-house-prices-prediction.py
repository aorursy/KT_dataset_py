# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # Data Visualization

import seaborn as sns # Advance Data Visualization

%matplotlib inline



#OS packages

import os



#Label Encoding of Columns

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()



#Kera Neural Network Package

from keras.models import Sequential, load_model

from keras.layers.core import Dense, Dropout, Activation

from keras.utils import np_utils

from keras.wrappers.scikit_learn import KerasRegressor
# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_Train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

df_Test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
#To find the head of the Data

df_Train.head()
#Information of the Dataset Datatype

df_Train.info()
#Information of the Dataset Continuous Values

df_Train.describe()
#Columns List

df_Train.columns
#Shape of the Train and Test Data

print('Shape of Train Data: ', df_Train.shape)

print('Shape of Test Data: ', df_Test.shape)
#Null values in the Train Dataset

print('Null values in Train Data: \n', df_Train.isnull().sum())
#Null Values in the Test Dataset

print('Null Values in Test Data: \n', df_Test.isnull().sum())
# We will concat both train and test data set

df_Train['is_train'] = 1

df_Test['is_train'] = 0



#df_Frames = [df_Train,df_Test]

df_Total = pd.concat([df_Train, df_Test])
#Percentage of the Missing Data



null_value = pd.concat([(df_Total.isnull().sum() /  df_Total.isnull().count())*100], axis=1, keys=['DF_TOTAL'], sort=False)

null_value[null_value.sum(axis = 1) > 0].sort_values(by = ['DF_TOTAL'], ascending = False)
#Deleting the Columns with more than 40% Null Values



df_Total.drop('PoolQC', axis = 1, inplace = True)

df_Total.drop('MiscFeature', axis = 1, inplace = True)

df_Total.drop('Alley', axis = 1, inplace = True)

df_Total.drop('Fence', axis = 1, inplace = True)

df_Total.drop('FireplaceQu', axis = 1, inplace = True)
#using Forward Fill to fill missing Values



df_Total['LotFrontage'] = df_Total['LotFrontage'].fillna(method="ffill",axis=0)

df_Total['GarageCond'] = df_Total['GarageCond'].fillna(method="ffill",axis=0)

df_Total['GarageYrBlt'] = df_Total['GarageYrBlt'].fillna(method="ffill",axis=0)

df_Total['GarageFinish'] = df_Total['GarageFinish'].fillna(method="ffill",axis=0)

df_Total['GarageQual'] = df_Total['GarageQual'].fillna(method="ffill",axis=0)

df_Total['GarageType'] = df_Total['GarageType'].fillna(method="ffill",axis=0)

df_Total['BsmtExposure'] = df_Total['BsmtExposure'].fillna(method="ffill",axis=0)

df_Total['BsmtCond'] = df_Total['BsmtCond'].fillna(method="ffill",axis=0)

df_Total['BsmtQual'] = df_Total['BsmtQual'].fillna(method="ffill",axis=0)

df_Total['BsmtFinType2'] = df_Total['BsmtFinType2'].fillna(method="ffill",axis=0)

df_Total['BsmtFinType1'] = df_Total['BsmtFinType1'].fillna(method="ffill",axis=0)

df_Total['MasVnrType'] = df_Total['MasVnrType'].fillna(method="ffill",axis=0)

df_Total['MasVnrArea'] = df_Total['MasVnrArea'].fillna(method="ffill",axis=0)

df_Total['MSZoning'] = df_Total['MSZoning'].fillna(method="ffill",axis=0)

df_Total['Functional'] = df_Total['Functional'].fillna(method="ffill",axis=0)

df_Total['BsmtHalfBath'] = df_Total['BsmtHalfBath'].fillna(method="ffill",axis=0)

df_Total['BsmtFullBath'] = df_Total['BsmtFullBath'].fillna(method="ffill",axis=0)

df_Total['Utilities'] = df_Total['Utilities'].fillna(method="ffill",axis=0)

df_Total['KitchenQual'] = df_Total['KitchenQual'].fillna(method="ffill",axis=0)

df_Total['TotalBsmtSF'] = df_Total['TotalBsmtSF'].fillna(method="ffill",axis=0)

df_Total['BsmtUnfSF'] = df_Total['BsmtUnfSF'].fillna(method="ffill",axis=0)

df_Total['GarageCars'] = df_Total['GarageCars'].fillna(method="ffill",axis=0)

df_Total['GarageArea'] = df_Total['GarageArea'].fillna(method="ffill",axis=0)

df_Total['BsmtFinSF2'] = df_Total['BsmtFinSF2'].fillna(method="ffill",axis=0)

df_Total['BsmtFinSF1'] = df_Total['BsmtFinSF1'].fillna(method="ffill",axis=0)

df_Total['Exterior2nd'] = df_Total['Exterior2nd'].fillna(method="ffill",axis=0)

df_Total['Exterior1st'] = df_Total['Exterior1st'].fillna(method="ffill",axis=0)

df_Total['SaleType'] = df_Total['SaleType'].fillna(method="ffill",axis=0)

df_Total['Electrical'] = df_Total['Electrical'].fillna(method="ffill",axis=0)
#Percentage of the Missing Data



null_value = pd.concat([(df_Total.isnull().sum() /  df_Total.isnull().count())*100], axis=1, keys=['DF_TOTAL'], sort=False)

null_value[null_value.sum(axis = 1) > 0].sort_values(by = ['DF_TOTAL'], ascending = False)
"""

#get dummies



Column_Object = df_Total.dtypes[df_Total.dtypes == 'object'].index

df_Total = pd.get_dummies(df_Total, columns = Column_Object, dummy_na = True)



"""
df_Total = pd.get_dummies(df_Total, columns=["MSZoning"])

df_Total = pd.get_dummies(df_Total, columns=["LotShape"])

df_Total = pd.get_dummies(df_Total, columns=["LandContour"])

df_Total = pd.get_dummies(df_Total, columns=["LotConfig"])

df_Total = pd.get_dummies(df_Total, columns=["LandSlope"])

df_Total = pd.get_dummies(df_Total, columns=["Neighborhood"])

df_Total = pd.get_dummies(df_Total, columns=["Condition1"])

df_Total = pd.get_dummies(df_Total, columns=["Condition2"])

df_Total = pd.get_dummies(df_Total, columns=["BldgType"])

df_Total = pd.get_dummies(df_Total, columns=["HouseStyle"])

df_Total = pd.get_dummies(df_Total, columns=["RoofStyle"])

df_Total = pd.get_dummies(df_Total, columns=["RoofMatl"])

df_Total = pd.get_dummies(df_Total, columns=["Exterior1st"])

df_Total = pd.get_dummies(df_Total, columns=["Exterior2nd"])

df_Total = pd.get_dummies(df_Total, columns=["MasVnrType"])

df_Total = pd.get_dummies(df_Total, columns=["ExterQual"])

df_Total = pd.get_dummies(df_Total, columns=["ExterCond"])

df_Total = pd.get_dummies(df_Total, columns=["Foundation"])

df_Total = pd.get_dummies(df_Total, columns=["BsmtQual"])

df_Total = pd.get_dummies(df_Total, columns=["BsmtCond"])

df_Total = pd.get_dummies(df_Total, columns=["BsmtExposure"])

df_Total = pd.get_dummies(df_Total, columns=["BsmtFinType1"])

df_Total = pd.get_dummies(df_Total, columns=["BsmtFinType2"])

df_Total = pd.get_dummies(df_Total, columns=["Heating"])

df_Total = pd.get_dummies(df_Total, columns=["HeatingQC"])

df_Total = pd.get_dummies(df_Total, columns=["Electrical"])

df_Total = pd.get_dummies(df_Total, columns=["KitchenQual"])

df_Total = pd.get_dummies(df_Total, columns=["Functional"])

df_Total = pd.get_dummies(df_Total, columns=["GarageType"])

df_Total = pd.get_dummies(df_Total, columns=["GarageFinish"])

df_Total = pd.get_dummies(df_Total, columns=["GarageQual"])

df_Total = pd.get_dummies(df_Total, columns=["GarageCond"])

df_Total = pd.get_dummies(df_Total, columns=["PavedDrive"])

df_Total = pd.get_dummies(df_Total, columns=["SaleType"])

df_Total = pd.get_dummies(df_Total, columns=["SaleCondition"])



df_Total['Street'] = le.fit_transform(df_Total['Street'])

df_Total['Utilities'] = le.fit_transform(df_Total['Utilities'])

df_Total['CentralAir'] = le.fit_transform(df_Total['CentralAir'])
df_Total.shape
#Un-Merge code

df_Train_final = df_Total[df_Total['is_train'] == 1]

df_Test_final = df_Total[df_Total['is_train'] == 0]
x = df_Train_final

x = x.drop(['Id'], axis=1)

#x = x.drop(['patientid'], axis=1)

x = x.drop(['is_train'], axis=1)

x = x.drop(['SalePrice'], axis=1)

y = df_Train_final['SalePrice']

x_pred = df_Test_final

x_pred = x_pred.drop(['Id'], axis=1)

#x_pred = x_pred.drop(['patientid'], axis=1)

x_pred = x_pred.drop(['is_train'], axis=1)

x_pred = x_pred.drop(['SalePrice'], axis=1)
x.shape
x_pred.shape
model = Sequential()



model.add(Dense(128, input_shape=(267,)))

model.add(Activation('relu'))

model.add(Dropout(0.5))



model.add(Dense(64))

model.add(Activation('relu'))

model.add(Dropout(0.5))



model.add(Dense(16))

model.add(Activation('relu'))

model.add(Dropout(0.5))



model.add(Dense(1))



model.compile(loss='mean_squared_error', metrics=['mse'], optimizer='adam')
history = model.fit(x, np.ravel(y), epochs=500, verbose=2)
#Plotting the Accuracy Metrics

fig = plt.figure()

plt.subplot(2,1,1)

plt.plot(history.history['mse'])

plt.plot(history.history['loss'])

plt.title('Model Accuracy')

plt.ylabel('Mean Square Error')

plt.xlabel('Loss')

plt.legend(['Mean Square Error', 'Loss'], loc='lower right')

fig
y_pred = model.predict_classes(x_pred)
y_pred = y_pred.reshape(1459,)
submission_df = pd.DataFrame({'Id':df_Test['Id'], 'SalePrice':y_pred})

submission_df.to_csv('Sample Submission.csv', index=False)