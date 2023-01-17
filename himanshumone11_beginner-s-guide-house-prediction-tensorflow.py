import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
data = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

data_test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
data.info()
data.describe().transpose()
plt.figure(figsize=(12,8))

sns.heatmap(data.corr())
data.corr()
data.corr()['SalePrice'].sort_values(ascending=False)
plt.figure(figsize=(12,8))

sns.distplot(data['SalePrice'])
plt.figure(figsize=(12,8))

sns.scatterplot(x='SalePrice',y='GrLivArea',data=data)
plt.figure(figsize=(12,8))

data.groupby('YearBuilt').mean()['SalePrice'].plot()
pd.options.display.max_columns = None

pd.options.display.max_rows = None

data.isnull().sum()
data = data.fillna(0)
data.isnull().sum()
data.head(10)
data = data.drop('Id',axis=1)
data.select_dtypes(['object']).columns
data['MSZoning'].value_counts()
data['Street'].value_counts()
data['Alley'].value_counts()
dummies = pd.get_dummies(data[['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities',

       'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',

       'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',

       'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',

       'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',

       'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',

       'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',

       'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature',

       'SaleType', 'SaleCondition']],drop_first=True)

data = data.drop(['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities',

       'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',

       'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',

       'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',

       'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',

       'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',

       'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',

       'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature',

       'SaleType', 'SaleCondition'],axis=1)

data = pd.concat([data,dummies],axis=1)
data.head(10)
data.shape
from sklearn.model_selection import train_test_split
X = data.drop('SalePrice',axis=1).values

y = data['SalePrice'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Activation,Dropout

from tensorflow.keras.optimizers import Adam
X_train.shape
model = Sequential()



model.add(Dense(261,  activation='relu'))

model.add(Dropout(0.2))



# hidden layer

model.add(Dense(130, activation='relu'))

model.add(Dropout(0.2))



# hidden layer

model.add(Dense(65, activation='relu'))

model.add(Dropout(0.2))



# hidden layer

model.add(Dense(32, activation='relu'))

model.add(Dropout(0.2))



# hidden layer

model.add(Dense(16, activation='relu'))

model.add(Dropout(0.2))



# output layer

model.add(Dense(1))



# Compile model

model.compile(loss='mse', optimizer='adam')
from tensorflow.keras.callbacks import EarlyStopping
model.fit(x=X_train, 

          y=y_train, 

          epochs=400,

          batch_size=256,

          validation_data=(X_test, y_test), 

          )
losses = pd.DataFrame(model.history.history)
losses.plot()
from sklearn.metrics import mean_squared_error,mean_absolute_error,explained_variance_score
predictions = model.predict(X_test)
mean_absolute_error(y_test,predictions)
np.sqrt(mean_squared_error(y_test,predictions))
explained_variance_score(y_test,predictions)
# Our predictions

plt.scatter(y_test,predictions)



# Perfect predictions

plt.plot(y_test,y_test,'r')
single_house = data.drop('SalePrice',axis=1).iloc[0]
single_house
single_house.shape
X_train.shape
single_house = scaler.transform(single_house.values.reshape(-1, 261))
model.predict(single_house)
data.iloc[0]
import random

random.seed(101)

random_ind = random.randint(0,len(data))



new_house = data.drop('SalePrice',axis=1).iloc[random_ind]

new_house
new_house = scaler.transform(new_house.values.reshape(-1, 261))
model.predict(new_house)
data.iloc[random_ind]
random_ind
data_test.head(10)
dummies_test = pd.get_dummies(data_test[['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities',

       'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',

       'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',

       'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',

       'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',

       'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',

       'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',

       'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature',

       'SaleType', 'SaleCondition']],drop_first=True)

data_test = data_test.drop(['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities',

       'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',

       'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',

       'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',

       'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',

       'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',

       'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',

       'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature',

       'SaleType', 'SaleCondition'],axis=1)

data_test = pd.concat([data_test,dummies_test],axis=1)
data_test.shape
data_test = data_test.drop('Id',axis=1)
data_modified_for_test = data.drop(['BsmtExposure_Av', 'Condition2_RRAn', 'FireplaceQu_Ex', 'Heating_OthW', 'Utilities_NoSeWa', 'BsmtFinType2_ALQ', 'PoolQC_Ex', 'RoofMatl_Membran', 'HouseStyle_2.5Fin', 'RoofMatl_CompShg', 'Condition2_RRNn', 'RoofMatl_Roll', 'Heating_GasA', 'MiscFeature_Gar2', 'GarageCond_Ex', 'Condition2_RRAe', 'GarageFinish_Fin', 'Exterior1st_Stone', 'MasVnrType_BrkCmn', 'BsmtFinType1_ALQ', 'Exterior1st_ImStucc', 'MiscFeature_TenC', 'BsmtCond_Fa', 'PoolQC_Fa', 'Electrical_Mix', 'Electrical_FuseA', 'GarageQual_Fa', 'Fence_GdPrv', 'Alley_Grvl', 'RoofMatl_Metal', 'SalePrice', 'GarageQual_Ex', 'Exterior2nd_Other', 'BsmtQual_Ex', 'GarageType_2Types'],axis=1)
data_modified_for_test.shape
data_test.shape
data_modified_for_test.head(10)
X_1 = data_modified_for_test.values
X_train, X_test, y_train, y_test = train_test_split(X_1, y, test_size=0.2, random_state=101)
from sklearn.preprocessing import MinMaxScaler
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_train.shape
model = Sequential()



model.add(Dense(227,  activation='relu'))

model.add(Dropout(0.2))



# hidden layer

model.add(Dense(114, activation='relu'))

model.add(Dropout(0.2))



# hidden layer

model.add(Dense(57, activation='relu'))

model.add(Dropout(0.2))



# hidden layer

model.add(Dense(29, activation='relu'))

model.add(Dropout(0.2))



# hidden layer

model.add(Dense(14, activation='relu'))

model.add(Dropout(0.2))



# output layer

model.add(Dense(1))



# Compile model

model.compile(loss='mse', optimizer='adam')
from tensorflow.keras.callbacks import EarlyStopping
model.fit(x=X_train, 

          y=y_train, 

          epochs=400,

          batch_size=256,

          validation_data=(X_test, y_test), 

          )
losses = pd.DataFrame(model.history.history)
losses.plot()
predictions = model.predict(X_test)
mean_absolute_error(y_test,predictions)
np.sqrt(mean_squared_error(y_test,predictions))
# Our predictions

plt.scatter(y_test,predictions)



# Perfect predictions

plt.plot(y_test,y_test,'r')
single_house = data_modified_for_test.iloc[0]
single_house
single_house.shape
X_train.shape
single_house = scaler.transform(single_house.values.reshape(-1, 227))
model.predict(single_house)
data_modified_for_test.head(10)
data_test.isnull().sum()
data_test.head(10)
data_test = data_test.fillna(0)
data_modified_for_test = scaler.transform(data_test.values.reshape(-1, 227))
model.predict(data_modified_for_test)
single_house_test = data_test.iloc[0]
single_house_test
single_house_test = scaler.transform(single_house_test.values.reshape(-1, 227))
model.predict(single_house_test)
Predicted_values = model.predict(data_modified_for_test)
data_test_raw = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

Id = data_test_raw['Id']
Id.shape
Predicted_values.shape
Predicted_values_1 = Predicted_values.reshape(1459)
Predicted_values_1.shape
Id.head(10)
Predicted_values_1
df = pd.DataFrame({"ID" : Id, "SalePrice" : Predicted_values_1})

df.to_csv("prediction values.csv", index=False)