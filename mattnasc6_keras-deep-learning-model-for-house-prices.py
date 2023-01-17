# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import matplotlib.pyplot as plt

import tensorflow.keras



from keras.layers import Dense

from keras.models import Sequential
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.corr()
all_data = pd.concat((train.loc[:, 'MSSubClass':'SaleCondition'], test.loc[:, 'MSSubClass':'SaleCondition']))
all_data.sample(15)
all_data_na = (all_data.isnull().sum()/len(all_data)) * 100

all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending = False)

missing_data = pd.DataFrame({'Missing Ratio': all_data_na})

missing_data
all_data = all_data.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence'], axis = 1)
numeric = all_data.dtypes[all_data.dtypes != 'object'].index

categorical = all_data.dtypes[all_data.dtypes == 'object'].index
#It seems to me that Month of Sold shoud be categorical

all_data['MoSold'] = all_data['MoSold'].apply(str)



all_data['FireplaceQu'].value_counts()

all_data['FireplaceQu'] = all_data['FireplaceQu'].fillna('None')



all_data['LotFrontage'].value_counts()

all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))



all_data['GarageFinish'].value_counts()

all_data['GarageFinish'] = all_data['GarageFinish'].fillna('None')



all_data['GarageYrBlt'].value_counts()

all_data['GarageYrBlt'] = all_data['GarageYrBlt'].apply(str)

all_data['GarageYrBlt'] = all_data['GarageYrBlt'].fillna('None')



all_data['GarageQual'].value_counts()

all_data['GarageQual'] = all_data['GarageQual'].fillna('None')



all_data['GarageCond'].value_counts()

all_data['GarageCond'] = all_data['GarageCond'].fillna('None')



all_data['GarageType'].value_counts()

all_data['GarageType'] = all_data['GarageType'].fillna('None')



all_data['BsmtExposure'].value_counts()

all_data['BsmtExposure'] = all_data['BsmtExposure'].fillna('None')



all_data['BsmtCond'].value_counts()

all_data['BsmtCond'] = all_data['BsmtCond'].fillna('None')



all_data['BsmtQual'].value_counts()

all_data['BsmtQual'] = all_data['BsmtQual'].fillna('None')



all_data['BsmtFinType2'].value_counts()

all_data['BsmtFinType2'] = all_data['BsmtFinType2'].fillna('None')



all_data['BsmtFinType1'].value_counts()

all_data['BsmtFinType1'] = all_data['BsmtFinType1'].fillna('None')



all_data['MasVnrType'].value_counts()

all_data['MasVnrType'] = all_data['MasVnrType'].fillna('None')



all_data['MasVnrArea'].value_counts()

all_data['MasVnrArea'] = all_data['MasVnrArea'].fillna(0)



all_data['MSZoning'].value_counts()

all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])



all_data['BsmtFullBath'].value_counts()

all_data['BsmtFullBath'] = all_data['BsmtFullBath'].fillna(0)



all_data['BsmtHalfBath'].value_counts()

all_data['BsmtHalfBath'] = all_data['BsmtHalfBath'].fillna(0)



all_data['Utilities'].value_counts()

all_data = all_data.drop('Utilities', axis = 1)



all_data['Functional'].value_counts()

all_data['Functional'] = all_data['Functional'].fillna('Typ')



all_data['Exterior2nd'].value_counts()

all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])



all_data['Exterior1st'].value_counts()

all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])



all_data['SaleType'].value_counts()

all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])



all_data['BsmtFinSF1'].value_counts()

all_data['BsmtFinSF1'] = all_data['BsmtFinSF1'].transform(lambda x: x.fillna(x.median()))



all_data['BsmtFinSF2'].value_counts()

all_data['BsmtFinSF2'] = all_data['BsmtFinSF2'].transform(lambda x: x.fillna(x.median()))



all_data['BsmtUnfSF'].value_counts()

all_data['BsmtUnfSF'] = all_data['BsmtUnfSF'].transform(lambda x: x.fillna(x.median()))



all_data['Electrical'].value_counts()

all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])



all_data['KitchenQual'].value_counts()

all_data['KitchenQual'] = all_data['KitchenQual'].fillna('TA')



all_data['GarageCars'].value_counts()

all_data['GarageCars'] = all_data['GarageCars'].fillna(2)



all_data['GarageArea'].value_counts()

all_data['GarageArea'] = all_data['GarageArea'].transform(lambda x: x.fillna(x.median()))



all_data['TotalBsmtSF'].value_counts()

all_data['TotalBsmtSF'] = all_data['TotalBsmtSF'].transform(lambda x: x.fillna(x.median()))
all_data = pd.get_dummies(all_data)
X_train = all_data[: train.shape[0]]

X_test = all_data[train.shape[0]:]



y_train = train['SalePrice']
model = Sequential()

model.add(Dense(40, input_dim = X_train.shape[1], activation = 'relu'))

model.add(Dense(40, activation = 'relu'))

model.add(Dense(1, activation = 'linear'))
model.summary()
model.compile(optimizer = 'adam', loss = 'mean_squared_logarithmic_error')
epochs_hist = model.fit(X_train, y_train, epochs = 1000, batch_size = 60, verbose = 1, validation_split = 0.1)
epochs_hist.history.keys()
plt.plot(epochs_hist.history['loss'])

plt.plot(epochs_hist.history['val_loss'])

plt.title('Model loss during training')

plt.xlabel('Epoch number')

plt.ylabel('Training and validation loss')

plt.legend(['Training loss', 'Validation loss'])
y_predict = model.predict(X_test)

predicted = [item for sublist in y_predict for item in sublist]

predicted
solution = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted})

solution.to_csv('salvefamilia', index = False)