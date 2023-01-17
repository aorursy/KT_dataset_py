#!pip3 install tensorflow==1.15
#!pip3 install keras
#!pip3 install -U scikit-learn
import numpy as np
import matplotlib.pylab as plt
import pandas as pd
import os
from sklearn import preprocessing, model_selection

from keras.layers import Input, Dense, BatchNormalization, Dropout
from keras import optimizers, regularizers, initializers, Model
import keras.backend as K

import tensorflow as tf
print(tf.__version__)

%matplotlib inline
train = pd.read_csv('train.csv')
print(train.columns)
train.drop('Id', axis=1, inplace=True)
print('train set size: ', train.shape)
train.head()
not_nulls = {}
for colname in train.columns :
    not_nulls[colname] = train[colname].notnull().sum()/train.shape[0]
    
print(not_nulls)
cols = ['Alley','FireplaceQu','PoolQC','Fence','MiscFeature']
train = train.drop(cols, axis=1)
print('train set size after removing nan columns: ', train.shape)
fig = plt.figure(figsize=(11, 5))

ax = fig.add_subplot(2,4,1)
ax.scatter(train.LotArea, train.SalePrice/1e3)
ax.set_xlabel('LotArea')
ax.set_ylabel('SalePrice (thousand $)')

ax = fig.add_subplot(2,4,2)
ax.scatter(train.GrLivArea, train.SalePrice/1e3)
ax.set_xlabel('GrLivArea')

ax = fig.add_subplot(2,4,3)
ax.scatter(train.YearBuilt, train.SalePrice/1e3)
ax.set_xlabel('YearBuilt')

ax = fig.add_subplot(2,4,4)
ax.scatter(train.TotalBsmtSF, train.SalePrice/1e3)
ax.set_xlabel('TotalBsmtSF')

ax = fig.add_subplot(2,4,5)
ax.scatter(train['1stFlrSF'], train.SalePrice/1e3)
ax.set_xlabel('1stFlrSF')

ax = fig.add_subplot(2,4,6)
ax.scatter(train.TotRmsAbvGrd, train.SalePrice/1e3)
ax.set_xlabel('TotRmsAbvGrd')

ax = fig.add_subplot(2,4,7)
ax.scatter(train.GarageArea, train.SalePrice/1e3)
ax.set_xlabel('GarageArea')

ax = fig.add_subplot(2,4,8)
ax.scatter(train['2ndFlrSF'], train.SalePrice/1e3)
ax.set_xlabel('2ndFlrSF')

fig.tight_layout();
train = train.drop(train[(train['LotArea']>60000) & (train['SalePrice']<400000)].index)
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)
train = train.drop(train[(train['TotalBsmtSF']>4000)].index)
train = train.drop(train[(train['1stFlrSF']>4000)].index)
print('External condition: ', train.ExterCond.unique())
print('Sale condition: ', train.SaleCondition.unique())
fig = plt.figure(figsize=(6,2.5))

ax = fig.add_subplot(1,2,1)
train.ExterCond.value_counts().plot(kind='bar')
ax.set_ylabel('counts')

ax = fig.add_subplot(1,2,2)
train.SaleCondition.value_counts().plot(kind='bar');

fig.tight_layout()
train['LotFrontage'] = train['LotFrontage'].fillna(train['LotFrontage'].mean())
train['MasVnrArea'] = train['MasVnrArea'].fillna(train['MasVnrArea'].mean())
train['GarageYrBlt'] = train['GarageYrBlt'].fillna(train['GarageYrBlt'].mean())
fig = plt.figure(figsize=(13,5))

ax = fig.add_subplot(2,6,1)
train.MasVnrType.value_counts().plot(kind='bar')
ax.set_title('MasVnrType')

ax = fig.add_subplot(2,6,2)
train.BsmtQual.value_counts().plot(kind='bar');
ax.set_title('BsmtQual')

ax = fig.add_subplot(2,6,3)
train.BsmtCond.value_counts().plot(kind='bar');
ax.set_title('BsmtCond')

ax = fig.add_subplot(2,6,4)
train.BsmtExposure.value_counts().plot(kind='bar');
ax.set_title('BsmtExposure')

ax = fig.add_subplot(2,6,5)
train.BsmtFinType1.value_counts().plot(kind='bar');
ax.set_title('BsmtFinType1')

ax = fig.add_subplot(2,6,6)
train.BsmtFinType2.value_counts().plot(kind='bar');
ax.set_title('BsmtFinType2')

ax = fig.add_subplot(2,6,7)
train.Electrical.value_counts().plot(kind='bar');
ax.set_title('Electrical')

ax = fig.add_subplot(2,6,8)
train.GarageType.value_counts().plot(kind='bar');
ax.set_title('GarageType')

ax = fig.add_subplot(2,6,9)
train.GarageFinish.value_counts().plot(kind='bar');
ax.set_title('GarageFinish')

ax = fig.add_subplot(2,6,10)
train.GarageQual.value_counts().plot(kind='bar');
ax.set_title('GarageQual')

ax = fig.add_subplot(2,6,11)
train.GarageCond.value_counts().plot(kind='bar');
ax.set_title('GarageCond')

fig.tight_layout()
train['MasVnrType'] = train['MasVnrType'].fillna('None');
train['BsmtQual'] = train['BsmtQual'].fillna('TA');
train['BsmtCond'] = train['BsmtCond'].fillna('TA');
train['BsmtExposure'] = train['BsmtExposure'].fillna('No')
train['BsmtFinType1'] = train['BsmtFinType1'].fillna('Unf')
train['BsmtFinType2'] = train['BsmtFinType2'].fillna('Unf')
train['Electrical'] = train['Electrical'].fillna('SBrkr')
train['GarageType'] = train['GarageType'].fillna('Attchd')
train['GarageFinish'] = train['GarageFinish'].fillna('RFn')
train['GarageQual']= train['GarageQual'].fillna('TA')
train['GarageCond'] = train['GarageCond'].fillna('TA')
not_nulls = {}
for colname in train.columns :
    not_nulls[colname] = train[colname].notnull().sum()/train.shape[0]
    
print(not_nulls)
dt = {}
for col in train.columns : 
    dt[col] = train[col].dtype

print(dt)
train.head()
le = preprocessing.LabelEncoder()
for col in train.columns :
    if train[col].dtype == object : 
        le.fit(train[col])
        train[col] = le.transform(train[col])
 

dt = {}
for col in train.columns : 
    dt[col] = train[col].dtype

print(dt)
train.head()
SalePrice = train.SalePrice
train = train.drop('SalePrice', axis=1)
fig = plt.figure(figsize=(16,3))
train.corrwith(SalePrice).plot(kind='bar', color='r', width=0.8)
for col in train.columns :
    train[col] = (train[col] - train[col].mean())/train[col].std()
SalePrice = np.array(SalePrice).reshape(SalePrice.shape[0], 1)
mu = SalePrice.mean()
sigma = SalePrice.std()
SalePrice = (SalePrice - mu)/sigma
fig = plt.figure(figsize=(6,3))

plt.hist(SalePrice, width=0.5)
plt.xlabel('SalePrice')
plt.ylabel('Counts');
train = train.to_numpy()
train_x, xvalid_x, train_y, xvalid_y = model_selection.train_test_split(train, SalePrice, train_size=0.8, shuffle=True)
print(train_x.shape, train_y.shape, xvalid_x.shape, xvalid_y.shape)
print(type(train_x), type(train_y))
def price_model (input_shape) :
    x_input = Input(input_shape)

    x = Dense(128, activation='tanh', kernel_initializer='glorot_uniform')(x_input)
    x = BatchNormalization(epsilon=0.01, momentum=0.99)(x)
    
    x = Dense(256, activation='tanh', kernel_regularizer=regularizers.l2(0.01))(x)
    x = BatchNormalization(epsilon=0.01, momentum=0.99)(x)
    
    x = Dense(64, activation='tanh', kernel_regularizer=regularizers.l2(0.01))(x)
    x = BatchNormalization(epsilon=0.01, momentum=0.99)(x)
    
    x = Dense(64, activation='tanh', kernel_regularizer=regularizers.l2(0.01))(x)
    x = BatchNormalization(epsilon=0.01, momentum=0.99)(x)
    
    x = Dense(16, activation='tanh', kernel_regularizer=regularizers.l2(0.01))(x)
    x = BatchNormalization(epsilon=0.01, momentum=0.99)(x)

    x_output = Dense(1, activation=None, use_bias=True, kernel_regularizer=regularizers.l2(0.01),
              bias_regularizer=regularizers.l2(0.02))(x)

    model = Model(inputs=x_input, outputs=x_output, name='price_model')
    
    return model
priceModel = price_model(np.shape(train_x[1,:]))
print(priceModel.summary())
optim = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.99)
priceModel.compile(optimizer=optim, loss='mean_squared_error', metrics=['accuracy'])
priceModel.fit(x=train_x, y=train_y, batch_size=64, epochs=100, verbose=1, shuffle=True, validation_split=0.0)
pred = priceModel.evaluate(x=xvalid_x, y=xvalid_y, batch_size=64, verbose=1)
print('Loss: ' + str(pred[0]))
print('Accuracy: ' + str(pred[1]))
xvalid_pred = priceModel.predict(xvalid_x)

fig = plt.figure(figsize=(4,4))
plt.scatter((xvalid_y*sigma+mu)/1e3, (xvalid_pred*sigma+mu)/1e3)
plt.plot([0,1e3], [0,1e3], 'k--')
plt.xlabel('Sale price (per K$)')
plt.ylabel('Predicted sale price (per K$)');
test = pd.read_csv('test.csv')

test_ids = test.Id
test.drop('Id', axis=1, inplace=True)
print('test set size: ', test.shape)

not_nulls = {}
for colname in test.columns :
    not_nulls[colname] = test[colname].notnull().sum()/test.shape[0]
    
print(not_nulls)
cols = ['Alley','FireplaceQu','PoolQC','Fence','MiscFeature']
test = test.drop(cols, axis=1)
print('test set size after removing nan columns: ', test.shape)
test['LotFrontage'] = test['LotFrontage'].fillna(test['LotFrontage'].mean())
test['MasVnrArea'] = test['MasVnrArea'].fillna(test['MasVnrArea'].mean())
test['GarageYrBlt'] = test['GarageYrBlt'].fillna(test['GarageYrBlt'].mean())

test['MasVnrType'] = test['MasVnrType'].fillna('None');
test['BsmtQual'] = test['BsmtQual'].fillna('TA');
test['BsmtCond'] = test['BsmtCond'].fillna('TA');
test['BsmtExposure'] = test['BsmtExposure'].fillna('Mn')
test['BsmtFinType1'] = test['BsmtFinType1'].fillna('ALQ')
test['BsmtFinType2'] = test['BsmtFinType2'].fillna('Unf')
test['Electrical'] = test['Electrical'].fillna('SBrkr')
test['GarageType'] = test['GarageType'].fillna('Attchd')
test['GarageFinish'] = test['GarageFinish'].fillna('RFn')
test['GarageQual']= test['GarageQual'].fillna('TA')
test['GarageCond'] = test['GarageCond'].fillna('TA')
test['BsmtFinSF1'] = test['BsmtFinSF1'].fillna(test['BsmtFinSF1'].mean())
test['BsmtFinSF2'] = test['BsmtFinSF2'].fillna(test['BsmtFinSF2'].mean())
test['BsmtUnfSF'] = test['BsmtUnfSF'].fillna(test['BsmtUnfSF'].mean())
test['TotalBsmtSF'] = test['TotalBsmtSF'].fillna(test['TotalBsmtSF'].mean())
test['BsmtFullBath'] = test['BsmtFullBath'].fillna(test['BsmtFullBath'].mean())
test['BsmtHalfBath'] = test['BsmtHalfBath'].fillna(test['BsmtHalfBath'].mean())
test['GarageCars'] = test['GarageCars'].fillna(test['GarageCars'].mean())
test['GarageArea'] = test['GarageArea'].fillna(test['GarageArea'].mean())


not_nulls = {}
for colname in test.columns :
    not_nulls[colname] = test[colname].notnull().sum()/test.shape[0]
    
print(not_nulls)
le = preprocessing.LabelEncoder()
for col in test.columns :
    if test[col].dtype == object :
        test[col] = test[col].astype(str) 
        le.fit(test[col])
        test[col] = le.transform(test[col])
        
for col in test.columns :
    test[col] = (test[col] - test[col].mean())/test[col].std()
dt = {}
for col in test.columns : 
    dt[col] = test[col].dtype

print(dt)
test.head()
test_pred = priceModel.predict(test)

test_price_pred = test_pred*sigma + mu
test_price_pred = test_price_pred.reshape(test_pred.shape[0], 1)

submission = pd.read_csv('sample_submission.csv')
submission.SalePrice = test_price_pred
submission
submission.to_csv ('submission.csv', index = None, header = True)