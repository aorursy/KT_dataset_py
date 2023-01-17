# by Grossmend, 2018
import pandas as pd

import numpy as np

import time

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

from sklearn import ensemble, tree, linear_model

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.metrics import r2_score, mean_squared_error

from sklearn.utils import shuffle

from sklearn.model_selection import StratifiedKFold, KFold



from keras import models

from keras import layers

from keras import optimizers
# train data

train_data = pd.read_csv('/kaggle/input/train.csv')



# test data

test_data = pd.read_csv('/kaggle/input/test.csv')



# concat train and test data in one DataFrame with keys

all_data = pd.concat([train_data, test_data], keys=['train', 'test'], axis=0, sort=False)



# set option display number columns

pd.set_option('display.max_columns', train_data.shape[1])



# show first 10 row data

all_data.head(7)
# view size datasets

print('Size train_data:', all_data.loc['train'].shape)

print('Size test data:', all_data.loc['test'].shape)
# view missing data in train and test data by percentage

nan_values = pd.concat([(train_data.isnull().sum() /  train_data.isnull().count())*100,

                        (test_data.isnull().sum() / test_data.isnull().count())*100], axis=1, keys=['Train', 'Test'], sort=False)

print('true')

nan_values[nan_values.sum(axis=1) > 0].sort_values(by=['Train'], ascending=False)
# view info without 'SalePrices'

all_data[all_data.columns.difference(['SalePrice'])].info()
# show counts each types in data

all_data.get_dtype_counts()
# check duplecated field "id"

any(all_data['Id'].duplicated())
# show correlation of data

corrmat = train_data.drop('Id', axis=1).corr()

plt.subplots(figsize=(13,13))

mask = np.zeros_like(corrmat, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

sns.heatmap(corrmat, mask=mask, vmax=0.9, cmap="YlGnBu", square=True, cbar_kws={"shrink": .5}, linewidths=0.6);

plt.title('correlation of data');



# more settings: 'https://seaborn.pydata.org/generated/seaborn.heatmap.html'
# show most correlated features from field 'SalePrice'

corr=train_data.corr()["SalePrice"]

corr[np.argsort(corr, axis=0)[::-1]]
# show descriptive statistics summary field "SalePrice"

train_data['SalePrice'].describe()
# show distribution field 'SalePrice'

plt.subplots(figsize=(17,7));

sns.distplot(train_data['SalePrice'], color='black', bins=100);

plt.title('distribution "SalePrice"');
# show skewness and kurtosis (показатели ассиметрии и аксцесса)

print('Skewness:', train_data['SalePrice'].skew())

print('Kurtosis:', train_data['SalePrice'].kurt())
# log transformation of train labels (SalePrice)

train_labels = np.log(train_data['SalePrice'])

plt.subplots(figsize=(17,7));

sns.distplot(train_labels, color='green', bins=100);

plt.title('Log transformation "SalePrice"');
# delete field "SalePrice" from data

if 'SalePrice' in all_data:

    all_data.drop('SalePrice', inplace=True, axis=1)

else:

    print('no field "SalePrice"')

all_data.shape
# delete do not need fields (another analysis, see more: https://blog.grossmend.com/blog)



drop_list = [

    '3SsnPorch',

    'BsmtFinSF1',

    'BsmtFinSF2', 

    'BsmtFullBath',

    'BsmtHalfBath',

    'BsmtUnfSF',

    'EnclosedPorch',

    'Fence',

    'Functional',

    'GarageArea',

    'GarageCond',

    'GarageYrBlt',

    'Heating',

    'LowQualFinSF',

    'MasVnrArea',

    'MiscFeature',

    'MiscVal',

    'OpenPorchSF',

    'PoolArea',

    'PoolQC',

    'RoofMatl',

    'ScreenPorch',

    'Utilities',

    'WoodDeckSF',

]



all_data.drop(drop_list, axis=1, errors='ignore', inplace=True)

all_data.shape
# show data first 7 rows

all_data.head(7)
# show missing values each field

nan_values = pd.concat([(all_data.isnull().sum() /  all_data.isnull().count())*100], axis=1, keys=['all_data'], sort=False)

nan_values[nan_values.sum(axis=1) > 0].sort_values(by=['all_data'], ascending=False)
# missing nan values

if all_data.isnull().values.any():

    all_data['Alley'] = all_data['Alley'].fillna('no_access')

    all_data['FireplaceQu'] = all_data['FireplaceQu'].fillna('no_fp')

    all_data['LotFrontage'] = all_data['LotFrontage'].fillna(all_data['LotFrontage'].mean())

    all_data['MasVnrType'] = all_data['MasVnrType'].fillna(all_data['MasVnrType'].mode()[0])

    all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])

    all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])

    all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])

    if 'TotalBsmtSF' in all_data:

        all_data['TotalBsmtSF'] = all_data['TotalBsmtSF'].fillna(0)

    all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior1st'].mode()[0])

    all_data['GarageCars'] = all_data['GarageCars'].fillna(0.0)

    all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])

    all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])

    for col in ('GarageType', 'GarageFinish', 'GarageQual'): all_data[col] = all_data[col].fillna('no_garage') 

    for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'): all_data[col] = all_data[col].fillna('no_bsmt')

else:

    print('all values is not missing')
# add need columns and drop do not need columns after missing

if set(['TotalBsmtSF', '1stFlrSF', '2ndFlrSF']).issubset(all_data.columns):

    all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']

    all_data.drop(['TotalBsmtSF', '1stFlrSF', '2ndFlrSF'], axis=1, inplace=True)

else:

    print("no found fields ('TotalBsmtSF', '1stFlrSF', '2ndFlrSF')")
# view missing data in train and test data by percentage

nan_values = pd.concat([(all_data.isnull().sum() /  all_data.isnull().count())*100], axis=1, keys=['all_data'], sort=False)

nan_values = nan_values[nan_values.sum(axis=1) > 0]

if nan_values.shape[0] > 0:

    nan_values.sort_values(by=['all_data'], ascending=False)

else:

    print('no missing values')
# convert fields to categorical

all_data['MSSubClass'] = all_data['MSSubClass'].astype(str)

all_data['KitchenAbvGr'] = all_data['KitchenAbvGr'].astype(str)

all_data['OverallCond'] = all_data['OverallCond'].astype(str)

all_data['YrSold'] = all_data['YrSold'].astype(str)

all_data['MoSold'] = all_data['MoSold'].astype(str)
# getting number fields from object

for col in all_data.dtypes[all_data.dtypes == 'object'].index:

    all_data[col] = all_data[col].astype('category')

    all_data[col] = all_data[col].cat.codes
# delete "Id" field

all_data = all_data.drop('Id', axis=1)
# normalize all values

all_data=(all_data-all_data.mean())/all_data.std()

print(all_data.shape)

all_data.head(7)
# split dataset train data and test data for ML

X_model = all_data.loc['train'].select_dtypes(include=[np.number])

y_model = np.log(train_data['SalePrice'])



Y_finish = all_data.loc['test'].select_dtypes(include=[np.number])
# split data train and test

X_train, X_test, y_train, y_test = train_test_split(X_model, y_model, test_size=0.15)
def build_model(insh):

    

    model = models.Sequential()

    model.add(layers.Dense(32, activation='relu', input_shape=(insh,)))

    model.add(layers.Dropout(0.3))

    model.add(layers.Dense(16, activation='sigmoid'))

    model.add(layers.Dense(1))

    

    opt = optimizers.Adam(lr=0.002, beta_1=0.9, beta_2=0.99, epsilon=None, decay=0.0, amsgrad=False)

    model.compile(optimizer=opt, loss='mse', metrics=['mae'])

    

    return model
# fit NN model

input_shape = X_train.shape[1]

DL_model = build_model(input_shape)



# get initial weights model

initial_weights = DL_model.get_weights()



history = DL_model.fit(X_train.values,

                       y_train.values,

                       epochs=300,

                       batch_size=128,

                       verbose=0,

                       validation_data=(X_test, y_test))
# check scores model

DL_model.evaluate(X_test.values, y_test.values)
n = 100



plt.subplots(figsize=(17,7));

plt.plot(history.history['mean_absolute_error'][n:])

plt.plot(history.history['val_mean_absolute_error'][n:])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()



# summarize history for loss

plt.subplots(figsize=(17,7));

plt.plot(history.history['loss'][n:])

plt.plot(history.history['val_loss'][n:])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
# # kross k validation



# kfold = KFold(n_splits=5, shuffle=True)



# scores = []



# nn_model = None

# nn_model = build_model(X_model.shape[1])



# for _ in range(5):

#     shf = shuffle(X_model, y_model)

#     for train, test in kfold.split(shf[0].reset_index(drop=True), shf[1].reset_index(drop=True)):

#         nn_model.set_weights(initial_weights)

#         nn_model.fit(X_model.iloc[train].values,

#                      y_model.iloc[train].values,

#                      epochs=300,

#                      batch_size=128,

#                      verbose=0)

#         acc = nn_model.evaluate(X_model.iloc[test].values, y_model.iloc[test].values, verbose=0)[1]

#         scores.append(acc)

#         print('accuracy step ' + str(len(scores)) + ': ', acc)

# print('mean:', np.mean(scores))    
# create finish model and train on complete train data

input_shape = X_model.shape[1]

DL_model_finish = build_model(input_shape)

DL_model_finish.set_weights(initial_weights)



history = DL_model_finish.fit(X_model.values,

                              y_model.values,

                              epochs=300,

                              batch_size=128,

                              verbose=0)
# get finish predict values

prediction = np.exp(DL_model_finish.predict(Y_finish.values))
# save to CSV

pd.DataFrame({'Id': test_data['Id'], 'SalePrice': prediction.flatten()}).to_csv('submission_nn_keras.csv', index=False)