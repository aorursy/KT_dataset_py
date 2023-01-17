from __future__ import absolute_import

from __future__ import division

from __future__ import print_function



import os

import keras



import itertools



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from pylab import rcParams

import matplotlib



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split



import seaborn as sns



from scipy.stats import skew



from keras.layers import Dense

from keras.models import Sequential

from keras.regularizers import l1



def load_data(train_path, test_path):

    """

     Data loading

    :param train_path: path for the train set file

    :param test_path: path for the test set file

    :return: a 'pandas' array for each set

    """



    train_data = pd.read_csv(train_path)

    test_data = pd.read_csv(test_path)



    print("number of training examples = " + str(train_data.shape[0]))

    print("number of test examples = " + str(test_data.shape[0]))

    print("train shape: " + str(train_data.shape))

    print("test shape: " + str(test_data.shape))



    return train_data, test_data
train_data = '../input/house-prices-advanced-regression-techniques/train.csv'

test_data = '../input/house-prices-advanced-regression-techniques/test.csv'



train, test = load_data(train_data, test_data)



#train = train.select_dtypes(exclude=['object'])



#train.drop('Id', axis = 1, inplace = True)

#train.fillna(0, inplace = True)

#train.describe()

prices = pd.DataFrame({"price":train["SalePrice"], "log(price + 1)":np.log1p(train["SalePrice"])})

#prices.hist()

plt.figure(figsize=(8,4))

sns.distplot(np.log1p(train["SalePrice"]))

#plt.figure(figsize=(8,4))

#sns.distplot(train['SalePrice'])
plt.figure(figsize=(8,4))

sns.distplot(train["SalePrice"])
all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],

                      test.loc[:,'MSSubClass':'SaleCondition']))





#log transform the target:

train["SalePrice"] = np.log1p(train["SalePrice"])



#log transform skewed numeric features:

numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index



skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness

skewed_feats = skewed_feats[skewed_feats > 0.75]

skewed_feats = skewed_feats.index



all_data[skewed_feats] = np.log1p(all_data[skewed_feats])



all_data = pd.get_dummies(all_data)

all_data = all_data.fillna(all_data.mean())

#creating matrices for sklearn:

X_train = all_data[:train.shape[0]]

X_test = all_data[train.shape[0]:]

y = train.SalePrice

X_test.shape
X_train = StandardScaler().fit_transform(X_train)

X_train_ds, X_val_ds, y_train_ds, y_val_ds = train_test_split(X_train, y,random_state = 42)

y_train_ds.shape

X_val_ds.shape
model = Sequential()

model.add(Dense(200, input_dim=X_train_ds.shape[1], kernel_initializer='normal', activation='relu'))

model.add(Dense(100, kernel_initializer='normal', activation='relu'))

model.add(Dense(50, kernel_initializer='normal', activation='relu'))

model.add(Dense(25, kernel_initializer='normal', activation='relu'))

model.add(Dense(1, kernel_initializer='normal'))



model.compile(loss='mean_squared_error', optimizer='adam')



model.summary()

history = model.fit(X_train_ds, y_train_ds, validation_data = (X_val_ds, y_val_ds),epochs = 100)

# summarize history for loss

plt.figure(figsize=(8,4))

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
y_predict = model.predict(X_test)[:,0]

y_predict.shape

submission = pd.DataFrame({"Id": test["Id"],"SalePrice": y_predict})

submission.loc[submission['SalePrice'] <= 0, 'SalePrice'] = 0

fileName = "submission.csv"

submission.to_csv(fileName, index=False)