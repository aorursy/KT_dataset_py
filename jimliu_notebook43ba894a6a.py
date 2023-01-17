from keras.models import Sequential

from keras.utils import np_utils

from keras.layers.core import Dense, Activation, Dropout



from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split



import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib



import matplotlib.pyplot as plt

from scipy.stats import skew

from scipy.stats.stats import pearsonr
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],

                      test.loc[:,'MSSubClass':'SaleCondition']))
#prices = pd.DataFrame({"price":train["SalePrice"], "log(price + 1)":np.log1p(train["SalePrice"])})

#log transform the target:

#train["SalePrice"] = np.log1p(train["SalePrice"])



#log transform skewed numeric features:

numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index



skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness

skewed_feats = skewed_feats[skewed_feats > 0.75]

skewed_feats = skewed_feats.index



all_data[skewed_feats] = np.log1p(all_data[skewed_feats])
all_data = pd.get_dummies(all_data)
all_data = all_data.fillna(all_data.mean())
X_train = all_data[:train.shape[0]]

X_test = all_data[train.shape[0]:]

y = train.SalePrice
X_train = StandardScaler().fit_transform(X_train)

X_test = StandardScaler().fit_transform(X_test)

X_tr, X_val, y_tr, y_val = train_test_split(X_train, y, random_state = 5)
# Here's a Deep Dumb MLP (DDMLP)

model = Sequential()

model.add(Dense(2000, input_dim=X_train.shape[1]))

model.add(Activation('relu'))

model.add(Dropout(0.15))



model.add(Dense(1700))

model.add(Activation('relu'))

model.add(Dropout(0.15))



model.add(Dense(1500))

model.add(Activation('relu'))

model.add(Dropout(0.15))



model.add(Dense(1200))

model.add(Activation('relu'))

model.add(Dropout(0.15))



model.add(Dense(1000))

model.add(Activation('relu'))

model.add(Dropout(0.15))



model.add(Dense(700))

model.add(Activation('relu'))

model.add(Dropout(0.15))



model.add(Dense(500))

model.add(Activation('relu'))

model.add(Dropout(0.15))



model.add(Dense(128))

model.add(Activation('relu'))

model.add(Dropout(0.15))



model.add(Dense(1))
model.compile(loss = "mse", optimizer = "adam",metrics=['accuracy'])
model.summary()
hist = model.fit(X_tr, y_tr, validation_data = (X_val, y_val),nb_epoch=100, batch_size=100)
pd.Series(model.predict(X_test)[:,0]).hist()
pred = model.predict(X_test)[:,0]

all_res = pd.DataFrame({"Id":test['Id'],"SalePrice":pred})
all_res.to_csv("res_house.csv",header=True)