# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.layers import Dense
from keras.models import Sequential
from keras.regularizers import l1
from keras.layers.normalization import BatchNormalization
from keras import backend as K

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.stats import skew
from scipy.stats.stats import pearsonr

import matplotlib.pyplot as plt
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],
                      test.loc[:,'MSSubClass':'SaleCondition']))
all_data.info()
all_data.head()
# Categorical Feature
cat_feats = all_data.dtypes[all_data.dtypes == "object"].index
cat_feats
all_data[cat_feats].head()
# Numeric Feature
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
numeric_feats
all_data[numeric_feats].head()
# Ordinal Feature
ordinal_features = ['YrSold']
#log transform skewed numeric features:

skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index
print(skewed_feats)

all_data[skewed_feats] = np.log1p(all_data[skewed_feats])
#log transform the target:
train["SalePrice"] = np.log1p(train["SalePrice"])
# One Hot Encoder
all_data = pd.get_dummies(all_data)
#filling NA's with the mean of the column:
all_data = all_data.fillna(all_data.mean())
#creating matrices for sklearn:
X_train = all_data[:train.shape[0]]
X_test = all_data[train.shape[0]:]
y = train.SalePrice
X_train = StandardScaler().fit_transform(X_train)
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y, random_state = 3)
X_tr.shape
X_tr
X_val
#Model1
model = Sequential()
#model.add(Dense(256, activation="relu", input_dim = X_train.shape[1]))
model.add(Dense(1, input_dim = X_train.shape[1], W_regularizer=l1(0.001)))

model.compile(loss = "mse", optimizer = "adam")
#Model2
model = Sequential()
BatchNormalization()
model.add(Dense(1028,input_dim=288,activation='relu'))
BatchNormalization()
model.add(Dense(1028,input_dim=288,activation='relu'))
BatchNormalization()
#Dropout(0.2)
model.add(Dense(100,input_dim=288,activation='relu'))
BatchNormalization()
#Dropout(0.2)
model.add(Dense(50))
BatchNormalization()
model.add(Dense(1))
model.compile(optimizer='adam',loss='mse',metrics=['accuracy'])
model.summary()
hist = model.fit(X_tr, y_tr, validation_data = (X_val, y_val))
scores = np.sqrt(model.evaluate(X_val,y_val,verbose=0))
scores
pd.Series(model.predict(X_val)[:,0]).hist()
