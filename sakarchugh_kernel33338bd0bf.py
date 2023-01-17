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
import numpy as np

import pandas as pd

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

from keras.models import Sequential

from keras.layers import Dense

from keras.utils import np_utils

from keras.layers.core import Dropout

from keras.layers.core import Flatten

from keras.layers.normalization import BatchNormalization

from sklearn.utils import class_weight

from keras.callbacks import EarlyStopping
df2 = pd.read_csv("/kaggle/input/bitsf312-lab1/train.csv", sep=',')

#size = {"Small" : 1, "Medium" : 2, "Big": 3, "?" : np.nan}

#df2.Size = [size[item] for item in df2.Size]

for col in df2:

    for x in range(len(df2[col])):

        if df2[col][x]=='?':

            df2[col][x] = np.nan

df2["Size"].value_counts()

df2.head()

df2['Size'].fillna("Medium", inplace=True)

df2.head()
df2.dtypes

df2 = pd.concat([df2,pd.get_dummies(df2['Size'], prefix='size')],axis=1)

df2.drop(['Size'],axis=1, inplace=True)

df2.head()
for col in df2:

    df2[col] = pd.to_numeric(df2[col])

df2.dtypes
for col in df2:

    df2[col].fillna((df2[col].mean()), inplace=True)

df2["Number of Special Characters"] = df2["Number of Special Characters"].astype(int)

df2["Number of Quantities"] = df2["Number of Quantities"].astype(int)

df2["Number of Insignificant Quantities"] = df2["Number of Insignificant Quantities"].astype(int)

df2["Total Number of Words"] = df2["Total Number of Words"].astype(int)

df2["Number of Special Characters"] = df2["Number of Special Characters"].astype(int)

df2.dtypes
plt.figure(figsize=(20,20))

corr = df2.corr()

corr.style.background_gradient(cmap='RdYlGn')
dfY = df2["Class"]

#cols = [0, 2,4,5,11]

#cols = [0,4,11]

cols=[0,11]

dftest = df2.drop(df2.columns[cols], axis=1)

plt.figure(figsize=(20,20))

corr = dftest.corr()

corr.style.background_gradient(cmap='RdYlGn')

dftest
dftest = dftest.values

y = dfY.values

y.shape
# from sklearn.preprocessing import StandardScaler

# scaler = StandardScaler()

# X = dftest

# X_scale = scaler.fit_transform(X)

# X.shape
from sklearn import preprocessing

X = dftest

min_max_scaler = preprocessing.MinMaxScaler()

X_scale = min_max_scaler.fit_transform(X)

X_scale.shape
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X_scale, y, test_size=0.2)
dummy_ytrain = np_utils.to_categorical(Y_train)

dummy_ytest = np_utils.to_categorical(Y_test)
from sklearn.utils import class_weight

class_weights = class_weight.compute_class_weight('balanced',np.unique(Y_train), Y_train)

class_weights
model = Sequential()

model.add(Dense(32, input_dim=13, activation='relu'))

model.add(Dropout(rate=0.2))

model.add(Dense(32, activation='tanh'))

model.add(Dropout(rate=0.2))

model.add(Dense(32, activation='tanh'))

model.add(Dropout(rate=0.2))

model.add(Dense(32, activation='tanh'))

model.add(Dropout(rate=0.2))

model.add(Dense(6, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
model.fit(X_train, dummy_ytrain, epochs=250, validation_split=0.2, batch_size = 40, class_weight= class_weights)
model.evaluate(X_test, dummy_ytest)
df1 = pd.read_csv("/kaggle/input/bitsf312-lab1/test.csv", sep=',')

df_submit = pd.DataFrame()

df_submit["ID"]=df1["ID"]
df1 = pd.concat([df1,pd.get_dummies(df1['Size'], prefix='size')],axis=1)

df1.drop(['Size'],axis=1, inplace=True)

df1.head()
cols=[0]

#cols = [0, 2, 4, 5]

df1 = df1.drop(df1.columns[cols], axis=1)

df1.head()
dtest = df1.values

Xtest = dtest

min_max_scaler = preprocessing.MinMaxScaler()

Xtest_scale = min_max_scaler.fit_transform(Xtest)

y_submit = model.predict_classes(Xtest_scale, batch_size = 40)

y_submit


df_submit["Class"] = y_submit

df_submit.to_csv("Test5.csv",index=False)
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64

def create_download_link(df_submit, title = "Download CSV file", filename = "data.csv"):

    csv = df_submit.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)

create_download_link(df_submit)