# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from sklearn.preprocessing import LabelEncoder

import gc

from sklearn.linear_model import LinearRegression

import random

import datetime as dt

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Dropout, BatchNormalization

from keras.layers.advanced_activations import PReLU

from keras.layers.noise import GaussianDropout

from keras.optimizers import Adam

from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import Imputer

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df16 = pd.read_csv("../input/properties_2016.csv")

df17 = pd.read_csv("../input/properties_2017.csv")

train16 = pd.read_csv("../input/train_2016_v2.csv")

train17 = pd.read_csv("../input/train_2017.csv")

samplesub = pd.read_csv("../input/sample_submission.csv")
df16.head()
data16 = pd.merge(df16,train16)

data17 = pd.merge(df17,train17)

data = pd.concat([data16,data17],keys=('parcelid','transactiondate'))

data.head()
num_cols = [col for col in data.columns if (data[col].dtype in ['float64','int64'] and col not in ['parcelid','transactiondate']) or data[col].dtype.name=='category']

temp_df = data[num_cols]

corrmat = temp_df.corr(method='spearman')

f, ax = plt.subplots(figsize=(12, 12))



sns.heatmap(corrmat, vmax=1., square=True,cmap='PiYG')

plt.title("Variables correlation map", fontsize=15)

plt.show()
for c in data.columns:

    data[c]=data[c].fillna(-1)

    if data[c].dtype == 'object':

        lbl = LabelEncoder()

        lbl.fit(list(data[c].values))

        data[c] = lbl.transform(list(data[c].values))



data["transactiondate"] = pd.to_datetime(data["transactiondate"])

data["transactiondate_year"] = data["transactiondate"].dt.year

data["transactiondate_month"] = data["transactiondate"].dt.month

data['transactiondate_quarter'] = data['transactiondate'].dt.quarter

data["transactiondate"] = data["transactiondate"].dt.day



data = data.fillna(-1.0)
x_train = data.drop(['parcelid', 'logerror', 'transactiondate', 'propertyzoningdesc', 'propertycountylandusecode','fireplacecnt', 'fireplaceflag'], axis=1)

y_train = data["logerror"]



y_mean = np.mean(y_train)

print(x_train.shape, y_train.shape)

train_columns = x_train.columns



for c in x_train.dtypes[x_train.dtypes == object].index.values:

    x_train[c] = (x_train[c] == True)



samplesub['parcelid'] = samplesub['ParcelId']

df_test = samplesub.merge(df16, on='parcelid', how='left')

df_test["transactiondate"] = pd.to_datetime('2016-11-15')  # placeholder value for preliminary version

df_test["transactiondate_year"] = df_test["transactiondate"].dt.year

df_test["transactiondate_month"] = df_test["transactiondate"].dt.month

df_test['transactiondate_quarter'] = df_test['transactiondate'].dt.quarter

df_test["transactiondate"] = df_test["transactiondate"].dt.day     

x_test = df_test[train_columns]



for c in x_test.dtypes[x_test.dtypes == object].index.values:

    x_test[c] = (x_test[c] == True)
imputer= Imputer()

imputer.fit(x_train.iloc[:, :])

x_train = imputer.transform(x_train.iloc[:, :])

imputer.fit(x_test.iloc[:, :])

x_test = imputer.transform(x_test.iloc[:, :])



sc = StandardScaler()

x_train = sc.fit_transform(x_train)

x_test = sc.transform(x_test)



len_x=int(x_train.shape[1])

print(len_x)
# model taken from Andy Harless

nn = Sequential()

nn.add(Dense(units = 400 , kernel_initializer = 'normal', input_dim = len_x))

nn.add(PReLU())

nn.add(Dropout(.4))

nn.add(Dense(units = 160 , kernel_initializer = 'normal'))

nn.add(PReLU())

nn.add(BatchNormalization())

nn.add(Dropout(.6))

nn.add(Dense(units = 64 , kernel_initializer = 'normal'))

nn.add(PReLU())

nn.add(BatchNormalization())

nn.add(Dropout(.5))

nn.add(Dense(units = 26, kernel_initializer = 'normal'))

nn.add(PReLU())

nn.add(BatchNormalization())

nn.add(Dropout(.6))

nn.add(Dense(1, kernel_initializer='normal'))

nn.compile(loss='mae', optimizer=Adam(lr=4e-3, decay=1e-4))
nn.fit(np.array(x_train), np.array(y_train), batch_size = 1000, epochs = 20, verbose=2)

y_pred_ann = nn.predict(x_test)

nn_pred = y_pred_ann.flatten()



pd.DataFrame(nn_pred).head()
y_pred=[]



for i,predict in enumerate(nn_pred):

    y_pred.append(str(round(predict,4)))

y_pred=np.array(y_pred)



output = pd.DataFrame({'ParcelId': df16['parcelid'].astype(np.int32),

        '201610': y_pred, '201611': y_pred, '201612': y_pred,

        '201710': y_pred, '201711': y_pred, '201712': y_pred})



# set col 'ParceID' to first col

cols = output.columns.tolist()

cols = cols[-1:] + cols[:-1]

output = output[cols]

output.head()
output.to_csv("zillow_sub.csv",index=False)