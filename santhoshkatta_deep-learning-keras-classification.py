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
import pandas as pd

import numpy as np 

from numpy.random import seed

seed(1)

import matplotlib.pyplot as plt

from sklearn.utils import shuffle

import seaborn as sns

import matplotlib.gridspec as gridspec

from sklearn.metrics import recall_score, accuracy_score, precision_score, confusion_matrix, f1_score,cohen_kappa_score



import keras

from keras.models import Sequential

from keras.layers.core import Dense, Dropout, Activation

from keras.utils import np_utils
df = pd.read_csv("/kaggle/input/creditcardfraud/creditcard.csv")
df.describe()
data.isnull().sum()
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12,4))



bins = 50



ax1.hist(df.Time[df.Class == 1], bins = bins)

ax1.set_title('Fraud')



ax2.hist(df.Time[df.Class == 0], bins = bins)

ax2.set_title('Normal')



plt.xlabel('Time (in Seconds)')

plt.ylabel('Number of Transactions')

plt.show()
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12,4))



bins = 30



ax1.hist(df.Amount[df.Class == 1], bins = bins)

ax1.set_title('Fraud')



ax2.hist(df.Amount[df.Class == 0], bins = bins)

ax2.set_title('Normal')



plt.xlabel('Amount ($)')

plt.ylabel('Number of Transactions')

plt.yscale('log')

plt.show()
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12,6))



ax1.scatter(df.Time[df.Class == 1], df.Amount[df.Class == 1])

ax1.set_title('Fraud')



ax2.scatter(df.Time[df.Class == 0], df.Amount[df.Class == 0])

ax2.set_title('Normal')



plt.xlabel('Time (in Seconds)')

plt.ylabel('Amount')

plt.show()
df['Amount_max_fraud'] = 1

df.loc[df.Amount <= 2125.87, 'Amount_max_fraud'] = 0
v_features = df.ix[:,1:29].columns
plt.figure(figsize=(12,28*4))

gs = gridspec.GridSpec(28, 1)

for i, cn in enumerate(df[v_features]):

    ax = plt.subplot(gs[i])

    sns.distplot(df[cn][df.Class == 1], bins=50)

    sns.distplot(df[cn][df.Class == 0], bins=50)

    ax.set_xlabel('')

    ax.set_title('histogram of feature: ' + str(cn))

plt.show()
df = df.drop(['V28','V27','V26','V25','V24','V23','V22','V20','V15','V13','V8'], axis =1)
df['V1_'] = df.V1.map(lambda x: 1 if x < -3 else 0)

df['V2_'] = df.V2.map(lambda x: 1 if x > 2.5 else 0)

df['V3_'] = df.V3.map(lambda x: 1 if x < -4 else 0)

df['V4_'] = df.V4.map(lambda x: 1 if x > 2.5 else 0)

df['V5_'] = df.V5.map(lambda x: 1 if x < -4.5 else 0)

df['V6_'] = df.V6.map(lambda x: 1 if x < -2.5 else 0)

df['V7_'] = df.V7.map(lambda x: 1 if x < -3 else 0)

df['V9_'] = df.V9.map(lambda x: 1 if x < -2 else 0)

df['V10_'] = df.V10.map(lambda x: 1 if x < -2.5 else 0)

df['V11_'] = df.V11.map(lambda x: 1 if x > 2 else 0)

df['V12_'] = df.V12.map(lambda x: 1 if x < -2 else 0)

df['V14_'] = df.V14.map(lambda x: 1 if x < -2.5 else 0)

df['V16_'] = df.V16.map(lambda x: 1 if x < -2 else 0)

df['V17_'] = df.V17.map(lambda x: 1 if x < -2 else 0)

df['V18_'] = df.V18.map(lambda x: 1 if x < -2 else 0)

df['V19_'] = df.V19.map(lambda x: 1 if x > 1.5 else 0)

df['V21_'] = df.V21.map(lambda x: 1 if x > 0.6 else 0)
df = df.rename(columns={'Class': 'Fraud'})
pd.set_option("display.max_columns",101)

df.head()
Fraud = df[df.Fraud == 1]

Normal = df[df.Fraud == 0]
X_train = Fraud.sample(frac=0.8)

count_Frauds = len(X_train)

X_train = pd.concat([X_train, Normal.sample(frac = 0.8)], axis = 0)

X_test = df.loc[~df.index.isin(X_train.index)]
X_train = shuffle(X_train)

X_test = shuffle(X_test)
y_train = pd.DataFrame()

y_test = pd.DataFrame()

y_train = pd.concat([y_train, X_train.Fraud], axis=1)

y_test = pd.concat([y_test, X_test.Fraud], axis=1)
X_train = X_train.drop(['Fraud'], axis = 1)

X_test = X_test.drop(['Fraud'], axis = 1)
X_train.shape
print(len(X_train))

print(len(y_train))

print(len(X_test))

print(len(y_test))
features = X_train.columns.values

for feature in features:

    mean, std = df[feature].mean(), df[feature].std()

    X_train.loc[:, feature] = (X_train[feature] - mean) / std

    X_test.loc[:, feature] = (X_test[feature] - mean) / std
from tensorflow import keras

from tensorflow.keras import layers

model = keras.Sequential()





model.add(layers.Dense(40, input_shape=(37,)))

model.add(layers.Activation('relu'))  # An "layers.Activation" is just a non-linear function applied to the output

# of the layer above. Here, with a "rectified linear unit",

# we clamp all values below 0 to 0.



model.add(layers.Dropout(0.2))  # layers.Dropout helps protect the model from memorizing or "overfitting" the training data

model.add(layers.Dense(40))

model.add(layers.Activation('relu'))

model.add(layers.Dropout(0.2))

model.add(layers.Dense(1))

model.add(layers.Activation('sigmoid')) 
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

## Now its time to fit the model



model.fit(X_train, y_train,

          batch_size=128, epochs=10,

           verbose=2,

           validation_data=(X_test, y_test))
yhat_probs = model.predict(X_test, verbose=0)

yhat_classes = model.predict_classes(X_test, verbose=0)

yhat_probs = yhat_probs[:, 0]

yhat_classes = yhat_classes[:, 0]
accuracy = accuracy_score(y_test["Fraud"], yhat_classes)

print('Accuracy: %f' % accuracy)

precision = precision_score(y_test["Fraud"], yhat_classes)

print('Precision: %f' % precision)

recall = recall_score(y_test["Fraud"], yhat_classes)

print('Recall: %f' % recall)

f1 = f1_score(y_test["Fraud"], yhat_classes)

print('F1 score: %f' % f1)
kappa = cohen_kappa_score(y_test, yhat_classes)

print('Cohens kappa: %f' % kappa)

auc = roc_auc_score(y_test, yhat_probs)

print('ROC AUC: %f' % auc)

matrix = confusion_matrix(y_test, yhat_classes)

print(matrix)