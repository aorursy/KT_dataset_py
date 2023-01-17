import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import model_selection

from sklearn.linear_model import ElasticNet



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df_train = pd.read_csv('/kaggle/input/liverpool-ion-switching/train.csv')

df_test = pd.read_csv('/kaggle/input/liverpool-ion-switching/test.csv')

df_submission = pd.read_csv('/kaggle/input/liverpool-ion-switching/sample_submission.csv')
X_train = df_train[['signal']].values

y_train = df_train['open_channels'].values

X_test = df_test[['signal']].values



df_train.head()
plt.figure(figsize=(12, 6))

sns.countplot(x='open_channels', data=df_train)
plt.figure(figsize=(12, 6))

sns.kdeplot(df_train['signal'])
plt.figure(figsize=(12, 6))

sns.distplot(df_train['signal'])
plt.figure(figsize=(12, 12))

sns.jointplot(x='time', y='signal', data=df_train, kind='hex', gridsize=20)
%%time



model = ElasticNet(alpha=0.9, l1_ratio=0.1)

model.fit(X_train, y_train)
%%time 

preds = model.predict(X_test)



# preds_reshaped = np.reshape(preds, (int(len(preds)/10), -1))

# preds_reduced_mean = np.mean(preds_reshaped, axis=1)

#preds_around = np.around(preds, decimals=1).astype(int)

preds_around = np.rint(preds).astype(int)
df_submission['open_channels'] = preds_around

df_submission[df_submission['open_channels']<0]['open_channels'] = 0

df_submission.head()
sns.countplot(x='open_channels', data=df_train)
sns.countplot(x='open_channels', data=df_submission)
df_submission.head(10)
df_submission['time'] = [ "{:.4f}".format(df_submission['time'].values[x]) for x in range(2000000)]

df_submission.head(10)
df_submission.to_csv("submission.csv", index=False)