import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import torch

import torch.nn as nn

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import confusion_matrix, f1_score

from sklearn import svm



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
def plot_metrics(metrics):

    #set up axes

    fig, ax = plt.subplots(1,3,figsize=(18,6))

    for i,key in enumerate(['KNN','LR','SVM']):

        F1,CM = metrics[key]['F1'],metrics[key]['CM']

        df = pd.DataFrame(CM, index = ['Not Pulsar','Is Pulsar'],columns = ['False Pulsar','True Pulsar'])

        ax[i].set_title(f'%s - F1:%.3f'%(key,F1))

        sns.heatmap(df, annot=True, fmt='g',ax=ax[i],cbar=False)

    fig.tight_layout()
df = pd.read_csv('/kaggle/input/predicting-a-pulsar-star/pulsar_stars.csv')
df.shape

df.columns
#df.head()

#df.describe()
#split by class

df0 = df[df['target_class'] == 0]

df1 = df[df['target_class'] == 1]

#split seperatly to ensure we get the minority class into each set

train0, val0, test0 = np.split(df0.sample(frac=1), [int(.6*len(df0)), int(.8*len(df0))])

train1, val1, test1 = np.split(df1.sample(frac=1), [int(.6*len(df1)), int(.8*len(df1))])

#concat

train = pd.concat([train0,train1])

val = pd.concat([val0,val1])

test = pd.concat([test0,test1])

print(train.shape,val.shape,test.shape)
fig, ax = plt.subplots(2, 4,figsize=(20,8))

for i in range(2):

    for j in range(4):

        col_name = df.columns[i*4+j]

        bins = np.linspace(-10, 100, 30)

        ax[i][j].hist([train0[col_name], train1[col_name]], bins, label=['0', '1'])

        ax[i][j].legend(loc='upper right')

        ax[i][j].title.set_text(col_name)

plt.show()
cols = train.columns.difference(['target_class'])

corrmat = train[cols].corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=.8, square=True);
#scatterplot

sns.set()

sns.pairplot(train.sample(frac=0.2), height = 5,vars=cols,hue='target_class')

plt.show();
#here we can pick what features to use for the models

model_cols = cols
#seperate train into (X,y)

    

X_train = train[model_cols].values

y_train = train['target_class'].values



#get mean and std

mu = np.mean(X_train,axis=0)

s = np.std(X_train,axis=0)



#center X_train

X_train -= mu

X_train /= s



#apply centering used on train to val

X_val = (val[model_cols].values - mu)/s

y_val = val['target_class'].values



#apply centering used on train to test

X_test = (test[model_cols].values - mu)/s

y_test = test['target_class'].values
val_metrics = {}
KNN = KNeighborsClassifier(3)

KNN.fit(X_train,y_train)
y_hat_KNN = KNN.predict(X_val)

CM_KNN = confusion_matrix(y_val, y_hat_KNN)

KNN_score = f1_score(y_val,y_hat_KNN)

val_metrics['KNN'] = {

    'F1':KNN_score,

    'CM':CM_KNN

}
LR = LogisticRegression(random_state=0, solver='liblinear').fit(X_train, y_train)
y_hat_LR = LR.predict(X_val)

CM_LR = confusion_matrix(y_val, y_hat_LR)

LR_score = f1_score(y_val,y_hat_LR)

val_metrics['LR'] = {

    'F1':LR_score,

    'CM':CM_LR

}
SVM = svm.SVC(gamma='scale')

SVM.fit(X_train, y_train)  
y_hat_SVM = SVM.predict(X_val)

CM_SVM = confusion_matrix(y_val, y_hat_SVM)

SVM_score = f1_score(y_val,y_hat_SVM)

val_metrics['SVM'] = {

    'F1':SVM_score,

    'CM':CM_SVM

}
plot_metrics(val_metrics)
test_metrics = {}
y_hat = KNN.predict(X_test)

CM_KNN = confusion_matrix(y_test, y_hat)

KNN_score = f1_score(y_test,y_hat)

test_metrics['KNN'] = {

    'F1':KNN_score,

    'CM':CM_KNN

}
y_hat = LR.predict(X_test)

CM_LR = confusion_matrix(y_test, y_hat)

LR_score = f1_score(y_test,y_hat)

test_metrics['LR'] = {

    'F1':LR_score,

    'CM':CM_LR

}
y_hat = SVM.predict(X_test)

CM_SCM = confusion_matrix(y_test, y_hat)

SVM_score = f1_score(y_test,y_hat)

test_metrics['SVM'] = {

    'F1':SVM_score,

    'CM':CM_SVM

}
plot_metrics(test_metrics)