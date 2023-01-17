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

import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd

import datetime

from sklearn.metrics import mean_squared_error

from sklearn.metrics import roc_auc_score,roc_curve

from sklearn.model_selection import StratifiedKFold

import warnings

df_train=pd.read_csv('/kaggle/input/santander-customer-satisfaction/train.csv')

df_test=pd.read_csv('/kaggle/input/santander-customer-satisfaction/test.csv')

print(df_train.columns.values)

print(df_train.shape)

print(df_train.head())

print(df_train.info())
print(df_test.shape)

print(df_test.head())

print(df_test.info())
def missing_data(data):

    total=data.isnull().sum()

    percent=(data.isnull().sum()/data.isnull().count())

    #types = data.coulmns.dtype

    missingdata=pd.concat([total,percent],axis=1,keys=['total','percent'])

    print(np.transpose(missingdata))
missing_data(df_train)

missing_data(df_test)
corrmat=df_train.corr()

plt.show(sns.heatmap(corrmat,vmax=20,square=True))

print(corrmat['TARGET'].sort_values(ascending=False))
k=20

cols=corrmat.nlargest(k,'TARGET').index

cm=np.corrcoef(df_train[cols].values.T)

#print(cm)

sns.set(font_scale=1.25)

plt.show(sns.heatmap(cm,cbar=True,annot=True,square=True,fmt='.2f',annot_kws={'size': 5}, yticklabels=cols.values, xticklabels=cols.values))

var='var36'

data=pd.concat([df_train['TARGET'],df_train[var]],axis=1)

plt.scatter(x=df_train[var],y=df_train['TARGET'])

plt.xlabel(var,fontsize=20)

plt.ylabel('TARGET',fontsize=20)

plt.show()
def plot_feature_scatter (df1,df2,features):

     i=0

     sns.set_style('whitegrid')

     plt.figure()

     fig,ax=plt.subplots(4,4,figsize=(14,14))

     for feature in features:

         i=+1

         plt.subplot(4,4,i)

         plt.scatter(df1[feature],df2[feature],marker='+')

         plt.xlabel(feature,fontsize=9)

         plt.show()



features= df_train.columns.values[2:1]

plot_feature_scatter(df_train[::20],df_test[::20],features)
plt.show(sns.countplot(df_train['TARGET']))

print(df_train['TARGET'].value_counts()[1])

print(df_train['TARGET'].value_counts()[0])
plt.figure(figsize=(16,6))

features=df_train.columns.values[2:102]

plt.show(sns.distplot(df_train[features].mean(axis=1),color='red',bins=120,label='train'))

plt.show(sns.distplot(df_test[features].mean(axis=1),color='red',bins=120,label='train'))

x_train=df_train.drop(['ID','TARGET'],axis=1)

y_train=df_train['TARGET']

x_test=df_test.drop('ID',axis=1)

print(x_train.head(5))

print(y_train.head(5))

print(x_test.head(5))

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix

x_trainsplit,x_testsplit,y_trainsplit,y_testsplit,=train_test_split(x_train,y_train,test_size=0.33,random_state=44,shuffle=True)

print(x_trainsplit.shape)

print(y_trainsplit.shape)

print(x_testsplit.shape)

print(y_testsplit.shape)

KNNclassifermodel=KNeighborsClassifier(n_neighbors=5,weights='uniform',algorithm='auto')

KNNclassifermodel.fit(x_trainsplit,y_trainsplit)

print(KNNclassifermodel.score(x_trainsplit,y_trainsplit))

print(KNNclassifermodel.score(x_testsplit,y_testsplit))

y_predict=KNNclassifermodel.predict(x_test)

print(y_predict)
submission=pd.DataFrame({

    'ID':df_test['ID'],

    'TARGET':y_predict

    

})

print(submission.head(10))