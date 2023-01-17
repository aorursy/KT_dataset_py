# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')
df.head(5)
df.describe()
import matplotlib.pyplot as plt
import seaborn as sns
df.groupby('Class').mean()['Amount']
sns.boxplot(x=df.groupby('Class').mean().index,y='Amount',data=df.groupby('Class').mean())
sns.violinplot(x='Class',y='Amount',data=df)
sns.countplot(x='Class',data=df)
fig,axes = plt.subplots(nrows=1,ncols=2,figsize=(10,6))
sns.distplot(df[df['Class']==0]['Time'],kde=True,ax=axes[0])
sns.distplot(df[df['Class']==1]['Time'],kde=True,ax=axes[1])
fig,axes = plt.subplots(nrows=1,ncols=2,figsize=(10,6),sharex=True)
sns.distplot(df[df['Class']==0]['Amount'],kde=True,ax=axes[0],label='0')
sns.distplot(df[df['Class']==1]['Amount'],kde=True,ax=axes[1],label='1')
df_fraud = df[df['Class']==1]
df_non_fraud = df[df['Class']==0].iloc[:492]
df_balanced = pd.concat([df_fraud,df_non_fraud],axis=0)
df_balanced.head()
len(df_balanced)
plt.figure(figsize=(6,4))
sns.heatmap(df_balanced.corr())
sns.countplot(x='Class',data = df_balanced)
plt.figure(figsize=(25,10))
sns.heatmap(df_balanced.corr())
#V3,V6,V9,V10,V12,V14,V16,V17
fig,axes = plt.subplots(nrows=2,ncols=4,figsize=(20,12))
sns.boxplot(x='Class',y='V3',data=df_balanced,ax=axes[0][0])
axes[0][0].set_title('V3')

sns.boxplot(x='Class',y='V6',data=df_balanced,ax=axes[0][1])
axes[0][1].set_title('V6')

sns.boxplot(x='Class',y='V9',data=df_balanced,ax=axes[0][2])
axes[0][2].set_title('V9')

sns.boxplot(x='Class',y='V10',data=df_balanced,ax=axes[0][3])
axes[0][3].set_title('V10')

sns.boxplot(x='Class',y='V12',data=df_balanced,ax=axes[1][0])
axes[1][0].set_title('V12')

sns.boxplot(x='Class',y='V14',data=df_balanced,ax=axes[1][1])
axes[1][1].set_title('V14')

sns.boxplot(x='Class',y='V16',data=df_balanced,ax=axes[1][2])
axes[1][2].set_title('V16')

sns.boxplot(x='Class',y='V17',data=df_balanced,ax=axes[1][3])
axes[1][3].set_title('V17')

len(df_balanced[df_balanced['V3']<-28].index)
len(df_balanced[df_balanced['V6']<-5].index)
len(df_balanced[df_balanced['V9']<-9].index)
len(df_balanced[df_balanced['V10']<-20].index)
len(df_balanced[df_balanced['V12']<-17].index)
len(df_balanced[df_balanced['V14']<-17].index)
len(df_balanced[df_balanced['V16']<-12.5].index)
len(df_balanced[df_balanced['V17']<-22].index)
len(df_balanced)
df_balanced.drop(df_balanced[df_balanced['V3']<-28].index,axis=0,inplace=True)

df_balanced.drop(df_balanced[df_balanced['V6']<-5].index,axis=0,inplace=True)

df_balanced.drop(df_balanced[df_balanced['V9']<-9].index,axis=0,inplace=True)

df_balanced.drop(df_balanced[df_balanced['V10']<-20].index,axis=0,inplace=True)

df_balanced.drop(df_balanced[df_balanced['V12']<-17].index,axis=0,inplace=True)

df_balanced.drop(df_balanced[df_balanced['V14']<-17].index,axis=0,inplace=True)

df_balanced.drop(df_balanced[df_balanced['V16']<-12.5].index,axis=0,inplace=True)

df_balanced.drop(df_balanced[df_balanced['V17']<-22].index,axis=0,inplace=True)
plt.figure(figsize=(6,4))
sns.heatmap(df_balanced.corr())
sns.distplot(df_balanced['V3'])
from sklearn.model_selection import train_test_split
X=df_balanced.drop('Class',axis=1)
y=df_balanced['Class']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train,y_train)
pred = model.predict(X_test)
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
print(classification_report(y_test,pred))
print(confusion_matrix(y_test,pred))
df_balanced
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping

X=df_balanced.drop('Class',axis=1).values
y=df_balanced['Class'].values
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

model = Sequential()

model.add(Dense(28,activation = 'relu'))
model.add(Dense(14,activation = 'relu'))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer ='adam')

model.fit(X_train,y_train,epochs =400,validation_data=(X_test,y_test))

early_stop = EarlyStopping(monitor='val_loss',mode='min',verbose =1,patience=20)
loss = pd.DataFrame(model.history.history)
loss.plot()
pred = model.predict_classes(X_test)
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
print(classification_report(y_test,pred))
print(confusion_matrix(y_test,pred))
