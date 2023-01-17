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
train=pd.read_csv('../input/mobile-price-classification/train.csv')
train.head(2)
train.shape
train.isnull().sum()
train.head(1)
print(train.price_range.value_counts())

print(train.blue.value_counts())

print(train.dual_sim.value_counts())

print(train.four_g.value_counts())

print(train.mobile_wt.value_counts())
train.describe().T
import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
train.hist(figsize=(15,15))
train.columns
sns.pointplot(x='price_range',y='clock_speed',data=train)
sns.pointplot(y='four_g',x='price_range',data=train)
sns.pointplot(x='price_range',y='int_memory',data=train)
sns.barplot(x='three_g',y='price_range',data=train)

#check three g support or not using barplot

label=['3 g support ','not support']

values=train['three_g'].value_counts().values

fig,ax1=plt.subplots()

ax1.pie(values,labels=label,shadow=True,startangle=90,autopct='%1.1f%%')
train.head(1)

sns.barplot(x='price_range',y='four_g',data=train)

label=['4-g support','not-support']

values=train['four_g'].value_counts().values

fig,ax1=plt.subplots()

ax1.pie(values,labels=label,shadow=True,startangle=90,autopct='%1.1f%%')
sns.swarmplot(x='price_range',y='clock_speed',data=train)
sns.boxplot(x='wifi',y='price_range',data=train)
train.head(1)

#plt.figure(figsize=(10,6))

train['fc'].hist(alpha=0.5,color='blue',label='Front camera')

train['pc'].hist(alpha=0.5,color='green',label='primary camera')

plt.legend()

plt.xlabel('megapixel')
sns.jointplot(x='ram',y='price_range',data=train,kind='kde')
train.head(1)

sns.jointplot(y='n_cores',x='price_range',data=train,kind='kde')
sns.jointplot(x='talk_time',y='price_range',kind='kde',data=train)
plt.figure(figsize=(15,15))

sns.heatmap(train.corr(),cmap='RdYlGn',annot=True)
train.head(1)

x=train.drop('price_range',axis=1)

y=(train['price_range'])
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=.2)
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier

rfc=RandomForestClassifier(n_estimators=100,criterion='gini',min_samples_split=2,bootstrap=True)

rfc.fit(X_train,y_train)

Y_pred=rfc.predict(X_test)

print('y predicted',Y_pred)

rfc.score(X_train,y_train)
rfc.score(X_test,y_test)
#gattung more accurating result using boosting clasifier

gc=GradientBoostingClassifier()

gc.fit(X_train,y_train)

gc.fit(X_test,y_test)

gc.score(X_test,y_test)
import xgboost as xgb
xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=42)

xgb_model.fit(X_train, y_train)

xgb_model.fit(X_test,y_test)

Y_pred=xgb_model.predict(X_test)

Y_pred
xgb_model.score(X_test,y_test)
pd.crosstab(y_test,Y_pred)
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report

sns.heatmap(confusion_matrix(y_test,Y_pred),annot=True)
accuracy_score(y_test,Y_pred,normalize=False)
print(classification_report(y_test,Y_pred))