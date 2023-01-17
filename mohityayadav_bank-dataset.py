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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# hide warnings
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 200)


%matplotlib inline
data=pd.read_csv('/kaggle/input/bankbalanced/bank.csv')
data.head()
data.info()
plt.figure(figsize=(26,5))
plt.subplot(1,6,1)
sns.countplot(x='job',data=data)
plt.xticks(rotation=70)
plt.subplot(1,6,2)
sns.countplot(x='marital',data=data)
plt.xticks(rotation=65)
plt.subplot(1,6,3)
sns.countplot(x='education',data=data)
plt.figure(figsize=(26,5))
plt.subplot(1,6,1)
sns.countplot(x='default',data=data)
plt.xticks(rotation=70)
plt.subplot(1,6,2)
sns.countplot(x='housing',data=data)
plt.xticks(rotation=65)
plt.subplot(1,6,3)
sns.countplot(x='loan',data=data)
plt.xticks(rotation=75)
plt.figure(figsize=(26,5))
plt.subplot(1,6,1)
sns.countplot(x='contact',data=data)
plt.xticks(rotation=70)
plt.subplot(1,6,2)
sns.countplot(x='month',data=data)
plt.xticks(rotation=65)
plt.subplot(1,6,3)
sns.countplot(x='education',data=data)
plt.figure(figsize=(26,5))
plt.subplot(1,6,1)
sns.countplot(x='default',data=data)
plt.xticks(rotation=70)
plt.subplot(1,6,2)
sns.countplot(x='poutcome',data=data)
plt.xticks(rotation=65)


plt.figure(figsize=(26,5))
plt.subplot(1,6,1)
sns.boxplot(data['age'])
plt.subplot(1,6,2)
sns.boxplot(data['balance'])
plt.subplot(1,6,3)
sns.boxplot(data['campaign'])
plt.figure(figsize=(26,5))
plt.subplot(1,6,1)
sns.boxplot(data['duration'])

plt.subplot(1,6,2)
sns.boxplot(data['pdays'])

q1 = data['balance'].quantile(0.05)
q4 = data['balance'].quantile(0.95)
data= data[(data['balance']>=q1) & (data['balance']<=q4)]
data.shape
q1 = data['duration'].quantile(0.05)
q4 = data['duration'].quantile(0.95)
data2 = data[(data['duration']>=q1) & (data['duration']<=q4)]
data2.shape
q1 = data['campaign'].quantile(0.05)
q4 = data['campaign'].quantile(0.95)
data = data[(data['campaign']>=q1) & (data['campaign']<=q4)]
data.shape
q1 = data['age'].quantile(0.02)
q4 = data['age'].quantile(0.98)
data = data[(data['age']>=q1) & (data['age']<=q4)]
data.shape
#checking for outliers 

plt.figure(figsize=(26,5))
plt.subplot(1,6,1)
sns.boxplot(data2['age'])
plt.subplot(1,6,2)
sns.boxplot(data2['balance'])
plt.subplot(1,6,3)
sns.boxplot(data2['campaign'])
plt.figure(figsize=(26,5))
plt.subplot(1,6,1)
sns.boxplot(data2['duration'])

plt.subplot(1,6,2)
sns.boxplot(data2['pdays'])

data.drop(['default'],axis=1,inplace=True) #skewed

varlist=['housing','loan','deposit']

def binary_map(x):
    return x.map({'yes': 1, "no": 0})

# Applying the function to the df list
data[varlist] = data[varlist].apply(binary_map)
cat_df=data[['job','marital','education','contact','month','poutcome']]
cat_df.head()
data_dummies=pd.get_dummies(cat_df,drop_first=True)
data_dummies.head()
data.drop(list(cat_df.columns),axis=1,inplace=True)
data=pd.concat([data,data_dummies],axis=1)
data.head()
X=data.drop(['deposit'],axis=1)
y=data.deposit
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.7,random_state=100)
#scaling the train and test data

from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
#class imbalance
100*y_train.value_counts(normalize=True)
from  sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

rf_model=RandomForestClassifier(class_weight='balanced',criterion='gini',min_samples_leaf=1,min_samples_split=16,n_estimators=700)
rf_model.fit(X_train,y_train)
y_pred=rf_model.predict(X_test)
print('Accuracy :  {}'.format(metrics.accuracy_score(y_test,y_pred)))
print('Sensitivity :  {}'.format(metrics.recall_score(y_test,y_pred)))

