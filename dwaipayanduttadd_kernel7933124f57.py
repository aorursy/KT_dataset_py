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

import matplotlib as mlp

import matplotlib.pyplot as plt

import math

import datetime

import seaborn as sns

df=pd.read_csv('../input/creditcardfraud/creditcard.csv')

df.head()
df.info()
df[['Amount', 'Time']].describe().T
df['Class'].value_counts().plot(kind='pie', figsize=[8,8], autopct='%1.1f%%')

plt.legend(['Genuine', 'Fraud'])
df.shape
df=df.drop_duplicates()

df.shape

#Removing duplicates, we get a new DataFrame of new 283726 rows
df[(df['Amount']==0)&(df['Class']==0)]['Class'].count()
df[(df['Amount']==0)&(df['Class']==1)]['Class'].count()
df.drop(df[df['Amount']==0].index, inplace=True)
df[df['Amount']==0].count()#All the rows with Amount 0 is removed
df.reset_index(inplace=True)#to reset the index numbers
plt.figure(figsize=(10,10))

sns.distplot(df['Amount'], bins=100)
plt.figure(figsize=(10,10))

sns.distplot(df[df['Class']==0]['Amount'], bins=100)
plt.figure(figsize=(10,10))

sns.distplot(df[df['Class']==1]['Amount'], bins=100)
print(df[df['Class']==1]['Amount'].value_counts().head(10))

print(df[df['Class']==1]['Amount'].max())

print(df[df['Class']==1]['Amount'].min())
np.percentile(df[df['Class']==1]['Amount'], (25,50, 75))
#For the fraud transactions, we can see that the difference between 75%ile and  50%ile is much greater than 50%ile and 25%ile
sns.boxplot(x='Class', y='Amount', data=df)
sns.boxplot(x='Class', y='Time', data=df)
#Distribution as per time

plt.figure(figsize=(10,10))

sns.distplot(df['Time'])
df['Time_Hr']=df['Time'].apply(lambda sec: (sec/3600))

df.shape

df.head()
plt.figure(figsize=(10,10))

sns.distplot(df[(df['Time_Hr']<=24)]['Time_Hr'])

sns.distplot(df['Time_Hr'])
#Checking only the fraud transactions within 24 hrs and 48 hrs

plt.figure(figsize=(10,10))

sns.distplot(df[(df['Class']==1)&(df['Time_Hr']<=24)]['Time_Hr'])
sns.distplot(df[(df['Class']==1)&(df['Time_Hr']>24)]['Time_Hr'])
df['Class'].value_counts(normalize=True)
q1,q3=np.percentile(df['Amount'], (25,75))

iqr=q3-q1

lower_bound=q1-(1.5*iqr)

upper_bound=q3+(1.5*iqr)

print(lower_bound)

print(upper_bound)

print(q1)

print(q3)

print(iqr)
outlier_amount=df[(df['Amount']<lower_bound)|(df['Amount']>upper_bound)]['Amount']

fraud_transaction=df[(df['Class']==1)&((df['Amount']>lower_bound)&(df['Amount']>upper_bound))]

genuine_transactions=df[(df['Class']==0)&((df['Amount']>lower_bound)&(df['Amount']<upper_bound))]

print('total count of outliers is: ', outlier_amount.count())

print('total count of fraud transactions is:', fraud_transaction.count())

print('total count of genuine transactions is:', genuine_transactions.count())
df_new=df.drop(outlier_amount.index)
df_new.reset_index(inplace=True)
print(df_new.shape)

print(df_new['Class'].value_counts())

print(df_new['Class'].value_counts(normalize=True))
print(df.shape)

print(df['Class'].value_counts())

print(df['Class'].value_counts(normalize=True))
df_new=df[(df['Amount']>=lower_bound)&(df['Amount']<=upper_bound)]

sns.boxplot(x='Class', y='Amount', data=df_new)
df_new['Class'].value_counts(normalize=True)
df_new['Class'].value_counts()
sns.distplot(df_new['Amount'])
sns.distplot(df_new[df_new['Class']==1]['Amount'], bins=10)

sns.distplot(df_new[df_new['Class']==0]['Amount'], bins=10)
np.percentile((df_new[df_new['Class']==1]['Amount']), (75,100))
sns.heatmap(df_new[['Time', 'Amount', 'Class']].corr(), linewidth=0.5, vmax=1, square=True, cmap='viridis', annot=True)
plt.figure(figsize=(10,10))

sns.scatterplot(x='Time_Hr', y='Amount', data=df_new, hue='Class')
from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler()

df_new['ScaledAmount']=scaler.fit_transform(df_new[['Amount']])

df_new['ScaledAmount']
df_new['log_transformed_amt']=np.log(df_new.Amount+0.01)
sns.boxplot(x='Class', y='log_transformed_amt', data=df_new)
df_new.shape
from sklearn.model_selection import train_test_split

y=df_new['Class'].values

x=df_new.drop(columns=['Time', 'Time_Hr', 'Amount', 'log_transformed_amt']).values

xTrain, xTest, yTrain, yTest=train_test_split(x,y, test_size=0.3, random_state=0)
from sklearn.linear_model import LogisticRegression

logreg=LogisticRegression()

logreg.fit(xTrain, yTrain)
ypred=logreg.predict(xTest)

from sklearn.metrics import accuracy_score

print(accuracy_score(ypred,yTest))
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve, auc

roc_auc_score(yTest, ypred)
import sklearn.metrics as metric

print(confusion_matrix(yTest, ypred))
from sklearn import metrics

fpr, tpr, thresholds=metrics.roc_curve(yTest, ypred, pos_label=2)
class_name=[0,1]

fig, ax=plt.subplots()

tick_marks=np.arange(len(class_name))

plt.xticks(tick_marks, class_name)

plt.yticks(tick_marks, class_name)

sns.heatmap(pd.DataFrame(confusion_matrix(yTest, ypred)), annot=True, cmap='viridis', fmt='g', linewidth=0.5)

ax.xaxis.set_label_position("top")

plt.tight_layout()

plt.title('Confusion Matrix', y=1.5)

plt.ylabel('Actial Label')

plt.xlabel('Predicted Label')
print("Accuracy:", metrics.accuracy_score(yTest, ypred))

print("Precision:", metrics.precision_score(yTest, ypred))

print("Recall:", metrics.recall_score(yTest, ypred))
y_pred_prob=logreg.predict_proba(xTest)[::,1]

fpr, tpr, threshold=metrics.roc_curve(yTest, y_pred_prob)

auc=metrics.roc_auc_score(yTest, y_pred_prob)

plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area=50.2f)')

plt.plot([0,1], [0,1], color='navy', linestyle='--')

plt.xlabel('FPR')

plt.ylabel('TPR')