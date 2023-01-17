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
data=pd.read_csv('/kaggle/input/Churn_Modelling.csv')
data.info()
data = data.drop(['RowNumber','CustomerId','Surname'],axis=1)
data.head()
data.describe().T
import matplotlib.pyplot as plt

import seaborn as sns

plt.rcParams['font.sans-serif']=['SimHei']

fig = plt.figure(figsize=(12,8))

plt.title('Sample Label Number Statistics')

ax1 = fig.add_subplot(121)

sns.countplot(data['Exited'])

ax2 = fig.add_subplot(122)

data['Exited'].value_counts().plot(kind='pie',autopct='%.2f')

plt.show()
sns.boxplot(y='CreditScore',x ='Exited',data=data)

plt.show()
def my_plot(data,col,figsize=(8,5),yticks=np.arange(0,1,.1)):

    positive = data.loc[data['Exited'] == 1]

    exited_rate = (positive[col].value_counts() / data[col].value_counts()).sort_index().fillna(0)

    exited_numbers = data.groupby([col, 'Exited']).size().reset_index().pivot(columns='Exited', index=col, values=0).reindex(index=exited_rate.index).fillna(0)

    #fig = plt.figure(figsize=figsize)

    ax1 = fig.add_subplot(111)

    exited_numbers.plot(kind='bar',stacked=True,ax=ax1)

    ax2 = ax1.twinx()

    ax2.plot(exited_rate.index.map(lambda x: str(x)),exited_rate,'og--')

    ax2.set_yticks(yticks)

    plt.show()
positive = data.loc[data['Exited'] == 1]

exited_rate = (positive['Age'].value_counts() / data['Age'].value_counts()).sort_index().fillna(0)

exited_numbers = data.groupby(['Age', 'Exited']).size().reset_index().pivot(columns='Exited', index='Age', values=0).reindex(index=exited_rate.index).fillna(0)

fig = plt.figure(figsize=(10,5))

ax1 = fig.add_subplot(111)

exited_numbers.plot(kind='bar',stacked=True,ax=ax1)

ax2 = ax1.twinx()

#ax2.plot(exited_rate.index.map(lambda x: str(x)),exited_rate,'og--') 

## In my local IDE, I can run this line of code smoothly. But kaggle will report errors

plt.show()
sns.boxplot(data=data,x='Exited',y='Balance')

plt.show()
data['NumOfProducts'].value_counts()

positive = data.loc[data['Exited'] == 1]

negative = data.loc[data['Exited'] == 0]

data[['NumOfProducts','Exited']].pivot_table(index='NumOfProducts',columns='Exited',aggfunc=len,margins=True).fillna(0)
data[['HasCrCard','Exited']].pivot_table(index='Exited',columns='HasCrCard',aggfunc=len,margins=True)
data[['IsActiveMember','Exited']].pivot_table(index='Exited',columns='IsActiveMember',aggfunc=len,margins=True)
sns.boxplot(x='Exited',y='EstimatedSalary',data=data)

plt.show()
import statsmodels.api as sm
data=data.replace({'Female':0,'Male':1})
X_train = data.drop('Exited',axis=1)

y = data['Exited']

X_train.shape,y.shape
log = sm.Logit(y,pd.get_dummies(X_train))

result=log.fit()
log = sm.Logit(y,pd.get_dummies(X_train))
print(result.summary2())
X_train_2 = data.drop(['Exited','Tenure','EstimatedSalary','HasCrCard'],axis=1)
log_2 = sm.Logit(y,pd.get_dummies(X_train_2))

result_2 = log_2.fit()
print(result_2.summary2())
pred= result_2.predict(pd.get_dummies(X_train_2))
yhat = 1*(pred>0.20)
from sklearn.metrics import confusion_matrix
print(confusion_matrix(np.array(data['Exited']),np.array(yhat),labels=[0,1]))