# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

sns.set_style(style="whitegrid")



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/creditcard.csv')
train.head()
# NO null value

train.isnull().any().sum()
train[['Time','Amount','Class']].describe()
train.columns
#unbalance class



sns.countplot(x='Class',data=train)

# to show only comaparision 

plt.yscale('log')
# fraud and normal class 

frauds_t=train[train['Class']==1]



normal_t=train[train['Class']==0]
# Class total input // unbalance  distribution 

total_transaction= len(train)



total_fraud =train['Class'].sum();

total_normal=total_transaction-total_fraud

print('Total Transaction ={}'.format(total_transaction))

print('fraud Data ={}, percentage ={}' .format(total_fraud,(total_fraud/total_transaction)*100) )

print('Normal Data ={} , percentage= {}'.format(total_normal,(total_normal/total_transaction)*100))

print('Fraud data Shape= {}'.format(frauds_t.shape))

print('Normal data Shape= {}'.format(normal_t.shape))

fig ,ax1 =  plt.subplots(nrows=2,ncols=1,figsize=(22,12),sharex=True)



sns.distplot(frauds_t['Amount'],ax=ax1[0],rug=True,bins=70)

sns.distplot(normal_t['Amount'],ax=ax1[1],rug=True,bins=70)





ax1[0].set_title('Fraud')



ax1[1].set_title('normal')



plt.xlim(0,2000)
fig ,ax= plt.subplots(2,1,figsize=(20,8),sharex=True)

ax[0].hist(frauds_t['Amount'],bins=30)

ax[0].set_title('Frauds')

ax[1].hist(normal_t['Amount'],bins=30)

ax[1].set_title('normal')

plt.xlabel('Amounts($)')

plt.yscale('log')

plt.tight_layout()
#check if any  fraud is abnormal for diff time (48 hour interval (day,night ,morning .. anything))

# fraud seems to normal  distribution our time  as normal transaction 

fig ,ax1 =  plt.subplots(nrows=2,ncols=1,figsize=(20,8))



ax1[0].set_title('Normal Transaction') 

ax1[1].set_title('fraud Transaction') 



sns.scatterplot(normal_t['Time'],normal_t['Amount'],ax=ax1[0])



sns.scatterplot(frauds_t['Time'],frauds_t['Amount'],ax=ax1[1])

fig.tight_layout()

#sns.jointplot(x="Amount", y="Time", data=frauds_t);
# approx similar in both 

fig ,ax1 =  plt.subplots(nrows=2,ncols=1,figsize=(22,8),sharex=True)

ax1[0].set_title('Normal Transaction') 

ax1[1].set_title('fraud Transaction') 

sns.distplot(normal_t['Time'],ax=ax1[0],)

sns.distplot(frauds_t['Time'],ax=ax1[1])

fig.tight_layout()
from sklearn.preprocessing import  StandardScaler 
train['Amount'] = StandardScaler().fit_transform(train['Amount'].values.reshape(-1, 1))

frauds_t =train[train['Class']==1]

normal_t =train[train['Class']==0]

from sklearn.model_selection import train_test_split
#counting  by class 

class_0_count,class_1_count=train.Class.value_counts()

# randomly return sample size class_1_count(Fraud)

class_0_under = normal_t.sample(class_1_count)

# make new dataframe (50/50)

df_test =pd.concat([class_0_under,frauds_t])

print(df_test.Class.value_counts())

sns.countplot(df_test['Class'])
plt.figure(figsize=(25,12))



sns.heatmap(df_test.corr(),annot=True)
X_train, X_test, y_train, y_test = train_test_split(df_test.drop(labels=['Time','Class'],axis=1), df_test['Class'], test_size=0.2)
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_auc_score

from sklearn.metrics import  confusion_matrix ,classification_report,roc_auc_score

#Empirical good default values are max_features=n_features for regression problems, and max_features=sqrt(n_features) for classification tasks

clf= RandomForestClassifier(verbose=2,n_estimators=300)



clf.fit(X_train,y_train)

def model_representation(y_true,y_predicted):

    matrix=confusion_matrix(y_true, y_predicted)

    print(matrix)

    print(roc_auc_score(y_true,y_predicted))

    print(classification_report(y_true,y_predicted))

    TP=matrix[0,0]

    FP= matrix[0,1]

    FN =matrix[1,0]

    TN=matrix[1,1]

    print('TP={}\n FP={}\n FN={}\n TN={}\n '.format(TP,FP,FN,TN))

    sns.heatmap(matrix,annot=True)

    plt.xlabel('predicted value')

    plt.ylabel('actual value')

    plt.title('Confusion Matrix')
y_pre=clf.predict(X_test)



model_representation(y_test,y_pre)


X = train.drop(labels=['Time','Class'],axis=1)

y= train['Class']



y_pre=clf.predict(X)



model_representation(y,y_pre)
