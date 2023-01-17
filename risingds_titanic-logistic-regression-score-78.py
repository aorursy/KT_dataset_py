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
# importing useful libraries

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
# reading the data

train= pd.read_csv('/kaggle/input/titanic/train.csv')

test= pd.read_csv('/kaggle/input/titanic/test.csv')
train.info()
# dropping two records where Embarked is missing

train = train[train.Embarked.notna()]
train['male'] = train.Sex.apply(lambda x: 1 if x=='male' else 0)
sns.barplot('Pclass','Age', data= train)
# creating imputation data

ageimpute = train.groupby('Pclass')['Age'].transform('mean')

ageimpute
# going forward working with copy of original DF

train1 = train
train1['Age'].fillna(ageimpute, inplace=True)
train['EmbarkC'] = train1.Embarked.apply(lambda x: 1 if x=='C' else 0)
train['EmbarkQ'] = train1.Embarked.apply(lambda x: 1 if x=='Q' else 0)
train1['numfam'] = train1['SibSp']+train['Parch']+1
train1.head()
train1.describe()
train1['farebin']= pd.qcut(train1['Fare'],10, labels=range(1,11))
train1['agebin']= pd.qcut(train1['Age'],10, labels=range(1,11))
train1.numfam.hist()
train1.head()
print('Male')

print('% survived in Age 0-10:',train1[(train1.Age<=10) & (train1.male==1)].Survived.mean())

print('% survived in Age 10-20:',train1[(train1.Age>10) & (train1.Age<=20) & (train1.male==1) ].Survived.mean())

print('% survived in Age 20-30:',train1[(train1.Age>20) & (train1.Age<=30) & (train1.male==1)].Survived.mean())

print('% survived in Age 30-40:',train1[(train1.Age>30) & (train1.Age<=40) & (train1.male==1)].Survived.mean())

print('% survived in Age 40-50:',train1[(train1.Age>40) & (train1.Age<=50) & (train1.male==1)].Survived.mean())

print('% survived in Age 50-60:',train1[(train1.Age>50) & (train1.Age<=60) & (train1.male==1)].Survived.mean())

print('% survived in Age 60-70:',train1[(train1.Age>60) & (train1.Age<=70) & (train1.male==1)].Survived.mean())

print('% survived in Age >70:',train1[train1.Age>70 & (train1.male==1)].Survived.mean())

print('% survived in Age 10-60:',train1[(train1.Age>10) & (train1.Age<=60) & (train1.male==1)].Survived.mean())

print('% survived in Age >60:',train1[(train1.Age>60) & (train1.male==1)].Survived.mean())

print('% survived in Age 0-15:',train1[(train1.Age<=15) & (train1.male==1)].Survived.mean())

print('% survived in Age 10-15:',train1[(train1.Age>10) & (train1.Age<=15) & (train1.male==1)].Survived.mean())



print('\n')

print('Female')

print('% survived in Age 0-10:',train1[(train1.Age<=10) & (train1.male==0)].Survived.mean())

print('% survived in Age 10-20:',train1[(train1.Age>10) & (train1.Age<=20) & (train1.male==0) ].Survived.mean())

print('% survived in Age 20-30:',train1[(train1.Age>20) & (train1.Age<=30) & (train1.male==0)].Survived.mean())

print('% survived in Age 30-40:',train1[(train1.Age>30) & (train1.Age<=40) & (train1.male==0)].Survived.mean())

print('% survived in Age 40-50:',train1[(train1.Age>40) & (train1.Age<=50) & (train1.male==0)].Survived.mean())

print('% survived in Age 50-60:',train1[(train1.Age>50) & (train1.Age<=60) & (train1.male==0)].Survived.mean())

print('% survived in Age 60-70:',train1[(train1.Age>60) & (train1.Age<=70) & (train1.male==0)].Survived.mean())

print('% survived in Age >70:',train1[train1.Age>70 & (train1.male==0)].Survived.mean())

print('% survived in Age 10-60:',train1[(train1.Age>10) & (train1.Age<=60) & (train1.male==0)].Survived.mean())

print('% survived in Age >60:',train1[(train1.Age>60) & (train1.male==0)].Survived.mean())

print('% survived in Age 0-15:',train1[(train1.Age<=15) & (train1.male==0)].Survived.mean())

print('% survived in Age 10-15:',train1[(train1.Age>10) & (train1.Age<=15) & (train1.male==0)].Survived.mean())



train1.groupby('male').Survived.mean()
sns.barplot('farebin','Survived',data=train1[train1.male==1])
sns.barplot('farebin','Survived',data=train1[train1.male==0])
sns.barplot('numfam','Survived',data=train1)
train1['bigfam'] = train1['numfam'].apply(lambda x: 1 if x>4 else 0)
train1['single'] = train1['numfam'].apply(lambda x: 1 if x==1 else 0)
train1['boy'] = train1.apply(lambda row: 1 if (row['male']==1) & (row['Age']<=10) else 0, axis=1)
train1.Pclass.value_counts()
train['class1'] = train1.Pclass.apply(lambda x: 1 if x==1 else 0)
train['class2'] = train1.Pclass.apply(lambda x: 1 if x==2 else 0)
train1['splitname'] = train1['Name'].apply(lambda x: x.split(',')[1].strip())
train1['title'] = train1['splitname'].apply(lambda x: x.split(' ')[0])
train1.male.value_counts()
train1.groupby('title')['Survived'].agg(['count','mean']).sort_values('mean')
train1['master'] = train1['title'].apply(lambda x: 1 if x=='Master.' else 0)
train1['rev'] = train1['title'].apply(lambda x: 1 if x=='Rev.' else 0)
train1.corr().Survived.sort_values()
X = train1[['class1','male','EmbarkC','master','single']]



y= train1['Survived']
X.head()
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=890)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, f1_score
# logistics



from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression()

logmodel.fit(X_train,y_train)

y_pred = logmodel.predict(X_test)



print(f1_score(y_test,y_pred))
# for submission

test.info()
test['title'] = test['Name'].apply(lambda x: x.split(',')[1].strip().split(' ')[0])
test['title'].value_counts()
test['master'] = test['title'].apply(lambda x: 1 if x=='Master.' else 0)

test['rev'] = test['title'].apply(lambda x: 1 if x=='Rev.' else 0)
test['Age'].fillna(test.groupby('Pclass')['Age'].transform('mean'), inplace=True)
test['Fare'].fillna(test.groupby('Pclass')['Fare'].transform('mean'), inplace=True)
test.info()
test['numfam'] = test['SibSp']+test['Parch']+1
test['EmbarkC'] = test.Embarked.apply(lambda x: 1 if x=='C' else 0)

test['EmbarkQ'] = test.Embarked.apply(lambda x: 1 if x=='Q' else 0)

test['male'] = test.Sex.apply(lambda x: 1 if x=='male' else 0)

test['bigfam'] = test['numfam'].apply(lambda x: 1 if x>4 else 0)

test['single'] = test['numfam'].apply(lambda x: 1 if x==1 else 0)

test['class1'] = test.Pclass.apply(lambda x: 1 if x==1 else 0)

test['class2'] = test.Pclass.apply(lambda x: 1 if x==2 else 0)
test.corr()
test['boy'] = test.apply(lambda row: 1 if (row['male']==1) & (row['Age']<=10) else 0, axis=1)
test.head()
test.columns
test_final = test[['class1','male','EmbarkC','master','single']]
fullscaler = StandardScaler()



X_scaled = fullscaler.fit_transform(X)



test_final = fullscaler.transform(test_final)
sublogmodel = LogisticRegression()

sublogmodel.fit(X_scaled,y)

y_prediction = sublogmodel.predict(test_final)
test['Survived'] = pd.Series(y_prediction)
submit_logmodel = test[['PassengerId','Survived']]
submit_logmodel.to_csv('/kaggle/working/submit_logmodel.csv', index=False)