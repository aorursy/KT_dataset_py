#import of libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

#loading data
df_train=pd.read_csv('../input/train.csv')
df_test=pd.read_csv('../input/test.csv')
df_train.head()


print('The number of samples into the test data is {}.'.format(df_train.shape[0]))
print('The number of samples into the test data is {}.'.format(df_test.shape[0]))
df_train.describe()

df_train.isnull().sum()

df_test.isnull().sum()
ax = df_train["Age"].hist(bins=15, density=True, stacked=True, color='teal', alpha=0.6)
df_train["Age"].plot(kind='density', color='orange')
ax.set(xlabel='Age')
plt.xlim(-10,85)
plt.show()
print(df_train['Embarked'].value_counts())
sns.countplot(x='Embarked', data=df_train, palette='spring')
plt.show()
sns.barplot('Pclass', 'Survived', data=df_train, color="darkturquoise")
plt.show()
sns.barplot('Embarked', 'Survived', data=df_train, color="blue")
plt.show()
sns.barplot('Sex', 'Survived', data=df_train, color="green")
plt.show()
sns.heatmap(df_train.corr(), annot=True, cmap="RdYlGn")
plt.show()
df1_train=df_train.copy()
df1_train['Family']=df1_train['Parch']+df1_train['SibSp']
del df1_train['Cabin']
del df1_train['Parch']
del df1_train['SibSp']
del df1_train['PassengerId']
del df1_train['Ticket']

df1_train["Age"].fillna(df1_train["Age"].median(skipna=True),inplace=True)
df1_train['Embarked'].fillna('S',inplace=True)
df1_train=pd.get_dummies(df1_train, columns=["Pclass","Embarked","Sex"])

sName=df_train['Name']
title=[]

for name in sName:
      title.append(name.split(',')[1].split('.')[0].lstrip())
print(title[:5])
df1_train['Title']=pd.Series(title)
print(pd.crosstab(df1_train.Title,df1_train.Survived))

from collections import Counter
c=Counter(title)
print(c)
d={}
d['any']=0
d['Mr']=1
d['Master']=2
d['Mrs']=3
d['Miss']=4


for i in range(len(title)):
    if title[i] not in ['Mr','Miss','Master','Mrs']:
        title[i]='any'
stemp=list(map((lambda x:d[x]),title))
df1_train['mTitle']=pd.Series(stemp)
sNameTest=df_test['Name']

title=[]
for name in sNameTest:
      title.append(name.split(',')[1].split('.')[0].lstrip())
print(title[:5])
df_test['Title']=pd.Series(title)
print(pd.crosstab(df1_train.Title,df1_train.Survived))
c=Counter(title)
print(c)
for i in range(len(title)):
    if title[i] not in ['Mr','Miss','Master','Mrs']:
        title[i]='any'
stemp=list(map((lambda x:d[x]),title))
df_test['mTitle']=pd.Series(stemp)
df_test.head()

df_test['Fare'].fillna(df_test['Fare'].median(skipna=True),inplace=True)
df_test["Age"].fillna(df_test["Age"].median(skipna=True),inplace=True)
df_test['Family']=df_test['Parch']+df_test['SibSp']
del df_test['Parch']
del df_test['SibSp']
del df_test['Ticket']
del df_test['Cabin']
df_test=pd.get_dummies(df_test, columns=["Pclass","Embarked","Sex"])
df_test.head()
from sklearn.linear_model import LogisticRegression 
features = ["Age","Fare","Pclass_1","Pclass_2","Embarked_C","Embarked_S","Sex_male","Family","mTitle"]
X=df1_train[features]
y=df1_train['Survived']
clf=LogisticRegression()
clf.fit(X,y)
Xt=df_test[features]
yr=clf.predict(Xt)
result=pd.DataFrame()
result['PassengerId']=df_test['PassengerId']
result['Survived']=pd.Series(yr)
result['Survived'].value_counts()
#result.to_csv('new.csv',index=False)