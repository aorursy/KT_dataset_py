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
import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

import numpy as np

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score
df=pd.read_csv('../input/titanic/train.csv')
dt=pd.read_csv('../input/titanic/test.csv')
df.head(3)
df.info()
corr=df.corr()
sns.heatmap(corr,annot=True)
df.isnull().sum()
df['Survived'].unique()
def bar_chart(features):

    survived=df[df['Survived']==1][features].value_counts()

    dead=df[df['Survived']==0][features].value_counts()

    barx=pd.DataFrame([survived,dead])

    barx.index=['survived','dead']

    barx.plot(kind='bar',stacked=True,figsize=(10,5),grid=True)
bar_chart('Sex')
bar_chart('Pclass')
bar_chart('SibSp')
bar_chart('Parch')
bar_chart('Embarked')
df.head(3)
df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
dt['Title'] = dt['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
df['Title'].value_counts()
bar_chart('Title')
title_mapping={"Mr": 0,"Miss": 1,"Mrs" : 1,"Master": 3, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mlle": 3,"Countess": 3,

                 "Ms": 3, "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona" : 3, "Mme": 3,"Capt": 3,"Sir": 3 }

df['Title']=df['Title'].map(title_mapping)
dt['Title']=dt['Title'].map(title_mapping)
df['Title'].value_counts()
df.drop('Name',axis=1,inplace=True)
dt.drop('Name',axis=1,inplace=True)
df.head(3)
le=LabelEncoder()
ohe=OneHotEncoder()
df['Sex']=le.fit_transform(df['Sex'])

dt['Sex']=le.fit_transform(dt['Sex'])

df['Sex'].value_counts()
bar_chart('Sex')
df.groupby("Title")["Age"].mean()
df['Age'].fillna(df.groupby('Title')['Age'].transform('mean'),inplace=True)

dt['Age'].fillna(dt.groupby('Title')['Age'].transform('mean'),inplace=True)

df.groupby('Title')['Age'].mean()
df.head(3)
df['Age'].isnull().sum()
dt['Age'].isnull().sum()
dt['Age'].isnull().sum()
facet=sns.FacetGrid(df,hue='Survived',aspect=4)

facet.map(sns.kdeplot,'Age',shade=True)

facet.set(xlim=(0,20))

facet.add_legend()

plt.show()
facet=sns.FacetGrid(df,hue='Survived',aspect=4)

facet.map(sns.kdeplot,'Age',shade=True)

facet.set(xlim=(20,30))

facet.add_legend()

plt.show()
facet=sns.FacetGrid(df,hue='Survived',aspect=4)

facet.map(sns.kdeplot,'Age',shade=True)

facet.set(xlim=(30,40))

facet.add_legend()

plt.show()
facet=sns.FacetGrid(df,hue='Survived',aspect=4)

facet.map(sns.kdeplot,'Age',shade=True)

facet.set(xlim=(40,50))

facet.add_legend()

plt.show()
df.head(3)
df.groupby('Survived')['Age'].mean()
df.groupby('Survived')['Sex'].mean()
dfx=df.copy()
dfx.head(3)
dfx['Embarked'].value_counts()
dfx.loc[df['Age']<=5,'Age']=0

dfx.loc[(df['Age']>5) & (dfx['Age']<20),'Age']=1

dfx.loc[(df['Age']>20) & (dfx['Age']<31),'Age']=2

dfx.loc[(df['Age']>31) & (dfx['Age']<40),'Age']=3

dfx.loc[(df['Age']>40) & (dfx['Age']<80),'Age']=4

dfx.loc[df['Age']>60,'Age']=5
dt.loc[dt['Age']<=5,'Age']=0

dt.loc[(dt['Age']>5) & (dt['Age']<20),'Age']=1

dt.loc[(dt['Age']>20) & (dt['Age']<31),'Age']=2

dt.loc[(dt['Age']>31) & (dt['Age']<40),'Age']=3

dt.loc[(dt['Age']>40) & (dt['Age']<80),'Age']=4

dt.loc[dt['Age']>60,'Age']=5
dfx.head(3)
dt.head(3)
dfx['Embarked'].isnull().sum()
dt['Embarked'].isnull().sum()
df=dfx.copy()
Pclass1 = df[df['Pclass']==1]['Embarked'].value_counts()

Pclass2 = df[df['Pclass']==2]['Embarked'].value_counts()

Pclass3 = df[df['Pclass']==3]['Embarked'].value_counts()

dfy = pd.DataFrame([Pclass1, Pclass2, Pclass3])

dfy.index = ['1st class','2nd class', '3rd class']

dfy.plot(kind='pie',subplots=True,stacked=True, figsize=(15,5))
df['Embarked']=df['Embarked'].fillna('S')
df['Embarked'].isnull().sum()
ohe=OneHotEncoder(handle_unknown='ignore')
df['Embarked'].value_counts()
df['Embarked']=le.fit_transform(df['Embarked'])
dt['Embarked']=le.fit_transform(dt['Embarked'])
df['Embarked'].value_counts()
dt['Embarked'].value_counts()
df.head(3)
df['Fare'].isnull().sum()
dt['Fare'].isnull().sum()

print('Average Fare was:',dt['Fare'].mean())
print('Highest Fare was:',df['Fare'].max())

print('Lowest Fare was:',df['Fare'].min())

print('Average Fare was:',df['Fare'].mean())

print('Average Fare was:',df['Fare'].median())
df['Fare'].fillna(14.4542,inplace=True)
dt['Fare'].fillna(14.4542,inplace=True)
facet=sns.FacetGrid(df,hue='Survived',aspect=4)

facet.map(sns.kdeplot,'Fare',shade=True)

facet.set(xlim=(0,df['Fare'].max()))

facet.add_legend()

plt.show()
facet=sns.FacetGrid(df,hue='Survived',aspect=4)

facet.map(sns.kdeplot,'Fare',shade=True)

facet.set(xlim=(0,20))

facet.add_legend()

plt.show()
facet=sns.FacetGrid(df,hue='Survived',aspect=4)

facet.map(sns.kdeplot,'Fare',shade=True)

facet.set(xlim=(0,30))

facet.add_legend()

plt.show()
df3=df.copy()
df3.loc[df3['Fare']<=5,'Fare']=0

df3.loc[(df3['Fare']>5) & (df3['Fare']<=10),'Fare']=1

df3.loc[(df3['Fare']>10) & (df3['Fare']<=20),'Fare']=2

df3.loc[df3['Fare']>20,'Fare']=3
dt.loc[dt['Fare']<=5,'Fare']=0

dt.loc[(dt['Fare']>5) & (dt['Fare']<=10),'Fare']=1

dt.loc[(dt['Fare']>10) & (dt['Fare']<=20),'Fare']=2

dt.loc[dt['Fare']>20,'Fare']=3
df3.head(3)
dt.head(3)
df3['Fare'].value_counts()
dt['Fare'].value_counts()
df3['Cabin']=df3['Cabin'].str[:1]
dt['Cabin']=dt['Cabin'].str[:1]
df3['Cabin'].value_counts()
dt['Cabin'].value_counts()
Pclass1 = df3[df3['Pclass']==1]['Cabin'].value_counts()

Pclass2 = df3[df3['Pclass']==2]['Cabin'].value_counts()

Pclass3 = df3[df3['Pclass']==3]['Cabin'].value_counts()

dfPclass = pd.DataFrame([Pclass1, Pclass2, Pclass3])

dfPclass.index = ['1st class','2nd class', '3rd class']

dfPclass.plot(kind='pie',subplots=True,stacked=True, figsize=(30,20))
df3['Cabin'].isnull().sum()
dt['Cabin'].isnull().sum()
cabin_mapping = {"A": 0, "B": 0.4, "C": 0.8, "D": 1.2, "E": 1.6, "F": 2, "G": 2.4, "T": 2.8}

df3['Cabin']=df['Cabin'].map(cabin_mapping)
df3['Cabin'].value_counts()
df3.drop('Cabin',axis=1)
dt.drop('Cabin',axis=1)
df4=df3.copy()
dt11=dt.copy()
df4['FamilySize']=df4['Parch']+df4['SibSp']+1
dt['FamilySize']=dt['Parch']+dt['SibSp']+1
df4.head(3)
dt.head(2)
facet=sns.FacetGrid(df4,hue='Survived',aspect=4)

facet.map(sns.kdeplot,'FamilySize',shade=True)

facet.set(xlim=(0,df4['FamilySize'].max()))

facet.add_legend()

plt.show()
facet=sns.FacetGrid(df4,hue='Survived',aspect=4)

facet.map(sns.kdeplot,'FamilySize',shade=True)

facet.set(xlim=(0,2))

facet.add_legend()

plt.show()
family_mapping = {1: 0, 2: 0.4, 3: 0.8, 4: 1.2, 5: 1.6, 6: 2, 7: 2.4, 8: 2.8, 9: 3.2, 10: 3.6, 11: 4}

df4['FamilySize']=df4['FamilySize'].map(family_mapping)
dt['FamilySize']=dt['FamilySize'].map(family_mapping)
df4.head(3)
dt.head(2)
features_drop=['Ticket','SibSp','Parch','Cabin']
df4=df4.drop(features_drop,axis=1)
dt=dt.drop(features_drop,axis=1)
df4.head(3)
df5=df4.copy()
dt.head(3)
dt.drop('PassengerId',axis=1)
dt.head(2)
dt1=dt.drop('PassengerId',axis=1)
dt1.head(2)
df5.head(3)
x1=df5.drop('Survived',axis=1)
x2=x1.drop('PassengerId',axis=1)
x2.head(3)
y=df4['Survived']
y.head(3)
x2.info()
dt1.info()
k_fold=KFold(n_splits=10,shuffle=True,random_state=42)
estimators=[10,20,50,70,120,300,500,1200,1350,1500,4000]
for i in estimators:

    rf=RandomForestClassifier(n_estimators=i)

    scoring = 'accuracy'

    score = cross_val_score(rf,x2,y,cv=k_fold, n_jobs=1, scoring=scoring)

    print('The score for ',i,' :')

    print(round(np.mean(score)*100,2))
rf=RandomForestClassifier(n_estimators=70)
rf.fit(x2,y)
predict_21=rf.predict(dt1)
submission = pd.DataFrame({

        "PassengerId": dt["PassengerId"],

        "Survived": predict_21

    })
submission.to_csv('submission_21.csv',index=False)
df11=df3.copy()
df3.head(1)
drop_features_df3=['Ticket','Cabin','PassengerId']

df12=df11.drop(drop_features_df3,axis=1)

dt12=dt11.drop(drop_features_df3,axis=1)
df12.head(2)
df13=df12.copy()
dt12.head(2)
y11=df12['Survived']

y11.head(2)
x11=df13.drop('Survived',axis=1)

x11.head(2)