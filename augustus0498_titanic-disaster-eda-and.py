import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import os

print(os.listdir("../input"))
df=pd.read_csv('../input/train.csv')
df.head()
df['Embarked'] = df['Embarked'].replace('C','Cherbourg')

df['Embarked'] = df['Embarked'].replace('S','Southampton')

df['Embarked'] = df['Embarked'].replace('Q','Queenstown')
df.describe()
df.info()
df.isnull().any()
sns.countplot(df['Embarked'])
df['Embarked'] = df['Embarked'].fillna('Southampton')
Sex = pd.get_dummies(df.Sex)

Sex.drop(Sex.columns[1],axis=1,inplace=True)

df = pd.concat([df,Sex],axis = 1)



Embarked = pd.get_dummies(df.Embarked)

Embarked.drop(Embarked.columns[1],axis=1,inplace=True)

df = pd.concat([df,Embarked],axis = 1)
df.describe()
sns.countplot(x='Survived',hue='Sex',data=df)
sns.barplot(x='Survived',y='Age',hue='Sex',data=df)
def bar_chart(feature):

   survived = df[df['Survived']==1][feature].value_counts()

   dead = df[df['Survived']==0][feature].value_counts()

   df1 = pd.DataFrame([survived,dead])

   df1.index = ['Survived','Dead']

   df1.plot(kind='bar',stacked=True, figsize=(5,5))
sns.countplot(x=df['Survived'],hue=df['SibSp'])
sns.countplot(x=df['Survived'],hue=df['Parch'])
sns.countplot(x=df['Survived'],hue=df['Pclass'])
sns.countplot(x=df['Survived'],hue=df['Embarked'])
sns.catplot(data = df, x ="Survived", y = "Age", 

               col = 'Pclass', # per store type in cols

               hue = 'female',

              sharex=False)
print('Class 1 females')

print('total',df[(df['Pclass']==1) & (df['female']==1)]['Survived'].count())

print('survived',df[(df['Pclass']==1) & (df['female']==1)]['Survived'].sum())



print('\n')

print('Class 2 females')

print('total',df[(df['Pclass']==2) & (df['female']==1)]['Survived'].count())

print('survived',df[(df['Pclass']==2) & (df['female']==1)]['Survived'].sum())



print('\n')

print('Class 3 females')

print('total',df[(df['Pclass']==3) & (df['female']==1)]['Survived'].count())

print('survived',df[(df['Pclass']==3) & (df['female']==1)]['Survived'].sum())
print('Class 1 children')

print('total',df[(df['Pclass']==1) & (df['Age']<=10)]['Survived'].count())

print('survived',df[(df['Pclass']==1) & (df['Age']<=10)]['Survived'].sum())

print('\n')

print('Class 2 children')

print('total',df[(df['Pclass']==2) & (df['Age']<=10)]['Survived'].count())

print('survived',df[(df['Pclass']==2) & (df['Age']<=10)]['Survived'].sum())

print('\n')

print('Class 3 children')

print('total',df[(df['Pclass']==3) & (df['Age']<=10)]['Survived'].count())



print('survived',df[(df['Pclass']==3) & (df['Age']<=10)]['Survived'].sum())
print('Cherbourg females')

print('total',df[(df['Embarked']=='Cherbourg') & (df['female']==1)]['Survived'].count())

print('survived',df[(df['Embarked']=='Cherbourg') & (df['female']==1)]['Survived'].sum())



print('\n')

print('Southampton females')

print('total',df[(df['Embarked']=='Southampton') & (df['female']==1)]['Survived'].count())

print('survived',df[(df['Embarked']=='Southampton') & (df['female']==1)]['Survived'].sum())



print('\n')

print('Queenstown females')

print('total',df[(df['Embarked']=='Queenstown') & (df['female']==1)]['Survived'].count())

print('survived',df[(df['Embarked']=='Queenstown') & (df['female']==1)]['Survived'].sum())
emdf =df

emdf['Embarked'] = emdf['Embarked'].replace('Cherbourg',0)

emdf['Embarked'] = emdf['Embarked'].replace('Southampton',1)

emdf['Embarked'] = emdf['Embarked'].replace('Queenstown',2)
sns.catplot(data = df, x ="Survived", y = "Age", 

               col = 'Pclass', # per store type in cols

               hue = 'female',

               row = 'Embarked',

               sharex=False)
sns.distplot(df['Age'].dropna())
df['Age'] = df['Age'].fillna(np.mean(df['Age']))
sns.distplot(df['Age'].dropna())
sns.catplot(data = df, x ="Survived", y = "Age", sharex=False)
sns.catplot(data = df[(df['female']==1) | (df['Age']<=10)], x ="Survived", y = "Age", sharex=False)
sns.distplot(df['Fare'])
df['Fare'] = df['Fare'].replace(0,np.median(df['Fare']))
plt.hist(df['Fare'])
sns.countplot(x=df['Fare'],hue=df['Pclass'])
sns.distplot(np.log(df['Fare']))
ndf = df

ndf.info()
ndf.describe()
pd.crosstab([df.Embarked,df.Pclass],[df.Sex,df.Survived],margins=True).style.background_gradient(cmap='Blues')
plt.figure(figsize=(10,7))

sns.heatmap(ndf.corr(),cmap='Blues',annot=True)
tedf = pd.read_csv('../input/test.csv')

tedf.isnull().any()
tedf.info()
tedf['Embarked'] = tedf['Embarked'].replace('C','Cherbourg')

tedf['Embarked'] = tedf['Embarked'].replace('S','Southampton')

tedf['Embarked'] = tedf['Embarked'].replace('Q','Queenstown')
Sex = pd.get_dummies(tedf.Sex)

Sex.drop(Sex.columns[1],axis=1,inplace=True)

tedf = pd.concat([tedf,Sex],axis = 1)



Embarked = pd.get_dummies(tedf.Embarked)

Embarked.drop(Embarked.columns[1],axis=1,inplace=True)

tedf = pd.concat([tedf,Embarked],axis = 1)
tedf.isnull().any()
features = ['Pclass','Age','Fare','female','Cherbourg','Southampton']
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn import metrics
def conflog(features):

    X = ndf[features]

    y = ndf.Survived

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=125)

    linreg = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial')

    linreg.fit(X_train,y_train)

    y_pred = linreg.predict(X_test)   

    print(metrics.classification_report(y_test,y_pred))

    print(metrics.accuracy_score(y_test,y_pred))
print(conflog(features))
print(conflog(['Pclass','female','Cherbourg','Southampton']))
from sklearn.naive_bayes import GaussianNB
def confNB(features):

    X = ndf[features]

    y = ndf.Survived

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=125)

    nb = GaussianNB()

    nb.fit(X_train,y_train)

    y_pred = nb.predict(X_test)   

    print(metrics.classification_report(y_test,y_pred))

    print(metrics.accuracy_score(y_test,y_pred))
print(confNB(features))
print(confNB(['Pclass','female','Cherbourg','Southampton']))
from sklearn.tree import DecisionTreeClassifier
def confDT(features):

    X = ndf[features]

    y = ndf.Survived

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=125)

    dt = DecisionTreeClassifier()

    dt.fit(X_train,y_train)

    y_pred = dt.predict(X_test)   

    print(metrics.classification_report(y_test,y_pred))

    print(metrics.accuracy_score(y_test,y_pred))
print(confDT(features))
print(confDT(['Pclass','female','Cherbourg','Southampton']))
from sklearn.svm import SVC
def confsvm(features):

    X = ndf[features]

    y = ndf.Survived

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=125)

    svmf = SVC(gamma='scale')

    svmf.fit(X_train,y_train)

    y_pred = svmf.predict(X_test)   

    print(metrics.classification_report(y_test,y_pred))

    print(metrics.accuracy_score(y_test,y_pred))
print(confsvm(features))
print(confsvm(['Pclass','female','Cherbourg','Southampton']))
X_train = ndf[['Pclass','female','Cherbourg','Southampton']]

y_train = ndf.Survived

X_test = tedf[['Pclass','female','Cherbourg','Southampton']]

svmf = SVC(gamma='scale')

svmf.fit(X_train,y_train)

y_pred = svmf.predict(X_test)
y_pred