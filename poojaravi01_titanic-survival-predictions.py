import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression

from sklearn import svm

from sklearn.neighbors import KNeighborsClassifier

import seaborn as sns

from xgboost import XGBClassifier

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn import metrics

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
dftrain=pd.read_csv('../input/titanic/train.csv')

dftest=pd.read_csv('../input/titanic/test.csv')
dftrain.head()
dftest.head()
sns.countplot(data=dftrain,x='Survived',palette='Set2')
sns.countplot(data=dftrain,x='Survived',hue='Sex',palette='Set2')
sns.countplot(data=dftrain,x='Survived',hue='Pclass',palette='Set2')
sns.countplot(data=dftrain,x='Survived',hue='Embarked',palette='Set2')
plt.figure(figsize=(10,5))

plt.subplot(2,2,1)

sns.distplot(dftrain['Age'],bins=50)

plt.subplot(2,2,2)

sns.distplot(dftrain['Fare'],bins=50,color='red')
plt.figure(figsize=(15,10))

sns.heatmap(dftrain.corr(),annot=True,linewidth=0.2)
dftrain.isnull().sum()
dftrain.drop('Cabin',axis=1,inplace=True)
dftest.isnull().sum()
dftest.drop('Cabin',axis=1,inplace=True)
dftrain.head()
dftrain.Sex[dftrain.Sex=='male']=0

dftrain.Sex[dftrain.Sex=='female']=1

dftest.Sex[dftest.Sex=='male']=0

dftest.Sex[dftest.Sex=='female']=1
dftrain.Embarked[dftrain.Embarked=='S']=0

dftrain.Embarked[dftrain.Embarked=='C']=1

dftrain.Embarked[dftrain.Embarked=='Q']=2



dftest.Embarked[dftest.Embarked=='S']=0

dftest.Embarked[dftest.Embarked=='C']=1

dftest.Embarked[dftest.Embarked=='Q']=2

finalId=dftest['PassengerId']
dftrain.drop(['PassengerId','Ticket'],axis=1,inplace=True)

dftest.drop(['PassengerId','Ticket'],axis=1,inplace=True)
dftest['Fare']=dftest['Fare'].fillna(np.mean(dftest['Fare']))

dftest['Age']=dftest['Age'].fillna(np.mean(dftest['Age']))

dftrain['Age']=dftrain['Age'].fillna(np.mean(dftrain['Age']))

dftrain['Embarked']=dftrain['Embarked'].fillna(0)
dftrain.isna().sum()
dftest.isna().sum()
df_title = [i.split(",")[1].split(".")[0].strip() for i in dftrain["Name"]]

dftrain["Title"] = pd.Series(df_title)

dftrain["Title"] = dftrain["Title"].replace(['Lady', 'the Countess','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

dftrain["Title"] = dftrain["Title"].map({"Master":0, "Miss":1, "Ms" : 1 , "Mme":1, "Mlle":1, "Mrs":1, "Mr":2, "Rare":3})

dftrain.drop('Name',axis=1,inplace=True)

dftrain['Familysize']=dftrain['SibSp']+dftrain['Parch']+1

dftrain['Alone'] = dftrain['Familysize'].map(lambda s: 1 if s == 1 else 0)
df_title = [i.split(",")[1].split(".")[0].strip() for i in dftest["Name"]]

dftest["Title"] = pd.Series(df_title)

dftest["Title"] = dftest["Title"].replace(['Lady', 'the Countess','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

dftest["Title"] = dftest["Title"].map({"Master":0, "Miss":1, "Ms" : 1 , "Mme":1, "Mlle":1, "Mrs":1, "Mr":2, "Rare":3})

dftest.drop('Name',axis=1,inplace=True)

dftest['Familysize']=dftest['SibSp']+dftest['Parch']+1

dftest['Alone'] = dftest['Familysize'].map(lambda s: 1 if s == 1 else 0)
dftrain.head()
dftest.head()
X=dftrain.drop('Survived',axis=1)

y=dftrain['Survived']

Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,test_size=0.3,random_state=42)
scaler=StandardScaler()

Xtrainscaled=scaler.fit_transform(Xtrain)

Xtestscaled=scaler.transform(Xtest)

dftestscaled=scaler.transform(dftest)
model=LogisticRegression()

model.fit(Xtrainscaled,ytrain)

ypred=model.predict(Xtestscaled)

n=accuracy_score(ytest,ypred)

print('Accuracy of Logistic regression model: {}%'.format(round(n*100,2)))
mod=svm.SVC()

mod.fit(Xtrainscaled,ytrain)

y1pred=mod.predict(Xtestscaled)

sc=accuracy_score(ytest,y1pred)

print('Accuracy of SVM model: {}%'.format(round(sc*100,2)))
base=DecisionTreeClassifier(max_depth=4,min_samples_split=2,criterion='gini')

base.fit(Xtrainscaled,ytrain)

y2pred=base.predict(Xtestscaled)

scr=accuracy_score(ytest,y2pred)

print('Accuracy of decision tree classifier: {}%'.format(round(scr*100,2)))

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(ytest,y2pred)))
mdl=RandomForestClassifier()

mdl.fit(Xtrainscaled,ytrain)

yr=model.predict(Xtestscaled)

scc=accuracy_score(ytest,yr)

print('Accuracy of Random forest classifier: {}%'.format(round(scc*100,2)))
bs=DecisionTreeClassifier(max_depth=4,min_samples_split=2,criterion='entropy',splitter='best')

gb=XGBClassifier(base_estimator=bs,n_estimators=200,learning_rate=0.001,max_depth=3)

gb.fit(Xtrainscaled,ytrain)

ypred=gb.predict(Xtestscaled)

scor=accuracy_score(ytest,ypred)

print('Accuracy score: {}%'.format(round(scor,2)*100))
print(confusion_matrix(ytest,ypred))
print(classification_report(ytest,ypred))
prediction=gb.predict(dftestscaled)

answerdf=pd.DataFrame({'PassengerId':finalId,'Survived':prediction})

answerdf.to_csv('TitanicSubmission.csv',index=False)