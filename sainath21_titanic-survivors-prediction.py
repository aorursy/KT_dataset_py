import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

import seaborn as sns

sns.set_style('whitegrid')

%matplotlib inline



from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC
train_df=pd.read_csv('../input/train.csv')

test_df=pd.read_csv('../input/test.csv')



titanic_df=pd.concat([train_df,test_df])

titanic_df.head()
titanic_df.describe()
titanic_df.isnull().any()
sns.heatmap(data=titanic_df.isnull(),cmap='viridis')

titanic_df.isnull().describe()
titanic_df['Ticket'].describe()
titanic_df=titanic_df.drop(['PassengerId','Cabin','Name','Ticket'],axis=1)

titanic_df['Embarked'].describe()

titanic_df['Embarked'].fillna('S',inplace=True)

titanic_df['Fare'].fillna(titanic_df['Fare'].mean(),inplace=True)

titanic_df['Age'].fillna(np.random.randint(titanic_df['Age'].min(),titanic_df['Age'].max()),inplace=True)



titanic_df['Age']=titanic_df['Age'].astype(int)

titanic_df['Fare']=titanic_df['Fare'].astype(int)
fg,(ax1,ax2,ax3)=plt.subplots(1,3,figsize=(15,5))

sns.countplot(titanic_df['Sex'],ax=ax1)

ax1.set_title('Number of males/females boarded')

sns.countplot(x='Sex',hue='Survived',data=titanic_df,ax=ax2)

ax2.set_title('Number of males and females survived/not survived')

sns.barplot('Sex','Survived',data=titanic_df,ax=ax3)

ax3.set_title('Number of males and females survived(mean)')



encoder=LabelEncoder()

titanic_df['Sex']=encoder.fit_transform(titanic_df['Sex'])
fg,(ax1,ax2,ax3)=plt.subplots(1,3,figsize=(15,5))

sns.countplot(titanic_df['Embarked'],ax=ax1)

sns.countplot(x='Embarked',hue='Survived',data=titanic_df,ax=ax2)

sns.barplot('Embarked','Survived',data=titanic_df,ax=ax3)



dummies=pd.get_dummies(titanic_df['Embarked'])

titanic_df['Embarked=C']=dummies['C']

titanic_df['Embarked=Q']=dummies['Q']

titanic_df.drop(['Embarked'],axis=1,inplace=True)
fg,(ax1,ax2,ax3)=plt.subplots(1,3,figsize=(15,5))

sns.countplot(titanic_df['Pclass'],ax=ax1)

sns.countplot(x='Pclass',hue='Survived',data=titanic_df,ax=ax2)

sns.barplot('Pclass','Survived',data=titanic_df,ax=ax3)



dummies=pd.get_dummies(titanic_df['Pclass'])

titanic_df['Pclass=1']=dummies[1]

titanic_df['Pclass=2']=dummies[2]

titanic_df.drop(['Pclass'],axis=1,inplace=True)
titanic_df['AgeGroup']=pd.cut(titanic_df['Age'],5)

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(15,5))

sns.countplot(titanic_df['AgeGroup'],ax=ax1)

sns.countplot(x='AgeGroup',hue='Survived',data=titanic_df,ax=ax2)

titanic_df['isYoung'] = np.where(titanic_df["Age"]<16,1,0)

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(15,5))

sns.countplot(titanic_df['isYoung'],ax=ax1)

sns.countplot(x='isYoung',hue='Survived',data=titanic_df,ax=ax2)



titanic_df.drop(['Age','AgeGroup'],axis=1,inplace=True)
titanic_df['Fare'].describe()
titanic_df['FareGroup']= None

titanic_df.loc[(titanic_df['Fare']<7),'FareGroup']=3

titanic_df.loc[(titanic_df['Fare']>=7) & (titanic_df['Fare']<31),'FareGroup']=2

titanic_df.loc[(titanic_df['Fare']>=31),'FareGroup']=1

sns.countplot('FareGroup',hue='Survived',data=titanic_df)

titanic_df.drop(['Fare'],axis=1,inplace=True)
titanic_df['FamilySize']=titanic_df['Parch']+titanic_df['SibSp']+1

sns.factorplot(x='FamilySize',y='Survived',hue='Sex',data=titanic_df,aspect=4)

titanic_df['SmallFamily']= None

titanic_df['isAlone']=0

titanic_df.loc[(titanic_df['FamilySize']==1),'isAlone']=1

titanic_df.loc[(titanic_df['FamilySize']<5)& (titanic_df['FamilySize']>=1),'SmallFamily']=1

titanic_df.loc[(titanic_df['FamilySize']>=5),'SmallFamily']=0

titanic_df.drop(['Parch','SibSp','FamilySize'],axis=1,inplace=True)
X_train=titanic_df.iloc[:891,:].drop(['Survived'],axis=1)

Y_train=titanic_df.iloc[:891,:]['Survived']

X_test=titanic_df.iloc[891:,:].drop(['Survived'],axis=1)
lm=LogisticRegression()

lm.fit(X_train,Y_train)

predictions1=lm.predict(X_test)

lm.score(X_train,Y_train)


rf=RandomForestClassifier(n_estimators=50)

rf.fit(X_train,Y_train)

predictions2= rf.predict(X_test)

rf.score(X_train, Y_train)
knn = KNeighborsClassifier(n_neighbors = 5)

knn.fit(X_train, Y_train)

predictions3= knn.predict(X_test)

knn.score(X_train, Y_train)
gaussian = GaussianNB()

gaussian.fit(X_train, Y_train)

predictions4= gaussian.predict(X_test)

gaussian.score(X_train, Y_train)
svc = SVC()

svc.fit(X_train, Y_train)

predictions5= svc.predict(X_test)

svc.score(X_train, Y_train)
submission = pd.DataFrame({

        "PassengerId":test_df["PassengerId"],

        "Survived": predictions2.astype(int)})

submission.to_csv('submission.csv', index=False)