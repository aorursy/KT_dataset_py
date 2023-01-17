import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

df=pd.read_csv("../input/train.csv")

test=pd.read_csv("../input/test.csv")
print(df.head(5))

print(df.columns)

print(df.describe(include='all'))

print(df.isnull().sum())
survived_class = df[df['Survived']==1]['Pclass'].value_counts()

dead_class = df[df['Survived']==0]['Pclass'].value_counts()

df_class=pd.DataFrame([survived_class,dead_class])

df_class.index=['Survived','Dead']

df_class.columns=['Class 1','Class 2','Class 3']

print(df_class)

df_class.plot(kind='bar')

plt.ylabel('No. of people',size=15,color='green')

plt.xlabel('Survival',size=20,color='blue')

plt.show()

Class1_survived= df_class.iloc[0,0]/df_class.iloc[:,0].sum()*100

Class2_survived = df_class.iloc[0,1]/df_class.iloc[:,1].sum()*100

Class3_survived = df_class.iloc[0,2]/df_class.iloc[:,2].sum()*100

print('Percentage of Class1 passenger survived is ',round(Class1_survived),'%')

print('Percentage of Class2 passenger survived is ',round(Class2_survived),'%')

print('Percentage of Class3 passenger survived is ',round(Class3_survived),'%')
survived_gender=df[df['Survived']==1]['Sex'].value_counts()

dead_gender=df[df['Survived']==0]['Sex'].value_counts()

df_gender=pd.DataFrame([survived_gender,dead_gender])

df_gender.columns=['Survived','Dead']

df_gender.index=['Female','Male']

print(df_gender)

df_gender.plot(kind='bar')

plt.ylabel('No. of people',size=15,color='green')

plt.xlabel('Sex',size=20,color='blue')

plt.show()

female_survived=df_gender.iloc[0,0]/df_gender.iloc[0,:].sum()*100

male_survived=df_gender.iloc[1,0]/df_gender.iloc[1,:].sum()*100

print('Percentage of male passengers survived is ',round(male_survived),'%')

print('Percentage of female passengers survived is ',round(female_survived),'%')
bins = [ 0, 5, 12, 18, 24, 35, 60, np.inf]

labels = [ 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']

df['AgeGroup'] = pd.cut(df["Age"], bins, labels = labels)

print(df[['AgeGroup','Survived']].groupby(['AgeGroup'],as_index=False).mean())

sns.barplot(x="AgeGroup",y="Survived",data=df)

plt.xlabel('AgeGroup',color='blue',size=18)

plt.ylabel('Survival Rate',color='green',size=18)

plt.title('Age vs Survival Rate',color='Black',size=20)

plt.show()
bins = [0,100,250,600]

labels=['Economic Class','Business Class','First Class']

df['Class']=pd.cut(df['Fare'],bins,labels=labels)

print(df[['Class','Survived']].groupby(['Class'],as_index=False).mean())

sns.barplot(x="Class",y="Survived",data=df)

plt.xlabel('Class',color='blue',size=18)

plt.ylabel('Survival Rate',color='green',size=18)

plt.title('First Class Passengers got the maximum survival rate',color='Black',size=20)

plt.show()
print(len(df['Cabin']))
df=df.drop(['Cabin'],axis=1)

test=test.drop(['Cabin'],axis=1)

df.columns
print("Number of people embarking in Southampton (S):")

southampton = df[df["Embarked"] == "S"].shape[0]

print(southampton)



print("Number of people embarking in Cherbourg (C):")

cherbourg = df[df["Embarked"] == "C"].shape[0]

print(cherbourg)



print("Number of people embarking in Queenstown (Q):")

queenstown = df[df["Embarked"] == "Q"].shape[0]

print(queenstown)
df=df.drop(['Ticket'],axis=1)

test=test.drop(['Ticket'],axis=1)

test.columns
#create a combined group of both datasets

combine = [df, test]



#extract a title for each Name in the train and test datasets

for dataset in combine:

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)



print(pd.crosstab(df['Title'], df['Sex']))
for dataset in combine:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\

 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')



    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    

df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
#Filling missing values

df['Age'] = df.groupby(['Title'])['Age'].transform(lambda x: x.fillna(x.mean()))

test['Age'] = test.groupby(['Title'])['Age'].transform(lambda x: x.fillna(x.mean()))



#AgeCategories

df['Age'] = df['Age'].astype(int)

test['Age']    = test['Age'].astype(int)



df.loc[ df['Age'] <= 16, 'Age'] = 0

df.loc[(df['Age'] > 16) & (df['Age'] <= 32), 'Age'] = 1

df.loc[(df['Age'] > 32) & (df['Age'] <= 48), 'Age'] = 2

df.loc[(df['Age'] > 48) & (df['Age'] <= 64), 'Age'] = 3

df.loc[(df['Age'] > 64), 'Age'] = 4



test.loc[ test['Age'] <= 16, 'Age'] = 0

test.loc[(test['Age'] > 16) & (test['Age'] <= 32), 'Age'] = 1

test.loc[(test['Age'] > 32) & (test['Age'] <= 48), 'Age'] = 2

test.loc[(test['Age'] > 48) & (test['Age'] <= 64), 'Age'] = 3

test.loc[(test['Age'] > 64), 'Age'] = 4

df['Age'].head(5)
for dataset in combine:

    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)



test.head(5)
df=df.drop(['Name'],axis=1)

test=test.drop(['Name'],axis=1)

df.columns
df.drop('AgeGroup',axis=1,inplace=True)
df = pd.concat([df.drop('Sex', axis=1), pd.get_dummies(df['Sex'])], axis=1)

test = pd.concat([test.drop('Sex', axis=1), pd.get_dummies(test['Sex'])], axis=1)

test.head(5)
df.drop('Class',axis=1,inplace=True)

df.head()
df['Embarked'].replace({'S':1,'C':2,'Q':3},inplace=True)

df['Embarked']=df['Embarked'].fillna(1)

test['Embarked'].replace({'S':1,'C':2,'Q':3},inplace=True)

test['Embarked']=test['Embarked'].fillna(1)



test.head(5)
#fill in missing Fare value in test set based on mean fare for that Pclass 

for x in range(len(test["Fare"])):

    if pd.isnull(test["Fare"][x]):

        pclass = test["Pclass"][x] #Pclass = 3

        test["Fare"][x] = round(df[df["Pclass"] == pclass]["Fare"].mean(), 4)

        

#map Fare values into groups of numerical values

df['FareBand'] = pd.qcut(df['Fare'], 4, labels = [1, 2, 3, 4])

test['FareBand'] = pd.qcut(test['Fare'], 4, labels = [1, 2, 3, 4])



#drop Fare values

df = df.drop(['Fare'], axis = 1)

test = test.drop(['Fare'], axis = 1)
df=df.drop(['Title'],axis=1)

test=test.drop(['Title'],axis=1)

test.columns
from sklearn.model_selection import train_test_split



predictors=df.drop(['Survived','PassengerId'],axis=1)

target=df['Survived']

x_train,x_cv,y_train,y_cv=train_test_split(predictors,target,test_size=0.35,random_state=0)
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score



knn = KNeighborsClassifier()

knn.fit(x_train, y_train)

y_pred = knn.predict(x_cv)

acc_knn = round(accuracy_score(y_pred,y_cv) * 100, 2)

print(acc_knn)
# Logistic Regression

from sklearn.linear_model import LogisticRegression



logreg = LogisticRegression()

logreg.fit(x_train, y_train)

y_pred = logreg.predict(x_cv)

acc_logreg = round(accuracy_score(y_pred, y_cv) * 100, 2)

print(acc_logreg)
# Random Forest

from sklearn.ensemble import RandomForestClassifier



randomforest = RandomForestClassifier()

randomforest.fit(x_train, y_train)

y_pred = randomforest.predict(x_cv)

acc_randomforest = round(accuracy_score(y_pred, y_cv) * 100, 2)

print(acc_randomforest)
# Support Vector Machines

from sklearn.svm import SVC



svc = SVC()

svc.fit(x_train, y_train)

y_pred = svc.predict(x_cv)

acc_svc = round(accuracy_score(y_pred, y_cv) * 100, 2)

print(acc_svc)
models = pd.DataFrame({

    'Method': ['KNN', 'Logistic Regression', 

              'Random Forest', 'Support Vector Machine'],

    'Score': [acc_knn, acc_logreg, 

              acc_randomforest, acc_svc]})

models.sort_values(by='Score', ascending=False)
from sklearn.neural_network.multilayer_perceptron import MLPClassifier

svc = MLPClassifier()

svc.fit(x_train, y_train)

y_pred = svc.predict(x_cv)

acc_svc = round(accuracy_score(y_pred, y_cv) * 100, 2)

print(acc_svc)

from sklearn.neighbors import NearestCentroid



svc = NearestCentroid()

svc.fit(x_train, y_train)

y_pred = svc.predict(x_cv)

acc_svc = round(accuracy_score(y_pred, y_cv) * 100, 2)

print(acc_svc)

svc = SVC()

svc.fit(x_train, y_train)

y_pred = svc.predict(test.drop('PassengerId',axis=1))

print(y_pred)
submission = pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':y_pred})

submission.head(5)
filename = 'TPredictions.csv'



submission.to_csv(filename,index=False)



print('Saved file: ' + filename)