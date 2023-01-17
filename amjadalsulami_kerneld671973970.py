# data analysis 

import pandas as pd

import numpy as np

 

# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

sns.set(style="whitegrid")

# preprocessing

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.preprocessing import StandardScaler

# algorithm of machine learning and evaluation the models

from sklearn.model_selection import train_test_split #to create validation data set

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score 
# get training and testing data.

df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')

# save the ids for later to use in the test

Id_passenger=df_test['PassengerId'] 
#check train df

df_train.head()
#check trest df

df_test.head()
#check train,test df keys

print(df_train.keys())

print(df_test.keys())
#check null values

print('Train null value')

print(df_train.isnull().sum())

print('TEST null value')

print(df_test.isnull().sum())
#check train,test df shapes

df_train.shape,df_test.shape
#check train,test df dtypes

print(df_train.dtypes)

print(df_test.dtypes)
# get Summarie and statistics

df_train.describe()
df_train=df_train.drop(['Cabin','Ticket'],axis=1)

df_test=df_test.drop(['Cabin','Ticket'],axis=1)
sns.kdeplot(df_train['Age'], shade=True);
df_train['Age']=df_train['Age'].fillna(df_train['Age'].median())

df_test['Age']=df_test['Age'].fillna(df_test['Age'].median())
df_train['Age'].isnull().sum(),df_test['Age'].isnull().sum()
sns.kdeplot(df_test['Fare'], shade=True);
df_test['Fare']=df_test['Fare'].fillna(df_test['Fare'].median())
df_train['Fare'].isnull().sum(),df_test['Fare'].isnull().sum()

print('Train null value')

print(df_train.isnull().sum())

print('TEST null value')

print(df_test.isnull().sum())
df_train['Embarked']=df_train['Embarked'].fillna('Q')

# Heatmap to see correlation between numerical values (SibSp Parch Age and Fare values) and Survived 

fig, ax = plt.subplots(figsize=(10, 8))

corr = df_train[["Survived","SibSp","Parch","Age","Fare"]].corr()

sns.heatmap(corr, cmap=sns.diverging_palette(220, 10, as_cmap=True),square=True, ax=ax)
parchXsurvived = sns.barplot(x="Parch",y="Survived",data=df_train)
sibSPXsurvived = sns.barplot(x="SibSp",y="Survived",data=df_train)
ageXsurvived = sns.kdeplot(df_train['Age'][(df_train["Survived"] == 1)])





ageXsurvived = sns.kdeplot(df_train['Age'][(df_train["Survived"] == 0)])



ageXsurvived=ageXsurvived.legend(['Survived',"Not Survived"])
sexXsurvived = sns.barplot(x="Sex",y="Survived",data=df_train)
df_train[["Sex","Survived"]].groupby('Sex').mean()
sns.barplot(x="Pclass", y="Survived", data=df_train);
sns.barplot(x="Pclass", y="Survived", hue="Sex", data=df_train);
sns.barplot(x="Embarked", y="Survived",data=df_train);
sns.catplot("Pclass", col="Embarked",  data=df_train,kind="count");
#set up a labelencoder

label_train_sex= LabelEncoder()

# convert columns [Sex,Embarked] to numerical data

df_train['Sex']=label_train_sex.fit_transform(df_train['Sex'])

#set up a labelencoder

label_train_embarked= LabelEncoder()

df_train['Embarked']=label_train_embarked.fit_transform(df_train['Embarked'])
#set up a labelencoder

label_test_sex= LabelEncoder()

# convert columns [Sex,Embarked] to numerical data

df_test['Sex']=label_test_sex.fit_transform(df_test['Sex'])

#set up a labelencoder

label_test_embarked= LabelEncoder()

df_test['Embarked']=label_test_embarked.fit_transform(df_test['Embarked'])
# start creating family_member_no" column in train and test df by adding 2 columns values SibSp+Parch + 1 (which is the current passenger)

df_train["family_member_no"] = df_train["SibSp"] + df_train["Parch"] + 1

df_test["family_member_no"] = df_test["SibSp"] + df_test["Parch"] + 1
# if passenger is single traveller we might get more accurate info because its a general case so I will set single traveller as 1

df_train["family_member_no"] = df_train["SibSp"] + df_train["Parch"] + 1



df_test["family_member_no"] = df_test["SibSp"] + df_test["Parch"] + 1
# I will get this info from family_member_no and i will use apply and lambda to do it

df_train["Single"] = df_train.family_member_no.apply(lambda a: 1 if a == 1 else 0)

df_test["Single"] = df_test.family_member_no.apply(lambda a: 1 if a == 1 else 0)

#Inshalize standerscaler 

ss= StandardScaler()

# change value as array and reshap them after that to pass them to the scaler fit_transform method

age_tr = np.array(df_train["Age"]).reshape(-1, 1)

fare_tr = np.array(df_train["Fare"]).reshape(-1, 1)

age_ts = np.array(df_test["Age"]).reshape(-1, 1)

fare_ts = np.array(df_test["Fare"]).reshape(-1, 1)

# fit_and transform column value and reduce their magnitude using ss

df_train["Age"] = ss.fit_transform(age_tr)

df_train["Fare"]= ss.fit_transform(fare_tr )

df_test["Age"]["Age"] = ss.fit_transform(age_ts)

df_test["Fare"] = ss.fit_transform(fare_ts)

# drop name feature from the 2 df

df_train = df_train.drop('Name', axis=1) 

df_test = df_test.drop('Name', axis=1)
df_train.head()
df_test.head()
#define features for  training and test

X_train = df_train.drop(["PassengerId", "Survived"], axis=1) 

y_train = df_train["Survived"]  

X_test = df_test.drop("PassengerId", axis=1) 
X_train.head()
# to ensure that model doesn't overfit with the data  train_test_split well be used. 

X_for_train, X_for_test, y_for_train, y_for_test= train_test_split(X_train, y_train, test_size=0.3, random_state=42) 

# RandomForestClassifier

rd_f = RandomForestClassifier(n_estimators=20, criterion='entropy',random_state=42)

rd_f.fit(X_for_train, y_for_train)

predictions = rd_f.predict(X_for_test)



rd_f_model=accuracy_score(y_for_test,predictions)

print("RandomForestClassifier accurecy score: " ,rd_f_model)
# LogisticRegression

log_model= LogisticRegression()

log_model.fit(X_for_train, y_for_train)

predictions = log_model.predict(X_for_test)

acc_log_model=accuracy_score(y_for_test,predictions)

print("Logistic Regression score: " ,acc_log_model)
#KNeighborsClassifier

clf=KNeighborsClassifier(p=2, n_neighbors=10)

clf.fit(X_for_train,y_for_train)

predictions=clf.predict(X_for_test)

print('KNeighborsClassifier score: ',accuracy_score(y_for_test,predictions))
#Supply or submit the results.

final_prediction= pd.Series(rd_f.predict(X_test), name="Survived")



results = pd.concat([Id_passenger,final_prediction],axis=1)



results.to_csv("results.csv",index=False)