import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

sns.set_style("darkgrid")
#load the data using pandas

df_train= pd.read_csv("/kaggle/input/titanic/train.csv")

df_test=pd.read_csv("/kaggle/input/titanic/train.csv")

df_train.head()
#what are the features

df_train.info()
#central tendency of data. Nan incase of categorical data

df_train.describe(include="all")
#checking dimensions

df_train.shape
#checking empty cells

df_train.isnull().sum()
data=df_train

#plotting survival rate w.r.t gender

sns.barplot(x='Sex',y='Survived',data=data)

plt.show();
#plotting survival rate w.r.t Pclass

sns.barplot(x='Pclass',y='Survived',data=data)

plt.show();
#plotting survival rate w.r.t SibSp

sns.barplot(x='SibSp',y='Survived',data=data)

plt.show();
#plotting survival rate w.r.t Parch

sns.barplot(x='Parch',y='Survived',data=data)

plt.show();


#sort the ages into logical categories



bins = [ 0, 5, 12, 18, 24, 35, 60, np.inf]

labels = [ 'infant', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']

df_train['AgeGroup'] = pd.cut(df_train["Age"], bins, labels = labels)

df_test['AgeGroup'] = pd.cut(df_test["Age"], bins, labels = labels)



#draw a bar plot of Age w.r.t survival

sns.set()

plt.figure(figsize = (12,6))

sns.barplot(x="AgeGroup", y="Survived", data=data,palette='magma')

plt.show()
df_train.columns
#First we will have a look at test data also

df_test.describe(include='all')
# We will start with dropping 'Cabin' column, as a lot of data is missing

df_train.drop("Cabin",axis=1,inplace=True)

df_test.drop("Cabin",axis=1,inplace=True)
# Count the number of values of Embarked feature to fill the empty spaces with mode

print(f'Number of people living in Southampton are(S) {df_train[df_train["Embarked"]=="S"].shape[0]}')

print(f'Number of people living in Cherbourg are(C) {df_train[df_train["Embarked"]=="C"].shape[0]}')

print(f'Number of people living in Queenstown are(Q) {df_train[df_train["Embarked"]=="Q"].shape[0]}')
# Majority people belong to S. so fillinf the empty cells

df_train.fillna({"Embarked":"S"},inplace=True)
#Name column, dropping it because it is useless in both datsets

df_train.drop("Name",axis=1,inplace=True)

df_test.drop("Name",axis=1,inplace=True)
#We will impute the mean in df_test for 'Fare' column

df_test['Fare']=df_test["Fare"].fillna(df_test["Fare"].mean())
# Fill the missing value of Age column with mean value.

df_train['Age'] = df_train['Age'].fillna(df_train.groupby('Sex')['Age'].transform('mean'))
df_test['Age'] = df_test['Age'].fillna(df_test.groupby('Sex')['Age'].transform('mean'))
df_test.isna().sum()
#Map the categorical column 'Sex' with numerical data

sex_mapping = {"male": 0, "female": 1}



df_train['Sex'] = df_train['Sex'].map(sex_mapping)

df_test['Sex'] = df_test['Sex'].map(sex_mapping)



df_train.head()
#Mapping for Embarked feature

embarked_mapping = {"S": 1, "C": 2, "Q": 3}

df_train['Embarked'] = df_train['Embarked'].map(embarked_mapping)

df_test['Embarked'] = df_test['Embarked'].map(embarked_mapping)



df_train.head()
df_train.isnull().sum()
#dropping AgeGroup from both datasets, as it was for data visualization purpose only

df_train.drop("AgeGroup",axis=1,inplace=True)
df_test.drop("AgeGroup",axis=1,inplace=True)
#dropping Ticket from both datasets, as its not needed

df_train.drop("Ticket",axis=1,inplace=True)
df_test.drop("Ticket",axis=1,inplace=True)
X= df_train.drop(["PassengerId","Survived"],axis=1) # Our samples

y= df_train["Survived"] # Our targets



X.shape,y.shape
from sklearn.model_selection import train_test_split



X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2)
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

np.random.seed(42)





rfc=RandomForestClassifier(n_estimators=100)

rfc.fit(X_train,y_train)

rfc_preds= rfc.predict(X_test)



accuracy_score(y_test,rfc_preds)
from sklearn.ensemble import GradientBoostingClassifier



gbk = GradientBoostingClassifier()

gbk.fit(X_train, y_train)

gbk_preds = gbk.predict(X_test)

accuracy_score(y_test,gbk_preds)