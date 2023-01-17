#Let's import necessary headers
import pandas as pd
import matplotlib as plt
%matplotlib inline
import numpy as np
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier as DTC
#Now let start import dataset's
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
#Let's check them out
df_train.head()
df_test.head()
#Let's create a full dataset by combining both
#First let's check out columns in both dataset
print(df_train.columns)
print(df_test.columns)
#Now to Create full dataset
df_AllData = pd.concat([df_train, df_test])
df_AllData.columns
#That's great,  Let's check number of rows in All Data
len(df_AllData)
#Now let's create unique index
df_AllData['Pid'] = df_AllData['PassengerId']
df_AllData.set_index(['Pid'], inplace=True)
#First column to process - PCLASS
#Let's check for any null values in the column
df_AllData.Pclass.isnull().sum()
#Great,  We don't have any null values in the field,  Let's now see what are the values in the field.
df_AllData.Pclass.value_counts().plot.bar();
#Let's checkout which class people survived the most in train dataset.
df_train.groupby('Pclass')['Survived'].mean().plot.bar();
#Looks like class 1 people had more chance that 2 and 3.
#Let's start our first modeling
x_train = df_train[['Pclass']]
y = df_train['Survived']
x_test = df_test[['Pclass']]
dtree = DTC()
dtree.fit(x_train,y)
prediction = dtree.predict(x_test)

#Let's add prediction results to test dataset.
df_test['Survived'] = prediction
#We just did the prediction,  Let's prepare submission dataset for Kaggle
(df_test[['PassengerId', 'Survived']]).to_csv('TitaticSolution_v1.csv', index=False)
#Scored - 0.65550
#Let's proceed further to process column - 'Sex'
#Check for any null values
df_AllData.Sex.isnull().sum()
#Let's check how many of them each.
df_AllData.Sex.value_counts().plot.bar();
#Looks like we got more male passengers than female.
#Let cross check each group survivial percentage.
df_train.groupby('Sex')['Survived'].mean().plot.bar();
#Well that made clear female had more survival chance than men. I wonder how would that percentage in present day ;)
#Let's switch string values to number in order to do our modeling.
df_AllData['Sex'] = df_AllData['Sex'].map({'male':0, 'female':1})
#Let's check the transistion
df_AllData.Sex.value_counts()
#That's Great,  Let's split train and test dataset from All Data.
df_train = df_AllData.loc[1:891,:]
df_test = df_AllData.loc[892:1309,:]
#That's Great,  Let's improve our prediction.
x_train = df_train[['Pclass', 'Sex']]
y = df_train['Survived']
x_test = df_test[['Pclass', 'Sex']]
dtree = DTC()
dtree.fit(x_train, y)
prediction = dtree.predict(x_test)
#Let's add prediction results to test dataset.
df_test['Survived'] = prediction.astype(int)
#Let's prepare submission dataset for Kaggle
(df_test[['PassengerId', 'Survived']]).to_csv('TitaticSolution_v2.csv', index=False)
#Scored - 0.75598
#Well let's proceed further processing column - 'Parch'
#Checking for any null values in the column
df_AllData['Parch'].isnull().sum()
#Checking for values present in the columns
df_train.Parch.value_counts().plot.bar();
#Reviewing Parch and Survived columns
df_train.groupby('Parch')['Survived'].mean().plot.bar();
#Again let's improve our prediction using Parch
x_train = df_train[['Pclass','Sex','Parch']]
y = df_train['Survived']
y_train = df_test[['Pclass','Sex','Parch']]
dtree.fit(x_train,y)
prediction = dtree.predict(y_train)

#Let's add prediction results to test dataset.
df_test['Survived'] = prediction.astype(int)
(df_test[['PassengerId', 'Survived']]).to_csv('TitaticSolution_v3.csv', index=False)
#Let's start processing next column - 'Embarked'
#Let's check for any null values in the field
df_AllData.Embarked.isnull().sum()
#Let locate the null value rows
df_AllData[df_AllData.Embarked.isnull()]
#Let's find some value close to the values in rows to replace this null value for Embarked
df_AllData.loc[(df_AllData.Sex == 1) & (df_AllData.Survived == 1) & (df_AllData.Pclass == 1) & (df_AllData.Parch == 0) ,'Embarked'].mode()
df_AllData.loc[(df_AllData.Fare>79) & (df_AllData.Fare < 85),'Embarked'].mode()
#Let's replace the null value
df_AllData['Embarked'] = df_AllData['Embarked'].fillna('C')
#Time to confirm
df_AllData.Embarked.isnull().any()
#Now to replace category value in number value for modeling
df_AllData['Embarked'] = df_AllData['Embarked'].map({'S':0, 'C':'1', 'Q':2})
#Checking the data again
df_AllData.Embarked.value_counts()
#Now to split train and test datasets
df_train = df_AllData.loc[1:891,:]
df_test = df_AllData.loc[892:1309,:]
#Again let's improve our model and prediction
x_train = df_train[['Pclass', 'Sex', 'Parch', 'Embarked']]
y = df_train['Survived']
x_test = df_test[['Pclass', 'Sex', 'Parch', 'Embarked']]
dtree = DTC()
dtree.fit(x_train, y)
prediction = dtree.predict(x_test)

#Let's add prediction results to test dataset.
df_test['Survived'] = prediction.astype(int)
(df_test[['PassengerId', 'Survived']]).to_csv('TitanicSolution_v4.csv', index=False)
#Next Column to Process - FARE
#Checking for any null vaues in the column
df_AllData[df_AllData.Fare.isnull()]
#Let try to predict value for missing fare
df_AllData.loc[(df_AllData.Age == 60) & (df_AllData.Sex == 0) & (df_AllData.Embarked == 0) ,'Fare'].mean()
#Lets replace null value
df_AllData['Fare'] = df_AllData.Fare.fillna(32.775)
#Spliting Training and Test Data

df_train = df_AllData.loc[1:891,:]
df_test = df_AllData.loc[892:1309, :]
#Again let's improve Model and predict results

x_train = df_train[['Pclass', 'Sex', 'Parch', 'Embarked', 'Fare']]
y = df_train['Survived']
x_test = df_test[['Pclass', 'Sex', 'Parch', 'Embarked', 'Fare']]
dtree.fit(x_train, y)
prediction = dtree.predict(x_test)

#Let's add prediction results to test dataset.
df_test['Survived'] = prediction.astype(int)
(df_test[['PassengerId', 'Survived']]).to_csv('TitanicSolution_v5.csv', index=False)
#Let's keep going. Next column to process - 'Cabin'
#Now to check for null values
df_AllData.Cabin.isnull().sum()
#Well we got more null values to process.  Let investigate further.
df_AllData.Cabin.value_counts(dropna = False)
#Lets try to group them
df_AllData.Cabin.str[0].value_counts(dropna=False).plot.bar();
#Let's categorize null values as separate category
df_AllData['Cabin'] = df_AllData['Cabin'].fillna('X')
#Now to switch category value to numeric value
df_AllData['Cabin'] = df_AllData['Cabin'].str[0].map({'A':0, 'B':1, 'C':2, 'D':3, 'E':4, 'F':5, 'G':6, 'T':7, 'X':8})
#Now to split train and test dataset from all data
df_train = df_AllData.loc[1:891,:]
df_test = df_AllData.loc[892:1309,:]
#Let's check against their survival
df_train.groupby(df_train.Cabin)['Survived'].mean().plot.bar();
#Time to add Cabin to our model and try to predict the output
x_train = df_train[['Pclass', 'Sex', 'Parch', 'Embarked', 'Fare', 'Cabin']]
y = df_train['Survived']
x_test = df_test[['Pclass', 'Sex', 'Parch', 'Embarked', 'Fare', 'Cabin']]
dtree = DTC()
dtree.fit(x_train,y)
prediction = dtree.predict(x_test)

#Now to add prediction to our dataset
df_test['Survived'] = prediction.astype(int)
#Prepare results for submission
(df_test[['PassengerId', 'Survived']]).to_csv('TitanicSolution_v6.csv', index=False)
#Check for any Null values
df_AllData.Name.isnull().sum()
#Great field doesn't have any null values.  Let's check the values in the column
df_AllData.Name.value_counts()
#Best way to make use of these names is by grouping them via their salutations
df_AllData['Salutation'] =  df_AllData.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
df_AllData.Salutation.value_counts().plot.bar();
df_AllData.Salutation.replace(to_replace=['Rev','Dr','Col','Major','Mlle','Ms','Countess','Capt','Dona','Don','Sir','Lady','Jonkheer','Mme'], value=0, inplace=True)
df_AllData.Salutation.replace('Mr', 1, inplace=True)
df_AllData.Salutation.replace('Miss', 2, inplace=True)
df_AllData.Salutation.replace('Mrs', 3, inplace=True)
df_AllData.Salutation.replace('Master', 4, inplace=True)
df_AllData.Salutation.value_counts(dropna=False).plot.bar();
#Let's split train and test datasets
df_train = df_AllData.loc[1:891,:]
df_test = df_AllData.loc[892:1309,:]
#Again let's improve our model and prediction
x_train = df_train[['Pclass', 'Sex', 'Parch', 'Salutation', 'Embarked', 'Fare']]
y = df_train['Survived']
x_test = df_test[['Pclass', 'Sex', 'Parch', 'Salutation', 'Embarked', 'Fare']]
dtree = DTC()
dtree.fit(x_train,y)
prediction = dtree.predict(x_test)

df_test['Survived'] = prediction.astype(int)
(df_test[['PassengerId', 'Survived']]).to_csv('TitanicSolution_v7.csv', index=False)
#Moving on to next column - Age
#Let's check for any null values
df_AllData.Age.isnull().sum()
#Let's explore the values in the field
df_AllData.Age.value_counts(dropna=False)
#Let's try to group these values, first round them up
df_AllData.Age = df_AllData.Age.round()
df_AllData.corr().Age
x_test=df_AllData[['Fare','Parch', 'Sex', 'Salutation', 'Sibsp']]
y = df_AllData.Age.dropna()