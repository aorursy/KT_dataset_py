#importing libraries for numpy and dataframe

import pandas as pd

import numpy as np



#importing libraries for data visualization

import matplotlib.pyplot as plt

from matplotlib.pyplot import xticks

import seaborn as sns

%matplotlib inline



#importing library for data scaling

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import scale



#importing library to suppress warnings

import warnings

warnings.filterwarnings('ignore')



#importing libraries for Logistic Regression

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.feature_selection import RFE

import statsmodels.api as sm



#Checking VIF values for the feature variables

from statsmodels.stats.outliers_influence import variance_inflation_factor



#Creation of confusion matrix

from sklearn import metrics



import os, sys, csv
#Reading the train dataset

train_df = pd.read_csv("../input/train.csv",encoding='ISO-8859-1')
#Previewing the dataframe by checking the first 5 records

train_df.head()
#To determine the number of rows and columns present in the train data

train_df.shape
#To understand the datatype of the columns present in the dataframe

train_df.info()
#To understand the statistical measures of the numerical columns present in the train dataframe

train_df.describe()
#Checking for missing data in the train dataset

missing = round((train_df.isna().sum()/len(train_df))*100,2)

total = train_df.isna().sum()

missing_data = pd.DataFrame({'Total Missing' : total,'Percentage Missing' : missing})

missing_data
#Dropping Cabin from the dataframe

train_df.drop(['Cabin'],axis=1,inplace=True)
#Previewing the dataframe by checking the first 5 records

train_df.head()
#To treat missing values of Age

train_df['Age'].hist(bins=10,density=True).plot(kind='Density')
#Calculating the mean of the column Age

train_df['Age'].mean()
#Let us calculate the mean of the column Age

train_df['Age'].median(skipna=True)
#Replacing all the NA values in age with median

train_df['Age'].fillna(train_df['Age'].median(skipna=True),inplace = True)
#We have only 2 NA in Embarked.

sns.countplot(train_df['Embarked'])
train_df['Embarked'].fillna(train_df['Embarked'].value_counts().idxmax(),inplace = True)
#Let us again look into the number of missing values after imputing values for NA

missing = round((train_df.isna().sum()/len(train_df))*100,2)

total = train_df.isna().sum()

missing_data = pd.DataFrame({'Total Missing' : total,'Percentage Missing' : missing})

missing_data
#We can see the cleaned data now.

train_df.head()
train_df['Family'] = np.where((train_df['SibSp']+train_df['Parch'])>0 , 1 , 0)
# We can drop SibSp and Parch

train_df.drop(['SibSp','Parch'],axis=1,inplace=True)
train_df.head()
#Ticket,Name,PassengerId are not going to play any role in model building so we can drop the columns.

train_df.drop(['PassengerId','Name','Ticket'],axis=1,inplace = True)
#Previewing dataframe after dropping the columns

train_df.head()
#Let us categorize age into 3 categories -> Children are below 20, between 20 and 50 are Adults and above 50 are Elders

train_df['Age_Category'] = pd.cut(x = train_df['Age'],bins=[0,20,50,100],labels = ['Children','Adults','Elders'])
train_df.head(10)
#Sex vs Survived

sns.countplot(x='Sex', hue='Survived' , data=train_df)
#Pclass vs Survived

sns.countplot(x='Pclass',hue='Survived',data=train_df)
#Survived vs Age

plt.figure(figsize=(50,15))

sns.countplot(x='Age',hue='Survived',data=train_df)

plt.show()
#Survived vs Embarked

sns.countplot(x='Embarked',hue='Survived',data=train_df)
#Survived vs Family

sns.countplot(x='Family',hue='Survived',data=train_df)
#Survived vs Age Category

sns.countplot(x='Age_Category',hue='Survived',data=train_df)
#Lets read the train_csv file from the local

test_df = pd.read_csv("../input/test.csv",encoding='ISO-8859-1')

#Previewing data of test dataset

test_df.head()
test_df.info()
test_df.describe()
#To know the number of rows and columns of the test dataset.

test_df.shape
#Checking for missing data in the train dataset

missing_test = round((test_df.isna().sum()/len(test_df))*100,2)

total_test = test_df.isna().sum()

missing_data_test = pd.DataFrame({'Total Missing' : total_test,'Percentage Missing' : missing_test})

missing_data_test
test_df.drop('Cabin',axis=1,inplace=True)
test_df['Age'].fillna(28,inplace=True)
test_df['Fare'].fillna(test_df['Fare'].mean(),inplace=True)
#Checking for missing data in the train dataset

missing_test = round((test_df.isna().sum()/len(test_df))*100,2)

total_test = test_df.isna().sum()

missing_data_test = pd.DataFrame({'Total Missing' : total_test,'Percentage Missing' : missing_test})

missing_data_test
#Both SibSp and Parch indicate the number of family members. Hence, where the sum of both the columns is greater than 0, we will consider it as 1.

test_df['Family'] = np.where((test_df['SibSp']+test_df['Parch'])>0 , 1 , 0)
PassengerId=test_df['PassengerId']
test_df.drop(['Name','SibSp','Parch','PassengerId','Ticket'],axis=1,inplace = True)
test_df.head()
#Let us categorize age into 3 categories -> Children are below 20, between 20 and 50 are Adults and above 50 are Elders

test_df['Age_Category'] = pd.cut(x = test_df['Age'],bins=[0,20,50,100],labels = ['Children','Adults','Elders'])
#Creating dummy variables for train dataset and test data

final_train = pd.get_dummies(train_df,columns=['Pclass','Sex','Embarked','Age_Category'])

final_test = pd.get_dummies(test_df,columns=['Pclass','Sex','Embarked','Age_Category'])
#Previewing final train dataset

final_train.head()
#Previewing final test dataset

final_test.head()
#Creating X_train dataset which would have the predicting features

X_train = final_train[['Age','Fare','Family','Pclass_1','Pclass_2','Pclass_3','Sex_female','Sex_male','Embarked_C','Embarked_Q','Embarked_S','Age_Category_Children','Age_Category_Adults','Age_Category_Elders']]
#Previewing the X_train dataset

X_train.head()
#Creating y_train dataset that contains the dependent variable

y_train = final_train[['Survived']]
logm1 = sm.GLM(y_train,(sm.add_constant(X_train)),family = sm.families.Binomial())

logm1.fit().summary()
col=X_train.columns
col = col.drop('Fare',1)
logm2 = sm.GLM(y_train,(sm.add_constant(X_train[col])),family = sm.families.Binomial())

logm2.fit().summary()
col = col.drop('Family',1)
logm3 = sm.GLM(y_train,(sm.add_constant(X_train[col])),family = sm.families.Binomial())

logm3.fit().summary()
col = col.drop('Age_Category_Elders',1)
logm4 = sm.GLM(y_train,(sm.add_constant(X_train[col])),family = sm.families.Binomial())

logm4.fit().summary()
col = col.drop('Age_Category_Children',1)
logm5 = sm.GLM(y_train,(sm.add_constant(X_train[col])),family = sm.families.Binomial())

logm5.fit().summary()
col = col.drop('Age_Category_Adults',1)
logm6 = sm.GLM(y_train,(sm.add_constant(X_train[col])),family = sm.families.Binomial())

logm6.fit().summary()
col = col.drop('Embarked_S',1)
logm7 = sm.GLM(y_train,(sm.add_constant(X_train[col])),family = sm.families.Binomial())

logm7.fit().summary()
col = col.drop('Embarked_Q',1)
logm8 = sm.GLM(y_train,(sm.add_constant(X_train[col])),family = sm.families.Binomial())

logm8.fit().summary()
col = col.drop('Pclass_2',1)
logm9 = sm.GLM(y_train,(sm.add_constant(X_train[col])),family = sm.families.Binomial())

logm9.fit().summary()
#Creating a dataframe that will contain VIF values of all the features

vif = pd.DataFrame()

vif['Features'] = X_train[col].columns

vif['VIF']=[variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]

vif['VIF']=round(vif['VIF'],2)

vif = vif.sort_values(by='VIF',ascending = False)

vif
#Since VIF of Sex_Male is greater than 5 , we would drop Sex_male

col = col.drop('Sex_male',1)
#Creating a dataframe that will contain VIF values of all the features

vif = pd.DataFrame()

vif['Features'] = X_train[col].columns

vif['VIF']=[variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]

vif['VIF']=round(vif['VIF'],2)

vif = vif.sort_values(by='VIF',ascending = False)

vif
X_test = final_test
X_test.head()
#Applying logistic regression model on X_test to predict Y.

logreg = LogisticRegression()

logreg.fit(X_train,y_train)

y_pred = logreg.predict(X_test)
#Calculating the metrics

accuracy = round(logreg.score(X_train,y_train)*100,2)

print('The accuracy for Logistic Regression model is :',accuracy)
#Converting the array in Dataframe format

submission = pd.DataFrame({'PassengerId':PassengerId,'Survived':y_pred})

submission.to_csv("submission.csv",index=False)