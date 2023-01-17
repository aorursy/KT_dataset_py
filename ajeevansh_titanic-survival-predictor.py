###  Importing relevant libraries



import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt



import warnings

warnings.filterwarnings("ignore")



import statsmodels

import statsmodels.api as sm

from statsmodels.stats.outliers_influence import variance_inflation_factor ####### VIF

import sklearn

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler # Inbuilt functions for Feature Scaling

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score

from sklearn.linear_model import LogisticRegression

from sklearn.feature_selection import RFE
df_train = pd.read_csv("../input/train.csv")

df_test = pd.read_csv("../input/test.csv")
df_train.head()
print("-----------------------------------------------------------------")

print(df_train.describe())

print("-----------------------------------------------------------------")

print(df_train.info())
print("-----------------------------------------------------------------")

print(df_test.describe())

print("-----------------------------------------------------------------")

print(df_test.info())
df_train = df_train.drop(['Cabin'],axis=1)

df_test = df_test.drop(['Cabin'],axis=1)
# Checking the percentage of missing values

round(100*(df_train.isnull().sum()/len(df_train.index)), 2)
df_train['Age'].fillna(df_train['Age'].mean(), inplace=True)
# Checking the percentage of missing values

round(100*(df_train.isnull().sum()/len(df_train.index)), 2)
df_train.dropna(subset=['Embarked'], inplace=True)
# Checking the percentage of missing values

round(100*(df_train.isnull().sum()/len(df_train.index)), 2)
df_train.head()
# Checking the percentage of missing values

round(100*(df_test.isnull().sum()/len(df_test.index)), 2)
df_test['Age'].fillna(df_test['Age'].mean(), inplace=True)
df_test['Fare'].fillna(df_test['Fare'].mean(), inplace=True)
# Checking the percentage of missing values

round(100*(df_test.isnull().sum()/len(df_test.index)), 2)
sns.heatmap(df_train.corr(), annot=True, cmap='YlGnBu')
df_train.head()
df_train['FamilySize'] = df_train['SibSp'] + df_train['Parch'] + 1

df_test['FamilySize'] = df_test['SibSp'] + df_test['Parch'] + 1
df_train['Alone'] = [1 if x == 1 else 0 for x in df_train['FamilySize']]
df_train.head()
df_test['Alone'] = [1 if x == 1 else 0 for x in df_test['FamilySize']]
df_train = df_train.drop(['SibSp','Parch','FamilySize','Name'],axis=1)

df_test = df_test.drop(['SibSp','Parch','FamilySize','Name'],axis=1)
df_train.head()
df_test.head()
df_train = df_train.drop(['Ticket'],axis=1)

df_test = df_test.drop(['Ticket'],axis=1)
df_train.head()
df_test.head()
#df_train['Title'] = df_train.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

#df_test['Title'] = df_test.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
df_test.head()
df_train.head()
df_test.head()
# Draw a nested barplot to show survival for class and sex

g = sns.catplot(x="Pclass", y="Survived", hue="Sex", data=df_train,

                height=6, kind="bar", palette="muted")

g.despine(left=True)

g.set_ylabels("survival probability")
# Draw a nested barplot to show survival for class and sex

g = sns.catplot(x="Pclass", y="Survived", hue="Embarked", data=df_train,

                height=6, kind="bar", palette="muted")

g.despine(left=True)

g.set_ylabels("survival probability")
# Draw a nested barplot to show survival for class and sex

g = sns.catplot(x="Pclass", y="Survived", hue="Alone", data=df_train,

                height=6, kind="bar", palette="muted")

g.despine(left=True)

g.set_ylabels("survival probability")
# Draw a nested barplot to show survival for class and sex

g = sns.catplot(x="Sex", y="Survived", hue="Alone", data=df_train,

                height=6, kind="bar", palette="muted")

g.despine(left=True)

g.set_ylabels("survival probability")
df_train.head()
df_test.head()
# List of variables to map



varlist =  ['Pclass']



# Defining the map function

def binary_map(x):

    return x.map({1: "1stClass", 2: "2ndClass",3:"3rdClass"})



# Applying the function to the housing list

df_train[varlist] = df_train[varlist].apply(binary_map)
# Creating a dummy variable for some of the categorical variables and dropping the first one.

dummy1 = pd.get_dummies(df_train[['Sex', 'Pclass' , 'Embarked']], drop_first=True)



# Adding the results to the master dataframe

df_train = pd.concat([df_train, dummy1], axis=1)
df_train.head()
df_train = df_train.drop(['PassengerId','Pclass','Sex','Embarked'],axis=1)
df_train.head()
# List of variables to map



varlist =  ['Pclass']



# Defining the map function

def binary_map(x):

    return x.map({1: "1stClass", 2: "2ndClass",3:"3rdClass"})



# Applying the function to the housing list

df_test[varlist] = df_test[varlist].apply(binary_map)
# Creating a dummy variable for some of the categorical variables and dropping the first one.

dummy1 = pd.get_dummies(df_test[['Sex', 'Pclass' , 'Embarked']], drop_first=True)



# Adding the results to the master dataframe

df_test = pd.concat([df_test, dummy1], axis=1)
df_test = df_test.drop(['Pclass','Sex','Embarked'],axis=1)
df_test.head()
df_test.shape
df_train.shape
df_train.columns
df_test.columns
df_train.head()
X_train = df_train.drop("Survived", axis=1)

Y_train = df_train["Survived"]

X_test  = df_test.drop("PassengerId", axis=1).copy()

X_train.shape, Y_train.shape, X_test.shape
X_train.head()
Y_train.head()
X_test.head()
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()



X_train[['Age','Fare']] = scaler.fit_transform(X_train[['Age','Fare']])



X_train.head()
import statsmodels.api as sm
# Logistic regression model

logm1 = sm.GLM(Y_train,(sm.add_constant(X_train)), family = sm.families.Binomial())

logm1.fit().summary()
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
from sklearn.feature_selection import RFE

rfe = RFE(logreg, 5)             # running RFE with 13 variables as output

rfe = rfe.fit(X_train, Y_train)
rfe.support_
list(zip(X_train.columns, rfe.support_, rfe.ranking_))
col = X_train.columns[rfe.support_]
X_train_sm = sm.add_constant(X_train[col])

logm2 = sm.GLM(Y_train,X_train_sm, family = sm.families.Binomial())

res = logm2.fit()

res.summary()
# Check for the VIF values of the feature variables. 

from statsmodels.stats.outliers_influence import variance_inflation_factor
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()

vif['Features'] = X_train[col].columns

vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
X_test.shape
y_train_pred = res.predict(X_train_sm).values.reshape(-1)
y_train_pred[:10]
y_train_pred_final = []

yif = pd.DataFrame(y_train_pred_final)

yif['Survival_Prob'] = y_train_pred
yif.shape
y_train_pred.shape
yif.head()
# Creating new column 'predicted' with 1 if Churn_Prob > 0.5 else 0

yif['predicted'] = yif.Survival_Prob.map(lambda x: 1 if x > 0.5 else 0)

yif.head()
yif['Survival']=Y_train.values
yif.head()
from sklearn import metrics
# Let's check the overall accuracy.

print(metrics.accuracy_score(yif.Survival, yif.predicted))
X_test[['Age','Fare']] = scaler.transform(X_test[['Age','Fare']])
X_test = X_test[col]

X_test.head()
X_test_sm = sm.add_constant(X_test)
y_test_pred = res.predict(X_test_sm)
y_test_pred[:10]
# Converting y_pred to a dataframe which is an array

y_pred_1 = pd.DataFrame(y_test_pred)
# Let's see the head

y_pred_1.head()
# Converting y_test to dataframe

y_test = []

y_test_df = pd.DataFrame(y_test)
# Appending y_test_df and y_pred_1

y_pred_final = pd.concat([df_test['PassengerId'], y_pred_1],axis=1)
y_pred_final.head()
# Renaming the column 

y_pred_final= y_pred_final.rename(columns={ 0 : 'Survival_Prob'})
y_pred_final.head()
y_pred_final['final_predicted'] = y_pred_final.Survival_Prob.map(lambda x: 1 if x > 0.42 else 0)
y_pred_final.head()
submission = pd.DataFrame({

        "PassengerId": y_pred_final["PassengerId"],

        "Survived": y_pred_final['final_predicted']

    })

submission.to_csv('gender_submission.csv', index=False)