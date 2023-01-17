#import libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import matplotlib.gridspec as gridspec

import matplotlib.pylab as pl

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import RandomizedSearchCV

from sklearn.model_selection import KFold

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import cross_val_score 

import warnings

warnings.filterwarnings("ignore") 
#dataset

data= pd.read_csv("../input/insurance-dataset/insurance_data.csv")

data.head()
# plot

plt.figure(figsize= (9,4))

plt.scatter(data["age"], data['bought_insurance'])

plt.xlabel("age", fontsize=14)

plt.ylabel("insurance ?: 0=No and 1=Yes", fontsize=14)
# separating independent and dependent variable

x= data[["age"]]

y= data[["bought_insurance"]]



# defining Linear Regression model

from sklearn.linear_model import LinearRegression

reg= LinearRegression()



# fitting the model

reg.fit(x,y)
# plotting best fit line

plt.figure(figsize= (9,4))

plt.scatter(data["age"], data['bought_insurance'])

plt.plot(data["age"], reg.predict(x), color="orange")

plt.xlabel("age", fontsize=14)

plt.ylabel("insurance: 0=No and 1=Yes", fontsize=14)

plt.legend([ "best fit regression line", "actual"])
print("intercept :", reg.intercept_)

print("coefficient :", reg.coef_)
# probability for age 15 and 75

print("probability(yes=15) is" , reg.predict([[15]]))

print("probability(yes=75) is" , reg.predict([[75]]))
# creating same dataset with outlier

data.to_csv("insurance_data(with_outlier).csv", index=False)
# to create an outlier, I have added a data point for age=120 and 1 as the corresponding output

import csv



with open("insurance_data(with_outlier).csv", "a") as csvfile:

    writer= csv.writer(csvfile)

    writer.writerow([120,1])  
# same dataset with outliers

data2= pd.read_csv("insurance_data(with_outlier).csv")

data2.tail()
# fitting linear regression model on the new dataset with outlier

reg2= LinearRegression()

reg2.fit(data2[["age"]], data2[["bought_insurance"]])
plt.figure(figsize= (9,4))

plt.scatter(data2[["age"]], data2[["bought_insurance"]])

plt.plot(data2[["age"]], reg2.predict(data2[["age"]]), color="orange")

plt.xlabel("age", fontsize=14)

plt.ylabel("insurance: 0=No and 1=Yes", fontsize=14)

plt.legend([ "best fit regression line", "actual"])



# intercept and coefficient of regression line

print("intercept :", reg2.intercept_)

print("coefficient :", reg2.coef_)
print("probability(yes=40) without outlier was {}. This person was buying the insurance.".format(reg.predict([[40]])))

print("probability(yes=40) with outlier is {}. Now this person will not buy the insurance.".format(reg2.predict([[40]])))
# upload dataset

df= pd.read_csv("../input/titanicdataset-traincsv/train.csv")



# shape of dataset

print("dataset shape:", df.shape)

df.head()
gs= gridspec.GridSpec(2,2)

plt.figure(figsize=(15,12))



plt.style.use('fivethirtyeight')

ax=pl.subplot(gs[0,0])

sns.countplot(x= "Survived", data=df, palette="husl")





ax=pl.subplot(gs[0,1])

sns.countplot(x="Survived", hue="Sex", data=df)



ax=pl.subplot(gs[1,0])

sns.countplot(x="Survived", hue="Pclass", data=df)



ax= pl.subplot(gs[1,1])

sns.countplot(x= "Survived", hue="Embarked", data=df, palette="husl")
# percentage survived

df["Survived"].value_counts()/ len(df)*100
df.isnull().sum()
sns.boxplot(df["Age"])
# function to impute median

def impute_age(dataframe, feature, median):

    dataframe[feature]= dataframe[feature].fillna(median)
impute_age(df, "Age", df["Age"].median())
print("percentage of missing values in Cabin :",df["Cabin"].isnull().mean() )



df.drop(columns= ["Cabin"], axis=1, inplace= True)
# function to impute mode

def impute_mode(dataframe, feature):

    dataframe[feature]= dataframe[feature].fillna(dataframe[feature].mode()[0])
impute_mode(df, "Embarked")
sns.countplot(df["Survived"])

plt.title("target variable")
print("percentage of class in target:\n", df["Survived"].value_counts()/ len(df)*100)
df.drop(columns= ["PassengerId", "Name", "Ticket"], axis=1, inplace=True)
# encoding categorical fetaures

df= pd.get_dummies(df, drop_first=True)



# dummy encoded dataset

df.head()
# features

x= df.iloc[:, 1:]



# target

y= df.iloc[:, 0]
x_train, x_test, y_train, y_test= train_test_split(x,y, test_size=0.2, random_state=10)
# selecting the classifier

reg= LogisticRegression()



# fitting model on train data

reg.fit(x_train, y_train)
# checking model performance

y_predicted= reg.predict(x_test)



cm= confusion_matrix(y_test, y_predicted)

print(cm)

sns.heatmap(cm, annot=True)

print(accuracy_score(y_test, y_predicted))

print(classification_report(y_test, y_predicted))