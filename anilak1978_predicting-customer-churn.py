import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt



df = pd.read_csv("https://raw.githubusercontent.com/anilak1978/customer_churn/master/Churn_Modeling.csv")

df.head()
# Looking for missing data

missing_data=df.isnull()

for column in missing_data.columns.values.tolist():

    print(column)

    print(missing_data[column].value_counts())

    print("")
# Looking at data types

df.dtypes
# Looking at the basic information

df.info()
# looking at the summary

df.describe()
# Looking at correlation between numerical variables

df.corr()
# Visualizing relationship of Age and Estimated Salary

plt.figure(figsize=(20,20))

sns.relplot(x="Age", y="EstimatedSalary", hue="Geography", data=df)

plt.title("Age VS Estimated Salary")

plt.xlabel("Age")

plt.ylabel("Estimated Salary")
# looking at age distribution

plt.figure(figsize=(10,10))

sns.distplot(df["Age"])
# Looking at Gender Distribution

plt.figure(figsize=(10,8))

sns.countplot(x="Gender", data=df)

plt.title("Gender Distribution")

plt.xlabel("Gender")

plt.ylabel("Count")
# Looking at Geography and Gender Distribution against Estimated Salary

plt.figure(figsize=(20,20))

sns.catplot(x="Geography", y="EstimatedSalary", hue="Gender", kind="box", data=df)

plt.title("Geography VS Estimated Salary")

plt.xlabel("Geography")

plt.ylabel("Estimated Salary")
# Looking at linear relationship between Age and CreditScore

plt.figure(figsize=(10,10))

sns.regplot(x="Age", y="CreditScore", data=df)
#looking at correlation between attributes in detail

corr=df.corr()

plt.figure(figsize=(10,10))

sns.heatmap(corr, annot=True)
df.columns
# Selecting and Preparing the Feature Set and Target

X = df[["CreditScore", "Geography", "Gender", "Age", "Tenure", "EstimatedSalary"]].values

y=df[["Exited"]]

X[0:5], y[0:5]
# preprocessing categorical variables

from sklearn import preprocessing

geography=preprocessing.LabelEncoder()

geography.fit(["France", "Spain", "Germany"])

X[:,1]=geography.transform(X[:,1])



gender = preprocessing.LabelEncoder()

gender.fit(["Female", "Male"])

X[:,2]=gender.transform(X[:,2])



X[0:5]
# split train and test data

from sklearn.model_selection import train_test_split

X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.2, random_state=3)
# create model using DecisionTree Classifier and fit training data

from sklearn.tree import DecisionTreeClassifier

dt_model = DecisionTreeClassifier()

dt_model.fit(X_trainset, y_trainset)
# create prediction

dt_pred = dt_model.predict(X_testset)

dt_pred[0:5]
# Evaluating the prediction model

from sklearn import metrics

metrics.accuracy_score(y_testset, dt_pred)
# create Random Forest Decision Tree model

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators=100)

rf_model.fit(X_trainset, y_trainset.values.ravel())
# create prediction using rf_model

rf_pred = rf_model.predict(X_testset)

rf_pred[0:5]
# evaluate the model

metrics.accuracy_score(y_testset, rf_pred)