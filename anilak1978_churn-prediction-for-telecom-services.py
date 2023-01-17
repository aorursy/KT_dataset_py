# loading Required Libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv("https://raw.githubusercontent.com/anilak1978/customer-churn/master/WA_Fn-UseC_-Telco-Customer-Churn.csv")

df.head()
# check for missing values

df.isnull().sum()
# Total Charges have empty values that needs to be handled

spaced_values = df["TotalCharges"].str.contains(' ')

spaced_values.value_counts()
# replace these 11 values to np.nan, drop them and update the data type

df["TotalCharges"]=df["TotalCharges"].replace(" ", np.nan)

df.dropna(subset=["TotalCharges"], inplace=True)

df.reset_index(drop=True, inplace=True)

df[["TotalCharges"]]=df[["TotalCharges"]].astype("float")
# check the data types to make sure they are correct

df.dtypes
# check unique values for each column

for column in df.columns.values.tolist():

    print(column)

    print(df[column].unique())

    print("")
# some of the column values needs to be updated

update_col = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]

for i in update_col:

    df[i]=df[i].replace({"No internet service": "No"})
# For the sake of analysis update Churn to numerical variables

df["Churn"]=df["Churn"].replace({"No": 0, "Yes": 1})

df[["Churn"]]=df[["Churn"]].astype("float")

df["Churn"].dtypes
# Look at the categorical variables distribution

categorical_variables = ["gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", "MultipleLines", "InternetService",

                        "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "Contract", "PaperlessBilling",

                        "PaymentMethod"]



for i in categorical_variables:

    plt.figure(figsize=(10,5))

    sns.countplot(x=df[i], data=df)

    plt.title("Categorical Variable Distribution")
# Looking at the churn rate for each categorical variable

for i in categorical_variables:

    plt.figure(figsize=(10,5))

    data=df.groupby(i)["Churn"].mean().reset_index()

    sns.barplot(x=data[i], y="Churn", data=data)
# looking at distribution on numerical variables

numerical_variables = ["TotalCharges", "tenure", "MonthlyCharges"]

for i in numerical_variables:

    plt.figure(figsize=(10,5))

    sns.distplot(df[i])
# Looking at the correlation between numerical variables

corr=df.corr()

plt.figure(figsize=(10,5))

sns.heatmap(corr, annot=True)
# looking at linear relationship on numerical variables

for i in numerical_variables:

    plt.figure(figsize=(10,5))

    data=df.groupby(i)["Churn"].mean().reset_index()

    sns.regplot(x=data[i], y="Churn", data=data)
df.columns
# creating feature set and target

X = df[["Partner", "tenure", "InternetService", "Contract", "PaperlessBilling", "MonthlyCharges"]].values

y=df[["Churn"]]

X[0:5]
df["PaymentMethod"].unique()
# preprocessing using LabelEncoder()

from sklearn import preprocessing



partner=preprocessing.LabelEncoder()

partner.fit(["No", "Yes"])

X[:,0]=partner.transform(X[:,0])



internetservice=preprocessing.LabelEncoder()

internetservice.fit(["DSL", "Fiber optic", "No"])

X[:,2]=internetservice.transform(X[:,2])



contract=preprocessing.LabelEncoder()

contract.fit(["Month-to-month", "One year", "Two year"])

X[:,3]=contract.transform(X[:,3])



paperlessbilling=preprocessing.LabelEncoder()

paperlessbilling.fit(["No", "Yes"])

X[:,4]=paperlessbilling.transform(X[:,4])



X[0:5]
# split the dataset to train and test for fitting and model creation

from sklearn.model_selection import train_test_split

X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.2, random_state=3)
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators=100)

rf_model.fit(X_trainset, y_trainset.values.ravel())
# create prediction

rf_pred=rf_model.predict(X_testset)

rf_pred[0:5]
# use metrics accuracy score for model evaluation

from sklearn import metrics

metrics.accuracy_score(y_testset, rf_pred)