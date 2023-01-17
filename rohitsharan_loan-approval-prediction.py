import pandas as pd

from sklearn.impute import SimpleImputer

import numpy as np

from  sklearn.preprocessing import OrdinalEncoder

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from sklearn import tree

from sklearn import model_selection

import seaborn as sns

from xgboost import XGBClassifier

from xgboost import plot_importance

np.random.seed(0)
file=pd.read_csv("../input/loan-data-set/loan_data_set.csv")

data=file

data.isnull().sum()
impute_missing=SimpleImputer(missing_values=np.NaN, strategy='most_frequent')

impute_missing.fit(data)

data=impute_missing.transform(data)

data=pd.DataFrame(data=data,columns=file.columns)

data.isnull().sum()
encoding=OrdinalEncoder()

X=data

X=encoding.fit_transform(X)

X=pd.DataFrame(data=X,columns=file.columns)

Y=X["Loan_Status"]

X=X.loc[:,X.columns!="Loan_Status"]

copy=X
model=DecisionTreeClassifier()

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,random_state=0)

model.fit(X_train,Y_train)

plt.figure(figsize=(20,20))

tree.plot_tree(model.fit(X_train,Y_train))

print("Decision tree accuracy:::::",model.score(X_test,Y_test)*100,"%")
Scaler=StandardScaler()

standard_data=Scaler.fit_transform(X)

X=pd.DataFrame(data=standard_data,columns=copy.columns)

X
model=DecisionTreeClassifier()

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,random_state=0)

model.fit(X_train,Y_train)

print(model.score(X_test,Y_test)*100,"%")
plt.figure(figsize=(17,17))

sns.boxplot(data=pd.DataFrame(X))

plt.show()
model1=XGBClassifier()

model1.fit(X,Y)

plot_importance(model1)

plt.show()
sort=model1.get_booster().get_score()

print(sort)

selected_features=["Dependents","Married","Loan_Amount_Term","Credit_History","Property_Area","CoapplicantIncome","LoanAmount","ApplicantIncome"]

## removed --Gender,education and loan id

X=X[selected_features]

X

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,random_state=0)

model.fit(X_train,Y_train)

print(model.score(X_test,Y_test)*100,"%")