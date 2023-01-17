import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
df = pd.read_csv("../input/Admission_Predict.csv")
df.head()
df.columns
Features = df[["GRE Score","TOEFL Score","University Rating","SOP","LOR ","CGPA","Research"]]
Features.head()
Target = df[["Chance of Admit "]]
df.isnull().sum()
df.corr(method ='kendall') 
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(Features, Target, test_size=0.33, random_state=42)
from sklearn.linear_model import LinearRegression

reg = LinearRegression().fit(X_train, y_train)

reg.score(X_train, y_train)

print("Training Score ",reg.score(X_train,y_train))
print("Test Score ",reg.score(X_test,y_test))

Features_new = df[["GRE Score","TOEFL Score","University Rating","SOP","CGPA"]]

Features_new.head()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(Features_new, Target, test_size=0.33, random_state=42)
from sklearn.linear_model import LinearRegression

reg = LinearRegression().fit(X_train, y_train)

reg.score(X_train, y_train)

print("Training Score ",reg.score(X_train,y_train))
print("Test Score ",reg.score(X_test,y_test))