import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

print(os.listdir("../input"))



Data_Set1=pd.read_csv("../input/diabetes.csv")

Data_Set1.tail()                     
#check for null values-- nothing found

Data_Set1.isnull().sum()
#prepare x axis 

x=Data_Set1[["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]]

x.head()
#prepare y axis

y=Data_Set1["Outcome"]

y.head()
#Correlation states how the features are related to each other or the target variable.

#Correlation can be positive ie, increase in one value of feature increases the value of the target variable or

#negative ie,increase in one value of feature decreases the value of the target variable

#Heatmap makes it easy to identify which features are most related to the target variable.

#get correlations of each features in dataset

corr = Data_Set1.corr()

top_corr_features = corr.index

plt.figure(figsize=(20,20))

#plot heat map

g=sns.heatmap(Data_Set1[top_corr_features].corr(),annot=True,cmap="RdYlGn")

#Verdict : Glucose, BMI and Age are having high correlation with diabetes
