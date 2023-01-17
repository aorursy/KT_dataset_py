import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

dataset=pd.read_csv("../input/Admission_Predict_Ver1.1.csv")
dataset.head()
#Omitting the first column
updated_dataset=dataset.iloc[:,1:9]
updated_dataset.head()
print(updated_dataset.shape)
updated_dataset.describe()
#Checking for NA values
updated_dataset.isna().sum()
updated_dataset.corr(method="pearson")
plt.subplots(figsize=(20,4))
sns.barplot(x="GRE Score",y="Chance of Admit ",data=dataset)
plt.subplots(figsize=(25,5))
sns.barplot(x="TOEFL Score",y="Chance of Admit ",data=dataset)
plt.subplots(figsize=(20,4))
sns.barplot(x="University Rating",y="Chance of Admit ",data=dataset)
plt.subplots(figsize=(15,5))
sns.barplot(x="SOP",y="Chance of Admit ",data=dataset)
#plt.subplots(figsize=(15,4))
#sns.barplot(x="CGPA",y="Chance of Admit ",data=dataset)
plt.subplots(figsize=(15,5))
sns.barplot(x="Research",y="Chance of Admit ",data=dataset)
X=updated_dataset.iloc[:,:7]
y=updated_dataset["Chance of Admit "]
print(X.shape)
print(y.shape)
X.head()
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=50)
from sklearn.linear_model import LinearRegression
#Linear Regression
Linear=LinearRegression()
Linear.fit(X_train,y_train)
y_pred=Linear.predict(X_test)
y_pred
from sklearn.metrics import mean_absolute_error,r2_score
print("R2 score of the model is ",r2_score(y_pred,y_test))
print("mean_absolute_error  of the model is ",mean_absolute_error(y_pred,y_test))
