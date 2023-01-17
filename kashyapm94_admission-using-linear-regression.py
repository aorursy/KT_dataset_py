import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
np.random.seed(1234)
dataset=pd.read_csv("../input/Admission_Predict_Ver1.1.csv")
dataset.head()
dataset.info()
dataset.isnull().values.any()
dataset.columns
dataset.rename(columns = {'Serial No.':'SNo', 'GRE Score':'GRE', 'TOEFL Score':'TOEFL', 'University Rating':'UniRate',
                          'Chance of Admit ':'Chance'}, inplace = True)
dataset = dataset.set_index('SNo')
dataset.head()
var_corr = dataset.corr()
plt.subplots(figsize=(20,10))
sns.heatmap(var_corr, xticklabels=var_corr.columns, yticklabels=var_corr.columns, cmap = 'RdBu', annot=True, linewidths = 0.9)
chance_car_corr = var_corr.iloc[:,-1]
print(chance_car_corr)
plt.subplots(figsize=(15,6))
sns.regplot(x="GRE",y="Chance",data=dataset, color = 'red')
plt.subplots(figsize=(15,6))
sns.regplot(x="TOEFL",y="Chance",data=dataset, color = 'green')
plt.subplots(figsize=(15,6))
sns.regplot(x="CGPA",y="Chance",data=dataset, color = 'blue')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 7].values
print('Shape of X:', X.shape)
print('Shape of y:', y.shape)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=1/5, random_state=8)
print('Shape of X_train:', X_train.shape)
print('Shape of y_train:', y_train.shape)
print('Shape of X_test:', X_test.shape)
print('Shape of y_test:', y_test.shape)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Predicting Test set results
y_pred = regressor.predict(X_test)
print(y_pred)
print('Shape of y_pred:', y_pred.shape)
from sklearn.metrics import mean_absolute_error,r2_score
r2 = r2_score(y_test, y_pred)
print('r_square score for the test set is', r2)
MAE = mean_absolute_error(y_pred,y_test)
print('MAE is', MAE)