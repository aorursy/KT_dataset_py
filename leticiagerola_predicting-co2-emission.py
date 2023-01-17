# importing necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
# atributing dataset to a dataframe df
df= pd.read_csv('/kaggle/input/oc2emission/FuelConsumptionCo2.csv')
df.head(10)
# describing statiscal measures
df.describe()
df.shape
# plotting correlation matrix
plt.figure(figsize=(20,8))
sns.heatmap(data=df.corr(),annot=True,linewidths=0.2,cmap='coolwarm', square=True)
# ploting graph engine size x CO2 emissions
plt.figure(figsize=(13,5))
sns.lineplot(x=df['ENGINESIZE'], y=df['CO2EMISSIONS'])
plt.xlabel('Motor engine')
plt.ylabel('CO2 emissions')
plt.show()
# importing necessary libraries
from sklearn import linear_model
from sklearn.metrics import r2_score, mean_squared_error,mean_absolute_error
from sklearn.model_selection import train_test_split
# features into variables
engine= df[['ENGINESIZE']]
co2 = df[['CO2EMISSIONS']]
# spliting data in train and test with train_test_split
engine_treino, engine_test, co2_treino, co2_test = train_test_split(engine, co2, test_size=0.2, random_state=42)
print(type(engine_treino))
# ploting the correlation between features
plt.scatter(engine_treino, co2_treino, color='blue')
plt.xlabel('engine')
plt.ylabel('co2 emission')
plt.show()
# creating a linear regression model
# LinearRegression is a method of sklearn
modelo = linear_model.LinearRegression()
# linear regression formula: (Y = A + B.X)
# training the model to obtain the values of A and B (always do it in the train dataset)
modelo.fit(engine_treino, co2_treino)
# exibiting the coeficients A and B that the model generated
print(f'(A) intercept: {modelo.intercept_} | (B) inclination: {modelo.coef_}')
# print linear regression line on our TRAIN dataset
plt.scatter(engine_treino, co2_treino, color='blue')
plt.plot(engine_treino, modelo.coef_[0][0]*engine_treino + modelo.intercept_[0], '-r') 
# LR formula: inclination(B) * engine_treino(X) + intercept(A)
plt.ylabel('CO2 emissions')
plt.xlabel('Engine')
plt.show()
predictCO2 = modelo.predict(engine_test)
# print linear regression line on our TEST dataset
plt.scatter(engine_test, co2_test, color='green')
plt.plot(engine_test, modelo.coef_[0][0]*engine_test + modelo.intercept_[0], '-r')
plt.ylabel('CO2 emissions')
plt.xlabel('Engine')
plt.show()
# Showing metrics to check the acuracy of our model
print(f'Sum of squared error (SSE): {np.sum((predictCO2 - co2_test)**2)}') # SSE: sum all of the  residuals and square them. 
print(f'Mean squared error (MSE): {mean_squared_error(co2_test, predictCO2)}') # MSE: avg of SSE
print(f'Mean absolute error (MAE): {mean_absolute_error(co2_test, predictCO2)}')
print (f'Sqrt of mean squared error (RMSE):  {sqrt(mean_squared_error(co2_test, predictCO2))}') # RMSE: sqrt of the MSE
print(f'R2-score: {r2_score(predictCO2, co2_test)}') # r2-score: explains the variance of the variable Y when it comes to X