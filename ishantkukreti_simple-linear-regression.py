import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

#read file from kaggle 

df=pd.read_csv("../input/calcofi/bottle.csv")

df.shape
df.head()
df.describe()
df=df[["Salnty","T_degC"]]
df.head()
print("\nSalinty\n",df.Salnty.isnull().value_counts())

print("\nTemprature\n",df.T_degC.isnull().value_counts())
df=df.dropna(axis=0)

df.reset_index(drop=True,inplace=True)
df=df[0:750]#slicing array for as it has a lot of rows 
df.shape


plt.figure(figsize=(12,12))

plt.scatter(df.Salnty,df.T_degC, color='blue')

plt.xlabel("Tempurature",fontsize=25)

plt.ylabel("Salinity",fontsize=25)
x=np.array(df['Salnty']).reshape(750,1)

y=np.array(df['T_degC']).reshape(750,1)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 100)

from sklearn.linear_model import LinearRegression

regr=LinearRegression()

regr.fit(X_train,y_train)

print ('Coefficients: ', regr.coef_)

print ('Intercept: ',regr.intercept_)
accuracy = regr.score(X_test, y_test)

print(accuracy)
plt.figure(figsize=(12,12))

plt.scatter(x, y,  color='blue')

plt.plot(X_train, regr.predict(X_train), '-r',linewidth="4")

plt.xlabel("Tempurature",fontsize=25)

plt.ylabel("Salinity",fontsize=25)

plt.title("Regression plot",fontsize=25)

from sklearn.metrics import r2_score

ypred = regr.predict(X_test)

print("Mean absolute error: %.2f" % np.mean(np.absolute(ypred - y_test)))

print("Residual sum of squares (MSE): %.2f" % np.mean((ypred - y_test) ** 2))

print("R2-score: %.2f" % r2_score(y_test , ypred) )