import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import pandas as pd
import math
# reading the csv files
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
df_train.info()
df_train.columns
df_train.describe()
x_train = df_train["x"]
y_train = df_train["y"]
x_test = df_test["x"]
y_test = df_test["y"]
#_train = x_train.reshape(len(x_train),1)
#_train = y_train.reshape(len(y_train),1)
#_test = x_test.reshape(len(x_test),1)
#_test = y_test.reshape(len(y_test),1)
plt.scatter(x_train, y_train,  color='black')
plt.title('Train data')
plt.xlabel('Size')
plt.ylabel('Price')
plt.xticks(())
plt.yticks(())

plt.show()
df_train_data = df_train.dropna()


X = df_train_data[['x']].as_matrix()
Y = df_train_data[['y']].as_matrix()

lm=linear_model.LinearRegression()
lm.fit(X,Y)
lm.coef_



lm.intercept_

df_test_data = df_test.dropna()
Xtest=df_test_data[['x']].as_matrix()
Ytest=df_test_data[['y']].as_matrix()
print('Coeff of determination:',lm.score(Xtest,Ytest))
print('correlation is:',math.sqrt(lm.score(Xtest,Ytest)))
plt.scatter(X,Y)
yhat=1.000656386*X-0.10726546
fig=plt.plot(X,yhat,lw=4,c='black',label='regression line')
plt.title("trainig data")
plt.xlabel('Size',fontsize=20)
plt.ylabel('Price',fontsize=20)
plt.show()
plt.scatter(Xtest,Ytest)
yhat=1.000656386*Xtest-0.10726546
fig=plt.plot(Xtest,yhat,lw=4,c='black',label='regression line')
plt.title("testing data")
plt.xlabel('Size',fontsize=20)
plt.ylabel('Price',fontsize=20)
plt.show()