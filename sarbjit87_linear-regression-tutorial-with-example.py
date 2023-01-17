import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats

from sklearn.datasets import load_boston
boston = load_boston()
print("Shape for Data is", boston.data.shape) 

print("Shape for Target is", boston.target.shape) 

print("Names of header :\n", boston.feature_names)

print("Dataset description :\n", boston.DESCR)
# Create a Pandas Data Frame from this data set



df = pd.DataFrame(boston.data)

df.head(2)
df.columns = boston.feature_names

df['MEDV'] = boston.target

df.head(2)
df.info()
df.isnull().sum()
# Descriptive Statistics

df.describe()
df.hist(bins=30, figsize=(18,18))
plt.figure(figsize=(25, 25))

for i, col in enumerate(boston.feature_names):

    plt.subplot(4, 4 , i+1)

    sns.regplot(x=col, y='MEDV', data=df)

    plt.title("%s vs MEDV" %(col))
sns.set(rc={'figure.figsize':(8,8)})

sns.heatmap(df.corr().round(2), square=True, cmap='RdYlGn', annot=True)
pearson_coef1, p_value1 = stats.pearsonr(df['RM'], df['MEDV'])

print("The Pearson Correlation Coefficient for RM is", pearson_coef1, " with a P-value of P = ", p_value1)  



pearson_coef2, p_value2 = stats.pearsonr(df['LSTAT'], df['MEDV'])

print("The Pearson Correlation Coefficient for LSTAT is", pearson_coef2, " with a P-value of P = ", p_value2) 



print("Is there strong evidence if corelation is signifcant for RM :- ", p_value1 < 0.001) 

print("Is there strong evidence if corelation is signifcant for LSTAT :- ", p_value2 < 0.001) 
from sklearn.linear_model import LinearRegression



# Create Linear Regression Object

lm1 = LinearRegression()

X1 = df[['RM']]

Y1 = df[['MEDV']] # Target



# Fit (Train) the model

lm1.fit(X1,Y1)



print("Intercept for the model is", lm1.intercept_, "and the scope is",lm1.coef_)



# Prediction

Yout1 = lm1.predict(X1)



# Actual and Predicted values (first five)

print("Predicted Values:",Yout1[0:5])

print("Actual Values:",Y1.values[0:5])

lm2 = LinearRegression()

X2 = df[['RM', 'LSTAT']]

Y2 = df[['MEDV']]



# Fit (Train) the model

lm2.fit(X2,Y2)



print("Intercept for the model is", lm2.intercept_, "and the scope is",lm2.coef_)



# Prediction

Yout2 = lm2.predict(X2)



# Actual and Predicted values (first five)

print("Predicted Values:",Yout2[0:5])

print("Actual Values:",Y2.values[0:5])
# Seaborn library to be used for Residual Plot

plt.figure(figsize=(6,6))

sns.residplot(df['RM'],df['MEDV'])

plt.show()
plt.figure(figsize=(6,6))

ax1 = sns.distplot(df['MEDV'], hist=False, color="r", label="Actual")

sns.distplot(Yout2, hist=False, color="b", label="Fitted", ax=ax1)
from sklearn.metrics import mean_squared_error



# Simple Linear Regression



mse1 = mean_squared_error(Y1,Yout1)

print("Mean square error for simple linear regression is",mse1)

print("R-Square value for simple linear regression is", lm1.score(X1,Y1))

print("\n")



# Multiple Linear Regression



mse2 = mean_squared_error(Y2,Yout2)

print("Mean square error for mulitple linear regression is",mse2)

print("R-Square value for multiple linear regression is", lm2.score(X2,Y2))
# First step that we will take is to seperate target data



y_data = df['MEDV']

x_data = df.drop('MEDV',axis=1)



from sklearn.model_selection import train_test_split



# Split the data into test and training (15% as test data)



x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.15, random_state=10)



print("Number of test samples :", x_test.shape[0])

print("Number of training samples:",x_train.shape[0])



lm = LinearRegression()



# Fit (Train) the model using the training data 



lm.fit(x_train[['RM','LSTAT']],y_train)



# Prediction using Training Data

yout_train = lm.predict(x_train[['RM','LSTAT']])

print(yout_train[0:5])



# Prediction using Test Data

yout_test = lm.predict(x_test[['RM','LSTAT']])

print(yout_test[0:5])

print("\n")



# Model accuracy using Test Data

mse_test = mean_squared_error(y_test,yout_test)

print("Mean square error is",mse_test)

print("R-Square value using test data is", lm.score(x_test[['RM','LSTAT']],y_test))

print("\n")



# Model accuracy using Training Data

mse_train = mean_squared_error(y_train,yout_train)

print("Mean square error is",mse_train)

print("R-Square value using training data is", lm.score(x_train[['RM','LSTAT']],y_train))
from sklearn.model_selection import cross_val_score, KFold



rcross = cross_val_score(lm, x_data, y_data, cv=KFold(n_splits=5,shuffle=True))



print(rcross)

print("The mean of the folds are", rcross.mean())