#Installing libraries

!pip install regressors
import numpy as np 

import pandas as pd 

from regressors import stats

from sklearn import linear_model as lm

import statsmodels.formula.api as sm

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression 

from sklearn.preprocessing import PolynomialFeatures

from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score

from sklearn.linear_model import LogisticRegression

from sklearn import metrics



import os

print(os.listdir("../input"))
#Data Preprocessing 

d = pd.read_csv("../input/survey.csv")

d = d.rename(index=str,columns={"Wr.Hnd":"WrHnd"})

d = d[["WrHnd","Height"]]

d = d.dropna()



#Model Fit 

inputDF = d[["WrHnd"]]

outcomeDF = d[["Height"]]

model = lm.LinearRegression()

results = model.fit(inputDF,outcomeDF)



#Regression coefficients

print(model.intercept_, model.coef_)
print("Adjusted R-Squared:\n",stats.adj_r2_score(model, inputDF, outcomeDF))



print("P-value:\n",stats.coef_pval(model, inputDF, outcomeDF))
#Data Preprocessing 

d = pd.read_csv("../input/survey.csv")

d = d.rename(index=str,columns={"Wr.Hnd":"WrHnd"})

d = d[["WrHnd","Height"]]

d = d.dropna()



#Adjusted R-Squared & P-vlaue genarated using Statsmodels

res = sm.ols(formula="Height ~ WrHnd",data=d).fit()

print(res.summary())

#Adjusted R-Squared & P-vlaue genarated for cubic polynomial transformation

res = sm.ols(formula="Height ~ WrHnd + I(WrHnd*WrHnd)+ I(WrHnd*WrHnd*WrHnd)",data=d).fit()

print(res.summary())
#Adjusted R-Squared & P-vlaue genarated for logarithmic transformation

res = sm.ols(formula = "Height ~ np.log(WrHnd)",data=d).fit()

print(res.summary())
#Declaring a dataframe

d = {'sno': [1,2,3,4,5,6],

     'Temperature': [0, 20, 40, 60, 80, 100], 

     'Pressure': [0.0002, 0.0012, 0.0060, 0.0300, 0.0900, 0.2700]}



df = pd.DataFrame(d)

df.head()
#Model Fit - Linear Regression

inputDF = df.iloc[:, 1:2].values 

outputDF = df.iloc[:, 2].values



lin = LinearRegression()  

lin.fit(inputDF, outputDF)
#Model Fit - Polynomial Regression

poly = PolynomialFeatures(degree = 4) 

inputDF_poly = poly.fit_transform(inputDF) 



poly.fit(inputDF_poly, outputDF) 

lin2 = LinearRegression() 

lin2.fit(inputDF_poly, outputDF) 
#Scatter Plot - Linear Regression

plt.scatter(inputDF, outputDF, color = 'blue') 

  

plt.plot(inputDF, lin.predict(inputDF), color = 'red') 

plt.title('Linear Regression') 

plt.xlabel('Temperature') 

plt.ylabel('Pressure') 

  

plt.show() 
#Scatter Plot - Polynomial Regression

plt.scatter(inputDF, outputDF, color = 'blue') 

  

plt.plot(inputDF, lin2.predict(poly.fit_transform(inputDF)), color = 'red') 

plt.title('Polynomial Regression') 

plt.xlabel('Temperature') 

plt.ylabel('Pressure') 

  

plt.show() 
#Model 1 - Evaluation using train_test_split

df = pd.read_csv("../input/mtcars.csv")



inputDF = df[["hp","am"]]

outputDF = df[["mpg"]]



X_train, X_test, y_train, y_test = train_test_split(inputDF, outputDF, test_size=0.2, random_state=0) 



model = lm.LinearRegression()

results = model.fit(X_train,y_train)



print("R - Squared value:\n",stats.adj_r2_score(model, X_train, y_train)) 



print(model.intercept_, model.coef_)
#Model 1 - Prediction

y_pred = model.predict(X_test) 

print("Predicted value:\n", y_pred) 

print("Originial value:\n", y_test) 

print("RMSE:\n", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))      
#Model 2 - Evaluation using train_test_split

inputDF = df[["hp"]]

outputDF = df[["mpg"]]



X_train, X_test, y_train, y_test = train_test_split(inputDF, outputDF, test_size=0.2, random_state=0) 

regressor = LinearRegression()  

regressor.fit(X_train, y_train) 

print("R - Squared value:\n",stats.adj_r2_score(regressor, X_train, y_train)) 

print(regressor.intercept_)

print(regressor.coef_)  
#Model 2 - Prediction

y_pred = regressor.predict(X_test)

print("Predicted value:\n", y_pred) 

print("Originial value:\n", y_test) 

print("RMSE:\n", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
d=pd.read_csv("../input/default.csv")

d.head()

d["balance"].describe()
#Add a new column DefaultYes which is 1 for Yes and 0 for No

d['DefaultYes'] = d['default'].map({'Yes': 1, 'No': 0})

d.head()
#Scikit-learn - Linear Regression

regressor = LinearRegression()  

inputDf = d[['balance']]

outputDf = d[['DefaultYes']]

regressor.fit(inputDf, outputDf) 

print(regressor.intercept_)

print(regressor.coef_[0]) 
#Input Dataframe

x1new = pd.DataFrame(np.hstack((np.arange(0,3000))))

x1new.columns=["balance"]

yp2new = regressor.predict(x1new)
#Scatter Plot

plt.scatter(d["balance"],d["DefaultYes"])

plt.plot(x1new,yp2new,color="red")

plt.show()
#Logistic Regression

inputDf = d[['balance']]

outputDf = d[['DefaultYes']].values.ravel()
#Model Fit - Logistic Regression

logisticRegr = LogisticRegression(solver='lbfgs')

logisticRegr.fit(inputDf, outputDf)

print(logisticRegr.intercept_)

print(logisticRegr.coef_)
#New Dataframe and prediction

x1new = pd.DataFrame(np.hstack((np.arange(0,3000))))

x1new.columns=["balance"]

yp2new = logisticRegr.predict(x1new)
#Plot

plt.scatter(d["balance"],d["DefaultYes"])

plt.plot(x1new,yp2new,color="red")

plt.show()
d = pd.read_csv("../input/default.csv")

d = d[["default","balance","income","student"]]

d.head()
#Add a new column DefaultYes which is 1 for Yes and 0 for No

d['DefaultYes'] = d['default'].map({'Yes': 1, 'No': 0})

d.head()
d = d.drop(['default'], axis=1)

d.head()
#Categorical Predictors

d = pd.get_dummies(d, prefix=['student'], columns=['student'])

d.head()
#Model - 1 Fit

inputDF = d[["balance","income","student_No","student_Yes"]]

outputDF = d[["DefaultYes"]].values.ravel()



logisticRegr = LogisticRegression(solver='lbfgs')

x_train, x_test, y_train, y_test = train_test_split(inputDF, outputDF, test_size=0.25, random_state=0)

logisticRegr.fit(inputDF, outputDF)



print(logisticRegr.intercept_)

print(logisticRegr.coef_)
#Model - 1 Validation

y_pred = logisticRegr.predict(x_test)

#print(r2_score(y_test, y_pred)) 

print("R - Squared value:\n",stats.adj_r2_score(logisticRegr, x_train, y_train)) 

print("RMSE:\n", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
#Model - 2 Fit

inputDF = d[["income","student_No","student_Yes"]]

outputDF = d[["DefaultYes"]].values.ravel()



logisticRegr = LogisticRegression(solver='lbfgs')

x_train, x_test, y_train, y_test = train_test_split(inputDF, outputDF, test_size=0.25, random_state=0)

logisticRegr.fit(x_train, y_train)



print(logisticRegr.intercept_)

print(logisticRegr.coef_)
#Model - 2 Validation

y_pred = logisticRegr.predict(x_test)

print("R - Squared value:\n",stats.adj_r2_score(logisticRegr, x_train, y_train)) 

print("RMSE:\n", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

#print(r2_score(y_test, y_pred)) 