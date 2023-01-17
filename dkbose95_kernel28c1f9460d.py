import pandas as pd



import numpy as np



from matplotlib import pyplot as plt

import seaborn as sns

 

%matplotlib inline 

%config InlineBackend.figure_format = 'retina' 

import warnings  

warnings.filterwarnings('ignore') 

import os 

print(os.listdir("../input")) 
cars=pd.read_csv("../input/car data.csv")

cars.sample()
cars['Car_Name'].value_counts()
cars.isnull().any()
from sklearn.preprocessing import LabelEncoder

lb = LabelEncoder()

cars["trans_code"] = lb.fit_transform(cars["Transmission"])

cars["Transmission"].replace(cars["trans_code"], inplace=True)

cars["seller_code"] = lb.fit_transform(cars["Seller_Type"])

cars["Seller_Type"].replace(cars["seller_code"], inplace=True)

cars["fuel_code"] = lb.fit_transform(cars["Fuel_Type"])

cars["Fuel_Type"].replace(cars["fuel_code"], inplace=True)
year=cars["Year"]

mile=cars["Kms_Driven"]

rate=mile/((2019-year)*365)

cars["Rating"]=rate
var = 'Kms_Driven'

data = pd.concat([cars["Selling_Price"], cars[var]], axis=1)

data.plot.scatter(x=var, y='Selling_Price');
var = 'Year'

data = pd.concat([cars["Selling_Price"], cars[var]], axis=1)

data.plot.scatter(x=var, y='Selling_Price',);
var = 'Present_Price'

data = pd.concat([cars["Selling_Price"], cars[var]], axis=1)

data.plot.scatter(x=var, y='Selling_Price',);
from sklearn.model_selection import train_test_split

import statsmodels.api as sm
X=cars[["Present_Price","fuel_code","seller_code","trans_code","Year","Kms_Driven","Owner"]]

y=cars["Selling_Price"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

X = sm.add_constant(X)

model = sm.OLS(y_train.astype(float), X_train.astype(float)).fit()

predictions = model.predict(X_test)

model.summary()
X=cars[["Present_Price","fuel_code","seller_code","trans_code","Rating","Owner"]]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

X = sm.add_constant(X)

model = sm.OLS(y_train.astype(float), X_train.astype(float)).fit()

predictions = model.predict(X_test)

model.summary()
X=cars[["Present_Price","fuel_code","seller_code","trans_code","Year","Kms_Driven","Owner"]]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

X = sm.add_constant(X)

model1 = sm.OLS(y_train.astype(float), X_train.astype(float)).fit()

r1=model1.rsquared

print('R squared value=', +r1)



predictions = model1.predict(X_test)

predictions=pd.DataFrame(predictions)

predictions=predictions.reset_index()

test_index=y_test.reset_index()["Selling_Price"]

ax=test_index.plot(label="originals",figsize=(12,6),linewidth=2,color="r")

ax=predictions[0].plot(label = "predictions",figsize=(12,6),linewidth=2,color="g")

plt.legend(loc='upper right')

plt.title("Linear Regressor")

plt.xlabel("index")

plt.ylabel("values")

plt.show()
X=cars[["Present_Price","fuel_code","trans_code","seller_code","Rating","Owner"]]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

X = sm.add_constant(X)

model2 = sm.OLS(y_train.astype(float), X_train.astype(float)).fit()

r2=model2.rsquared

print('R squared value=', +r2)
X=cars[["fuel_code","seller_code","trans_code","Year","Kms_Driven","Owner"]]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

X = sm.add_constant(X)

model3 = sm.OLS(y_train.astype(float), X_train.astype(float)).fit()

r3=model3.rsquared

print('R squared value=', +r3)
X=cars[["Present_Price","fuel_code","trans_code","Year","Kms_Driven","Owner"]]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

X = sm.add_constant(X)

model4 = sm.OLS(y_train.astype(float), X_train.astype(float)).fit()

r4=model4.rsquared

print('R squared value=', +r4)
X=cars[["Present_Price","seller_code","trans_code","Year","Kms_Driven","Owner"]]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

X = sm.add_constant(X)

model5 = sm.OLS(y_train.astype(float), X_train.astype(float)).fit()

r5=model5.rsquared

print('R squared value=', +r5)
X=cars[["Present_Price","fuel_code","seller_code","Year","Kms_Driven","Owner"]]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

X = sm.add_constant(X)

model6 = sm.OLS(y_train.astype(float), X_train.astype(float)).fit()

r6=model6.rsquared

print('R squared value=', +r6)
##Plot results of linear regression by considering different features



data=[r1,r2,r3,r4,r5,r6]



plt.subplots(figsize = (15,8))

ax = plt.plot(data)

plt.axis([0, 5, 0.6, 1])

plt.ylabel('Rsquared value',size=25)

plt.xlabel('Trial',size=25)

labels = (['default', 'w rating', 'w/o present \n price','w/o seller','w/o fuel','w/o \n transmission'])

val = [0,1,2,3,4,5]  

plt.xticks(val, labels);

plt.show()
from sklearn.ensemble import AdaBoostRegressor

from sklearn.datasets import make_regression
regr = AdaBoostRegressor()

X=cars[["Present_Price","fuel_code","trans_code","Year","Kms_Driven","Owner"]]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

model=regr.fit(X_train, y_train) 

x3=regr.score(X,y)

print('R squared value=', +x3)

predictions = model.predict(X_test)

predictions=pd.DataFrame(predictions)



predictions=predictions.reset_index()

test_index=y_test.reset_index()["Selling_Price"]

ax=test_index.plot(label="originals",figsize=(12,6),linewidth=2,color="r")

ax=predictions[0].plot(label = "predictions",figsize=(12,6),linewidth=2,color="g")

plt.legend(loc='upper right')

plt.title("ADABOOST Regressor")

plt.xlabel("index")

plt.ylabel("values")

plt.show()
from sklearn.tree import DecisionTreeRegressor

dtr = DecisionTreeRegressor()

from sklearn.metrics import r2_score

from sklearn.model_selection import cross_val_score
X=cars[["Present_Price","fuel_code","trans_code","Year","Kms_Driven","Owner"]]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1)

dtr.fit(X_train,y_train)

predicts=dtr.predict(X_test)

prediction=pd.DataFrame(predicts)

R_2=r2_score(y_test,prediction)





    

    # Printing results  

print(dtr,"\n") 

print("R squared value=",R_2,"\n")



    

    # Plot for prediction vs originals

test_index=y_test.reset_index()["Selling_Price"]

ax=test_index.plot(label="originals",figsize=(12,6),linewidth=2,color="r")

ax=prediction[0].plot(label = "predictions",figsize=(12,6),linewidth=2,color="g")

plt.legend(loc='upper right')

plt.title("Decision Tree Regressor")

plt.xlabel("index")

plt.ylabel("values")

plt.show()
regr_2 = AdaBoostRegressor(DecisionTreeRegressor())



model=regr_2.fit(X_train, y_train)



y_2 = regr_2.predict(X_test)



x4=regr_2.score(X_test,y_test)

print('R squared value=', +x4)



predictions=pd.DataFrame(y_2)

predictions=predictions.reset_index()

test_index=y_test.reset_index()["Selling_Price"]

ax=test_index.plot(label="originals",figsize=(12,6),linewidth=2,color="r")

ax=predictions[0].plot(label = "predictions",figsize=(12,6),linewidth=2,color="g")

plt.legend(loc='upper right')

plt.title("Decision tree with ADABOOST Regressor")

plt.xlabel("index")

plt.ylabel("values")

plt.show()
from tabulate import tabulate

print(tabulate([['Linear regression', r1], ['Adaboost linear regression', x3],['Decision tree regressor',R_2],['Decision tree with adaboost',x4]], headers=['Model used', 'Rsquared value']))