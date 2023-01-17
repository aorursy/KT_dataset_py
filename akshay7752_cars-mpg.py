#Problem statement- To predict MPG(miles per gallon) of car on various features such as 'model', 'mpg', 'cyl', 'disp', 'hp', 'drat', 'wt', 'qsec', 'vs', 'am','gear', 'carb'

#Importing libraries



import pandas as pd

from sklearn import linear_model

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score, mean_squared_error

from sklearn.preprocessing import PolynomialFeatures

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge

from sklearn.linear_model import Lasso



#import Dataset



path="../input/mtcars.csv"

cars=pd.read_csv(path)

print(cars)
#To get data insights

cars.head()
#EDA



#Univariate



#quantittative variables



#1) To understand distribution of mpg

sns.distplot(cars.mpg,kde = False)
#2) To understand distribution of disp

sns.distplot(cars.disp,kde = False)
#3) To understand distribution of hp

sns.distplot(cars.hp,kde = False)
#4) To understand distribution of disp

sns.distplot(cars.drat,kde = False)
#5) To understand distribution of disp

sns.distplot(cars.wt,kde = False)
#5) To understand distribution of disp

sns.distplot(cars.qsec,kde = False)
#Categoricl variables



#6) To understand distribution of gear

sns.countplot(data=cars,x=cars.gear)
#7) To understand distribution of carb

sns.countplot(data=cars,x=cars.carb)
#8) To understand distribution of vs

sns.countplot(data=cars,x=cars.vs)
#9) To understand distribution of am

sns.countplot(data=cars,x=cars.am)
#10) To understand distribution of am

sns.countplot(data=cars,x=cars.cyl)
#Bivariate 



#1) To understand relation between  weight and mpg

sns.lmplot(x='wt',y='mpg',data=cars,hue='cyl')
#2) To understand relation between  displacement and mpg

sns.lmplot(x='disp',y='mpg',data=cars,hue='cyl')
#3) To understand relation between  Horsepower and mpg

sns.lmplot(x='hp',y='mpg',data=cars,hue='cyl')
#4) To understand relation between  drat and mpg

sns.lmplot(x='drat',y='mpg',data=cars,hue='cyl')
#5) To understand relation between  qsec and mpg

sns.lmplot(x='qsec',y='mpg',data=cars,hue='cyl')
#6) To understand relation between  vs and mpg

sns.violinplot(x='vs',y='mpg',data=cars)

#sns.boxplot(x='vs',y='mpg',data=cars)
#7) To understand relation between  am and mpg

sns.violinplot(x='am',y='mpg',data=cars)

#sns.boxplot(x='am',y='mpg',data=cars)
#8) To understand relation between  gear and mpg

sns.violinplot(x='gear',y='mpg',data=cars)

#sns.boxplot(x='gear',y='mpg',data=cars)
#9) To understand relation between  carb and mpg

sns.violinplot(x='carb',y='mpg',data=cars)

#sns.boxplot(x='carb',y='mpg',data=cars)
#Regression



cars.columns

x=cars.drop(['mpg','model'],axis=1)

y=cars.mpg

print(x.head())

y.head()
#linear regression



x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)

linreg=linear_model.LinearRegression()

linreg.fit(x_train,y_train)



y_pred_train=linreg.predict(x_train)

print(y_pred_train)

train_accuracy=r2_score(y_train,y_pred_train)

print(train_accuracy)

train_err=mean_squared_error(y_train,y_pred_train)

print(train_err)



y_pred_test=linreg.predict(x_test)

print(y_pred_test)

test_accuracy=r2_score(y_test,y_pred_test)

print(test_accuracy)

test_err=mean_squared_error(y_test,y_pred_test)

print(test_err)
#polynomial regression for degree=2



poly=PolynomialFeatures(degree=2)

x_poly=poly.fit_transform(x)



x1_train,x1_test,y1_train,y1_test=train_test_split(x_poly,y,random_state=1,test_size=0.2)



linreg1=linear_model.LinearRegression()

linreg1.fit(x1_train,y1_train)



y1_pred_test=linreg1.predict(x1_test)

print(y1_pred_test)

test_accuracy1=r2_score(y1_test,y1_pred_test)

print(test_accuracy1)

test_err1=mean_squared_error(y1_test,y1_pred_test)

print(test_err1)



y1_pred_train=linreg1.predict(x1_train)

print(y1_pred_train)

train_accuracy1=r2_score(y1_train,y1_pred_train)

print(train_accuracy1)

train_err1=mean_squared_error(y1_train,y1_pred_train)

print(train_err1)

#polynomial regression for degree=3



poly=PolynomialFeatures(degree=3)

x_poly3=poly.fit_transform(x)



x2_train,x2_test,y2_train,y2_test=train_test_split(x_poly3,y,random_state=1,test_size=0.2)



linreg2=linear_model.LinearRegression()

linreg2.fit(x2_train,y2_train)



y2_pred_test=linreg2.predict(x2_test)

print(y2_pred_test)

test_accuracy2=r2_score(y2_test,y2_pred_test)

print(test_accuracy2)

test_err2=mean_squared_error(y2_test,y2_pred_test)

print(test_err2)



y2_pred_train=linreg2.predict(x2_train)

print(y2_pred_train)

train_accuracy2=r2_score(y2_train,y2_pred_train)

print(train_accuracy2)

train_err2=mean_squared_error(y2_train,y2_pred_train)

print(train_err2)
#Regulrization Techiniques



#1)Ridge Regression



x3_train,x3_test,y3_train,y3_test=train_test_split(x,y,test_size=0.2,random_state=1)



rr=Ridge(alpha=100)

rr.fit(x3_train,y3_train)



y_pred_rr_test=rr.predict(x3_test)

print(y_pred_rr_test)

test_accuracy_rr=r2_score(y3_test,y_pred_rr_test)

print(test_accuracy_rr)

test_err_rr=mean_squared_error(y3_test,y_pred_rr_test)

print(test_err_rr)



y_pred_rr_train=rr.predict(x3_train)

print(y_pred_rr_train)

train_accuracy_rr=r2_score(y3_train,y_pred_rr_train)

print(train_accuracy_rr)

train_err_rr=mean_squared_error(y3_train,y_pred_rr_train)

print(train_err_rr)
#2)Lasso Regression



x4_train,x4_test,y4_train,y4_test=train_test_split(x,y,test_size=0.2,random_state=3)



lr=Lasso(alpha=100)

lr.fit(x4_train,y4_train)



y_pred_lr_test=lr.predict(x4_test)

print(y_pred_lr_test)

test_accuracy_lr=r2_score(y4_test,y_pred_lr_test)

print(test_accuracy_lr)

test_elr_lr=mean_squared_error(y4_test,y_pred_lr_test)

print(test_elr_lr)



y_pred_lr_train=lr.predict(x4_train)

print(y_pred_lr_train)

train_accuracy_lr=r2_score(y4_train,y_pred_lr_train)

print(train_accuracy_lr)

train_elr_lr=mean_squared_error(y4_train,y_pred_lr_train)

print(train_elr_lr)

#Model Selection

#From r2 score and MSE model 1 with linear regression is best model for prediction

#Prediction on best model

y_pred_test=linreg.predict(x_test)

print(y_pred_test)