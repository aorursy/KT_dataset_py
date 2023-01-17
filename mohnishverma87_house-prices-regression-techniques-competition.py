# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import math

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns





from sklearn.preprocessing import PolynomialFeatures

from sklearn import metrics

from sklearn.metrics import mean_squared_log_error

#from sklearn.metrics import 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/"))



# Any results you write to the current directory are saved as output.
# evaluation=evaluation.truncate(before=-1, after=-1)
evaluation = pd.DataFrame({"Model":[],"Details":[],"RMSE":[],"R2-train":[],"Adj-R2-train":[],"R2-test":[],"Adj-R2-test":[],"RMSLE":[],"VARIANCE":[]})

evaluation
dataset = pd.read_csv("../input/train.csv")
# Differentiate numerical features (minus the target) and categorical features

print("Numerical features : " + str(len(dataset.select_dtypes(exclude = ["object"]).columns)))

print("Categorical features : " + str(len(dataset.select_dtypes(include = ["object"]).columns)))
# We will take only Numerical Features as of now:

features = [] 

for i in dataset.select_dtypes(exclude = ["object"]).columns:

        features.append(i)
dataset[features].head()
dataset[features].info()
def adjustedR2(r2,n,k):

    return r2 - (k - 1)/(n - k)*(1 - r2)
#features=[]

#for i in dataset.columns:

#    features.append(i)
dataset[features].corr().sort_values(["SalePrice"],ascending=False).head()
# Lets have Simple Linear Regression B/W sqft_living' and 'price'

from sklearn.model_selection import train_test_split

train_data,test_data = train_test_split(dataset,test_size=0.2,random_state=42)

X_train = train_data["OverallQual"].values.reshape(-1,1)

y_train = train_data[["SalePrice"]].values.reshape(-1,1)

X_test = test_data["OverallQual"].values.reshape(-1,1)

y_test = test_data["SalePrice"].values.reshape(-1,1)
from sklearn.linear_model import LinearRegression

linreg = LinearRegression()

#Applying k-fold cross validation[The purpose of cross-validation is model checking, not model building.]

# Divide the training data into 10 folds and create a accuracies score list by fitting b/w 9fold training and 1fold testing data

# In short fittting X_train and y_train 10 times, and creating 10 accuracies

# accuracies.mean()----> BIAS(R2 Score)

#accuracies.std() -----> VARIANCE

from sklearn.model_selection import cross_val_score

#score = 'mean_absolute_error'

accuracies = cross_val_score(linreg,X=X_train,y=y_train, cv=10)



linreg.fit(X_train,y_train)

y_pred = linreg.predict(X_test)
mse = np.sqrt(metrics.mean_squared_error(y_test,y_pred))

R2_test = accuracies.mean()

Adj_R2_test_array= []

for i in accuracies:

    Adj_R2_test_array.append(adjustedR2(i,X_train.shape[0],X_train.shape[1]))

Adj_R2_test_array = np.asarray(Adj_R2_test_array)

Adj_R2_test = Adj_R2_test_array.mean()

#BIAS = accuracies.mean()

VARIANCE = accuracies.std()

#RMSLE = np.sqrt(mean_squared_log_error(y_test[0],y_pred_exp[0]))

print(mse, R2_test, Adj_R2_test)
evaluation=evaluation.append({"Model":"Simple Linear, CV=10","Details":"","RMSE":mse, "R2-train":"","Adj-R2-train":Adj_R2_test,"R2-test":R2_test,"Adj-R2-test":"","RMSLE":"","VARIANCE":VARIANCE},ignore_index=True)
evaluation
#evaluation.drop(evaluation[evaluation.Model=="Polynomial Linear-3, CV=10"].index, inplace=True)
import statsmodels.api as sm

X1 = sm.add_constant(X_train)

result = sm.OLS(y_train,X1).fit()

print(result.rsquared,result.rsquared_adj)
plt.scatter(X_test,y_test,color='darkgreen',label="Data", alpha=.1)

plt.plot(X_test,linreg.predict(X_test),color="red",label="Predicted Regression Line")
# Lets have a multivariate Linear Regression
train_data,test_data = train_test_split(dataset,test_size=0.2,random_state=42)
# We will take only those independent variables which are corelated to the output variable "price"

features1=["OverallQual","GrLivArea","GarageCars","GarageArea","TotalBsmtSF","1stFlrSF","FullBath","TotRmsAbvGrd","YearBuilt","YearRemodAdd"]

linreg2 = LinearRegression()

X_train = train_data[features1].values

y_train = train_data[["SalePrice"]].values

X_test = test_data[features1].values

y_test = test_data[["SalePrice"]].values



from sklearn.model_selection import cross_val_score

#score = 'mean_absolute_error'

accuracies = cross_val_score(linreg2,X=X_train,y=y_train, cv=10)

linreg2.fit(X_train,y_train)

y_pred = linreg2.predict(X_test)
correlations=dataset[features1].corr()

mask = np.zeros_like(correlations, dtype=np.bool)

mask[np.triu_indices_from(mask)]=True

fig,ax = plt.subplots(figsize=(15,15))

plt.title("Pearson Correlation Matrix", fontsize=5)

sns.heatmap(correlations,vmax=1,square=True,annot=True,mask=mask)
mse2 = np.sqrt(metrics.mean_squared_error(y_test,y_pred))

R2_test2 = accuracies.mean()

Adj_R2_test_array= []

for i in accuracies:

    Adj_R2_test_array.append(adjustedR2(i,X_test.shape[0],X_test.shape[1]))

Adj_R2_test_array = np.asarray(Adj_R2_test_array)

Adj_R2_test2 = Adj_R2_test_array.mean()

VARIANCE = accuracies.std()

evaluation=evaluation.append({"Model":"Multivariate Linear, CV=10","Details":"","RMSE":mse2, "R2-train":"","Adj-R2-train":"","R2-test":R2_test2,"Adj-R2-test":Adj_R2_test2,"RMSLE":"","VARIANCE":VARIANCE},ignore_index=True)
evaluation
# Lets have a Polynomial Linear Regression

linreg3 = LinearRegression()

polyreg = PolynomialFeatures(degree = 2)

X_train = train_data[features1].values

y_train = train_data[["SalePrice"]].values

X_test = test_data[features1].values

y_test = test_data[["SalePrice"]].values



X_poly_train = polyreg.fit_transform(X_train) 

X_poly_test = polyreg.fit_transform(X_test)
from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(linreg3,X=X_poly_train,y=y_train, cv=10)



linreg3.fit(X_poly_train,y_train)

y_pred = linreg3.predict(X_poly_test)
mse3 = np.sqrt(metrics.mean_squared_error(y_test,y_pred))

R2_test3 = accuracies.mean()

Adj_R2_test_array= []

for i in accuracies:

    Adj_R2_test_array.append(adjustedR2(i,X_poly_test.shape[0],X_poly_test.shape[1]))

Adj_R2_test_array = np.asarray(Adj_R2_test_array)

Adj_R2_test3 = Adj_R2_test_array.mean()

VARIANCE = accuracies.std()

evaluation=evaluation.append({"Model":"Polynomial Linear-2, CV=10","Details":"","RMSE":mse3, "R2-train":"","Adj-R2-train":"","R2-test":R2_test3,"Adj-R2-test":Adj_R2_test3,"RMSLE":"","VARIANCE":VARIANCE},ignore_index=True)
evaluation
# WE WILL EVALUATE XGBOOST MODEL

train_data,test_data = train_test_split(dataset,test_size=0.2,random_state=42)

features1=["OverallQual","GrLivArea","GarageCars","GarageArea","TotalBsmtSF","1stFlrSF","FullBath","TotRmsAbvGrd","YearBuilt","YearRemodAdd"]

X_train = train_data[features1].values

y_train = train_data[["SalePrice"]].values

X_test = test_data[features1].values

y_test = test_data[["SalePrice"]].values



from xgboost import XGBClassifier

xgb = XGBClassifier()

xgb.fit(X_train,y_train)
y_pred = xgb.predict(X_test)
mse4 = np.sqrt(metrics.mean_squared_error(y_test,y_pred))

R2_test4 = accuracies.mean()

Adj_R2_train_array= []

for i in accuracies:

    Adj_R2_train_array.append(adjustedR2(i,X_train.shape[0],X_train.shape[1]))

Adj_R2_train_array = np.asarray(Adj_R2_train_array)

Adj_R2_train = Adj_R2_train_array.mean()

VARIANCE = accuracies.std()

evaluation=evaluation.append({"Model":"XGBoost-train","Details":"","RMSE":mse4, "R2-train":"","Adj-R2-train":Adj_R2_train,"R2-test":R2_test4,"Adj-R2-test":"","RMSLE":"","VARIANCE":VARIANCE},ignore_index=True)
evaluation
# SO XGBoost seems to have better prediction, we will take this model.

test_dataset = pd.read_csv("../input/test.csv")

test_dataset1=test_dataset[features1]

test_dataset1.head()
test_dataset1["GarageCars"].fillna( method ='ffill', inplace = True) 

test_dataset1["GarageArea"].fillna( method ='ffill', inplace = True) 

test_dataset1["TotalBsmtSF"].fillna( method ='ffill', inplace = True)

test_dataset1.info()
train_data,test_data = train_test_split(dataset,test_size=0.2,random_state=42)

features1=["OverallQual","GrLivArea","GarageCars","GarageArea","TotalBsmtSF","1stFlrSF","FullBath","TotRmsAbvGrd","YearBuilt","YearRemodAdd"]

X_train = train_data[features1].values

y_train = train_data[["SalePrice"]].values

X_test = test_dataset1.values



from xgboost import XGBClassifier

xgb = XGBClassifier()

xgb.fit(X_train,y_train)

y_pred = xgb.predict(X_test)
linreg3 = LinearRegression()

polyreg = PolynomialFeatures(degree = 2)

X_train = train_data[features1].values

y_train = train_data[["SalePrice"]].values

X_test = test_dataset1.values





X_poly_train = polyreg.fit_transform(X_train) 

X_poly_test = polyreg.fit_transform(X_test)



from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(linreg3,X=X_poly_train,y=y_train, cv=10)



linreg3.fit(X_poly_train,y_train)

y_pred = linreg3.predict(X_poly_test)
submit = pd.DataFrame()

test_Id = test_dataset["Id"]

submit['Id'] = test_Id

submit['SalePrice'] = y_pred

submit.to_csv('price-op.csv', index=False)