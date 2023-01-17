import pandas as pd

from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LinearRegression

import xgboost as xgb
bf_train= pd.read_csv("../input/traindataset/train.csv")
bf_train.isnull().sum()
#Dropping the Column Product_ID

bf_train=bf_train.drop("Product_ID", axis=1)
lb=LabelEncoder()
#Converting the Categorical Variables to Numberic Variables Using Label Encoder

bf_train["Gender"]=lb.fit_transform(bf_train["Gender"])

bf_train["Age"]=lb.fit_transform(bf_train["Age"])

bf_train["City_Category"]=lb.fit_transform(bf_train["City_Category"])

bf_train["Stay_In_Current_City_Years"]=lb.fit_transform(bf_train["Stay_In_Current_City_Years"])
#Filling the missing values or null values with 0

bf_train["Product_Category_2"]=bf_train["Product_Category_2"].fillna(0)

bf_train["Product_Category_3"]=bf_train["Product_Category_3"].fillna(0)
#Forming x_train and y_train 

x_train=bf_train.drop("Purchase", axis=1)

y_train=bf_train["Purchase"]
#Linear Regression Model

lr_model=LinearRegression()

lr_model.fit(x_train,y_train)

lr_model.score(x_train,y_train)
#Using XGBoost Regressor (A boosting  & Regularising method) as the score of Linear Regression Model is very low.

xgb_model=xgb.XGBRegressor()

xgb_model.fit(x_train,y_train)

xgb_model.score(x_train,y_train)
bf_test=pd.read_csv("../input/testdataset/test.csv")
#Dropping Product_ID column in Test Dataset

bf_test=bf_test.drop("Product_ID", axis=1)
#Converting Categorical Data in Test Data

bf_test["Gender"]=lb.fit_transform(bf_test["Gender"])

bf_test["Age"]=lb.fit_transform(bf_test["Age"])

bf_test["City_Category"]=lb.fit_transform(bf_test["City_Category"])

bf_test["Stay_In_Current_City_Years"]=lb.fit_transform(bf_test["Stay_In_Current_City_Years"])
#Filling missing values/null values with 0

bf_test["Product_Category_2"]=bf_test["Product_Category_2"].fillna(0)

bf_test["Product_Category_3"]=bf_test["Product_Category_3"].fillna(0)
prediction=xgb_model.predict(bf_test)
prediction