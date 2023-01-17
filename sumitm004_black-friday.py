import pandas as pd

import numpy as np

from sklearn.metrics import mean_squared_error

import math
traindf=pd.read_csv("../input/BlackFriday.csv")

traindf.columns
traindf.shape
traindf.head()
traindf['City_Category'].value_counts()
traindf['Age'].value_counts()
traindf.isna().sum()
traindf.nunique(axis=0)
# Filling_the_Null_values



traindf['Product_Category_2'].fillna(value=traindf['Product_Category_2'].mean(),inplace=True);

traindf['Product_Category_3'].fillna(value=traindf['Product_Category_3'].mean(),inplace=True);
traindf.isnull().sum(axis=0)
traindf.head()
traindf=traindf.astype({"Age": str})

traindf['Age'].dtype
#Setting_UserID_as_index



traindf.set_index("User_ID",inplace=True)

traindf.head()
# Changing_the_Gender_and_CityCategory_into_Numerical_Data_type_for_training



dic1={'F':1,'M':2}

dic2={'A':1,'B':2,'C':3}

traindf=traindf.replace({'Gender':dic1,'City_Category':dic2})
# Changing_the_Object_into_Numerical_Data_type_for_training



df1=traindf['Stay_In_Current_City_Years'].unique()

df1=df1.tolist()



li_1=list(range(1,6))

dic3=dict(zip(df1,li_1))

traindf=traindf.replace({'Stay_In_Current_City_Years':dic3})
traindf.head()
# Changing_the_Age_into_Numerical_Data_type_for_training



df2=traindf['Age'].unique()

df2=df2.tolist()



li_2=list(range(1,8))

dic4=dict(zip(df2,li_2))

traindf=traindf.replace({'Age':dic4})
traindf.head()
# Using_labelEncoder_on_Product_ID



from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

traindf['Product_ID']=le.fit_transform(traindf['Product_ID'])



# Splitting_the_Training_Dataset



Ytr=traindf['Purchase']

Xtr=traindf.drop(columns=['Purchase'])



from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(Xtr,Ytr,random_state=0)
# 1_Using_Linear_Regression
from sklearn.linear_model import Ridge

from sklearn.metrics import mean_squared_error

Rid=Ridge().fit(X_train,y_train)
y_pred=Rid.predict(X_test)
Rid.score(X_test,y_test)
# 2_Using_polynomial_Regression_with_different_Degrees
from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import LinearRegression

poly=PolynomialFeatures(degree=2)

X_poly=poly.fit_transform(Xtr)

X_train,X_test,y_train,y_test=train_test_split(X_poly,Ytr,random_state=0)

linreg=LinearRegression().fit(X_train,y_train)
y_pred=linreg.predict(X_test)

math.sqrt(mean_squared_error(y_test,y_pred))
linreg.score(X_test,y_test)
# 3_Using_Lasso_Regression
from sklearn.linear_model import Lasso

linlasso=Lasso(alpha=1,max_iter=100).fit(X_train,y_train)
y_pred=linlasso.predict(X_test)

math.sqrt(mean_squared_error(y_test,y_pred))
# 4_Using_Boosted_Gradient_Descent
import xgboost as xgb



xgb_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.05,

                max_depth = 10, alpha = 10, n_estimators = 1000)
# Training_the_model



xgb_reg.fit(X_train,y_train)
# Finding_the_RMSE_for_XGB_Model



predictions_xgb=xgb_reg.predict(X_test)

math.sqrt(mean_squared_error(y_test,predictions_xgb))