# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

import seaborn as sns







df_train = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

df_test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")

display(df_train.head())

display(df_train.columns)
display(df_test.head())



display(df_test.columns)
display(df_test.info())
fig,ax=plt.subplots(figsize=(30,15))



sns.heatmap(df_train.isnull(),ax=ax)

fig,ax2=plt.subplots(figsize=(30,15))

sns.heatmap(df_test.isnull(),ax=ax2)

numeric_train = df_train.select_dtypes(exclude=['object']).drop(["SalePrice",'Id'], axis = 1)

display(numeric_train.head())



id_test = df_test["Id"]

y_train = df_train["SalePrice"]



numeric_test = df_test.select_dtypes(exclude=['object']).drop('Id', axis = 1)

display(numeric_test.head())





display(numeric_train.shape)

display(numeric_test.shape)
numeric_train.isnull().sum()
numeric_test.isnull().sum()
object_train = df_train.select_dtypes(include=['object'])

object_test = df_test.select_dtypes(include=['object'])



display(object_train.shape)

display(object_test.shape)

unique_1=[]

for col in object_train:

    unique_1.append(object_train[col].nunique())

    

print(unique_1)
unique_2=[]

for col in object_test:

    unique_2.append(object_test[col].nunique())

    

print(unique_2)
object_index_list = np.array(object_train.columns)[np.array(unique_1) == np.array(unique_2)]







object_train = object_train[object_index_list]

object_test = object_test[object_index_list]



object_train.isnull().sum()
object_dummies_train = pd.get_dummies(object_train)



object_dummies_test = pd.get_dummies(object_test)



display(object_dummies_train.shape)

display(object_dummies_test.shape)
#imputer



import numpy as np

from sklearn.impute import SimpleImputer





imp_mean = SimpleImputer(strategy="mean")

imp_median = SimpleImputer(strategy="median")

imp_mode = SimpleImputer(strategy="most_frequent")



imp_mean.fit(numeric_train)

numeric_train = imp_mean.transform(numeric_train)



imp_mean.fit(numeric_test)

numeric_test = imp_mean.transform(numeric_test)



numeric_train = pd.DataFrame(numeric_train)

numeric_test = pd.DataFrame(numeric_test)



X_train = pd.concat([numeric_train,object_dummies_train],axis=1)



display(X_train)



X_test = pd.concat([numeric_test,object_dummies_test],axis=1)



display(X_test)
X_train.columns


display(X_train.shape)

display(y_train.shape)

display(X_test.shape)





from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet

from lightgbm import LGBMRegressor



from sklearn import model_selection

from sklearn.model_selection import cross_val_score

from catboost import CatBoostRegressor 

from xgboost import XGBRegressor



lin_reg = LinearRegression()

lasso_reg = Lasso(alpha=30,max_iter=10000,random_state=9,normalize=True)

ridge_reg = Ridge(alpha=30,max_iter=10000,random_state=9,fit_intercept=False,normalize=True)

catboost = CatBoostRegressor() 

xg_reg = XGBRegressor(colsample_bytree=0.4, gamma=0,

learning_rate=0.07, max_depth=3, min_child_weight=1.5, n_estimators=10000,

reg_alpha=0.75, reg_lambda=0.45, subsample=0.6, seed=42)

lgbr = LGBMRegressor()





lgbr.fit(X_train,y_train)

lin_reg.fit(X_train,y_train)

lasso_reg.fit(X_train,y_train)

ridge_reg.fit(X_train,y_train)

catboost.fit(X_train,y_train)

xg_reg.fit(X_train,y_train)



print(model_selection.cross_val_score(lin_reg,X_train,y_train,cv=5,scoring="neg_mean_squared_log_error").mean())

print(model_selection.cross_val_score(lin_reg,X_train,y_train,cv=5).mean())

print(model_selection.cross_val_score(lasso_reg,X_train,y_train,cv=5,scoring="neg_mean_squared_log_error").mean())

print(model_selection.cross_val_score(lasso_reg,X_train,y_train,cv=5).mean())

print(model_selection.cross_val_score(ridge_reg,X_train,y_train,cv=5,scoring="neg_mean_squared_log_error").mean())

print(model_selection.cross_val_score(ridge_reg,X_train,y_train,cv=5).mean())

print(model_selection.cross_val_score(lgbr,X_train,y_train,cv=5,scoring="neg_mean_squared_log_error").mean())

print(model_selection.cross_val_score(lgbr,X_train,y_train,cv=5).mean())

#print(model_selection.cross_val_score(xg_reg,X_train,y_train,cv=5,scoring="neg_mean_squared_log_error").mean())

#print(model_selection.cross_val_score(xg_reg,X_train,y_train,cv=5).mean())
y_predict_lin = lin_reg.predict(X_test)

y_predict_lasso = lasso_reg.predict(X_test)



y_predict_ridge = ridge_reg.predict(X_test)



y_predict_cat = catboost.predict(X_test)



y_predict_xg = xg_reg.predict(X_test)



y_predict_lgbr = lgbr.predict(X_test)







y_predict = (y_predict_xg + y_predict_cat + y_predict_lgbr+y_predict_lasso)/4

y_predict
output = pd.DataFrame({"Id":id_test,"SalePrice":y_predict})
output.head()
output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")