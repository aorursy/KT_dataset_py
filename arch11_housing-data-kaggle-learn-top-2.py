# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load







# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np 

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt
df_train = pd.read_csv("/kaggle/input/home-data-for-ml-course/train.csv")

df_test = pd.read_csv("/kaggle/input/home-data-for-ml-course/test.csv")



display(df_train.head())

display(df_test.head())



display(df_train.shape, df_test.shape)
id_test = df_test["Id"]
from sklearn.impute import KNNImputer

knn_imp = KNNImputer()



df_train[df_train.select_dtypes(exclude="object").columns] = knn_imp.fit_transform(df_train.select_dtypes(exclude="object"))



df_train[df_train.select_dtypes(include="object").columns] = df_train[df_train.select_dtypes(include="object").columns].fillna(0)



df_test[df_test.select_dtypes(exclude="object").columns] = knn_imp.fit_transform(df_test.select_dtypes(exclude="object"))



df_test[df_test.select_dtypes(include="object").columns] = df_test[df_test.select_dtypes(include="object").columns].fillna(0)
display(df_train.head())

display(df_test.head())

l1 = []

for col in df_train.select_dtypes(include="object").columns:

    l1.append(df_train[col].nunique())

    

l2 = []

for col in df_test.select_dtypes(include="object").columns:

    l2.append(df_test[col].nunique())

    





df_train = pd.concat([df_train.select_dtypes(exclude="object"),df_train.select_dtypes(include="object").loc[:,np.array(l1) == np.array(l2)]],axis=1)

df_test = pd.concat([df_test.select_dtypes(exclude="object"),df_test.select_dtypes(include="object").loc[:,np.array(l1) == np.array(l2)]],axis=1)

y_train = df_train[["SalePrice"]]

X_train = df_train.drop("SalePrice",axis=1)

X_test= df_test.copy()
a = (pd.get_dummies(X_train.select_dtypes(include="object")).columns == pd.get_dummies(X_test.select_dtypes(include="object")).columns)

X_train = pd.concat([X_train.select_dtypes(exclude="object"),pd.get_dummies(X_train.select_dtypes(include="object")).loc[:,a]],axis=1)

X_test= pd.concat([X_test.select_dtypes(exclude="object"),pd.get_dummies(X_test.select_dtypes(include="object")).loc[:,a]],axis=1)





display(X_train.shape,y_train.shape,X_test.shape)

X_train.drop("Id",inplace=True,axis=1)

X_test.drop("Id",inplace=True,axis=1)
from sklearn.linear_model import LinearRegression, Ridge,Lasso

from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,ExtraTreesRegressor

from sklearn.svm import SVR

from catboost import CatBoostRegressor

from xgboost import XGBRegressor

from lightgbm import LGBMRegressor

from sklearn.model_selection import cross_val_score

from sklearn import model_selection







ridge = Ridge(alpha=11)

lasso = Lasso(alpha=105,max_iter=1000)

rf = RandomForestRegressor()

xg_reg = XGBRegressor(colsample_bytree=0.4, gamma=0,

learning_rate=0.07, max_depth=3, min_child_weight=1.5, n_estimators=10000,

reg_alpha=0.75, reg_lambda=0.45, subsample=0.6, seed=42)

#ada = AdaBoostRegressor(base_estimator=)

lgbr = LGBMRegressor()

catboost = CatBoostRegressor(verbose=0)

catboost.fit(X_train,y_train,verbose=0)



clf = [ridge,lasso,lgbr,rf,xg_reg]



for i in clf:

    i.fit(X_train,y_train.values.ravel())

    print(str(i).split("(")[0] +" accuracy: " + str(model_selection.cross_val_score(i,X_train,y_train.values.ravel(),cv=5,verbose=0).mean()))

print("\n")

for i in clf:

     print(str(i).split("(")[0] +" nmsle: " + str(model_selection.cross_val_score(i,X_train,y_train.values.ravel(),cv=5,verbose=0,scoring="neg_mean_absolute_error").mean()))
#learning_rate=0.07, max_depth=3, n_estimators=1000,seed=42
y_predict_lasso = lasso.predict(X_test)



y_predict_ridge = ridge.predict(X_test)



y_predict_cat = catboost.predict(X_test)



y_predict_xg = xg_reg.predict(X_test)



y_predict_lgbr = lgbr.predict(X_test)



y_predict_rf = rf.predict(X_test)

y_predict = (y_predict_xg + y_predict_cat + y_predict_lasso+ y_predict_lgbr )/4
output = pd.DataFrame({"Id":id_test,"SalePrice":y_predict})
output.head()
output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")