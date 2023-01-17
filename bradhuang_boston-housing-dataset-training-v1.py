# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv("/kaggle/input/boston-housing-dataset/train.csv")

df_test = pd.read_csv("/kaggle/input/boston-housing-dataset/test.csv")

df_train.head(5)
#通過info（）方法可以快速獲取數據集的簡單描述，特別是總行數、每個屬性的類型和非空值的數量

df_train.info()

#沒有空值, 共406筆資料
#describe（）方法可以顯示數值屬性的摘要

#觀察所有欄位的筆數, 平均值, 標準差, 最小, 最大, 四分位數

df_train.describe() 
%matplotlib inline

import matplotlib.pyplot as plt

#調用hist（）方法，繪製每個屬性的直方圖

df_train.hist(bins=20, figsize=(20,15)) #分50桶

plt.show()
corr_matrix = df_train.corr()#計算所有欄位之間的相關係數

corr_matrix.style.background_gradient(cmap='coolwarm') #pandas內建直接可以顯示heatmap
#單看房價中位數與其他欄位的關係

corr_matrix["MEDV"].abs().sort_values(ascending=False)
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import FunctionTransformer



def removeRedundantColumn(df_iput):

    return df_iput.drop(columns=['DIS', 'CHAS', 'ID'])



remove_redundantColumn = FunctionTransformer(removeRedundantColumn, validate=False)
#完整的pipeline

full_pipeline = Pipeline([

        ('remove_redundantColumn', remove_redundantColumn),#移除多餘欄位

    ])
df_train_transformed = full_pipeline.fit_transform(df_train)

df_train_x = df_train_transformed.drop(columns=['MEDV'])

df_train_y = df_train["MEDV"].copy()
from sklearn.metrics import mean_squared_error #import RMSE進行評分
from sklearn.linear_model import LinearRegression



lin_reg = LinearRegression() #建立線性迴歸物件

lin_reg.fit(df_train_x, df_train_y) #使用線性迴歸物件進行訓練
mean_squared_error(lin_reg.predict(df_train_x), df_train_y)
from sklearn.tree import DecisionTreeRegressor



tree_reg = DecisionTreeRegressor(random_state=42)

tree_reg.fit(df_train_x, df_train_y)
mean_squared_error(tree_reg.predict(df_train_x), df_train_y)
from sklearn.ensemble import RandomForestRegressor



forest_reg = RandomForestRegressor(n_estimators=10, random_state=42)

forest_reg.fit(df_train_x, df_train_y)
mean_squared_error(forest_reg.predict(df_train_x), df_train_y)
#使用cross_val_score進行交叉驗證，分十組，一次拿九組訓練，一組訓練，總共訓練十次

#驗證LinearRegression的分數

from sklearn.model_selection import cross_val_score

scores = cross_val_score(lin_reg, df_train_x, df_train_y, cv=10, scoring='neg_mean_squared_error')

scores.mean()
#驗證DecisionTreeRegressor的分數

from sklearn.model_selection import cross_val_score

scores = cross_val_score(tree_reg, df_train_x, df_train_y, cv=10, scoring='neg_mean_squared_error')

scores.mean()
#驗證RandomForestRegressor的分數

from sklearn.model_selection import cross_val_score

scores = cross_val_score(forest_reg, df_train_x, df_train_y, cv=10, scoring='neg_mean_squared_error')

scores.mean()
# 使用Scikit-Learn的GridSearchCV優化參數

from sklearn.model_selection import GridSearchCV



# 列出打算嘗試的參數清單，它會一個一個測試

param_grid = [

    # try 12 (3×4) combinations of hyperparameters

    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},

    # then try 6 (2×3) combinations with bootstrap set as False

    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},

  ]



forest_reg = RandomForestRegressor(random_state=42)

# 網格搜索將探索RandomForestRegressor超參數值的12＋6＝18種組合，

# 並對每個模型進行五次訓練（因爲我們使用的是5折交叉驗證）。

# 總共會完成18×5＝90次訓練

grid_search = GridSearchCV(forest_reg, param_grid, cv=5,

                           scoring='neg_mean_squared_error', return_train_score=True, iid=False)

grid_search.fit(df_train_x, df_train_y)
#取得最佳參數

grid_search.best_params_
#取得最佳模型

best_reg = grid_search.best_estimator_

best_reg
#轉換df_test

df_test_transformed = full_pipeline.fit_transform(df_test)
#使用最佳模型進行預測

df_test_predicted = best_reg.predict(df_test_transformed)
#產生submit格式需求的資料

df_test_submit = pd.DataFrame({"ID": df_test['ID'], "MEDV":df_test_predicted})

df_test_submit.head(5)
df_test_submit.to_csv("submit.csv",index=False)