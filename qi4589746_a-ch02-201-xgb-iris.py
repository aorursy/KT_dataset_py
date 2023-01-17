# -*- coding: utf-8 -*-

"""

A_ch02_201_xgb_iris.py

"""

import pandas as pd

import numpy as np

import xgboost as xgboost

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split

#===========================================================

#Dataset

#讀取 'iris.csv' 成為 DataFrame irisdf

irisdf=pd.read_csv('../input/iris.csv')



#資料預處理及準備訓練資料 

#裁取特徵屬性

X=irisdf.loc[:,'sepal_length':'petal_width']

# X=irisdf.loc[:,0:3]



#轉換target欄位

#建立轉換字典 'class_mapping'

class_mapping={"setosa":0,"versicolor":1,"virginica":2}

#以字典 'class_mapping' 做轉換

irisdf['class']=irisdf['species'].map(class_mapping)

#target(labels) y

y=irisdf['class']

#train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



#modeling, 以預設定的參數塑模

xgb = XGBClassifier()

xgb.get_params() #檢視參數設定



#fitting, 以iris資料集訓練模型 

xgb.fit(X_train, y_train)



#scoring

xgb.score(X_train, y_train) #1.0

xgb.score(X_test, y_test) #1.0



#make prediction

#預測,新採集的一朵鳶尾花樣本 尺寸為 (5,2.9,1,0.2)

#以我們建立的模型預測判定此樣本品種:

#新採集的一朵鳶尾花樣本

new_data = {'sepal_length': [5], 'sepal_width': [2.9], 'petal_length': [1], 'petal_width': [0.2]}

new_data = pd.DataFrame(data=new_data)

# new_array_01=[5,2.9,1,0.2]



#將list轉為shape:(1,4) 的 ndarray

# new_array_01=np.array(new_array_01).reshape(1,-1)



#以treeModel01做預測

prediction_01=xgb.predict(new_data)

# prediction_01=xgb.predict(X_test)

print(prediction_01)



#Reshape your data either using array.reshape(-1, 1) 

#if your data has a single feature or

# array.reshape(1, -1) if it contains a single sample.

prediction_01[0] #0