import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from sklearn.metrics import mean_squared_error
from math import sqrt

train = pd.read_csv("../input/diamonds-datamad0620/train.csv")
test = pd.read_csv("../input/diamonds-datamad0620/predict.csv")
train.head()
test.head()
train.info()
test.info()
train['cut'].unique()
test['cut'].unique()
def encode_and_bind(original_dataframe, feature_to_encode):
    dummies = pd.get_dummies(original_dataframe[[feature_to_encode]])
    res = pd.concat([original_dataframe, dummies], axis=1)
    res = res.drop([feature_to_encode], axis=1)
    return(res) 
train = encode_and_bind(train,'cut')
test = encode_and_bind(test,'cut')
train = encode_and_bind(train,'clarity')
test = encode_and_bind(test,'clarity')
train = encode_and_bind(train,'color')
test = encode_and_bind(test,'color')
y = train['price']
train = train.drop('price' , axis = 1)
train = train.drop(['id'] ,axis =1)
for feature_name in ['carat' ,	'depth','table','x','y','z'] :
  mean = train[feature_name].mean()
  std = train[feature_name].std()
  train[feature_name] = (train[feature_name] - mean) / (std)
  test[feature_name] = (test[feature_name] - mean) / (std)
from sklearn.model_selection import train_test_split

Train, val, Label, val_label = train_test_split(train,y, random_state=32, test_size=0.15)

from sklearn.tree import ExtraTreesRegressor

m =ExtraTreesRegressor( bootstrap=True, ccp_alpha=0.0, criterion='mse',
                    max_depth=None, max_features='auto', max_leaf_nodes=None,
                    max_samples=None, min_impurity_decrease=0.0,
                    min_impurity_split=None, min_samples_leaf=1,
                    min_samples_split=2, min_weight_fraction_leaf=0.0,
                    n_estimators=500, n_jobs=-1, oob_score=True,
                    random_state=32, verbose=2, warm_start=True)
m.fit(Train,Label)
# m.score(val,val_label)  
!pip install xgboost
!pip install --upgrade xgboost
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb
model = xgb.XGBRegressor(objective='reg:squarederror', 
                         base_score=0.8 , 
                         booster = 'gbtree' , 
                         learning_rate=0.07, 
                         max_depth= 8, 
                         n_estimators=300 , 
                         random_state=32)
model.fit(Train,Label)
model.score(val,val_label)
from sklearn.metrics import mean_squared_error
from math import sqrt 
print(sqrt(mean_squared_error(val_label,reg.predict(val))))
print(sqrt(mean_squared_error(Label,reg.predict(Train))))

model.feature_importances_
!sudo pip install catboost
from sklearn.datasets import load_diabetes
from sklearn.linear_model import RidgeCV
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import StackingRegressor
import catboost
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor



estimators = [
('LGBM' , LGBMRegressor(boosting_type =  'gbdt',
                            class_weight = 'balanced', 
                            importance_type =  'split', 
                            learning_rate =  0.05,
                            n_estimators =  300,
                            num_leaves = 31, 
                            objective = 'regression', 
                            random_state = 32)),
('Cat' , CatBoostRegressor(depth = 6, iterations = 1850, learning_rate = 0.09 )),
('rf' ,  RandomForestRegressor(n_jobs = -1, random_state = seed)),
('et' , ExtraTreesRegressor(n_jobs = -1, random_state = seed)),
('gb' , GradientBoostingRegressor(random_state = seed)),
('xgb' , xgb.XGBRegressor(objective='reg:squarederror', 
                         base_score=0.8 , 
                         booster = 'gbtree' , 
                         learning_rate=0.05, 
                         max_depth= 8 , 
                         n_estimators=300 , 
                         random_state=32 ))
 ]
reg = StackingRegressor(
    estimators=estimators,
     final_estimator= LinearRegression(n_jobs = -1),
       cv= 5 , verbose = 2 , passthrough = True 
)
reg.fit(Train,Label)
id = test['id']
test  =  test.drop('id' , axis = 1)
Result = reg.predict(test)

R = pd.DataFrame({'id' : id , 'price' : Result})
R
R.to_csv("result.csv" , index=False)