import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor 
from sklearn.model_selection import cross_val_score
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import GridSearchCV
X_train = pd.read_csv('../input/housing-price-cleaned/Cleaned_train.csv',index_col='Id')
X_test = pd.read_csv('../input/housing-price-cleaned/Cleaned_test.csv',index_col='Id')
y = X_train.SalePrice
X_test.index = X_test.index.astype('int32')
print(X_test.index.dtype)
X_train = X_train.drop(columns=['Unnamed: 0'])
X_test = X_test.drop(columns = ['Unnamed: 0'])
X_train
X_test
label_X_train, label_X_test = X_train.align(X_test, join='inner', axis=1)
label_X_train
label_X_test
third_mod = LGBMRegressor(learning_rate = 0.01,n_estimators = 850,boosting_type='gbdt')
third_mod.fit(label_X_train,y)
sec_mod = CatBoostRegressor()
sec_mod.fit(label_X_train,y)
'''param = {'learning_rate':[0.03,0.04,0.05,0.06],
        'n_estimators':[500,1000,1500],
         'random_state':[0],
         'missing':[0]
        }'''
mod = XGBRegressor(n_estimators = 1000, learning_rate = 0.04,random_state = 0,missing =0)
mod.fit(label_X_train,y)
th_scores = -1*cross_val_score(third_mod,label_X_train,y,cv=3,scoring = 'neg_root_mean_squared_error')
print(th_scores)
print(th_scores.mean())
scores = -1*cross_val_score(mod,label_X_train,y,cv=3,scoring = 'neg_root_mean_squared_error')
print(scores)
print(scores.mean())
sec_scores = -1*cross_val_score(sec_mod,label_X_train,y,cv=3,scoring = 'neg_root_mean_squared_error')
print(sec_scores)
print(sec_scores.mean())
pred = 0.1* mod.predict(label_X_test) + 0.8*sec_mod.predict(label_X_test) + 0.1*third_mod.predict(label_X_test)
print(pred)
output = pd.DataFrame({'Id': X_test.index , 'SalePrice':pred})
output.to_csv('temp.csv',index=False)
output = pd.DataFrame({'Id': X_test.index , 'SalePrice':pred})
output.to_csv('sub.csv',index=False)
