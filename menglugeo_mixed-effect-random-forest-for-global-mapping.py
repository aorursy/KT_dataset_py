import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import xgboost as xgb

from sklearn.model_selection import train_test_split

from sklearn import metrics

import os

from scipy import stats

from pandas import read_csv  

!pip install merf
ap = read_csv('../input/global5221/global_annual.csv')

from sklearn.impute import SimpleImputer

# Shuffle your dataset 

ap = ap.sample(frac=1)



# Define a size for your train set 

train_size = int(0.8* len(ap))

ap_pred = ap.filter (regex="pop|nig|trop|ele|wind|temp|ind|GH|road|coun")



ap_pred.cc = pd.Categorical(ap_pred.country)

 

ap_pred['country2'] = ap_pred.cc.astype('category').codes.tolist()
ap_pred= ap_pred.drop("country",axis =1 )


list(ap_pred.columns )
# Split dataset 

X_train = ap_pred [:train_size]

X_test  = ap_pred [train_size:]

Y_train = ap.loc[:train_size, ["value_mean"]]

Y_test  = ap.loc[train_size:,["value_mean"]]



X_train, X_test, Y_train, Y_test = train_test_split(ap_pred, ap['value_mean'], test_size=0.2, random_state=42)

 
from merf import MERF

import inspect

from sklearn.ensemble import RandomForestRegressor

inspect.signature(MERF)

merf = MERF(RandomForestRegressor(n_estimators = 1000), max_iterations = 100)

Z_train = np.ones((len(X_train), 1))



clusters_train = X_train['country2']

clusters_test= X_test['country2']

my_imputer = SimpleImputer()



X_train = my_imputer .fit_transform(X_train)  

X_test  = my_imputer .fit_transform(X_test)  

merf.fit(X_train,  Z_train, clusters_train, Y_train)





    
Z_test = np.ones((len(X_test), 1))

y_hat = merf.predict(X_test, Z_test, clusters_test)

y_hat
metrics.explained_variance_score(y_hat, Y_test)

metrics.r2_score(y_hat, Y_test)
from sklearn.ensemble import RandomForestRegressor

# Instantiate model with 1000 decision trees

rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)

# Train the model on training data

rf.fit(X_train, Y_train)

y_hatrf = rf.predict(X_test)
metrics.r2_score(y_hatrf, Y_test)
xg_reg = xgb.XGBRegressor(objective = "reg:squarederror",booster = "dart", learning_rate = 0.007, max_depth =6 , n_estimators = 3000,gamma =5, alpha =2) 

xg_reg.fit(X_train ,Y_train) # predictor at the station and station measurements

y_hatxgb = xg_reg.predict(X_test) # 1 degree tile

 
metrics.r2_score(y_hatxgb, Y_test)