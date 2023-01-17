import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df= pd.read_csv("/kaggle/input/hackerearth-effectiveness-of-std-drugs/dataset/train.csv")
df.head(2)
df.drop(["patient_id","name_of_drug","use_case_for_drug","review_by_patient","drug_approved_by_UIC"],axis=1,inplace=True)
min_rating = df.effectiveness_rating.min()
max_rating = df.effectiveness_rating.max()

def scale_rating(rating):
    rating -= min_rating
    rating = rating/(max_rating -1)
    rating *= 5
    rating = int(round(rating,0))
    
    if(int(rating) == 0 or int(rating)==1 or int(rating)==2):
        return 0
    else:
        return 1
    
df['new_eff_score'] = df.effectiveness_rating.apply(scale_rating)
X = df.drop("base_score",axis=1)
y= df.base_score
from sklearn.preprocessing import  MinMaxScaler
sc= MinMaxScaler()
X= sc.fit_transform(X)
y= y.values.reshape(-1,1)
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from hyperopt import hp
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
# from sklearn.metrics import neg_mean_squared_error
from sklearn.metrics import mean_squared_error
def model(params):
    regr = xgb.XGBRegressor(**params)
    return cross_val_score(regr,X,y,scoring='neg_mean_squared_error').mean()
# XGB parameters
space = {
    'learning_rate':    hp.choice('learning_rate',    np.arange(0.15, 0.25, 0.01)),
    'n_estimators':     hp.choice('n_estimators', np.arange(1000,1800,10, dtype=int)),
    'max_depth':        hp.choice('max_depth',        np.arange(5, 15, 1, dtype=int)),
}
def objective(params):
    loss = model(params)
    return {'loss': -loss,'status' : STATUS_OK}
trials =Trials()
best = fmin(fn=objective,space=space,algo=tpe.suggest,max_evals=100,trials=trials)
print(best)
print("learning_rate = ",np.arange(0.15, 0.25, 0.01)[1])
print("max_dept = ",np.arange(5, 15, 1, dtype=int)[0])
print("n_estimators =",np.arange(1000,1800,10, dtype=int)[63])