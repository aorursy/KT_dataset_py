import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from scipy import stats

from scipy.stats import norm
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
from sklearn import linear_model

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import GridSearchCV
gridSearch2 = pd.DataFrame.from_dict(clf_gbr2.cv_results_,orient='columns')
para_grid3 = {'n_estimators': [100, 300,500,800,1200],

            'max_depth' :[2,5,8,13,20],

             'learning_rate':[0.001, 0.01, 0.1]}
from sklearn.ensemble import GradientBoostingRegressor

clf_gbr_=GradientBoostingRegressor(learning_rate=0.1,n_estimators=500)

clf_gbr_.fit(train.values, target.values)

y_pred_gbr=clf_gbr_.predict(test.values)
test2 = pd.read_csv("../input/test.csv")
from sklearn import linear_model