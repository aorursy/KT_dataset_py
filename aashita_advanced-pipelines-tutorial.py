import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import warnings 
warnings.filterwarnings('ignore')

# Read Data
data = pd.read_csv('../input/melb_data.csv')
cols_to_use = ['Rooms', 'Distance', 'Landsize', 'BuildingArea', 'YearBuilt']
X = np.array(data[cols_to_use])
y = data.Price
train_X, val_X, train_y, val_y = train_test_split(X, y)

from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer

my_pipeline = Pipeline([('imputer', Imputer()), ('xgbrg', XGBRegressor())])
from sklearn.model_selection import GridSearchCV

param_grid = {
    "xgbrg__n_estimators": [10, 50, 100, 500],
    "xgbrg__learning_rate": [0.1, 0.5, 1],
}

fit_params = {"xgbrg__eval_set": [(val_X, val_y)], 
              "xgbrg__early_stopping_rounds": 10, 
              "xgbrg__verbose": False} 

searchCV = GridSearchCV(my_pipeline, cv=5, param_grid=param_grid, fit_params=fit_params)
searchCV.fit(train_X, train_y)  
searchCV.best_params_ 
searchCV.cv_results_['mean_train_score']
searchCV.cv_results_['mean_test_score']
searchCV.cv_results_['mean_train_score'].mean(), searchCV.cv_results_['mean_test_score'].mean()
