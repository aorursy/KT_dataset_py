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
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor #for the model
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer

from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_curve, auc #for model evaluation
from sklearn.metrics import classification_report #for model evaluation
from sklearn.metrics import confusion_matrix #for model evaluation
from sklearn.inspection import permutation_importance

from xgboost import XGBClassifier, XGBRegressor

from tqdm import tqdm,trange
import scipy.stats as st
X_full = pd.read_csv('/kaggle/input/breast-cancer-wisconsin-data/data.csv',index_col='id')
# X_full.describe()
X_full.info()
# X_full.head()
# X_full.isnull().sum()
# X_full['diagnosis'].value_counts()
# X_full.nunique()
# sns.countplot('diagnosis',data=X_full)
# Remove rows with missing target, separate target from predictors
X_full.dropna(axis=0, subset=['diagnosis'], inplace=True)
y = X_full.diagnosis
X_full.drop(['diagnosis','Unnamed: 32'], axis=1, inplace=True)
# Break off validation set from training data
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X_full, y, 
                                                                train_size=0.8, test_size=0.2,
                                                                random_state=0)

# Select categorical columns with relatively low cardinality (convenient but arbitrary)
# categorical_cols = [cname for cname in X_train_full.columns if
#                     X_train_full[cname].nunique() < 10 and 
#                     X_train_full[cname].dtype == "object"]

categorical_cols = [cname for cname in X_train_full.columns if
                    X_train_full[cname].nunique() < 60 and                     
                    X_train_full[cname].dtype == "object"]

# Select numerical columns
numerical_cols = [cname for cname in X_train_full.columns if 
                X_train_full[cname].dtype in ['int64', 'float64']]

# Keep selected columns only
my_cols = categorical_cols + numerical_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()
# X_test = X_test_full[my_cols].copy()



# Preprocessing for numerical data
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', MinMaxScaler())
])


# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

categorical_cols
# 1. RandomForestRegressor in pipeline
# Define the Model using n_estimators_best
model = RandomForestClassifier(n_estimators=100, random_state=0)


# Create and Evaluate the Pipeline
# Bundle preprocessing and modeling code in a pipeline
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)
                             ])

X_train_rf = X_train.copy()
X_valid_rf = X_valid.copy()

# Fit model
my_pipeline.fit(X_train_rf, y_train)

# get predicted prices on validation data
val_predictions = my_pipeline.predict(X_valid_rf)
# print(mean_squared_error(y_valid, val_predictions))

c_report=classification_report(y_valid,val_predictions)
print(c_report)
# 2. RandomForestRegressor with RandomizedSearchCV

#Randomized Search CV
# https://github.com/krishnaik06/Car-Price-Prediction

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1100, num = 6)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
# max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]


# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

print(random_grid)
# 2. RandomForestRegressor with RandomizedSearchCV
# Define model
regr = RandomForestClassifier()

# Data preprocessing pipeline
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

# # Transform the data
X_train_rf_r = X_train.copy()
X_valid_rf_r = X_valid.copy()

X_train_rf_r = my_pipeline.fit_transform(X_train_rf_r)
X_valid_rf_r = my_pipeline.transform(X_valid_rf_r)



# rf_random = RandomizedSearchCV(estimator = regr, param_distributions = random_grid, scoring='neg_mean_squared_error',
#                               n_iter = 10, cv = 4, verbose= 1, random_state= 0, n_jobs = 1)
rf_random = RandomizedSearchCV(estimator = regr, param_distributions = random_grid, 
                              n_iter = 10, cv = 4, verbose= 1, random_state= 0, n_jobs = 1)


rf_random.fit(X_train_rf_r,y_train)

# get predicted prices on validation data
# rs_val_predictions = rs_model.predict(X_valid)
# print(mean_squared_error(y_valid, rs_val_predictions))
rf_random.best_params_
# get predicted prices on validation data
rs_val_predictions = rf_random.predict(X_valid_rf_r)
# print(mean_squared_error(y_valid, rs_val_predictions))
c_report=classification_report(y_valid,rs_val_predictions)
print(c_report)
rf_random.best_score_
# 3. Xgboost in pipeline
# Define the Model using n_estimators_best
xgb_model = XGBClassifier(n_estimators=1000, learning_rate=0.05) # Typical values range from 100-1000


# Create and Evaluate the Pipeline
# Bundle preprocessing and modeling code in a pipeline
xgb_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', xgb_model)
                             ])


# Fit model
# xgb_pipeline.fit(X_train, y_train)
#  cyin: xgboost using pipeline: 
#     1. to seperate model and pipeline(used only for preprocessing) 
#     or
#     2. to set param in pipeline: estimator + __(2 underscore) + normal param names 
# https://stackoverflow.com/questions/58136107/xgbregressor-using-pipeline
#https://www.kaggle.com/questions-and-answers/101994

# Make a copy to avoid changing original data
X_valid_eval=X_valid.copy()
# Remove the model from pipeline
eval_set_pipe = Pipeline(steps = [('preprocessor', preprocessor)])
# fit transform X_valid.copy()
X_valid_eval = eval_set_pipe.fit(X_train, y_train).transform (X_valid_eval)


X_train_xg = X_train.copy()
X_valid_xg = X_valid.copy()

xgb_pipeline.fit(X_train_xg, y_train, model__early_stopping_rounds=5, model__eval_metric = "mae", 
                 model__eval_set=[(X_valid_eval, y_valid)],model__verbose=False)

# get predicted prices on validation data
xgb_val_predictions = xgb_pipeline.predict(X_valid_xg)
# print(mean_squared_error(y_valid, xgb_val_predictions))
c_report=classification_report(y_valid,xgb_val_predictions)
print(c_report)
# 3. still Xgboost in pipeline but just using default param
xgb_pipeline_raw = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', xgb_model)
                             ])

X_train_xg = X_train.copy()
X_valid_xg = X_valid.copy()

xgb_pipeline_raw.fit(X_train_xg, y_train)

# get predicted prices on validation data
xgb_val_predictions = xgb_pipeline_raw.predict(X_valid_xg)
# print(mean_squared_error(y_valid, xgb_val_predictions))
c_report=classification_report(y_valid,xgb_val_predictions)
print(c_report)
# 4. Xgboost with RandomizedSearchCV

# http://danielhnyk.cz/how-to-use-xgboost-in-python/
one_to_left = st.beta(10, 1)  
from_zero_positive = st.expon(0, 50)

params = {  
#     "n_estimators": st.randint(3, 40),
    "n_estimators": range(100,1000,200),
    "max_depth": st.randint(3, 40),
    "learning_rate": st.uniform(0.05, 0.4),
    "colsample_bytree": one_to_left,
    "subsample": one_to_left,
    "gamma": st.uniform(0, 10),
    'reg_alpha': from_zero_positive,
    "min_child_weight": from_zero_positive,
}

# xgbreg = XGBRegressor(nthreads=-1)  
# xgb_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
#                               ('model', xgbreg)
#                              ])

# gs = RandomizedSearchCV(xgb_pipeline, params, n_jobs=1)
# gs.fit(X_train, y_train,early_stopping_rounds=5, eval_set=[(X_valid, y_valid)], verbose=False) # cyin: has error
# gs.best_model_

#  cyin: xgboost using RandomizedSearchCV needs to seperate with pipeline otherwise error: 
#     use pipeline only for preprocessing 

# Define model
xgbreg = XGBClassifier(n_estimators=500, learning_rate=0.05)

# Data preprocessing pipeline
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

# Transform the data
X_train_xg_r = X_train.copy()
X_valid_xg_r = X_valid.copy()

X_train_xg_r = my_pipeline.fit_transform(X_train_xg_r)
X_valid_xg_r = my_pipeline.transform(X_valid_xg_r)


# xgb_model = RandomizedSearchCV(xgbreg, params, scoring='neg_mean_squared_error', n_jobs=1)  
xgb_model = RandomizedSearchCV(xgbreg, params, n_jobs=1)  

xgb_model.fit(X_train_xg_r, y_train)  
# rs_model.best_estimator_


# get predicted prices on validation data
xgb_val_predictions = xgb_model.predict(X_valid_xg_r)
# print(mean_squared_error(y_valid, xgb_val_predictions))
c_report=classification_report(y_valid,xgb_val_predictions)
print(c_report)
