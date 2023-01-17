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

        os.path.join(dirname, filename)



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



# Load data 

cov_dash_country = pd.read_csv('../input/uncover/UNCOVER_v4/UNCOVER/johns_hopkins_csse/johns-hopkins-covid-19-daily-dashboard-cases-by-country.csv');
cov_dash_country
cov_dash_country.describe()
y = cov_dash_country['mortality_rate']

X = cov_dash_country.drop(['mortality_rate'], axis=1)

print(' Loaded target:', y.name, '\n Loaded features:', X.columns)
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=1)
categorical_cols = [cat for cat in X_train.columns if X_train[cat].dtype=='object']

numerical_cols = [num for num in X_train.columns if X_train[num].dtype in ['int64', 'float64']]

'Successful separation of cat and num data: {}' .format((len(numerical_cols + categorical_cols) - len(X_train.columns)==0))
from sklearn.compose import ColumnTransformer 

from sklearn.pipeline import Pipeline 

from sklearn.impute import SimpleImputer 

from sklearn.preprocessing import OneHotEncoder





numerical_transformer = SimpleImputer(strategy = "constant")



categorical_transformer = Pipeline(steps=[ ('imputer', SimpleImputer(strategy='most_frequent')), 

                                              ('onehot', OneHotEncoder(handle_unknown='ignore')) ])





preprocessor = ColumnTransformer( transformers=[ ('num', numerical_transformer, numerical_cols), 

                                                ('cat', categorical_transformer, categorical_cols) ])



from sklearn.metrics import mean_absolute_error



def testmodels (model):

    #bundle preprocessing and modeling code in a pipeline

    my_pipeline = Pipeline(steps = [('preprocessing', preprocessor), ('model', model) ])

    #preprocessing of training data, and fit model

    my_pipeline.fit(X_train, y_train)

    #Preprocessing to get prediction

    preds = my_pipeline.predict(X_valid)

    #evaluate the model

    score = mean_absolute_error(y_valid, preds)

    

    return score
# import models 

from sklearn.ensemble import RandomForestRegressor

import xgboost as xgb



# defining different models 

rfr_model_0 = RandomForestRegressor(n_estimators=50, random_state=0)

rfr_model_1 = RandomForestRegressor(n_estimators=50, random_state=0)

rfr_model_2 = RandomForestRegressor(n_estimators=100, random_state=0)

rfr_model_3 = RandomForestRegressor(n_estimators=100, criterion='mae', random_state=0)

rfr_model_4 = RandomForestRegressor(n_estimators=200, min_samples_split=20, random_state=0)





xgb_model_5 = xgb.XGBRegressor()

xgb_model_6 = xgb.XGBClassifier(max_depth=20,n_estimators=2020,colsample_bytree=0.20,learning_rate=0.020,objective='binary:logistic', n_jobs=-1)

xgb_model_7 = xgb.XGBClassifier(n_estimators=100,learning_rate=0.020)





models = [rfr_model_0, rfr_model_1, rfr_model_2, rfr_model_3, rfr_model_4, 

          xgb_model_5, xgb_model_6, xgb_model_7]



# testing and saving the best model

model_performance = {}

for idx, model in enumerate(models):        

    model_performance.update({models[idx]:testmodels(model)})

    print("mae: {} --> Model number {}".format(testmodels(model), str(idx)))
final_model = Pipeline(steps = [('preprocessing', preprocessor), ('model', min(model_performance, key=model_performance.get))])

final_model.fit(X_train, y_train)

preds = final_model.predict(X_valid)

score = mean_absolute_error(y_valid, preds)

score