# Set up code checking

import os

if not os.path.exists("../input/train.csv"):

    os.symlink("../input/home-data-for-ml-course/train.csv", "../input/train.csv")  

    os.symlink("../input/home-data-for-ml-course/test.csv", "../input/test.csv") 

from learntools.core import binder

binder.bind(globals())

from learntools.ml_intermediate.ex4 import *
import pandas as pd

from sklearn.model_selection import train_test_split



#DATOS ENTRADA (TRAIN  VALID)

X_full = pd.read_csv('../input/train.csv', index_col='Id')

X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)  #Quitar las filas en las que el taget sea NAN

y = X_full.SalePrice  # TARGET

X_full.drop(['SalePrice'], axis=1, inplace=True)  # dataset de entrada,s in target

X_train_full, X_valid_full, y_train, y_valid = train_test_split(X_full, y, train_size=0.8, test_size=0.2, random_state=0)  # TRAIN + VALID



categorical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and  X_train_full[cname].dtype == "object"] # CATEGORICAS (con poca cardinalidad)

numerical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]  #NUMERICAS

my_cols = categorical_cols + numerical_cols  # COLUMNAS: categoricas + numericas

X_train = X_train_full[my_cols].copy()

X_valid = X_valid_full[my_cols].copy()



#DATOS de TEST

X_test_full = pd.read_csv('../input/test.csv', index_col='Id')

X_test = X_test_full[my_cols].copy()
from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error



numerical_transformer = SimpleImputer(strategy='constant')  #NUMERICAS

categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])  #CATEGORICAS

preprocessor = ColumnTransformer( transformers=[('num', numerical_transformer, numerical_cols), ('cat', categorical_transformer, categorical_cols) ]) #NUMERICAS + CATERGORICAS



model = RandomForestRegressor(n_estimators=100, random_state=0)  #MODELO PREDICTIVO



clf = Pipeline(steps=[('preprocessor', preprocessor),('model', model)])  #PIPELINE

clf.fit(X_train, y_train)  #PIPELINE - TRAIN

preds = clf.predict(X_valid)  #PIPELINE - Predict VALID

print('MAE:', mean_absolute_error(y_valid, preds))  #SCORE
numerical_transformer = SimpleImputer(strategy='constant')  #NUMERICAL

categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant')), ('onehot', OneHotEncoder(handle_unknown='ignore')) ]) #CATEGORICAL

preprocessor = ColumnTransformer(transformers=[('num', numerical_transformer, numerical_cols), ('cat', categorical_transformer, categorical_cols)]) #NUMERICAL + CATEGORICAL



model = RandomForestRegressor(n_estimators=100, random_state=0) #MODELO



my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),('model', model) ])  #PIPELINE

my_pipeline.fit(X_train, y_train)  #PIPELINE - TRAIN

preds = my_pipeline.predict(X_valid)  #PIPELINE - Predict VALID



score = mean_absolute_error(y_valid, preds) #SCORE

print('MAE:', score)
preds_test = my_pipeline.predict(X_test)  #PIPELINE - Predict TEST

output = pd.DataFrame({'Id': X_test.index,'SalePrice': preds_test})  #Format for competition

output.to_csv('submission.csv', index=False)  #Save to file