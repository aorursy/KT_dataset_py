#install jcopml --> %pip install jcopml

%pip install jcopml
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np

import pandas as pd



from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer





from jcopml.pipeline import num_pipe, cat_pipe

from jcopml.utils import save_model, load_model

from jcopml.plot import plot_missing_value

from jcopml.feature_importance import mean_score_decrease



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("/kaggle/input/graduate-admissions/Admission_Predict_Ver1.1.csv", index_col="Serial No.")

df.head()
X = df.drop(columns="Chance of Admit ")

y = df["Chance of Admit "]
from jcopml.automl import AutoRegressor
# Training with AutoRegressor

model = AutoRegressor(["GRE Score", "TOEFL Score", "CGPA", "SOP", "LOR "], ["University Rating", "Research"])

model.fit(X, y, cv=5, n_trial=100)
X = df.drop(columns="Chance of Admit ")

y = df["Chance of Admit "]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train.shape, X_test.shape, y_train.shape, y_test.shape
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import RandomizedSearchCV

from jcopml.tuning import random_search_params as rsp
# Training with RandomForest Algorithm

preprocessor = ColumnTransformer([

    ('numeric', num_pipe(), ["GRE Score", "TOEFL Score", "SOP", "LOR ", "CGPA"]),

    ('categoric', cat_pipe(encoder='onehot'), ["University Rating", "Research"]),

])





pipeline = Pipeline([

    ('prep', preprocessor),

    ('algo', RandomForestRegressor(n_jobs=-1, random_state=42))

])



model = RandomizedSearchCV(pipeline, rsp.rf_params, cv=3, n_iter=50, n_jobs=-1, verbose=1, random_state=42)

model.fit(X_train, y_train)



print(model.best_params_)

print(model.score(X_train, y_train), model.best_score_, model.score(X_test, y_test))
from sklearn.linear_model import ElasticNet

from sklearn.model_selection import RandomizedSearchCV

from jcopml.tuning import random_search_params as rsp
# Training with Elasticnet Algorithm

preprocessor = ColumnTransformer([

    ('numeric', num_pipe(), ["GRE Score", "TOEFL Score", "SOP", "LOR ", "CGPA"]),

    ('categoric', cat_pipe(encoder='onehot'), ["University Rating", "Research"]),

])





pipeline = Pipeline([

    ('prep', preprocessor),

    ('algo', ElasticNet())

])



model = RandomizedSearchCV(pipeline, rsp.enet_params, cv=3, n_iter=50, n_jobs=-1, verbose=1, random_state=42)

model.fit(X_train, y_train)



print(model.best_params_)

print(model.score(X_train, y_train), model.best_score_, model.score(X_test, y_test))
from jcopml.utils import save_model
save_model(model, "graduate_admission_v1.pkl")