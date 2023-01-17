# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from jcopml.plot import plot_missing_value

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
pd.set_option('display.max_columns', 500)
df = pd.read_csv('/kaggle/input/mobile-price-classification/train.csv')
df.head()
df.shape
from jcopml.plot import plot_correlation_matrix
plot_correlation_matrix(df, 'price_range', numeric_col=["battery_power", "clock_speed", "fc", 'int_memory', 'm_dep', 'mobile_wt', 'n_cores', 'pc', 'px_height',
       'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time'])
kolom_yg_dipakai = ["battery_power", "ram", "px_height", "px_width", "price_range"]
df_new = df[kolom_yg_dipakai]
print(df_new.shape)
df_new.head()
plot_missing_value(df_new, return_df=True)
[print(df[x].describe(), "\n") for x in df_new.columns if x!='price_range']

f, axes = plt.subplots(2, 2, figsize=(17, 7), sharex=False)
sns.distplot( df["battery_power"], bins=100, color="blue", ax=axes[0, 0])
sns.distplot( df["ram"], bins=100, color="olive", ax=axes[0, 1])
sns.distplot( df["px_height"], bins=100, color="gold", ax=axes[1, 0])
sns.distplot( df["px_width"], bins=100, color="teal", ax=axes[1, 1])
sns.scatterplot(x=df["ram"], y=df["price_range"])
sns.pairplot(df_new,hue='price_range')
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from jcopml.pipeline import num_pipe, cat_pipe
from jcopml.utils import save_model, load_model
from jcopml.plot import plot_missing_value
from jcopml.feature_importance import mean_score_decrease
# Separate the features and target columns
X = df_new.drop(columns=["price_range"])
y = df_new["price_range"]

# Create data train and data test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
from sklearn.svm import SVC 
from sklearn.model_selection import RandomizedSearchCV
from jcopml.tuning import grid_search_params as gsp
parameter_tune = {'algo__gamma': [1.e-03, 1.e-02, 1.e-01, 1.e+00, 1.e+01, 1.e+02, 1.e+03],
'algo__C': [1.e-03, 1.e-02, 1.e-01, 1.e+00, 1.e+01, 1.e+02, 1.e+03],
'algo__max_iter' : [100, 200, 300, 400, 500, 600]}

preprocessing = ColumnTransformer([
    ('numeric', num_pipe(), X_train.columns),
])

pipeline = Pipeline([
    ('prep', preprocessing),
    ('algo', SVC())
])

model = RandomizedSearchCV(pipeline, parameter_tune,cv=3, n_jobs=-1, verbose=1, n_iter=150)
model.fit(X_train, y_train)

print(model.best_params_)
print(model.score(X_train, y_train), model.best_score_, model.score(X_test, y_test))
parameter_tune = {'algo__gamma': [1.e-03, 1.e-02, 1.e-01, 1.e+00, 1.e+01, 1.e+02, 1.e+03],
'algo__C': [1.e-03, 1.e-02, 1.e-01, 1.e+00, 1.e+01, 1.e+02, 1.e+03],
'algo__max_iter' : [100, 200, 300, 400, 500, 600]}

preprocessing = ColumnTransformer([
    ('numeric', num_pipe(scaling='standard', transform='yeo-johnson'), X_train.columns),
])

pipeline = Pipeline([
    ('prep', preprocessing),
    ('algo', SVC())
])

model = RandomizedSearchCV(pipeline, parameter_tune,cv=3, n_jobs=-1, verbose=1, n_iter=200)
model.fit(X_train, y_train)

print(model.best_params_)
print(model.score(X_train, y_train), model.best_score_, model.score(X_test, y_test))
parameter_tune = {'algo__gamma': [1.e-03, 1.e-02, 1.e-01, 1.e+00, 1.e+01, 1.e+02, 1.e+03],
'algo__C': [1.e-03, 1.e-02, 1.e-01, 1.e+00, 1.e+01, 1.e+02, 1.e+03],
'algo__max_iter' : [100, 200, 300, 400, 500, 600]}

preprocessing = ColumnTransformer([
    ('numeric', num_pipe(scaling='robust'), X_train.columns),
])

pipeline = Pipeline([
    ('prep', preprocessing),
    ('algo', SVC())
])

model = RandomizedSearchCV(pipeline, parameter_tune,cv=3, n_jobs=-1, verbose=1, n_iter=200)
model.fit(X_train, y_train)

print(model.best_params_)
print(model.score(X_train, y_train), model.best_score_, model.score(X_test, y_test))
parameter_tune = {
'prep__numeric__poly__degree': [2, 3, 4],
 'prep__numeric__poly__interaction_only': [True, False],
'algo__gamma': [1.e-03, 1.e-02, 1.e-01, 1.e+00, 1.e+01, 1.e+02, 1.e+03],
'algo__C': [1.e-03, 1.e-02, 1.e-01, 1.e+00, 1.e+01, 1.e+02, 1.e+03],
'algo__max_iter' : [100, 200, 300, 400, 500, 600]}

preprocessing = ColumnTransformer([
    ('numeric', num_pipe(poly=2 ,scaling='robust'), X_train.columns),
])

pipeline = Pipeline([
    ('prep', preprocessing),
    ('algo', SVC())
])

model = RandomizedSearchCV(pipeline, parameter_tune,cv=3, n_jobs=-1, verbose=1, n_iter=200)
model.fit(X_train, y_train)

print(model.best_params_)
print(model.score(X_train, y_train), model.best_score_, model.score(X_test, y_test))
from jcopml.plot import plot_classification_report, plot_confusion_matrix, plot_roc_curve, plot_pr_curve
plot_classification_report(X_train, y_train, X_test, y_test, model, report=True)
plot_confusion_matrix(X_train, y_train, X_test, y_test, model)