# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
# print(os.listdir('input'))

# Any results you write to the current directory are saved as output.
pd.set_option('display.max_colwidth', -1)
from matplotlib import pyplot as plt
%matplotlib inline

import seaborn as sns
from sklearn.linear_model import ElasticNet, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import pickle
# df = pd.read_csv('input/train.csv')
df = pd.read_csv('../input/train.csv')
df.shape
pd.set_option('display.max_columns', 100)
df.head(10)
df.tail()
df.dtypes
df.hist(figsize=(20,20), xrot=-45)
plt.show()
df.describe()
df.describe(include=['object'])
for column in df.describe(include=['object']):
    sns.countplot(y=column, data=df)
    plt.grid()
    plt.show()
df.groupby('LotConfig').median()
df.nunique()
df = df.replace([np.inf, -np.inf], np.nan)
categoricals = df.select_dtypes(include=['object'])
for column in categoricals:
    df[column] = df[column].fillna('Missing')
df.select_dtypes(include='object').isnull().sum()
df.select_dtypes(exclude=['object']).isnull().sum()
numericals = df.select_dtypes(exclude=['object'])
for column in numericals:
        df[column + 'Missing'] = np.nan
        df[column + 'Missing'] = df[column].isnull().astype(int)
        df[column] = df[column].fillna(0)
df.select_dtypes(exclude=['object']).isnull().sum()
df = pd.get_dummies(df)
df.head()
df = df.replace([np.inf, -np.inf], np.nan)
df.columns[df.isna().any()]
y = df.SalePrice
X = df.drop('SalePrice', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=777)
pipelines = {
    'lasso' : make_pipeline(StandardScaler(), Lasso(random_state=777)),
    'ridge' : make_pipeline(StandardScaler(), Ridge(random_state=777)),
    'enet' : make_pipeline(StandardScaler(), ElasticNet(random_state=777)),
    'rf' : make_pipeline(StandardScaler(), RandomForestRegressor(random_state=777)),
    'gb' : make_pipeline(StandardScaler(), GradientBoostingRegressor(random_state=777))
}
lasso_hyperparameters = {
    'lasso__alpha' : [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50]
}

ridge_hyperparameters = {
    'ridge__alpha' : [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50]
}

enet_hyperparameters = {
    'elasticnet__alpha' : [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50],
    'elasticnet__l1_ratio' : [0.1, 0.3, 0.5, 0.7, 0.9]
}

rf_hyperparameters = {
    'randomforestregressor__n_estimators' : [100, 200, 300],
    'randomforestregressor__max_features' : ['auto', 'sqrt', 0.33]
}

gb_hyperparameters = {
    'gradientboostingregressor__n_estimators' : [100, 200, 300],
    'gradientboostingregressor__max_features' : ['auto', 'sqrt', 0.33],
    'gradientboostingregressor__max_depth' : [1, 3]
}
hyperparameters = {
    'lasso' : lasso_hyperparameters,
    'ridge' : ridge_hyperparameters,
    'enet' : enet_hyperparameters,
    'rf' : rf_hyperparameters,
    'gb' : gb_hyperparameters
}
fitted_models = {}

for name, pipeline in pipelines.items():
    model = GridSearchCV(pipelines[name], hyperparameters[name], cv=10, n_jobs=-1)
    model.fit(X_train, y_train)
    fitted_models[name] = model
    print(name, 'has been fitted.')
for name, model in fitted_models.items():
    print(name, model.best_score_)
for name in fitted_models:
    pred = fitted_models[name].predict(X_test)
    print(name)
    print('--------')
    print('R^2:', r2_score(y_test, pred))
    print('MAE:', mean_absolute_error(y_test, pred))
    print()
gb_pred = fitted_models['gb'].predict(X_test)
plt.scatter(gb_pred, y_test)
plt.xlabel('predicted')
plt.ylabel('actual')
plt.grid()
plt.show()
fitted_models['gb'].best_estimator_.get_params()
with open('final_model.pkl', 'wb') as f:
    pickle.dump(fitted_models['gb'].best_estimator_, f)
with open('final_model.pkl', 'rb') as f:
    model=pickle.load(f)
print(model)
submission_df = pd.read_csv("../input/test.csv")
# submission_df = pd.read_csv('input/test.csv')
submission_df.shape
submission_df.head()
def clean_data(df):
    df = df.replace([np.inf, -np.inf], np.nan)
    categoricals = df.select_dtypes(include=['object'])
    numericals = df.select_dtypes(exclude=['object'])
    for column in categoricals:
        df[column] = df[column].fillna('Missing')
    for column in numericals:
        df[column + 'Missing'] = np.nan
        df[column + 'Missing'] = df[column].isnull().astype(int)
        df[column] = df[column].fillna(0)
    df = pd.get_dummies(df)
    return df
cleaned_submission = clean_data(submission_df)
cleaned_submission.head()
cleaned_submission.columns[cleaned_submission.isna().any()]
pred = model.predict(cleaned_submission)
