%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def fetch_data():
    train_df = pd.read_csv('../input/train.csv')
    test_df = pd.read_csv('../input/test.csv')
    samp_df = pd.read_csv('../input/sample_submission.csv')
    return train_df, test_df, samp_df
train_df,test_df,samp_df = fetch_data()
train_df.head()
train_df.shape, test_df.shape
train_df.info()
train_df.loc[:, train_df.dtypes == object].columns
train_df.describe()
train_df['target'].hist(bins=40)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
train_x = train_df.iloc[:,2:]
train_y = train_df.iloc[:,1]
train_x.shape, train_y.shape
train_x.describe()
train_y.head()
pipeline = Pipeline([
    ('imputer', Imputer(strategy='median')),
    ('scaler', StandardScaler()),
])
train_x_prepared = pipeline.fit_transform(train_x).astype('float64')
train_x_prepared
train_y_prepared = train_y.values.astype('float64')
train_y_prepared
train_x_prepared.shape, train_y_prepared.shape
from sklearn.model_selection import cross_val_score
def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())
def rmsle(y, y0):
    assert len(y) == len(y0)
    return np.sqrt(np.mean(np.power(np.log1p(y)-np.log1p(y0), 2)))

y1 = np.random.rand(1000000)
y2 = np.random.rand(1000000)
err = rmsle(y1,y2)
err
from sklearn.metrics import make_scorer

rmsle_score = make_scorer(rmsle, greater_is_better=False)
from sklearn.svm import SVR

svm_reg = SVR()
svm_scores = cross_val_score(svm_reg,train_x_prepared,train_y_prepared,scoring=rmsle_score,cv=5)
svm_rmsle_scores = -svm_scores
display_scores(svm_rmsle_scores)
from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor(random_state=42)
forest_scores = cross_val_score(forest_reg, train_x_prepared, train_y_prepared, scoring=rmsle_score, cv=5)
forest_rmsle_scores = -forest_scores
display_scores(forest_rmsle_scores)
from sklearn.model_selection import GridSearchCV

param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
  ]

forest_reg = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring=rmsle_score, return_train_score=True)
grid_search.fit(train_x_prepared, train_y_prepared)
grid_search.best_params_
grid_search.best_estimator_
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print((-mean_score), params)
-grid_search.best_score_
test_df.head()
test_x = test_df.iloc[:,1:]
test_x.shape
test_x_prepared = pipeline.fit_transform(test_x).astype('float64')
preds = grid_search.best_estimator_.predict(test_x_prepared)
preds.shape
samp_df.head()
subm_df = samp_df.copy()
subm_df.shape
subm_df['target'] = preds
subm_df.head()
subm_df.to_csv('submission.csv',index=False)