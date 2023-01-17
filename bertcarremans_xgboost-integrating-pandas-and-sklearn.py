import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error

import xgboost as xgb

import os
print(os.listdir("../input"))
df = pd.read_csv('../input/train.csv', parse_dates=True, encoding='UTF-8')
# Change the nominal variables' dtype to categorical
df.vendor_id = df.vendor_id.astype('category')
df.store_and_fwd_flag = df.store_and_fwd_flag.astype('category')
X_submit = pd.read_csv('../input/test.csv', parse_dates=True, encoding='UTF-8')
X_submit.vendor_id = X_submit.vendor_id.astype('category')
X_submit.store_and_fwd_flag = X_submit.store_and_fwd_flag.astype('category')
vals = df.sample(10000)['trip_duration']  # sample to speed up the processing a bit

fig, ax = plt.subplots()
sns.distplot(vals, ax=ax)
ax.set(xlabel="Trip Duration",  title='Distribution of Trip Duration')
ax.legend()
plt.show()
df['log_trip_duration'] = np.log(df['trip_duration'].values + 1)
vals = df.sample(10000)['log_trip_duration']  # sample to speed up the processing a bit

fig, ax = plt.subplots()
sns.distplot(vals, ax=ax)
ax.set(xlabel="Log Trip Duration", xlim=[0,15], title='Distribution of Log Trip Duration')
ax.axvline(x=np.median(vals), color='m', label='Median', linestyle='--', linewidth=2)
ax.axvline(x=np.mean(vals), color='b', label='Mean', linestyle='--', linewidth=2)
ax.legend()
plt.show()
features_to_keep = ['passenger_count','pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','vendor_id', 'store_and_fwd_flag']
X, y = df[features_to_keep], df.log_trip_duration
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=38)
# Class to select Dataframe columns based on dtype
class TypeSelector(BaseEstimator, TransformerMixin):
    '''
    Returns a dataframe while keeping only the columns of the specified dtype
    '''
    def __init__(self, dtype):
        self.dtype = dtype
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        return X.select_dtypes(include=[self.dtype])
# Class to convert a categorical column into numeric values
class StringIndexer(BaseEstimator, TransformerMixin):
    '''
    Returns a dataframe with the categorical column values replaced with the codes
    Replaces missing value code -1 with a positive integer which is required by OneHotEncoder
    '''
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        return X.apply(lambda s: s.cat.codes.replace(
            {-1: len(s.cat.categories)}
        ))
'''
pipeline = Pipeline([
    ('features', FeatureUnion(n_jobs=1, transformer_list=[
        # Part 1
        ('boolean', Pipeline([
            ('selector', TypeSelector('bool')),
        ])),  # booleans close
        
        ('numericals', Pipeline([
            ('selector', TypeSelector(np.number)),
            ('scaler', StandardScaler()),
        ])),  # numericals close
        
        # Part 2
        ('categoricals', Pipeline([
            ('selector', TypeSelector('category')),
            ('labeler', StringIndexer()),
            ('encoder', OneHotEncoder(handle_unknown='ignore')),
        ]))  # categoricals close
    ])),  # features close
    ("clf", xgb.XGBRegressor(objective="reg:linear", booster="gbtree", nthread=4))
])  # pipeline close
'''
'''
# 'clf__learning_rate': np.arange(0.05, 1.0, 0.05),
# 'clf__n_estimators': np.arange(50, 200, 50)
param_grid = {
    'clf__max_depth': np.arange(3, 10, 1)
}
'''
'''
randomized_mse = RandomizedSearchCV(param_distributions=param_grid, estimator=pipeline, n_iter=2, scoring="neg_mean_squared_error", verbose=1, cv=3)

# Fit the estimator
randomized_mse.fit(X_train, y_train)
print(randomized_mse.best_score_)
print(randomized_mse.best_estimator_)
'''
'''
preds_test = randomized_mse.best_estimator_.predict(X_test)
mean_squared_error(y_test.values, preds_test)
'''
'''
preds_submit = randomized_mse.best_estimator_.predict(X_submit)
X_submit['trip_duration'] = np.exp(preds_submit) - 1
X_submit[['id', 'trip_duration']].to_csv('bc_xgb_submission.csv', index=False)
'''
from catboost import Pool, CatBoostRegressor
cat_features = [5,6]
# initialize Pool
train_pool = Pool(X_train, y_train, cat_features=cat_features)
test_pool = Pool(X_test, cat_features=cat_features) 
# specify the training parameters 
model = CatBoostRegressor(iterations=1000, loss_function='RMSE', random_seed=38, logging_level='Silent', learning_rate=0.1)
#train the model
model.fit(X_train, y_train, cat_features=cat_features)
# make the prediction using the resulting model
preds_test = model.predict(test_pool)
mean_squared_error(y_test.values, preds_test)
preds_submit = model.predict(X_submit[features_to_keep])
X_submit['trip_duration'] = np.exp(preds_submit) - 1
X_submit[['id', 'trip_duration']].to_csv('bc_catb_submission.csv', index=False)