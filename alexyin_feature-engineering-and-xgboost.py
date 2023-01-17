import numpy as np
import pandas as pd 
import xgboost as xgb
import sklearn

import matplotlib.pyplot as plt

for p in [np, pd, xgb, sklearn]:
    print (p.__name__, p.__version__)
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import PredefinedSplit, cross_val_score, GridSearchCV

from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, make_scorer
RANDOM_STATE = 71
# Evaluation criterion
def smape(pred, actual):
    """
    pred: a numpy array of predictions
    actual: a numpy array of actual values
    
    for a perfectly predicted zero observation, the smape is defined to be 0. 
    
    """
    
    selector = ~((pred == 0) & (actual == 0))
    numerator = np.abs(pred-actual)
    denom = (np.abs(pred) + np.abs(actual)) / 2
    return 100*np.sum((numerator[selector] / denom[selector])) / pred.shape[0]

smape_scorer = make_scorer(smape, greater_is_better=False)
# Test cases
for actual, pred in zip([np.array([1,4,0,5])]*3, 
                        [np.array([1,3,0,5]), np.array([0.5,4,1,6]), np.array([2,7,-1,4])]):
    print(smape(pred, actual))
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
sample_submission = pd.read_csv("../input/sample_submission.csv")
sample_submission.info()
train.info()
# Convert the date field
train.loc[:,'date'] = pd.to_datetime(train.date)
test.loc[:,'date'] = pd.to_datetime(test.date)
data = pd.concat([train, test], sort=False).fillna(0)   # test data has id column
def downcast_dtypes(df):
    '''
        Changes column types in the dataframe: 
                
                `float64` type to `float32`
                `int64`   type to `int32`
    '''
    
    # Select columns to downcast
    float_cols = [c for c in df if df[c].dtype == "float64"]
    int_cols =   [c for c in df if df[c].dtype == "int64"]
    
    # Downcast
    df[float_cols] = df[float_cols].astype(np.float32)
    df[int_cols]   = df[int_cols].astype(np.int32)
    
    return df
data = downcast_dtypes(data)
# Lag featurizer
class Lag_Featurizer(TransformerMixin):
    def __init__(self, index_col, time_col, value_col, output_col, output_time_index=False, shift=0, freq='1D'):
        self.index_col = index_col
        self.time_col = time_col
        self.value_col = value_col
        self.output_col = output_col
        self.output_time_index=output_time_index
        self.shift = shift
        self.freq = freq
        
    def fit(self, X):                
        pass
    
    def transform(self, X):
        assert isinstance(self.index_col, list)
        
        time_key = pd.Grouper(freq=self.freq)      
        time_index = self.index_col + [time_key]
        resampled = X.groupby(time_index)[self.value_col].sum().reset_index().set_index(self.time_col)
        shifted= resampled.groupby(self.index_col).shift(self.shift, freq=self.freq).drop(self.index_col, axis=1).reset_index().rename(columns={self.value_col:self.output_col})
        merged = pd.merge(X, shifted, how='left',left_on=self.index_col + [self.time_col], right_on=self.index_col + [self.time_col])
        if self.output_time_index:
            return merged.set_index(self.time_col)
        else:
            return merged
data = data.set_index('date')
lag_feature_pipeline = Pipeline(
[
    # lag store, item sales
    ('store_item_lag_1d', Lag_Featurizer(index_col=['store', 'item'],time_col='date',value_col='sales',output_col='sales_1d', output_time_index=True, shift=1 )),
    ('store_item_lag_2d', Lag_Featurizer(index_col=['store', 'item'],time_col='date',value_col='sales',output_col='sales_2d', output_time_index=True, shift=2 )),
    ('store_item_lag_3d', Lag_Featurizer(index_col=['store', 'item'],time_col='date',value_col='sales',output_col='sales_3d', output_time_index=True, shift=3 )),
    ('store_item_lag_4d', Lag_Featurizer(index_col=['store', 'item'],time_col='date',value_col='sales',output_col='sales_4d', output_time_index=True, shift=4 )),
    ('store_item_lag_1w', Lag_Featurizer(index_col=['store', 'item'],time_col='date',value_col='sales',output_col='sales_1w', output_time_index=True, shift=7 )),
    ('store_item_lag_2w', Lag_Featurizer(index_col=['store', 'item'],time_col='date',value_col='sales',output_col='sales_2w', output_time_index=True, shift=14)),
    ('store_item_lag_4w', Lag_Featurizer(index_col=['store', 'item'],time_col='date',value_col='sales',output_col='sales_4w', output_time_index=True, shift=28)),
    ('store_item_lag_3m', Lag_Featurizer(index_col=['store', 'item'],time_col='date',value_col='sales',output_col='sales_3m', output_time_index=True, shift=84)),
    ('store_item_lag_6m', Lag_Featurizer(index_col=['store', 'item'],time_col='date',value_col='sales',output_col='sales_6m', output_time_index=True, shift=168)),
    ('store_item_lag_1y', Lag_Featurizer(index_col=['store', 'item'],time_col='date',value_col='sales',output_col='sales_1y', output_time_index=True, shift=336)),
    
    #lag store sales
    ('store_lag_1d', Lag_Featurizer(index_col=['store'],time_col='date',value_col='sales',output_col='store_sales_1d', output_time_index=True, shift=1 )),
    ('store_lag_2d', Lag_Featurizer(index_col=['store'],time_col='date',value_col='sales',output_col='store_sales_2d', output_time_index=True, shift=2 )),
    ('store_lag_3d', Lag_Featurizer(index_col=['store'],time_col='date',value_col='sales',output_col='store_sales_3d', output_time_index=True, shift=3 )),
    ('store_lag_4d', Lag_Featurizer(index_col=['store'],time_col='date',value_col='sales',output_col='store_sales_4d', output_time_index=True, shift=4 )),
    ('store_lag_1w', Lag_Featurizer(index_col=['store'],time_col='date',value_col='sales',output_col='store_sales_1w', output_time_index=True, shift=7 )),
    ('store_lag_2w', Lag_Featurizer(index_col=['store'],time_col='date',value_col='sales',output_col='store_sales_2w', output_time_index=True, shift=14)),
    ('store_lag_4w', Lag_Featurizer(index_col=['store'],time_col='date',value_col='sales',output_col='store_sales_4w', output_time_index=True, shift=28)),
    ('store_lag_3m', Lag_Featurizer(index_col=['store'],time_col='date',value_col='sales',output_col='store_sales_3m', output_time_index=True, shift=84)),
    ('store_lag_6m', Lag_Featurizer(index_col=['store'],time_col='date',value_col='sales',output_col='store_sales_6m', output_time_index=True, shift=168)),
    ('store_lag_1y', Lag_Featurizer(index_col=['store'],time_col='date',value_col='sales',output_col='store_sales_1y', output_time_index=True, shift=336)),
    
    # lag item sales
    ('item_lag_1d', Lag_Featurizer(index_col=['item'],time_col='date',value_col='sales',output_col='item_sales_1d', output_time_index=True, shift=1 )),
    ('item_lag_2d', Lag_Featurizer(index_col=['item'],time_col='date',value_col='sales',output_col='item_sales_2d', output_time_index=True, shift=2 )),
    ('item_lag_3d', Lag_Featurizer(index_col=['item'],time_col='date',value_col='sales',output_col='item_sales_3d', output_time_index=True, shift=3 )),
    ('item_lag_4d', Lag_Featurizer(index_col=['item'],time_col='date',value_col='sales',output_col='item_sales_4d', output_time_index=True, shift=4 )),
    ('item_lag_1w', Lag_Featurizer(index_col=['item'],time_col='date',value_col='sales',output_col='item_sales_1w', output_time_index=True, shift=7 )),
    ('item_lag_2w', Lag_Featurizer(index_col=['item'],time_col='date',value_col='sales',output_col='item_sales_2w', output_time_index=True, shift=14)),
    ('item_lag_4w', Lag_Featurizer(index_col=['item'],time_col='date',value_col='sales',output_col='item_sales_4w', output_time_index=True, shift=28)),
    ('item_lag_3m', Lag_Featurizer(index_col=['item'],time_col='date',value_col='sales',output_col='item_sales_3m', output_time_index=True, shift=84)),
    ('item_lag_6m', Lag_Featurizer(index_col=['item'],time_col='date',value_col='sales',output_col='item_sales_6m', output_time_index=True, shift=168)),
    ('item_lag_1y', Lag_Featurizer(index_col=['item'],time_col='date',value_col='sales',output_col='item_sales_1y', output_time_index=True,shift=336))
    
]
)
data = lag_feature_pipeline.transform(data)
# drop all rows with nulls. Part of 2013 data is kept since the maximum lag is 336 days. 
data.dropna(inplace=True)
data.loc[:,'weekend'] = ((data.index.weekday == 5) |  (data.index.weekday == 6)) + 0
cols = [
    
    'sales',
    
    'sales_1d',
    'sales_2d',
    'sales_3d',
    'sales_4d',
    'sales_1w',
    'sales_2w',
    'sales_4w',
    'sales_3m',
    'sales_6m',
    'sales_1y',
    
    'store_sales_1d',
    'store_sales_2d',
    'store_sales_3d',
    'store_sales_4d',
    'store_sales_1w',
    'store_sales_2w',
    'store_sales_4w',
    'store_sales_3m',
    'store_sales_6m',
    'store_sales_1y',
    
    'item_sales_1d',
    'item_sales_2d',
    'item_sales_3d',
    'item_sales_4d',
    'item_sales_1w',
    'item_sales_2w',
    'item_sales_4w',
    'item_sales_3m',
    'item_sales_6m',
    'item_sales_1y',
    
    'weekend'
]
training = data.loc[:'2017-03',cols]
validation_split = np.where((training.index >= pd.Timestamp(2017,1,1)) & (training.index <= pd.Timestamp(2017,3,31)), 0, -1)
X_validation = training.loc[validation_split == 0, cols[1:]]
y_validation = training.loc[validation_split == 0, 'sales']
# training matrices
X_training = training.loc[:,cols[1:]]
y_training = training.loc[:,'sales']
print('Number of training instances = {0:d}'.format(X_training.shape[0]))
print('Number of features           = {0:d}'.format(X_training.shape[1]))
print('Date range = {0} to {1}'.format(training.index[0].strftime('%Y-%m-%d'), training.index[-1].strftime('%Y-%m-%d')))
testing = data.loc['2018-01':,cols]
X_testing = testing.loc[:,cols[1:]]
y_testing = testing.loc[:,'sales']
print('Number of test instances = {0:d}'.format(X_testing.shape[0]))
print('Number of features       = {0:d}'.format(X_testing.shape[1]))
print('Date range = {0} to {1}'.format(testing.index[0].strftime('%Y-%m-%d'), testing.index[-1].strftime('%Y-%m-%d')))
# Lasso
lasso = Lasso(random_state=RANDOM_STATE, max_iter=2000)
lasso_params = {'alpha': np.logspace(-3,3,7)}
reg = GridSearchCV(lasso, lasso_params, scoring=smape_scorer, n_jobs=1, cv=PredefinedSplit(validation_split), verbose=1)
reg.fit(X_training,y_training)
print('Best score = {0:.4f}; Best Parameter = {1}'.format(-reg.best_score_, reg.best_params_)) # Score is the negative because large score indicates better fit
pred_validation = reg.predict(X_validation)
print(smape(pred_validation, y_validation.values))
fig, ax = plt.subplots(figsize=(12,12))
ax.scatter(y_validation.values, pred_validation)
# Refit on the entire training data
training_full = data.loc[:'2017-12',cols]
X_training_full = training_full.loc[:,cols[1:]]
y_training_full = training_full.loc[:,'sales']
print('Number of training instances = {0:d}'.format(X_training_full.shape[0]))
print('Number of features           = {0:d}'.format(X_training_full.shape[1]))
print('Date range = {0} to {1}'.format(training_full.index[0].strftime('%Y-%m-%d'), training_full.index[-1].strftime('%Y-%m-%d')))
reg_best = reg.best_estimator_
reg_best.fit(X_training_full, y_training_full)
pred_reg_best = reg_best.predict(X_testing)
# Make predictions and a submission: 141.87
submission_lasso = pd.DataFrame({'Id': sample_submission.id, 'sales': pred_reg_best})
submission_lasso.to_csv('submission_lasso.csv', index=False)
rf = RandomForestRegressor(random_state=RANDOM_STATE, criterion='mae')
rf_params = {"n_estimators": np.arange(100, 1100, 100),
              "max_depth": np.arange(3, 10, 1),
              "min_samples_split": np.arange(10,150,1),
              "min_samples_leaf": np.arange(5,10,1),
              "max_leaf_nodes": np.arange(5,15,1)}
rf = GridSearchCV(rf, rf_params, scoring=smape_scorer, n_jobs=1, cv=PredefinedSplit(validation_split), verbose=10)
# rf.fit(X_training,y_training)
# print('Best score = {0:.4f}; Best Parameter = {1}'.format(-rf.best_score_, rf.best_params_))
# rf_best = rf.best_estimator_
# rf_best.fit(X_training_full, y_training_full)
# pred_rf_best = rf_best.predict(X_testing)
# submission_rf = pd.DataFrame({'Id': sample_submission.id, 'sales': pred_rf_best})
# submission_rf.to_csv('submission_rf.csv', index=False)
