# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

plt.style.use('fivethirtyeight')
plt.rcParams['font.size'] = 18
palette = sns.color_palette('Paired', 10)
import plotly
import plotly.plotly as py
import plotly.offline as offline
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
train = pd.read_csv('../input/train.csv', nrows = 500_000,parse_dates = ['pickup_datetime'])

train = train.dropna()
train.head()
train.describe()
fare_amount = train.fare_amount
data = [go.Histogram(x = fare_amount)]
layout = go.Layout(title='Distribution of fare')

fig = go.Figure(data = data, layout = layout )

iplot(fig)
print(f"There are {len(train[train['fare_amount'] < 0])} negative fares.")
print(f"There are {len(train[train['fare_amount'] == 0])} $0 fares.")
print(f"There are {len(train[train['fare_amount'] > 100])} fares greater than $100.")
train = train[train['fare_amount'].between(left = 2.5, right = 200)]
train['fare-bin'] = pd.cut(train['fare_amount'], bins = list(range(0, 50, 5))).astype(str)

# Uppermost bin
train.loc[train['fare-bin'] == 'nan', 'fare-bin'] = '[45+]'

# Adjust bin so the sorting is correct
train.loc[train['fare-bin'] == '(5, 10]', 'fare-bin'] = '(05, 10]'

train['fare-bin'].value_counts().sort_index().plot.bar(color = 'b', edgecolor = 'k');

data = [go.Histogram(x = train['passenger_count'],
                        marker=dict(
        color='rgb(158,202,225)',
        line=dict(
            color='rgb(8,48,107)',
            width=1.5,
        )
    ),
    opacity=0.6)]
layout = go.Layout(title='Passenger Count')

fig = go.Figure(data = data, layout = layout )

iplot(fig)
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
# Remove latitude and longtiude outliers
train = train.loc[train['pickup_latitude'].between(40, 42)]
train = train.loc[train['pickup_longitude'].between(-75, -72)]
train = train.loc[train['dropoff_latitude'].between(40, 42)]
train = train.loc[train['dropoff_longitude'].between(-75, -72)]

temp1 = train.sample(10000, random_state=100)

data = [
    go.Scattermapbox(
    lat = temp1['pickup_latitude'],
    lon = temp1['pickup_longitude'],
    customdata = temp1['key'],
    mode = 'markers',
    marker = dict(
        size = 4, 
        color = 'gold',
        opacity = 0.8
        ),
        )]
layout = go.Layout(autosize=False,
                   mapbox= dict(accesstoken="pk.eyJ1IjoiYWJoaTM0NTMiLCJhIjoiY2pucWQ4NDlrMDY3NTNrbndjczZnNnZ4eCJ9.EJX2rmBc8eeXtuJ_ouagpQ",
                                bearing=10,
                                pitch=60,
                                zoom=13,
                                center= dict(
                                         lat=40.721319,
                                         lon=-73.987130),
                                style= "mapbox://styles/abhi3453/cjnqed0x70z5i2ro24fyy53ak"),
                    width=900,
                    height=600, title = "Pickup locations in Newyork")

fig = dict(data = data, layout = layout)
iplot(fig)


data = [
    go.Scattermapbox(
    lat = temp1['dropoff_latitude'],
    lon = temp1['dropoff_longitude'],
    customdata = temp1['key'],
    mode = 'markers',
    marker = dict(
        size = 4, 
        color = 'orange',
        opacity = 0.8
        ),
        )]
layout = go.Layout(autosize=False,
                   mapbox= dict(accesstoken="pk.eyJ1IjoiYWJoaTM0NTMiLCJhIjoiY2pucWQ4NDlrMDY3NTNrbndjczZnNnZ4eCJ9.EJX2rmBc8eeXtuJ_ouagpQ",
                                bearing=10,
                                pitch=60,
                                zoom=13,
                                center= dict(
                                         lat=40.721319,
                                         lon=-73.987130),
                                style= "mapbox://styles/abhi3453/cjnqed0x70z5i2ro24fyy53ak"),
                    width=900,
                    height=600, title = "Drop off locations in Newyork")

fig = dict(data = data, layout = layout)
iplot(fig)

train['abs_lat_diff'] = (train['dropoff_latitude']- train['pickup_latitude']).abs()
train['abs_lon_diff'] = (train['dropoff_longitude']- train['pickup_longitude']).abs()

temp = train.sample(1000, random_state=100)
sns.lmplot('abs_lat_diff', 'abs_lon_diff', fit_reg = False,
           data = temp);
plt.title('Absolute latitude difference vs Absolute longitude difference');
temp = temp.sort_index()
no_diff = train[(train['abs_lat_diff']==0) & (train['abs_lon_diff']==0)]
no_diff.shape
sns.lmplot('abs_lat_diff','abs_lon_diff', data = train.sort_index(), hue='fare-bin', palette=palette, fit_reg=False)
plt.title('Absolute latitude difference vs Absolute longitude difference');

def minkowski_distance(x1, x2, y1, y2, p):
    return ((abs(x2 - x1) ** p) + (abs(y2 - y1)) ** p) ** (1 / p)
color_mapping = {fare_bin: palette[i] for i, fare_bin in enumerate(train['fare-bin'].unique())}
train['color'] = train['fare-bin'].map(color_mapping)

train['manhattan'] = minkowski_distance(train['pickup_longitude'], train['dropoff_longitude'],
                                       train['pickup_latitude'], train['dropoff_latitude'], 1)

# Calculate distribution by each fare bin
plt.figure(figsize = (12, 6))
for f, grouped in train.groupby('fare-bin'):
    sns.kdeplot(grouped['manhattan'], label = f'{f}', color = list(grouped['color'])[0]);

plt.xlabel('degrees'); plt.ylabel('density')
plt.title('Manhattan Distance by Fare Amount');
train['euclidean'] = minkowski_distance(train['pickup_longitude'], train['dropoff_longitude'],
                                       train['pickup_latitude'], train['dropoff_latitude'], 2)

# Calculate distribution by each fare bin
plt.figure(figsize = (12, 6))
for f, grouped in train.groupby('fare-bin'):
    sns.kdeplot(grouped['euclidean'], label = f'{f}', color = list(grouped['color'])[0]);

plt.xlabel('degrees'); plt.ylabel('density')
plt.title('Euclidean Distance by Fare Amount');
train.groupby('fare-bin')['euclidean'].agg(['mean', 'count'])

train.groupby('fare-bin')['euclidean'].mean().plot.bar(color = 'b');
plt.title('Average Euclidean Distance by Fare Bin');
train.groupby('passenger_count')['fare_amount'].mean().plot.bar(color = 'b');
plt.title('Avg Fare By passenger count');
test = pd.read_csv('../input/test.csv', parse_dates = ['pickup_datetime'])

test['abs_lat_diff'] = (test['dropoff_latitude']-test['pickup_latitude']).abs()
test['abs_lon_diff'] = (test['dropoff_longitude']- test['pickup_longitude']).abs()
test_id = list(test.pop('key'))

test.describe()

test['manhattan'] = minkowski_distance(test['pickup_longitude'], test['dropoff_longitude'],
                                       test['pickup_latitude'], test['dropoff_latitude'], 1)

test['euclidean'] = minkowski_distance(test['pickup_longitude'], test['dropoff_longitude'],
                                       test['pickup_latitude'], test['dropoff_latitude'], 2)
R = 6378

def haversine_np(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    All args must be of equal length.    
    
    source: https://stackoverflow.com/a/29546836

    """
    # Convert latitude and longitude to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    # Find the differences
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    # Apply the formula 
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    # Calculate the angle (in radians)
    c = 2 * np.arcsin(np.sqrt(a))
    # Convert to kilometers
    km = R * c
    
    return km
train['haversine'] =  haversine_np(train['pickup_longitude'], train['pickup_latitude'],
                         train['dropoff_longitude'], train['dropoff_latitude']) 

test['haversine'] = haversine_np(test['pickup_longitude'], test['pickup_latitude'],
                         test['dropoff_longitude'], test['dropoff_latitude'])
subset = train.sample(100000, random_state=100)
plt.figure(figsize = (10,6))
for f, grouped in subset.groupby('fare-bin'):
        sns.kdeplot(grouped['haversine'], label = f'{f}', color = list(grouped['color'])[0]);
    
plt.title('Distribution of Haversine Distance by Fare Bin');
train.groupby('fare-bin')['haversine'].mean().sort_index().plot.bar(color='r')
plt.title('Avg Haversine distance by Fare')
plt.ylabel('avg haversine distance')
correlation = train.corr()
correlation['fare_amount'].plot.bar(color = 'b')
plt.title('Correlation of Fare')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

lm = LinearRegression()
X_train, X_valid, y_train, y_valid = train_test_split(train, np.array(train['fare_amount']), 
                                                      stratify = train['fare-bin'],
                                                      random_state = 100, test_size = 100_000)
lm.fit(X_train[['abs_lat_diff', 'abs_lon_diff', 'haversine', 'passenger_count' ]], y_train)
print('Intercept', round(lm.intercept_, 4))
print('abs_lat_diff coef: ', round(lm.coef_[0], 4), 
      '\tabs_lon_diff coef:', round(lm.coef_[1], 4),
      '\tpassenger_count coef:', round(lm.coef_[2], 4))
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore', category = RuntimeWarning)

def metrics(train_pred, valid_pred, y_train, y_valid):
    """Calculate metrics:
       Root mean squared error and mean absolute percentage error"""
    
    # Root mean squared error
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    valid_rmse = np.sqrt(mean_squared_error(y_valid, valid_pred))
    
    # Calculate absolute percentage error
    train_ape = abs((y_train - train_pred) / y_train)
    valid_ape = abs((y_valid - valid_pred) / y_valid)
    
    # Account for y values of 0
    train_ape[train_ape == np.inf] = 0
    train_ape[train_ape == -np.inf] = 0
    valid_ape[valid_ape == np.inf] = 0
    valid_ape[valid_ape == -np.inf] = 0
    
    train_mape = 100 * np.mean(train_ape)
    valid_mape = 100 * np.mean(valid_ape)
    
    return train_rmse, valid_rmse, train_mape, valid_mape

def evaluate(model, features, X_train, X_valid, y_train, y_valid):
    """Mean absolute percentage error"""
    
    # Make predictions
    train_pred = model.predict(X_train[features])
    valid_pred = model.predict(X_valid[features])
    
    # Get metrics
    train_rmse, valid_rmse, train_mape, valid_mape = metrics(train_pred, valid_pred,
                                                             y_train, y_valid)
    
    print(f'Training:   rmse = {round(train_rmse, 2)} \t mape = {round(train_mape, 2)}')
    print(f'Validation: rmse = {round(valid_rmse, 2)} \t mape = {round(valid_mape, 2)}')
evaluate(lm, ['abs_lat_diff', 'abs_lon_diff','haversine', 'passenger_count'], 
        X_train, X_valid, y_train, y_valid)
preds = lm.predict(test[['abs_lat_diff', 'abs_lon_diff','haversine', 'passenger_count']])

sub = pd.DataFrame({'key': test_id, 'fare_amount': preds})
sub.to_csv('sub_lr_simple.csv', index = False)
sns.distplot(sub['fare_amount'])
sub[sub['fare_amount'] > 100]

test.loc[sub[sub['fare_amount'] > 100].index]

plt.figure(figsize = (12,12))
sns.heatmap(correlation,annot = True, vmin = -1, vmax = 1, fmt = '.3f', cmap=plt.cm.PiYG_r);
from sklearn.ensemble import RandomForestRegressor

random_forest = RandomForestRegressor(n_estimators = 20, max_depth = 20, 
                                      max_features = None, oob_score = True, 
                                      bootstrap = True, verbose = 1, n_jobs = -1)
random_forest.fit(X_train[['abs_lat_diff', 'abs_lon_diff', 'haversine', 'passenger_count']], y_train)
evaluate(random_forest, ['abs_lat_diff', 'abs_lon_diff', 'haversine', 'passenger_count'],X_train, X_valid, y_train, y_valid)
preds= random_forest.predict(test[['abs_lat_diff', 'abs_lon_diff', 'haversine', 'passenger_count']])
sub = pd.DataFrame({'key': test_id, 'fare_amount': preds})
sns.distplot(sub['fare_amount'])
plt.title('Distribution of Random Forest Predicted Fare Amount');
import re

def extract_dateinfo(df, date_col, drop=True, time=False, 
                     start_ref = pd.datetime(2009, 1, 1),
                     extra_attr = False):
    """
    Extract Date (and time) Information from a DataFrame
    Adapted from: https://github.com/fastai/fastai/blob/master/fastai/structured.py
    """
    df = df.copy()
    
    # Extract the field
    fld = df[date_col]
    
    # Check the time
    fld_dtype = fld.dtype
    if isinstance(fld_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
        fld_dtype = np.datetime64

    # Convert to datetime if not already
    if not np.issubdtype(fld_dtype, np.datetime64):
        df[date_col] = fld = pd.to_datetime(fld, infer_datetime_format=True)
    

    # Prefix for new columns
    pre = re.sub('[Dd]ate', '', date_col)
    pre = re.sub('[Tt]ime', '', pre)
    
    # Basic attributes
    attr = ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear', 'Days_in_month', 'is_leap_year']
    
    # Additional attributes
    if extra_attr:
        attr = attr + ['Is_month_end', 'Is_month_start', 'Is_quarter_end', 
                       'Is_quarter_start', 'Is_year_end', 'Is_year_start']
    
    # If time is specified, extract time information
    if time: 
        attr = attr + ['Hour', 'Minute', 'Second']
        
    # Iterate through each attribute
    for n in attr: 
        df[pre + n] = getattr(fld.dt, n.lower())
        
    # Calculate days in year
    df[pre + 'Days_in_year'] = df[pre + 'is_leap_year'] + 365
        
    if time:
        # Add fractional time of day (0 - 1) units of day
        df[pre + 'frac_day'] = ((df[pre + 'Hour']) + (df[pre + 'Minute'] / 60) + (df[pre + 'Second'] / 60 / 60)) / 24
        
        # Add fractional time of week (0 - 1) units of week
        df[pre + 'frac_week'] = (df[pre + 'Dayofweek'] + df[pre + 'frac_day']) / 7
    
        # Add fractional time of month (0 - 1) units of month
        df[pre + 'frac_month'] = (df[pre + 'Day'] + (df[pre + 'frac_day'])) / (df[pre + 'Days_in_month'] +  1)
        
        # Add fractional time of year (0 - 1) units of year
        df[pre + 'frac_year'] = (df[pre + 'Dayofyear'] + df[pre + 'frac_day']) / (df[pre + 'Days_in_year'] + 1)
        
    # Add seconds since start of reference
    df[pre + 'Elapsed'] = (fld - start_ref).dt.total_seconds()
    
    if drop: 
        df = df.drop(date_col, axis=1)
        
    return df
print(train['pickup_datetime'].min())
print(test['pickup_datetime'].min())
test = extract_dateinfo(test, 'pickup_datetime', drop = False, time = True)
test.head()
train = extract_dateinfo(train, 'pickup_datetime', drop = False, 
                         time = True)
train.head()
train.head()
sns.lmplot('pickup_Elapsed','fare_amount', hue = 'pickup_Year', palette=palette, data = train , fit_reg= False, scatter_kws= {'alpha': 0.05},markers='.',size=8)
plt.figure(figsize = (12,10))

for h, grouped in train.groupby('pickup_Hour'):
    sns.kdeplot(grouped['fare_amount'], label = f'{h} hour');
plt.title('fare by Hour of the Day');
    
                
plt.figure(figsize = (12,10))

for h, grouped in train.groupby('pickup_Dayofweek'):
    sns.kdeplot(grouped['fare_amount'], label = f'{h}');
plt.title('fare by Day of the week');
correlation = train.corr()
correlation['fare_amount'].plot.bar(figsize = (12,10))
X_train, X_valid, y_train, y_valid = train_test_split(train, np.array(train['fare_amount']), 
                                                      stratify = train['fare-bin'],
                                                      random_state = 100, test_size = 100_000)
time_features = ['pickup_frac_day', 'pickup_frac_week', 'pickup_frac_year', 'pickup_Elapsed']

features = ['abs_lat_diff', 'abs_lon_diff', 'haversine', 'passenger_count',
            'pickup_latitude', 'pickup_longitude', 
            'dropoff_latitude', 'dropoff_longitude'] + time_features
def model_rf(X_train, X_valid, y_train, y_valid, test, features,
             model = RandomForestRegressor(n_estimators = 20, max_depth = 20,
                                           n_jobs = -1),
             return_model = False):
    """Train and evaluate the random forest using the given set of features."""
    
    # Train
    model.fit(X_train[features], y_train)
    
    # Validation
    evaluate(model, features, X_train, X_valid, y_train, y_valid)
    
    # Make predictions on test and generate submission dataframe
    preds = model.predict(test[features])
    sub = pd.DataFrame({'key': test_id, 'fare_amount': preds})
    
    # Extract feature importances
    feature_importances = pd.DataFrame({'feature': features,
                                        'importance': model.feature_importances_}).\
                           sort_values('importance', ascending = False).set_index('feature')
    
    if return_model:
        return sub, feature_importances, model
    
    return sub, feature_importances
sub, f1 = model_rf(X_train, X_valid, y_train, y_valid, test, features)
f1.importance.plot.bar(figsize=(10,8), color='g')
plt.title('Feature Importance of RF model')
key = train.key
df = train.copy()
features = list(train.columns)
for f in ['pickup_datetime','fare_amount', 'fare-bin', 'color','key']:
    features.remove(f)

features
sub, fi, random_forest = model_rf(X_train, X_valid, y_train, y_valid, test,features = features, return_model = True)

valid_preds = random_forest.predict(X_valid[features])

plt.figure(figsize = (10, 6))
sns.kdeplot(y_valid, label = 'Actual')
sns.kdeplot(valid_preds, label = 'Predicted')
plt.legend(prop = {'size': 30})
plt.title("Distribution of Validation Fares");
def ecdf(x):
    """Empirical cumulative distribution function of a variable"""
    # Sort in ascending order
    x = np.sort(x)
    n = len(x)
    
    # Go from 1/n to 1
    y = np.arange(1, n + 1, 1) / n
    
    return x, y
xv, yv = ecdf(valid_preds)
xtrue, ytrue = ecdf(y_valid)

# Plot the ecdfs on same plot
plt.scatter(xv, yv, s = 0.02,  c = 'r', marker = '.', label = 'Predicted')
plt.scatter(xtrue, ytrue, s = 0.02, c = 'b', marker = '.', label = 'True')
plt.title('ECDF of Predicted and Actual Validation')

plt.legend(markerscale = 100, prop = {'size': 20});
from sklearn.model_selection import RandomizedSearchCV

param_grid = {
    'n_estimators' : np.linspace(10,100).astype(int),
    'max_depth' : [None] + list(np.linspace(5,30).astype(int)),
    'max_features' : ['auto', 'sqrt', None] + list(np.arange(0.5, 1, 0.1)),
    'max_leaf_nodes': [None] + list(np.linspace(10, 50, 500).astype(int)),
    'min_samples_split': [2, 5, 10],
    'bootstrap': [True, False]
}
estimator = RandomForestRegressor(random_state = 100)

rs = RandomizedSearchCV(estimator, param_grid, n_iter = 100, n_jobs=-1, scoring = 'neg_mean_absolute_error', cv = 3,verbose=1, random_state=100)
tune_data = train.sample(100_000, random_state = 100)

# Select features
time_features = ['pickup_frac_day', 'pickup_frac_week', 'pickup_frac_year', 'pickup_Elapsed']

features = ['abs_lat_diff', 'abs_lon_diff', 'haversine', 'passenger_count',
            'pickup_latitude', 'pickup_longitude', 
            'dropoff_latitude', 'dropoff_longitude'] + time_features

rs.fit(tune_data[features],np.array(tune_data['fare_amount']))
model = rs.best_estimator_
print(f'The best parameters were {rs.best_params_} with a negative mae of {rs.best_score_}')
model.n_jobs = -1
model.fit(X_train[features], y_train)

evaluate(model, features, X_train, X_valid, y_train, y_valid)
pred = np.array(model.predict(test[features])).reshape((-1))
sub = pd.DataFrame({'key': test_id, 'fare_amount': pred})
sub.to_csv('sub_rf_tuned.csv', index = False)
sub['fare_amount'].plot.hist();
plt.title('Predicted Test Fare Distribution');
import xgboost as xgb
from bayes_opt import BayesianOptimization
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train[features],
                                                    train['fare_amount'], test_size=0.25)

dtrain = xgb.DMatrix(X_train, label=y_train)
del(X_train)
dtest = xgb.DMatrix(X_test)
del(X_test)
def xgb_evaluate(max_depth, gamma, colsample_bytree):
    params = {'eval_metric': 'rmse',
              'max_depth': int(max_depth),
              'subsample': 0.8,
              'eta': 0.1,
              'gamma': gamma,
              'colsample_bytree': colsample_bytree}
    # Used around 1000 boosting rounds in the full model
    cv_result = xgb.cv(params, dtrain, num_boost_round=100, nfold=3)    
    
    # Bayesian optimization only knows how to maximize, not minimize, so return the negative RMSE
    return -1.0 * cv_result['test-rmse-mean'].iloc[-1]
xgb_bo = BayesianOptimization(xgb_evaluate, {'max_depth': (3, 7), 
                                             'gamma': (0, 1),
                                             'colsample_bytree': (0.3, 0.9)})
# Use the expected improvement acquisition function to handle negative numbers
# Optimally needs quite a few more initiation points and number of iterations
xgb_bo.maximize(init_points=3, n_iter=5, acq='ei')
params = xgb_bo.res['max']['max_params']
params['max_depth'] = int(params['max_depth'])
model2 = xgb.train(params, dtrain, num_boost_round=250)

# Predict on testing and training set
y_pred = model2.predict(dtest)
y_train_pred = model2.predict(dtrain)

# Report testing and training RMSE
print(np.sqrt(mean_squared_error(y_test, y_pred)))
print(np.sqrt(mean_squared_error(y_train, y_train_pred)))
import matplotlib.pyplot as plt
fscores = pd.DataFrame({'X': list(model2.get_fscore().keys()), 'Y': list(model2.get_fscore().values())})
fscores.sort_values(by='Y').plot.bar(x='X')
dtest = xgb.DMatrix(test[features])
y_pred_test = model2.predict(dtest)
holdout = pd.DataFrame({'key': test_id, 'fare_amount': y_pred_test})
holdout.to_csv('submission_xgb.csv', index=False)