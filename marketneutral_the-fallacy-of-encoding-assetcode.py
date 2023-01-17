import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import pandas as pd
import shap
from sklearn import preprocessing
plt.style.use('seaborn')
plt.rcParams['figure.figsize'] = 12, 7
# Make environment and get data
from kaggle.competitions import twosigmanews
env = twosigmanews.make_env()
(market_train_df, news_train_df) = env.get_training_data()
# Dropping assetName just to focus exclusively on one categorical variable
market_train_df.drop('assetName', axis=1, inplace=True)
def make_test_train(df, split=0.80):
    # Label encode the assetCode feature
    X = df[df.universe==1]
    le = preprocessing.LabelEncoder()
    X = X.assign(assetCode = le.fit_transform(X.assetCode))
    
    # split test and train
    train_ct = int(X.shape[0]*split)
    y_train, y_test = X['returnsOpenNextMktres10'][:train_ct], X['returnsOpenNextMktres10'][train_ct:]
    X = X.drop(['time', 'returnsOpenNextMktres10'], axis=1)
    X_train, X_test = X.iloc[:train_ct,], X.iloc[train_ct:,]
    return X, X_train, X_test, y_train, y_test
# Make the encoding and split
X, X_train, X_test, y_train, y_test = make_test_train(market_train_df)
def make_lgb(X_train, X_test, y_train, y_test, categorical_cols = ['assetCode']):
    # Set up LightGBM data structures
    train_cols = X_train.columns.tolist()
    dtrain = lgb.Dataset(X_train.values, y_train, feature_name=train_cols, categorical_feature=categorical_cols)
    dvalid = lgb.Dataset(X_test.values, y_test, feature_name=train_cols, categorical_feature=categorical_cols)
    return dtrain, dvalid
# Set up the LightGBM data structures
dtrain, dvalid = make_lgb(X_train, X_test, y_train, y_test)
# Set up the LightGBM params
lgb_params = dict(
    objective='regression_l1', learning_rate=0.1, num_leaves=127, max_depth=-1, bagging_fraction=0.75,
    bagging_freq=2, feature_fraction=0.5, lambda_l1=1.0, seed=1015
)
# Fit and predict
evals_result = {}
m = lgb.train(
    lgb_params, dtrain, num_boost_round=1000, valid_sets=(dvalid,), valid_names=('valid',), 
    verbose_eval=25, early_stopping_rounds=20, evals_result=evals_result
)
# Plot reported feature importance
lgb.plot_importance(m);
lgb.plot_importance(m, importance_type='gain');
shap_explainer = shap.TreeExplainer(m)
%%time
sample = X.sample(frac=0.50, random_state=100)
shap_values = shap_explainer.shap_values(sample)
%%time
shap.summary_plot(shap_values, sample)
# Create some random assetCodes (this is a nice snippet fro McKinney, Wes. Python for Data Analysis: Data Wrangling with Pandas, NumPy, and IPython (p. 340). O'Reilly Media. Kindle Edition.)
import random; random.seed(0)
import string
num_stocks = 1250
def rands(n):
    choices = string.ascii_uppercase
    return ''.join([random.choice(choices) for _ in range(n)])
assetCodes = np.array([rands(5) for _ in range(num_stocks)])
# Spoof intraday and overnight returns
days_in_year = 260
total_days = days_in_year*7
on_vol_frac = 0.2  # overnight volatility fraction

annualized_vol = 0.20
open_to_close_returns = np.random.normal(0.0, scale=annualized_vol*(1-on_vol_frac)/np.sqrt(days_in_year), size=(total_days, num_stocks))
close_to_open_returns = np.random.normal(0, scale=annualized_vol*(on_vol_frac)/np.sqrt(days_in_year), size=(total_days, num_stocks))
open_to_open_returns = close_to_open_returns + open_to_close_returns 
close_to_close_returns = close_to_open_returns + np.roll(open_to_close_returns, -1)

# Make price series
prices_close = 100*np.cumprod(1+close_to_close_returns, axis=0)
prices_open = prices_close*(1+close_to_open_returns)
import itertools

# Make into a DataFrame
dates = pd.date_range(end=pd.Timestamp('2017-12-31'), periods=total_days)
spoofed_df = pd.DataFrame(
    data={'close': prices_close.flatten('F'), 'open': prices_open.flatten('F')},
    index = pd.MultiIndex.from_tuples(
        list(itertools.product(assetCodes, dates)), names=('assetCode', 'time')
    )
)
spoofed_df['universe'] = 1.0
spoofed_df.head()

# Looks good!
spoofed_df = spoofed_df.reset_index().sort_values(['assetCode','time']).set_index(['assetCode', 'time'])
# make sure we did the open/close transform properly. Looks good.
spoofed_df.loc['MYNBI', ['open', 'close']]['1Q2013'].plot();
# # Make the "return" based features

spoofed_df = spoofed_df.assign(
     returnsClosePrevRaw1 = spoofed_df.groupby(level='assetCode').
     apply(lambda x: x.close/x.close.shift(1) -1)
     .reset_index(0, drop=True)
)

spoofed_df = spoofed_df.assign(
     returnsOpenPrevRaw1 = spoofed_df.groupby(level='assetCode').
     apply(lambda x: x.open/x.open.shift(1) -1)
     .reset_index(0, drop=True)
)

spoofed_df = spoofed_df.assign(
     returnsOpenPrevRaw10 = spoofed_df.groupby(level='assetCode').
     apply(lambda x: (x.open/x.open.shift(10)) - 1)
     .reset_index(0, drop=True)
)

spoofed_df = spoofed_df.assign(
     returnsClosePrevRaw10 = spoofed_df.groupby(level='assetCode').
     apply(lambda x: x.close/x.close.shift(10)-1)
     .reset_index(0, drop=True)
)

# Make the target variable
spoofed_df = spoofed_df.assign(
    returnsOpenNextMktres10 = spoofed_df.groupby(level='assetCode').
    apply(lambda x: (x.open.shift(-10)/x.open)-1)
    .reset_index(0, drop=True)
)
# Drop the edges where we don't have data to make returns
spoofed_df = spoofed_df.reset_index().dropna()
# Split the data
X, X_train, X_test, y_train, y_test = make_test_train(spoofed_df)

# Set up LightGBM data structures
dtrain, dvalid = make_lgb(X_train, X_test, y_train, y_test)
# Fit and predict
evals_result = {}
m = lgb.train(
    lgb_params, dtrain, num_boost_round=1000, valid_sets=(dvalid,), valid_names=('valid',), 
    verbose_eval=25, early_stopping_rounds=20, evals_result=evals_result
)
lgb.plot_importance(m);
lgb.plot_importance(m, importance_type='gain');
shap_explainer = shap.TreeExplainer(m)
%%time
sample = X.sample(frac=0.50, random_state=100)
shap_values = shap_explainer.shap_values(sample)
%%time
shap.summary_plot(shap_values, sample)
