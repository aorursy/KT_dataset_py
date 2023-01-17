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
# https://www.kaggle.com/rspadim/parse-ric-stock-code-exchange-asset
market_train_df['assetCode_asset'] = market_train_df['assetCode']
market_train_df['assetCode_exchange'] = market_train_df['assetCode']
tmp_map_a, tmp_map_b = {}, {}
for i in market_train_df['assetCode'].unique():
    a,b = i.split('.')
    tmp_map_a[i] = a
    tmp_map_b[i] = b
market_train_df['assetCode'] = market_train_df['assetCode'].astype('category')
market_train_df['assetCode_asset'] = market_train_df['assetCode_asset'].map(tmp_map_a).astype('category')
market_train_df['assetCode_exchange'] = market_train_df['assetCode_exchange'].map(tmp_map_b).astype('category')
print(market_train_df.dtypes)
# Dropping assetName just to focus exclusively on one categorical variable
market_train_df.drop('assetName', axis=1, inplace=True)
def make_test_train(df, split=0.80):
    # Label encode the assetCode feature
    X = df[df.universe==1]
    le = preprocessing.LabelEncoder()
    X = X.assign(assetCode = le.fit_transform(X.assetCode))
    X = X.assign(assetCode_asset = le.fit_transform(X.assetCode_asset))
    X = X.assign(assetCode_exchange = le.fit_transform(X.assetCode_exchange))
    
    
    # split test and train
    train_ct = int(X.shape[0]*split)
    y_train, y_test = X['returnsOpenNextMktres10'][:train_ct], X['returnsOpenNextMktres10'][train_ct:]
    X = X.drop(['time', 'returnsOpenNextMktres10'], axis=1)
    X_train, X_test = X.iloc[:train_ct,], X.iloc[train_ct:,]
    return X, X_train, X_test, y_train, y_test
# Make the encoding and split
X, X_train, X_test, y_train, y_test = make_test_train(market_train_df)
def make_lgb(X_train, X_test, y_train, y_test, categorical_cols = ['assetCode', 'assetCode_asset', 'assetCode_exchange']):
    # Set up LightGBM data structures
    train_cols = X_train.columns.tolist()
    dtrain = lgb.Dataset(X_train.values, y_train, feature_name=train_cols, categorical_feature=categorical_cols)
    dvalid = lgb.Dataset(X_test.values, y_test, feature_name=train_cols, categorical_feature=categorical_cols)
    print('train cols:', train_cols)
    print('categorical_feature:', dtrain.categorical_feature)
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
    verbose_eval=25, early_stopping_rounds=20, evals_result=evals_result,
    #categorical_feature=categorical_feature
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
# Make the encoding and split
X, X_train, X_test, y_train, y_test = make_test_train(market_train_df)
X.drop(['assetCode_asset', 'assetCode'], axis=1, inplace=True)
X_train.drop(['assetCode_asset', 'assetCode'], axis=1, inplace=True)
X_test.drop(['assetCode_asset', 'assetCode'], axis=1, inplace=True)
# Set up the LightGBM data structures
dtrain, dvalid = make_lgb(X_train, X_test, y_train, y_test, categorical_cols = ['assetCode_exchange'])
# Fit and predict
evals_result = {}
m = lgb.train(
    lgb_params, dtrain, num_boost_round=1000, valid_sets=(dvalid,), valid_names=('valid',), 
    verbose_eval=25, early_stopping_rounds=20, evals_result=evals_result,
    #categorical_feature=categorical_feature
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