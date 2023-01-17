# imports
import datetime
t_start = datetime.datetime.utcnow()
from collections import defaultdict
from functools import partial
from itertools import compress
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import f_regression, SelectKBest
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from xgboost.sklearn import XGBRegressor
from hyperopt import fmin, hp, STATUS_OK, tpe, Trials
import graphviz
import json
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import os
import pandas as pd
import pickle
import seaborn as sns
import time
import unittest
import xgboost as xgb

# suppress warnings – bad practice but they're annoying and not being annoyed is good pracice
import warnings
warnings.filterwarnings("ignore")

# Definitions and settings
nthread = multiprocessing.cpu_count()
run_plots = False
pd.set_option('display.float_format', lambda x: '%.3f' % x)
%config InlineBackend.figure_format = 'retina'
#%matplotlib notebook
%config IPCompleter.greedy=True
color = sns.color_palette()
sns.set()
cwd = os.getcwd()
def now():
    return datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

if cwd == '/kaggle/working':
    %matplotlib inline
    train_data_path = os.path.join('../input/','train.csv')
    test_data_path = os.path.join('../input/','test.csv')
    trials_file = 'trials.pk'
else:
    train_data_path = os.path.join('input/','train.csv')
    test_data_path = os.path.join('input/','test.csv')
    trials_file = os.path.join('../input/','trials.pk')
# get training data
train = pd.read_csv(train_data_path)
print(f"train: {train.shape}")
# get test data
test = pd.read_csv(test_data_path)
print(f"test: {test.shape}")
df_combined = pd.concat([train, test], axis=0)
#df_combined = df_combined.fillna('none')
maps = {}
for col in df_combined.select_dtypes(exclude='number'):
    maps[col] = dict(zip(df_combined[col].unique(), range(len(df_combined[col].unique()))))
# Check the training data types
print(f'Unique dtypes: {train.dtypes.unique()}\n')

# Check the training data for nans
print(f'nan count per feature:\n{train.isnull().sum()[train.isnull().sum() > 0]}\n')

# For reference, print the number of elements in train
num_vals = train.shape[0] * train.shape[1]
print(f'total elements in train: {num_vals}\n')

# Count the null values in train
num_nulls = train.isnull().sum().sum()
print(f'null values in train: {num_nulls}\n')

# percentage null values
print(f'percentage null values: {(num_nulls / num_vals) * 100:3.1f}%')
train.describe()
dtype_df = train.dtypes.reset_index()
dtype_df.columns = ["Count", "Column Type"]
dtype_df.groupby("Column Type").aggregate('count').reset_index()
categorical_df = train.select_dtypes('object').astype('category')
categorical_features = categorical_df.columns
ordinal_df = train.select_dtypes('int64').drop(['SalePrice'], axis=1)
def unique_vals(col):
    return set(col)

unique_ints = ordinal_df.apply(unique_vals, axis=0)
print(unique_ints)
# There's probably a much nicer way to do this with Pandas, but this will do for now.
parametric_features = []
for feature in unique_ints:
    name = unique_ints[unique_ints == feature].index.format()[0]
    count = len(feature)
    if count > 112:
        parametric_features.append(name)
parametric_df = pd.concat([train.select_dtypes('float'), train[parametric_features]], axis=1)
ordinal_df = ordinal_df.drop(parametric_features, axis=1)
ordinal_features = ordinal_df.columns
categorical_df.describe()
ordinal_df.describe()
parametric_df.describe()
for df in [categorical_df, ordinal_df, parametric_df]:
    print(df.shape)
# define a function to produce plots

def box_plot(feature, label='SalePrice', data=train):
    try:
        col_order = np.sort(data[feature].unique()).tolist()
    except TypeError:
        col_order = None
    plt.figure(figsize=(16,6))
    ax = sns.boxplot(x=feature, y=label, data=data, order=col_order)
    plt.xlabel(f'{feature}', fontsize=12)
    plt.ylabel(f'{label}', fontsize=12)
    plt.title(f'Distribution of {feature} with {label}', fontsize=15)
    xticks = ax.get_xticks()
    if len(xticks) > 500:
        plt.xticks(rotation=45)
        ax.set(xticks=xticks[0::100])
    elif len(xticks) > 50:
        plt.xticks(rotation=45)
        ax.set(xticks=xticks[0::5])
    plt.show()
if run_plots:
    for col in train.drop(['Id', 'SalePrice'], axis=1).columns:
        box_plot(col)
# plot a correlation matrix for the new version of X
corrmat = pd.concat([train.drop(['Id', 'SalePrice'], axis=1), train.SalePrice], axis=1).corr()
dims = corrmat.shape; plt_size = 15 + (0.18 * dims[0]); plt.figure(figsize=(plt_size, plt_size))
if run_plots:
    sns.heatmap(corrmat, annot=False, square=True);
corrmat = pd.concat([parametric_df, train.SalePrice], axis=1).corr(method='pearson')
dims = corrmat.shape; plt_size = 15 + (0.18 * dims[0]); plt.figure(figsize=(plt_size, plt_size))
if run_plots:
    sns.heatmap(corrmat, annot=True, fmt='4.2f', square=True);
saleprice_corrvec = corrmat.iloc[-1,:-1].sort_values()
print(saleprice_corrvec)
high_dv_corr_feats = saleprice_corrvec[saleprice_corrvec.abs() > 0.4]
print(high_dv_corr_feats.sort_values())
dims = high_dv_corr_feats.shape; plt_size = 15 + (0.18 * dims[0]); plt.figure(figsize=(plt_size, plt_size / 2))
if run_plots:
    high_dv_corr_feats.plot.bar();
corrmat = pd.concat([train.drop(['Id', 'SalePrice'], axis=1), train.SalePrice], axis=1).corr()
dims = corrmat.shape; plt_size = 15 + (0.18 * dims[0]); plt.figure(figsize=(plt_size, plt_size))
if run_plots:
    sns.heatmap(corrmat, annot=False, square=True);
parametric_features_corr = pd.concat([parametric_df, train.SalePrice], axis=1)
parametric_corr = parametric_features_corr.corr().iloc[-1,:-1].abs().sort_values()
# Plot the correlation values
plt.figure(figsize=(16,6))
if run_plots:
    parametric_corr.plot.bar();
y = train.SalePrice
print('y.shape: ', y.shape)
if run_plots:
    # histogram of y
    plt.figure(figsize=(16,6))
    sns.distplot(y, bins=100);
# Log transform SalePrice as this is what is used for scoring
# As described in the dataset, this "means that errors in predicting
# expensive houses and cheap houses will affect the result equally."
y = np.log1p(train.SalePrice)
if run_plots:
    # histogram of log y
    plt.figure(figsize=(16,6))
    sns.distplot(y, bins=100);
# define RMSE function as this is the evaluation metric for competition entries. 
def rmse(y_true, y_pred):
        return np.sqrt(np.mean(np.square(y_pred - y_true)))
def xgb_cvs(x, y, r=None):
    xgb_test = XGBRegressor(learning_rate=0.05, n_estimators=500, max_depth=3, colsample_bytree=0.4, randdom_state=r)
    cv_score = cross_val_score(xgb_test, x, y, cv = 5, n_jobs=-1)
    return cv_score.mean()
scores = []
X_baseline = train.drop(['Id', 'SalePrice'], axis=1).select_dtypes('number')
y_baseline = train.SalePrice
score_baseline = xgb_cvs(X_baseline, y_baseline)
scores.append(score_baseline)
print(scores)
test_ids = test.Id
train_ids = train.Id
def train_test_splitter(df, train_ids, test_ids):
    train = df.loc[df.Id.isin(train_ids)]
    test = df.loc[df.Id.isin(test_ids)]
    
    return train, test
all_inputs_df = pd.concat([train.drop('SalePrice', axis=1), test])
y = train.SalePrice
# create new area total house area feature
area_features = ['1stFlrSF','2ndFlrSF','3SsnPorch','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','EnclosedPorch',
                 'GarageArea','GrLivArea','OpenPorchSF','ScreenPorch','TotalBsmtSF','WoodDeckSF']
total_area_df = all_inputs_df[area_features].sum(axis=1)
total_area_df.column = 'TotalArea'
total_area_df.describe()
# compute correlation of total area in train to sale price
total_area_corr = pd.concat([train[area_features], y], axis=1).corr().iloc[0,1]
print(f'total area correlation: {total_area_corr:5.3f}')
total_house_area_features = ['TotalBsmtSF', '1stFlrSF', '2ndFlrSF']
# compute correlation of total area in train to sale price
total_house_area_corr = pd.concat([train[total_house_area_features], y], axis=1).corr().iloc[0,1]
print(f'total house area correlation: {total_house_area_corr:5.3f}')
all_inputs_df['TotalHouseSF'] = all_inputs_df[total_house_area_features].sum(axis=1)
all_inputs_df['YrSinceRemod'] = all_inputs_df.YrSold - all_inputs_df.YearRemodAdd
all_inputs_df['Age'] = all_inputs_df.YrSold - all_inputs_df.YearBuilt
new_features = ['TotalHouseSF', 'YrSinceRemod', 'Age']
ordinal_df_ = pd.concat([ordinal_df, y], axis=1)
ordinal_corr = ordinal_df_.corr(method='spearman').iloc[-1,:-1].abs().sort_values()
print(f'ordinal_corr : {ordinal_corr.shape}')

# plot the correlation values
plt.figure(figsize=(16,6))
if run_plots:
    ordinal_corr.plot.bar();
numeric_features = list(ordinal_features) + list(parametric_features) + new_features
print(list(ordinal_features))
print(list(parametric_features))
print(new_features)
def encode(df):

    for col in maps.keys():
        df[col] = df[col].map(maps[col])
    
    return df
all_inputs_df = encode(all_inputs_df)
X_train_df, X_test_df = train_test_splitter(all_inputs_df, train_ids, test_ids)
if run_plots:
    fig = plt.figure(figsize=(12,100))
    x = X_train_df
    xlen = len(x.columns)
    for i in np.arange(xlen):
        ax = fig.add_subplot(40,3,i+1)
        sns.regplot(x=x.iloc[:,i], y=y)

    plt.tight_layout()
    plt.show()
outlier_df = pd.concat([X_train_df, train.SalePrice], axis=1)
lowQualFinSF_outlier = outlier_df.LowQualFinSF.index[-1]
poolArea_outlier = outlier_df.SalePrice[outlier_df.SalePrice > 600000].index[-1]
bsmtFinSF1_outlier = outlier_df.BsmtFinSF1.index[-1]
totalHouseSF_outlier = outlier_df.TotalHouseSF.index[-1]
totalBsmtSF_outlier = outlier_df.TotalBsmtSF.index[-1]
print(f'lowQualFinSF_outlier: {lowQualFinSF_outlier}\n',
      f'poolArea_outlier: {poolArea_outlier}\n',
      f'bsmtFinSF1_outlier: {bsmtFinSF1_outlier}\n',
      f'totalHouseSF_outlier: {totalHouseSF_outlier}\n',
      f'totalBsmtSF_outlier: {totalBsmtSF_outlier}\n',
     )
other_outliers = 1459
X_train_df.PoolArea.loc[poolArea_outlier] = X_train_df.PoolArea.median()
X_train_df.LowQualFinSF.loc[other_outliers] = X_train_df.LowQualFinSF.median()
X_train_df.BsmtFinSF1.loc[other_outliers] = X_train_df.BsmtFinSF1.median()
X_train_df.TotalHouseSF.loc[other_outliers] = X_train_df.LowQualFinSF.median()
X_train_df.TotalBsmtSF.loc[other_outliers] = X_train_df.TotalBsmtSF.median()
def scaler(df, ss=None):
    columns = df.columns
    if not ss:
        ss = StandardScaler()
        df = ss.fit_transform(df)
    else:
        df = ss.transform(df)
    df = pd.DataFrame(df, columns=columns)
    features = df.columns.values
    df = df.fillna(df.median())
    return df, features, ss
X_train, X_train_features, ss = scaler(X_train_df)
X_test, X_test_features, ss = scaler(X_test_df, ss)
y_train = y
variances = X_train_df.var().sort_values()
low_var_cols = variances[variances<0.1].index
X_train = X_train.drop(low_var_cols, axis=1)
X_train = X_train.drop('Id', axis=1)
X_train_features = X_train.columns.values
score_feat_eng_pt1 = xgb_cvs(X_train, y_train)
scores.append(score_feat_eng_pt1)
latest_score_delta = scores[-2] - scores[-1]
print(scores)
print(latest_score_delta)
rfr = RandomForestRegressor(n_estimators=200, max_features='auto')
rfr.fit(X_train, y_train)

ranking = np.argsort(-rfr.feature_importances_)
rfr_ranked_features = X_train_features[ranking]
if run_plots:
    f, ax = plt.subplots(figsize=(20, 16))
    sns.barplot(x=rfr.feature_importances_[ranking], y=rfr_ranked_features, orient='h')
    ax.set_xlabel("feature importance")
    plt.tight_layout()
    plt.show()
skb_ranked_features = []
sel = SelectKBest(k='all', score_func=f_regression)
sel.fit(X_train, y_train)
ranking = np.argsort(-sel.scores_)
skb_ranked_features = X_train_features[ranking]
if run_plots:
    f, ax = plt.subplots(figsize=(20, 16))
    sns.barplot(x=sel.scores_[ranking], y=skb_ranked_features, orient='h')
    ax.set_xlabel("feature importance")
    plt.tight_layout()
    plt.show()
def vary_features_by_rfr_importance(n_features, df=X_train, rfr_ranked_features=rfr_ranked_features):
    n_features = min(n_features, len(rfr_ranked_features))
    return df[importance_ranked_features[0:n_features]]
def vary_features_by_skb_importance(n_features, df=X_train, skb_ranked_features=skb_ranked_features):
    n_features = min(n_features, len(skb_ranked_features))
    sel = SelectKBest(k='all', score_func=f_regression)
    sel.fit(X_train, y_train)
    mask = sel.get_support()
    idxs_selected = sel.get_support(indices=True)
    return df[idxs_selected]
space={'n_features': hp.quniform('n_features', 10, len(X_train_features), 1),
       'regressor': hp.choice('regressor',[
           {'regressor': 'xgb',
            'n_estimators': 1 + hp.randint('n_estimators',50000),
            'max_depth': 1 + hp.randint('max_depth',5000),
            'colsample_bylevel': hp.uniform('colsample_bylevel',0,1),
            'colsample_bytree':  hp.uniform('colsample_bytree',0,1)},
           {'regressor': 'lasso',
            'alpha': hp.uniform('alpha',2e-5,4e3)}
       ]),
       'learning_rate': hp.uniform('learning_rate',0,3),
       'random_state': hp.randint('random_state', 100)}

n_estimators = max_depth = random_state = 1
colsample_bylevel = colsample_bytree = 1.0
learning_rate = 0.1
max_evals = 2000
X_train, X_val, y_train, y_val = train_test_split(X_train,
                                                  y_train,
                                                  test_size=0.2,
                                                  shuffle=False)
class runModel(object):
    def __init__(self,
                 space=space,
                 X_train=X_train,
                 y_train=y_train,
                 X_val=X_val,
                 y_val=y_val,
                 skb_ranked_features=skb_ranked_features,
                 rfr_ranked_features=rfr_ranked_features,
                 trials_file=trials_file,
                 nthread=nthread):
        
        self.space = space
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        print(X_train.shape, X_val.shape)
        self.eval_set  = [( self.X_train, self.y_train)]
        self.skb_ranked_features = skb_ranked_features
        self.rfr_ranked_features = rfr_ranked_features
        self.trials_file = trials_file
        self.nthread = nthread
    
    def vary_features_by_skb_importance(self, n_features):
        
        self.sel = SelectKBest(k=n_features, score_func=f_regression)
        self.sel.fit(self.X_train, self.y_train)
        self.mask = self.sel.get_support()
        self.idxs_selected = self.sel.get_support(indices=True)
        
        return self.X_train.iloc[:,self.idxs_selected], self.X_val.iloc[:,self.idxs_selected]
    
    def objective(self, params):
        print(f'Params: {params}')
        self.params = params
        
        self.n_features = int(params['n_features'])

        self.X_train_, self.X_val_ = self.vary_features_by_skb_importance(self.n_features)
        
        self.eval_set  = [(self.X_train_, self.y_train)]
        self.regressor_params = params['regressor']
        self.regressor_type = params['regressor']['regressor']
        
        self.learning_rate = params['learning_rate']
        self.random_state = params['random_state']
        self.regressor_params.pop('regressor', None)
        if self.regressor_type == 'xgb':
            self.regressor_params['nthread'] = nthread
            self.model = XGBRegressor(**self.regressor_params)
            self.model.fit(self.X_train_,
                           self.y_train)
        else:
            self.model = Lasso(self.regressor_params['alpha'])
            self.model.fit(self.X_train_,
                           self.y_train)

        self.y_pred = self.model.predict(self.X_val_)
        self.loss = rmse(self.y_val, self.y_pred)
        print(f'Loss: {self.loss}')
        return {'loss': self.loss, 'status': STATUS_OK}
    
    def run_trials(self, trials_step=1, max_trials=20):
        self.trials_step = trials_step
        self.trials_file = trials_file
        self.trials_step = trials_step
        self.max_trials = max_trials

        try:
            if os.path.isfile(self.trials_file):
                print(f'Found saved trials at {self.trials_file}. Loading...')
                self.trials = pickle.load(open(self.trials_file, 'rb'))
            else:
                self.trials = Trials()
            self.max_trials = len(self.trials.trials) + self.trials_step
            print(f'Running from {len(self.trials.trials)} to {len(self.trials.trials) +self.trials_step} trials')
        except IndexError:
            print(f'No valid saved trials found at {trials_file}.  Starting a new trials file.')
            
        self.n_startup_jobs = min(int(self.max_trials/4), 20)
        self.best_params = fmin(self.objective,
                                self.space,
                                algo=partial(tpe.suggest,
                                             n_startup_jobs=self.n_startup_jobs),
                                max_evals=self.max_trials,
                                trials=self.trials,
                                verbose=1)

        print(f'Best: {self.best_params}')
        if self.trials_file:
            print(f'Saving new trials to {self.trials_file}')
            pickle.dump(self.trials, open(self.trials_file, 'wb'))

        return (self.best_params, self.trials)
run_trials = runModel(space=space,
           X_train=X_train,
           y_train=y_train,
           X_val=X_val,
           y_val=y_val,
           trials_file=trials_file,
           nthread=nthread)
trials_step = 10
max_trials = 40
loops = 4
for i in range(loops):
    best_params, trials = run_trials.run_trials(trials_step, max_trials)
t_end = datetime.datetime.utcnow()
t_run = t_end - t_start
print(str(t_run).split('.')[0])
test['TotalHouseSF'] = test[total_house_area_features].sum(axis=1)
test['YrSinceRemod'] = test.YrSold - test.YearRemodAdd
test['Age'] = test.YrSold - test.YearBuilt
X_test = encode(test)
X_test, X_test_features, ss = scaler(X_test, ss=ss)
bst_params = sorted(trials.trials, key=lambda x: x['result']['loss'])[0]
bst_params = bst_params['misc']['vals']
bst_params.pop('regressor', None)
bst_params.pop('alpha', None)
for k, v in bst_params.items():
    bst_params[k] = v[0]
X_test = X_test.drop(low_var_cols, axis=1)
X_test = X_test.drop('Id', axis=1)
X_test = X_test.fillna(X_test.median())
X_train_df = X_train_df.drop(low_var_cols, axis=1)
X_train_df = X_train_df.drop('Id', axis=1)
X_train_df = X_train_df.fillna(X_train_df.median())
X_test = X_test.fillna(X_test.median())
y_train = y
print('Training XGB regressor')
dtrain_reg = xgb.DMatrix(X_train_df, y_train)

num_rounds = 300

eval_list_reg = [(dtrain_reg, 'train')]
bst_reg = xgb.train(bst_params,
                    dtrain_reg,
                    num_rounds,
                    eval_list_reg,
                    early_stopping_rounds=10)

d_test = xgb.DMatrix(X_test)
y_pred_xgb_reg = bst_reg.predict(d_test,
                        ntree_limit=bst_reg.best_ntree_limit)
y_pred_xgb_reg
mean_pred = y_pred_xgb_reg
out_preds = pd.DataFrame()
out_preds['Id'] = test['Id']
out_preds['SalePrice'] = mean_pred
out_preds.to_csv('submission.csv', index=False)