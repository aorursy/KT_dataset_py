import numpy as np

import pandas as pd

import os

print('input files:\n', os.listdir("../input"))

import xgboost as xgb

from xgboost.sklearn import XGBRegressor



import matplotlib.pyplot as plt

import seaborn as sns

sns.set()



from itertools import product

from sklearn.preprocessing import StandardScaler, KBinsDiscretizer, OneHotEncoder, RobustScaler, FunctionTransformer

from sklearn.impute import SimpleImputer

from sklearn.metrics import mean_squared_error, mutual_info_score, normalized_mutual_info_score, adjusted_mutual_info_score

from sklearn.pipeline import Pipeline

from sklearn.feature_selection import mutual_info_regression

from sklearn.model_selection import cross_val_score, cross_validate, cross_val_predict

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor

from sklearn.svm import LinearSVR, NuSVR, SVR

from sklearn.linear_model import LinearRegression, Lasso, LassoCV

from sklearn.base import BaseEstimator, RegressorMixin



from scipy import stats

import datetime



%matplotlib inline

plt.rcParams['figure.figsize'] = (8,8)



import sklearn as sk

print('sklearn:',sk.__version__)

print('seaborn:',sns.__version__)

print('xgboost', xgb.__version__)



pd.options.display.max_rows = 999
def grid_plot(df, y, cols = None, hue = None, plot_func = sns.boxplot, n_col = 3, figsize = (25,30)):

    if cols is None:

        cols = df.columns

        

    shape = (int(np.ceil(len(cols)/n_col)), n_col)

    fig, axs = plt.subplots(shape[0],shape[1], figsize = figsize)

    

    for i,c in enumerate(cols):

        args = {'x' : c, 'y' : y}

        if hue is not None:

            args['hue'] = hue

            args['dodge'] = True



        plot_func(**args, data = df, ax = axs[i//n_col, i%n_col] if shape[0] > 1 else axs[i%n_col])

        

def df_info(df):

    features_info = pd.DataFrame(data = {'dtype' : df.dtypes, 

                             'na_count' : df.isnull().sum(axis = 0),

                             '0_count' : (df == 0).sum(axis = 0),

                             'unique_count': df.nunique(dropna=False)

                            })

    return features_info

df_raw = pd.read_csv('..//input//train.csv')

df_test = pd.read_csv('../input/test.csv')

target_col = 'SalePrice'

df = df_raw.copy().drop('Id', axis = 1)

features_info = df_info(df)

display(features_info)

display(df.head())
display(df[target_col].describe())

sns.distplot(df[target_col], fit = stats.norm)
df[target_col] = np.log(df_raw[target_col])

y = df[target_col]

sns.distplot(y, fit = stats.norm)
df.drop(target_col, axis = 1, inplace=True)
base_mse = mean_squared_error(y, y.mean() + np.zeros_like(y))

print('MSE:', base_mse)

print('RMSE:', base_mse**0.5)

#mean_squared_error(y, y.median() + np.zeros_like(y))
year_cols = ['YearBuilt' , 'YearRemodAdd', 'GarageYrBlt', 'YrSold']

month_cols = ['MoSold']

overall_rank_cols = ['OverallQual', 'OverallCond']



for col in overall_rank_cols + ['MSSubClass']:

    df[col] = df[col].astype(str)
df_cat = df.select_dtypes(exclude=[np.number])

df_num = df.select_dtypes(include=[np.number])

print('n_cat:', df_cat.shape[1])

print('n_num:', df_num.shape[1])
def na_info(df):    

    info = df_info(df)

    na = info[info['na_count'] > 0].sort_values('na_count', ascending = False)    

    na_index = na.index

    na['na_percent'] = na['na_count'] / len(df)

    display(na)

    if len(na) > 0:

        plt.figure(figsize = (6,6))

        sns.barplot(x = na['na_percent'] , y = na.index);
na_info(df_cat)
na_cols = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageType',

       'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtExposure',

       'BsmtFinType2', 'BsmtQual', 'BsmtCond', 'BsmtFinType1'] + ['Electrical']



df_cat['MasVnrType'] = df_cat['MasVnrType'].fillna('None')

df_cat[na_cols] = df_cat[na_cols].fillna('NA')



na_info(df_cat)
df_ohe = pd.get_dummies(df_cat)
mi_good_cats = list(df_ohe.columns[mutual_info_regression(df_ohe, y).argsort()][-1::-1])

print('mi_good_cats:', mi_good_cats[:10])

grid_plot(df_ohe, y , cols = mi_good_cats[:9], figsize = (20,12))
def fv_info(series, value, y, base_loss, loss):

    dic_ret = {}

    if pd.isnull(value):

        mask = series.isnull()

    else:

        mask = (series == value)



    n = mask.sum()

    dic_ret['feature'] = series.name

    dic_ret['value'] = str(value)

    dic_ret['count'] = n

    dic_ret['p'] = dic_ret['count']/len(series)

    dic_ret['y_in_mean'] = y[mask].mean()

    dic_ret['loss_in'] = loss(y[mask], dic_ret['y_in_mean']) if n > 0 else 0

    dic_ret['loss_out'] = loss(y[~mask], y[~mask].mean())

    dic_ret['loss'] = (dic_ret['p']*dic_ret['loss_in']**2 + (1 - dic_ret['p'])*dic_ret['loss_out']**2)**0.5

    dic_ret['delta_loss'] = base_loss - dic_ret['loss']

    

    return dic_ret





def split_info(df_imp, y, cols = 'all', loss = None):

    if loss is None:

        loss = lambda y_true, y_pred : mean_squared_error(y_true, y_pred + np.zeros_like(y_true))**0.5

    

    base_loss = loss(y, y.mean())

    

    if isinstance(cols, str) and cols == 'all':

        cols = list(df_imp.columns)

        

    dfs = []

    for col in cols:

        series = df_imp[col]

        values = list(set(series[~series.isnull()]))

        values += [None] if series.isnull().sum() > 0 else []

        order = ['feature', 'value', 'count', 'p', 'y_in_mean','loss_in', 'loss_out', 'loss', 'delta_loss']

        records = list(map(lambda value: fv_info(series, value, y, base_loss, loss), values))

        df_col = pd.DataFrame(records, columns=order)

        dfs.append(df_col)

        

    return pd.concat(dfs, axis = 0, ignore_index=True).sort_values('delta_loss', ascending = False)
ci = split_info(df_cat, y)



print('good delta loss')

mask2 = ci['delta_loss'] > 0.025

display(ci[mask2].head(10))

si_good_cats = set(ci[mask2]['feature'])



print('representative value and low loss')

mask1 = (ci['p'] > 0.04) & (ci['loss_in'] < 0.31)

display(ci[mask1].sort_values('loss_in').head(10))

si_rll_cats = set(ci[mask1]['feature'])



print('low representative + bad loss')

mask3 = (ci['loss_in'] > 0.022) & (ci['p'] <= 0.05)

display(ci[mask3].sort_values('loss_in').head(10))
si_cats = si_good_cats.union(si_rll_cats)

not_in_si = set(df_cat.columns).difference(si_cats)

print('Most informative categoricals (Split Information)\n' , si_cats)



#print('Not in si:', not_in_si)
#display(y[df_cat['OverallCond'] == '2'])



hue = None



# We can uncomment this to follow a specific point in all the plots



"""idx_select = 523

hue = pd.Series(index = df_cat.index, data = 0)

hue.loc[idx_select] = 1"""
grid_plot(df_cat, y, hue = hue, cols = list(si_cats), plot_func = sns.stripplot)
grid_plot(df_cat, y, hue = hue, cols = list(not_in_si), figsize=(20,35), plot_func = sns.stripplot)
fig, axs = plt.subplots(3, 2, figsize = (25,15))



sns.swarmplot(hue = df_cat['KitchenQual'], y = y, x = df_cat['OverallQual'], ax = axs[0,0])

sns.swarmplot(hue = df_cat['OverallCond'], y = y, x = df_cat['OverallQual'], ax = axs[0,1])

sns.swarmplot(hue = df_cat['BsmtQual'], y = y, x = df_cat['OverallQual'], ax = axs[1,0])

sns.swarmplot(hue = df_cat['ExterQual'], y = y, x = df_cat['OverallQual'], ax = axs[1,1])

sns.swarmplot(hue = df_cat['GarageQual'], y = y, x = df_cat['OverallQual'], ax = axs[2,0])

sns.swarmplot(hue = df_cat['GarageCond'], y = y, x = df_cat['OverallQual'], ax = axs[2,1])
df_num.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).T
na_info(df_num)
# No Masonty => MasVnrArea = 0

df_num['MasVnrArea'] = df_num['MasVnrArea'].fillna(0)



# No garage => GarageYrBlt = 0

df_num['GarageYrBlt'] = df_num['GarageYrBlt'].fillna(0)
# Categorical boxplots only for examples with LotFrontage equals NA



#mask = df_num['LotFrontage'].isnull()

#grid_plot(df_cat[mask], y[mask], plot_func = sns.stripplot, figsize=(20,60))
# Scatterplots looking for LotFrontage and numerical feature relationship



#mask = ~df_num['LotFrontage'].isnull()

#grid_plot(df_num[mask], df_num['LotFrontage'][mask], plot_func=sns.scatterplot, figsize=(20,40))
mask = ~df_num['LotFrontage'].isnull()

sns.stripplot(x = 'Neighborhood', y = df_num['LotFrontage'][mask], data = df_cat[mask])

#grid_plot(df_cat[mask], df_num['LotFrontage'][mask], plot_func = sns.stripplot, figsize=(20,50))
mask = df_num['LotFrontage'].isnull()

rep_dic = df[['Neighborhood', 'LotFrontage']].groupby('Neighborhood').median().to_dict()['LotFrontage']

df_num.loc[mask, 'LotFrontage'] = df_cat['Neighborhood'][mask].replace(rep_dic)
na_info(df_num)
num_info = df_info(df_num)

th_num_boxplot = 15

mask = num_info['unique_count'] <= th_num_boxplot

grid_plot(df_num, y, cols = num_info.index[mask], plot_func= sns.stripplot)
mask = ~ (df_num['GarageYrBlt'] == 0)

grid_plot(df_num[mask], y[mask], plot_func=sns.scatterplot, n_col=2, cols = year_cols, figsize=(20,10))
fig, ax = plt.subplots(figsize=(20, 25))

df_num.hist(ax = ax);



grid_plot(df_num, y, plot_func=sns.scatterplot, figsize=(20,45))
mi_good_numeric = list(df_num.columns[mutual_info_regression(df_num.fillna(df_num.mean()), y).argsort()][-1::-1])

print('mi_good_numeric:', mi_good_numeric)
skew_th = 0.7

ax = sns.barplot(x = df_num.apply(lambda s : stats.skew(s)).abs(), y = df_num.columns)

ax.vlines(skew_th, *ax.get_ylim())



mask_skew = df_num.apply(lambda s : stats.skew(s)).abs() > skew_th



print(df_num.columns[mask_skew])



#fig, ax = plt.subplots(figsize=(20, 25))

#df_num.loc[:,mask_skew].hist(ax = ax);
df_num_t = df_num.copy()

"""

df_num_t['LotFrontage'] = np.clip(df_num_t['LotFrontage'] , 0, 180)

df_num_t['1stFlrSF'] = np.clip(df_num_t['1stFlrSF'] , 0, 2200)

df_num_t['2ndFlrSF'] = df_num_t['2ndFlrSF']

df_num_t['BsmtFinSF1'] = np.clip(df_num_t['BsmtFinSF1'], 0, 1800)

df_num_t['GarageArea'] = df_num_t['GarageArea']

df_num_t['GrLivArea'] = np.clip(df_num_t['GrLivArea'], 0, 3000)

df_num_t['LotArea'] = np.log(np.clip(df_num_t['LotArea'], 0, 50000))

df_num_t['MasVnrArea'] = df_num_t['MasVnrArea']

df_num_t['OpenPorchSF'][df_num_t['OpenPorchSF'] > 0] = np.log(df_num_t['OpenPorchSF'][df_num_t['OpenPorchSF'] > 0])

df_num_t['TotalBsmtSF'] = np.clip(df_num_t['TotalBsmtSF'] , 0, 2200)

df_num_t['WoodDeckSF'] = df_num_t['WoodDeckSF']

"""



# log transform skewd features

df_num_t.loc[:, mask_skew] = np.log1p(df_num.loc[:, mask_skew])



# new histograms

fig, ax = plt.subplots(figsize=(20, 15))

df_num_t.hist(ax = ax);



# scatter plots with feature x SalePrice

grid_plot(df_num_t, y, plot_func=sns.scatterplot, figsize=(20,45))
df_corr = pd.concat([df_num, y], axis = 1).corr()



display(sns.heatmap(df_corr))



print('High correlations')

for ((c1,c2),b) in (df_corr.abs() > 0.7).unstack().iteritems():

    if c1 != c2 and b:

        print(c1,c2, df_corr.loc[c1,c2])

        
df_info(df_test.loc[:, df_test.isnull().sum() > 0])
from sklearn.model_selection import KFold



def create_submission(name, target_name, df_test, model, tranf_func = lambda x : x):

    if tranf_func is not None:

        df_test_t = tranf_func(df_test)

    

    y_test = model.predict(df_test_t)

    y_test = np.exp(y_test)

    df_sub = pd.DataFrame({'Id' : df_test['Id'] , target_name : y_test})

    print('Submission head:')

    display(df_sub.head())    

    filepath = name + '.csv'

    df_sub.to_csv(filepath, index=False)

    print('Submission created in:', filepath)        

        

def evaluate_model(model, df, y, cv = 5):

    kfold = KFold(n_splits = cv, shuffle=True, random_state=42)

    scores = cross_validate(model, df, y, scoring='neg_mean_squared_error', return_train_score=True, cv = kfold)

    

    print('RMSE train: ', (-scores['train_score'].mean())**0.5, ' +-', scores['train_score'].std()**0.5)

    print('RMSE test: ', (-scores['test_score'].mean())**0.5, ' +-', scores['test_score'].std()**0.5)

    

    model.fit(df,y)    

    y_pred = model.predict(df)

    

    print('Full RMSE train:', mean_squared_error(y, y_pred)**0.5)

    

    if isinstance(model, Pipeline):

        last_model = model.steps[-1][1]

        df_ = Pipeline(model.steps[:-1]).transform(df)

        df_ = pd.DataFrame(df_)

    else:

        last_model = model

        df_ = df

        

    if hasattr(last_model, 'feature_importances_'):

        print('Feature Importances')

        

        fig , ax = plt.subplots(figsize = (10,15))

        sns.barplot(x = 'importance', y = 'feature', 

                    data = pd.DataFrame({'feature' : df_.columns, 'importance' : last_model.feature_importances_})\

                    .sort_values('importance', ascending = False), ax = ax)

        plt.show()

        

    if hasattr(last_model, 'train_score_'):

        print('Train score')

        plt.plot(last_model.train_score_, label = 'train_score')

        plt.show()

    

    print('Real x Predicted')

    fig, ax = plt.subplots()

    sse = (y - y_pred)**2    

    ax = sns.scatterplot(x = y, y = y_pred, hue = sse, ax = ax)

    sns.lineplot(x = y, y = y, ax = ax)    

    ax.set_xlabel('y_real')

    ax.set_ylabel('y_pred')

    ax.legend_.texts[0].set_text('SSE')

    plt.show()

        

    return model



def create_error_df(df_raw, df_model, y, model, name):

    y_pred = model.predict(df_model)

    sse = (y - y_pred)**2

    sse_sign = np.sign(y - y_pred)*sse

    sns.scatterplot(x = y, y = y_pred, hue = sse)

    sns.lineplot(x = y, y = y)

    plt.show()

    df_error = df_raw.copy()

    df_error['y_pred'] = y_pred

    df_error['y'] = y

    df_error['sse'] = sse

    df_error['sse_sign'] = sse_sign

    df_error.index.name = 'idx'

    df_error.sort_values('sse', ascending = False, inplace = True)

    display(df_error.head())

    file = '{}_regression_errors.csv'.format(name)

    print('writing errors in:', file)

    df_error.to_csv(file)



kfold = KFold(n_splits=5, shuffle=True, random_state=42)
class T:

    """

    Impute values to numerical and categorical columns.

    Then apply OHE for selected categoricals.

    

    For numerical columns:

    - Apply log transform to selected columns.

    - Apply binarization to selected columns.

    - Apply discretization (KBinDiscretizer) to selected columns.

    

    

    """

    def __init__(self, bin_th = 0.0, numeric_imputer = None, categorical_imputer = None, log_eps = 1):

        self.bin_th = bin_th

        self.log_eps = log_eps

        

        if numeric_imputer is None:

            self.numeric_imputer = SimpleImputer(missing_values=np.nan, strategy='median')

        else:

            self.numeric_imputer = sk.clone(numeric_imputer)

        

        if categorical_imputer is None:

            self.categorical_imputer = Pipeline([('cat_imputer_None', SimpleImputer(None, strategy='constant', fill_value='NA')),

                                                 ('cat_imputer_nan', SimpleImputer(np.nan, strategy='constant', fill_value='NA'))])

        else:

            self.categorical_imputer = sk.clone(cateforical_imputer)

            

        self.ohe_ = OneHotEncoder(handle_unknown='ignore', sparse=False)

            

                                                

    def _impute_ctg(self, df):

        df_imp = df[self.ctg_cols].copy()

        df_imp[self._ctg_cols_default] = self.categorical_imputer.transform(df[self._ctg_cols_default])

        for col in [c for c in self.ctg_cols if c in self.imp_dict]:

            df_imp[col] = self.imp_dict[col].transform(df_imp[[col]])

        

        return df_imp

    

    def _impute_numerical(self, df):

        df_imp = df[self.num_cols].copy()

        

        df_imp[self._num_cols_default] = self.numeric_imputer.transform(df[self._num_cols_default])

        for col in [c for c in self.num_cols if c in self.imp_dict]:

            df_imp[col] = self.imp_dict[col].transform(df_imp[[col]])

        

        return df_imp

    

    def _fit_ohe(self, df):

        

        if len(self.to_ohe) == 0:

            return

        

        # impute to fit ohe after

        df_ohe = self._impute_ctg(df)[self.to_ohe]

        self.ohe_.fit(df_ohe)

        self.ohe_col_names = self.ohe_.get_feature_names(df_ohe.columns)

        

    

    def _fit_kbin(self, df):

        # tuples with (column_name, n_bins, strategy)

        

        if len(self.to_kbin) == 0:

            return

        

        df_kbin = self._impute_numerical(df)[[t[0] for t in self.to_kbin]]

        

        self.kbin_fit_dict = {}

            

        for col, n_bins, strategy in self.to_kbin:

            kb = KBinsDiscretizer(n_bins=n_bins, strategy=strategy, encode='onehot-dense')            

            kb.fit(df_kbin[[col]])

            

            #todo improve col names

            kb_cols = ['{}_{}{}_{}{}'.format(col, '<' if i == 0 else '', y1, 

                                             y2, '>' if i == (len(kb.bin_edges_[0])-2) else '')

                       for i,(y1,y2) in enumerate(zip(kb.bin_edges_[0] , kb.bin_edges_[0][1:]))]

            

            self.kbin_fit_dict[col] =  kb, kb_cols

        

    

    def fit(self, df, imputer_dict = None, to_keep = [], numerical_cols = [], categorical_cols = [],

            to_ohe = [] , to_log = [], to_bin = [], to_kbin = []):

        

        """

        to_keep: 

            list of columns that will copied as is.

        

        numerical_cols:

            list of columns that will have default numerical imputation (if not present in imputer_dict)

        

        categorical_cols:

            list of columns that will have default categorical imputation (if not present in imputer_dict)

            

        imputer_dict:

            dict with (column_name : imputer_transformer) that will be applied instead of default imputation strategy.

            

        to_ohe (list or str):

            list of columns to be ohe, or 'all' if all categorical_cols

        

        to_log:

            list of columns to be log transformed.

            

        to_bin:

            list of columns to be binarized.

        

        to_kbin:

            lisf of tuples (column_name, n_bins, strategy) with columns to be discretized.

            

        """

        

        self.imp_dict = {col_name : sk.clone(imp) for (col_name,imp) in imputer_dict.items()} if (imputer_dict is not None) else dict()

        self.to_keep = list(to_keep)

        self.num_cols = list(numerical_cols)

        self.ctg_cols = list(categorical_cols)

        self.to_ohe = list(to_ohe) if to_ohe != 'all' else list(categorical_cols)

        self.to_log = list(to_log)

        self.to_bin = list(to_bin)

        self.to_kbin = list(to_kbin)

        

        self._num_cols_default = [col for col in self.num_cols if col not in self.imp_dict]

        self._ctg_cols_default = [col for col in self.ctg_cols if col not in self.imp_dict]

        

        

        # fit imputers in dict

        for col_name, imp in self.imp_dict.items():

            imp.fit(df[[col_name]])

            

        # fit numerical imputer

        if len(self._num_cols_default) > 0:

            self.numeric_imputer.fit(df[self._num_cols_default])

        

        # fit categorical imputer

        if len(self._ctg_cols_default) > 0:

            self.categorical_imputer.fit(df[self._ctg_cols_default])



        # fit ohe

        self._fit_ohe(df)

        

        # fit kbin

        self._fit_kbin(df)

        

        

    def transform(self, df):

        df_ret = pd.DataFrame(index = df.index)

        

        if len(self.to_keep) > 0:

            df_ret[self.to_keep] = df[self.to_keep]

        

        # Numerical transformations

        df_num = self._impute_numerical(df) if len(self.num_cols) > 0 else pd.DataFrame(index = df.index)

        

        if len(self.to_log) > 0:

            df_num[self.to_log] = np.log(df_num[self.to_log] + self.log_eps)

        if len(self.to_bin) > 0:

            df_num[self.to_bin] = (df_num[self.to_bin] > self.bin_th).astype(int)

        

        if len(self.to_kbin) > 0:

            dfs_kbin = []

            for col, (kb_fitted,kb_cols) in self.kbin_fit_dict.items():

                dfs_kbin.append(pd.DataFrame(kb_fitted.transform(df_num[[col]]), columns=kb_cols, index = df_num.index))

            

            df_num.drop([t[0] for t in self.to_kbin], axis = 1, inplace = True)

            df_num = pd.concat([df_num] + dfs_kbin, axis = 1, verify_integrity=True)

        

        # Categorical transformations

        df_ctg = self._impute_ctg(df) if len(self.ctg_cols) > 0 else pd.DataFrame(index = df.index)

        

        if len(self.to_ohe) > 0:

            df_ohe = pd.DataFrame(data = self.ohe_.transform(df_ctg[self.to_ohe]), columns=self.ohe_col_names, index = df_ctg.index)

            df_ctg.drop(self.to_ohe, axis = 1, inplace = True)

            df_ctg = pd.concat([df_ctg, df_ohe], axis = 1, verify_integrity=True)

            

        df_ret = pd.concat([df_ret, df_ctg, df_num], axis = 1, verify_integrity=True)

        

        

        

        return df_ret
all_num_cols = list(df_num.columns) + ['OverallCond', 'OverallQual']

area_cols = list(df.columns[((df.columns.str.find('Area') >= 0) | (df.columns.str.find('SF') >= 0))])  

fill_0_cols = area_cols + ['BsmtFullBath', 'BsmtHalfBath'] + ['GarageYrBlt','GarageCars']



my_ctg_cols = df_cat.columns.difference(['OverallQual', 'OverallCond', 'Alley', 'Utilities', 'MSSubClass', 'Condition1', 'Condition2', 'Fence', 'Heating', 'PoolQC'])

ctg_qual_cols = ['KitchenQual', 'ExterQual','ExterCond', 'HeatingQC', 'GarageQual','GarageCond','BsmtQual','BsmtCond','FireplaceQu']

to_ohe = my_ctg_cols.difference(ctg_qual_cols + ['Neighborhood'])



my_t = T()



my_t.fit(df, to_keep=[], numerical_cols = all_num_cols

         , to_bin = []

         , to_log = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 

                     'GrLivArea', 'BsmtHalfBath', 'KitchenAbvGr', 'GarageYrBlt', 

                     'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 

                     'ScreenPorch', 'PoolArea']

         

         , to_kbin = [] #[(y,10,'uniform') for y in ['YearBuilt']]

         

         , categorical_cols = list(my_ctg_cols)

         

         , to_ohe = list(to_ohe)

         

         , imputer_dict = {'MasVnrType' : SimpleImputer(strategy='constant', fill_value='None')}.update(

             {col : SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0) for col in fill_0_cols})

        )
# some dictionaries that extracts data the trainning dataset as a whole and are used in preprocess



# neighborhood name -> median LotFrontage

neighborhood_to_lot_frontage = df[['Neighborhood', 'LotFrontage']].groupby('Neighborhood').median().to_dict()['LotFrontage']



# neighborhood name -> median log SalePrice

neighborhood_to_log_sale = np.log(df_raw.groupby('Neighborhood').median()['SalePrice']).to_dict()
# lets keep the median/mean SalePrice/TotalSF grouped by some categories

total_sf = df['1stFlrSF'] + df['2ndFlrSF'] + df['TotalBsmtSF'].fillna(0)

price_by_sf = df_raw[target_col] / total_sf



price_by_sf_cols = ['Neighborhood']

df_ = df[price_by_sf_cols].copy()

df_['price_by_sf'] = price_by_sf



price_by_sf_dicts = dict()

for col in price_by_sf_cols:

    price_by_sf_dicts[col] = df_.groupby(col).mean()['price_by_sf'].to_dict()
def preproc(df, fitted_transf = my_t, is_lm = False, remove_unimportant = True):

    df_ = df.copy()

    

    # special imputer for LotFrontage

    mask = df_['LotFrontage'].isnull()

    df_.loc[mask, 'LotFrontage'] = df_['Neighborhood'][mask].replace(neighborhood_to_lot_frontage).astype(float)



    # custom transformer

    df_ = fitted_transf.transform(df_)

    

    if not is_lm:

        # Changing neighborhood to mean neighborhood sale price.

        df_.replace({'Neighborhood' : neighborhood_to_log_sale}, inplace = True)

        df_['Neighborhood'] = df_['Neighborhood'].astype(float)



        # Changing some categorical to numerical using it's order

        qual_dic = {'NA' : 0 , 'Po' : 1, 'Fa' : 2, 'TA' : 3, 'Gd' : 4, 'Ex' : 5}

        df_.replace({col : qual_dic for col in ctg_qual_cols}, inplace = True)

        df_[ctg_qual_cols] = df_[ctg_qual_cols].astype(float)

    

    # Creating new features

    df_['TotalSF'] = df_['1stFlrSF'] + df_['2ndFlrSF'] + df_['TotalBsmtSF']

    df_['TotalFullBath'] = df_['FullBath'] + df_['BsmtFullBath']

    df_['TotalHalfBath'] = df_['HalfBath'] + df_['BsmtHalfBath']

    df_['Has_RemodAdd'] = (df_['YearRemodAdd'] != df_['YearBuilt']).astype(int)

    df_['GrLivAreaPerRms'] = df_['GrLivArea'] / df_['TotRmsAbvGrd']

    df_['YearBuilt2'] = df_['YearBuilt']**2    

    df_['OverallCQ'] = df_['OverallCond']*df_['OverallQual']

    

    # creating estimates of log(SalePrice) using median sale_price_by_sf dicts

    for col in price_by_sf_dicts.keys():

        df_['price_by_sf_by_{}'.format(col)] = df[col].apply(lambda n : price_by_sf_dicts[col][n])

        #df_['log_price_sf_estimate_{}'.format(col)] = np.log(df_['price_by_sf_by_{}'.format(col)]*df_['TotalSF'])

    

    if not is_lm:

        df_['ExterCQ'] = df_['ExterQual']*df_['ExterCond']

        df_['BsmtCQ'] = df_['BsmtCond']*df_['BsmtQual']

        df_['GarageCQ'] = df_['GarageCond']*df_['GarageQual']

        df_['CQ_all'] = df_['OverallCQ'] * df_['BsmtCQ'] * df_['KitchenQual'] * df_['GarageCQ'] * df_['ExterCQ']

    

    df_['Age1'] = df_['YrSold'] - df_['YearBuilt']

    df_['Age2'] = df_['YrSold'] - df_['YearRemodAdd']

    

    if remove_unimportant:

        # droped based on low feature importance

        df_.drop(['Electrical_Mix', 'SaleType_CWD', 'Exterior2nd_Other', 'Functional_Sev',

           'RoofMatl_ClyTile', 'Foundation_Wood', 'Electrical_NA',

           'Exterior2nd_CBlock', 'Exterior1st_AsphShn', 'Exterior1st_BrkComm',

           'SaleCondition_AdjLand', 'Exterior1st_CBlock', 'RoofMatl_Membran',

           'Exterior1st_ImStucc', 'RoofMatl_Metal', 'RoofMatl_Roll',

           'Exterior1st_Stone', 'RoofStyle_Shed', 'MiscFeature_TenC',

           'SaleType_Con', 'Exterior2nd_AsphShn', 'RoofMatl_WdShake',

           'MiscFeature_Othr', 'SaleType_Oth', 'SaleType_ConLw', 'LotConfig_FR3',

           'MiscFeature_Gar2', 'Electrical_FuseP', 'MasVnrType_NA',

           'Exterior2nd_ImStucc', 'Foundation_Stone', 'Exterior2nd_Brk Cmn',

           'Exterior2nd_Stone', 'RoofStyle_Mansard', 'SaleType_ConLI',

           'HouseStyle_2.5Fin', 'GarageType_2Types', 'Street_Pave', 'Street_Grvl',

           'PoolArea', 'LotShape_IR3', 'SaleType_ConLD', 'RoofMatl_Tar&Grv',

           'HouseStyle_2.5Unf', 'BsmtFinType2_GLQ', 'SaleCondition_Alloca',

           'HouseStyle_1.5Unf', 'RoofStyle_Gambrel', 'Functional_Maj2',

           'RoofStyle_Flat', 'LandSlope_Sev', 'RoofMatl_WdShngl', 'MSZoning_RH',

           'GarageType_Basment', 'BsmtFinType2_ALQ', 'GarageType_CarPort',

           'Functional_Min1', 'MasVnrType_BrkCmn', 'LowQualFinSF', '3SsnPorch',

           'Functional_Maj1', 'Exterior1st_WdShing', 'BsmtFinType2_BLQ',

           'BsmtFinType2_LwQ', 'Exterior2nd_AsbShng', 'Functional_Mod',

           'HouseStyle_SFoyer', 'Foundation_Slab', 'MSZoning_C (all)',

           'Exterior2nd_BrkFace', 'MiscFeature_Shed', 'Exterior1st_AsbShng',

           'BsmtFinType2_Rec', 'SaleCondition_Family', 'PavedDrive_P',

           'Functional_Min2', 'MiscFeature_NA', 'Exterior1st_Stucco',

           'SaleType_COD', 'BsmtHalfBath', 'HouseStyle_SLvl',

           'Exterior2nd_Wd Shng', 'Exterior2nd_Stucco', 'BsmtExposure_Mn',

           'BldgType_2fmCon', 'LotConfig_FR2', 'MSZoning_FV', 'BsmtFinType1_LwQ',

           'LandContour_HLS', 'LotShape_IR2', 'Electrical_FuseF',

           'RoofMatl_CompShg', 'MiscVal', 'BsmtExposure_NA', 'BsmtFinType2_Unf',

           'Exterior1st_Plywood', 'Exterior2nd_HdBoard', 'LandContour_Low',

           'BsmtFinType1_BLQ', 'BldgType_Twnhs'], axis = 1, inplace = True)

    

    return df_



df_train = df.copy()

df_train.drop([1298, 523],inplace=True)

y_train = y.copy()

y_train = y.drop([1298,523], axis = 0)



df_basic = preproc(df_train, my_t)



preproc_transformer = FunctionTransformer(func=preproc, validate=False)

print(df_basic.shape)
display(df_info(preproc_transformer.transform(df_train)))
%%time



rf_basic = RandomForestRegressor(n_estimators=5000, max_features=8, max_depth = 8, min_impurity_decrease = 0.0,  min_samples_split = 8, bootstrap=True, random_state=42)

rf_pipe = Pipeline([('preproc', preproc_transformer), ('rf', rf_basic)])

rf_pipe = evaluate_model(rf_pipe, df_train, y_train, cv = 5);
create_submission('rf_pipe', target_col, df_test, rf_pipe)
%%time



#Single model based on good models found by Random Grid Serach CV



gb_basic = GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,

             learning_rate=0.001, loss='ls', max_depth=4,

             max_features=20, max_leaf_nodes=None,

             min_impurity_decrease=0.0, min_impurity_split=None,

             min_samples_leaf=3, min_samples_split=8,

             min_weight_fraction_leaf=0.0, n_estimators=11000,

             n_iter_no_change=None, presort='auto', random_state=42,

             subsample=0.15, tol=0.0001, validation_fraction=0.2, verbose=0,

             warm_start=False)



gb_pipe = Pipeline([('preproc', preproc_transformer), ('gb', gb_basic)])



gb_pipe = evaluate_model(gb_pipe, df_train, y_train, cv = 5);
plt.plot(np.cumsum(gb_basic.oob_improvement_[5000:]))
create_submission('gb_pipe', target_col, df_test, gb_pipe)
%%time





xgb_basic = XGBRegressor(max_depth=4, learning_rate=0.01, n_estimators=2000, silent=True, objective='reg:linear', booster='gbtree',

                         n_jobs=1, nthread=None, gamma=0, min_child_weight=3, max_delta_step=0, subsample=0.3, colsample_bytree=0.7, 

                         colsample_bylevel=1, reg_alpha=1, reg_lambda=1, scale_pos_weight=1, base_score=0.5, 

                         random_state=42, missing=None)





xgb_pipe = Pipeline([('preproc', preproc_transformer),('xgb', xgb_basic)])

xgb_pipe = evaluate_model(xgb_pipe, df_train, y_train, cv = 5);
lm_t = T()



lm_t.fit(df, to_keep=[], numerical_cols = all_num_cols

         , to_bin = []

         , to_log = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 

                     'GrLivArea', 'BsmtHalfBath', 'KitchenAbvGr', 'GarageYrBlt', 

                     'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 

                     'ScreenPorch', 'PoolArea']

         

         , to_kbin = [] #[(y,10,'uniform') for y in ['YearBuilt']]

         

         , categorical_cols = list(my_ctg_cols)

         

         , to_ohe = list(to_ohe) + ['Neighborhood'] + ctg_qual_cols

         

         , imputer_dict = {'MasVnrType' : SimpleImputer(strategy='constant', fill_value='None')}.update(

             {col : SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0) for col in fill_0_cols})

        )



preproc_lm = lambda df : preproc(df, fitted_transf=lm_t, is_lm=True)

preproc_transformer_lm = FunctionTransformer(preproc_lm, validate = False)

lm_cols = preproc_lm(df_train).columns
%%time



lasso = Lasso(alpha = 0.0009, max_iter=11000, tol= 1e-4, random_state=42) #LassoCV(normalize=True, max_iter=1000, random_state=42, n_alphas=5000, cv = kfold)

lasso_pipe = Pipeline([('preproc', preproc_transformer_lm),('scaler', RobustScaler()), ('lasso', lasso)])



lasso_pipe = evaluate_model(lasso_pipe, df_train, y_train, cv = 5);



display(pd.DataFrame({'cols' : lm_cols , 'coef_' : np.abs(lasso_pipe.steps[-1][1].coef_) }).sort_values('coef_', ascending = False).head())

#print('alpha:', lasso_pipe.steps[-1][1].alpha_)
create_submission('lasso_pipe', 'SalePrice', df_test, lasso_pipe)
%%time



svr = SVR(kernel='rbf', gamma = 0.001, C = 6.0, shrinking=True)

svr_pipe = Pipeline([('preproc', preproc_transformer_lm),('scaler', RobustScaler()) , ('svm', svr)]) 

svr_pipe = evaluate_model(svr_pipe, df_train, y_train, cv = 5);
create_submission('svm_pipe', 'SalePrice', df_test, svr_pipe)
class simple_stack(BaseEstimator, RegressorMixin):

    

    def __init__(self, models, weights):

        self.models = models

        self.weights = weights



    def fit(self, X, y):

        self.fitted_models_ = [sk.clone(m).fit(X,y) for m in self.models]

        

        return self

    

    def predict(self, X):

        preds = list(map(lambda m: m.predict(X), self.fitted_models_))

        preds = np.vstack(preds).T #(n_samples,n_models)

        weights = np.array(self.weights, ndmin=1).reshape(1, -1) # (1, n_models)

        

        y_pred = np.sum(preds * weights, axis = 1)

        return y_pred
%%time

my_simple_stack = simple_stack(models=[gb_pipe, xgb_pipe, lasso_pipe, svr_pipe],weights=[0.4,0.2,0.2,0.2])



evaluate_model(my_simple_stack, df_train, y_train, cv = 5)
create_submission('simple_stack', 'SalePrice', df_test, my_simple_stack, lambda x : x)
class complex_stack(BaseEstimator, RegressorMixin):

    

    def __init__(self, models, cv = KFold(5, random_state=42), meta_model = None):

        self.models = models

        self.cv = cv

        self.meta_model = meta_model



    def fit(self, X, y):

        self.fitted_models_ = [sk.clone(m).fit(X,y) for m in self.models]

    

        cv_preds = list(map(lambda m : cross_val_predict(m,X,y,cv=self.cv) , self.models))

        cv_preds = np.vstack(cv_preds).T #(n_samples, n_models)

        

        # creating indicator feature for levels of predicted values

        mean_pred = cv_preds.mean(axis = 1, keepdims = True) #(n_samples, 1)

        self.kb_ = KBinsDiscretizer(5, encode='onehot-dense', strategy='uniform')

        pred_ohe = self.kb_.fit_transform(mean_pred) #(n_samples, 5)

        

        features = np.hstack([cv_preds, pred_ohe]) #(n_samples, n_models + 5)

        self.features_ = features

        

        if self.meta_model is None:

            self.meta_model_ = Lasso(alpha=0.01, max_iter=10000, random_state=42)

        else:

            self.meta_model_ = sk.clone(self.meta_model)

        

        self.meta_model_.fit(features,y)

        self.meta_model_score_ = self.meta_model_.score(features, y)

        

        return self

    

    def predict(self, X):

        preds = list(map(lambda m: m.predict(X), self.fitted_models_))

        preds = np.vstack(preds).T #(n_samples,n_models)

        pred_ohe = self.kb_.transform(preds.mean(axis=1, keepdims=True))

        features = np.hstack([preds,pred_ohe])

        

        y_pred = self.meta_model_.predict(features)

        

        return y_pred
%%time



meta_models = [#(RandomForestRegressor(max_depth=4, n_estimators=100, random_state=42), 'rf'), 

               #(SVR(kernel='rbf', gamma = 'scale', C = 1.0, shrinking=True), 'svm'),

               (Lasso(alpha=0.001, max_iter=1000, random_state=42),'lasso')

              ]



complex_models = dict()



for loss_gb, is_lm, remove_unimportant, meta_model in product(['huber', 'ls'], [True, False], [False,True], meta_models):

    

    meta_model, name = meta_model

    

    name = '{}_{}_{}_{}'.format(name, loss_gb, is_lm, remove_unimportant)

    print('='*60)

    print(name)

    

    # preprocs

    fitted_transformer = lm_t if is_lm else my_t

    

    preproc_transformer = FunctionTransformer(func=preproc, validate=False, kw_args={'is_lm' : is_lm, 'fitted_transf' : fitted_transformer, 'remove_unimportant' : remove_unimportant})

    #preproc_lm = lambda df : preproc(df, fitted_transf=lm_t, is_lm=True)

    preproc_transformer_lm = FunctionTransformer(func=preproc, validate=False, kw_args={'is_lm' : True, 'fitted_transf' : lm_t, 'remove_unimportant' : remove_unimportant})

    

    # gb

    gb_basic = GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,

             learning_rate=0.001, loss = loss_gb, max_depth=4,

             max_features=20, max_leaf_nodes=None,

             min_impurity_decrease=0.0, min_impurity_split=None,

             min_samples_leaf=3, min_samples_split=8,

             min_weight_fraction_leaf=0.0, n_estimators=18000,

             n_iter_no_change=None, presort='auto', random_state=42,

             subsample=0.1, tol=0.0001, validation_fraction=0.2, verbose=0,

             warm_start=False)



    gb_pipe = Pipeline([('preproc', preproc_transformer), ('gb', gb_basic)])

    

    # xgb

    xgb_basic = XGBRegressor(max_depth=4, learning_rate=0.01, n_estimators=2000, silent=True, objective='reg:linear', booster='gbtree',

                         n_jobs=1, nthread=None, gamma=0, min_child_weight=3, max_delta_step=0, subsample=0.3, colsample_bytree=0.7, 

                         colsample_bylevel=1, reg_alpha=1, reg_lambda=1, scale_pos_weight=1, base_score=0.5, 

                         random_state=42, missing=None)





    xgb_pipe = Pipeline([('preproc', preproc_transformer),('xgb', xgb_basic)])



    

    # svr

    svr = SVR(kernel='rbf', gamma = 0.001, C = 6.0, shrinking=True)

    svr_pipe = Pipeline([('preproc', preproc_transformer_lm),('scaler', RobustScaler()) , ('svm', svr)]) 

    

    # lasso

    lasso = Lasso(alpha = 0.0009, max_iter=11000, tol= 1e-4, random_state=42) #LassoCV(normalize=True, max_iter=1000, random_state=42, n_alphas=5000, cv = kfold)

    lasso_pipe = Pipeline([('preproc', preproc_transformer_lm),('scaler', RobustScaler()), ('lasso', lasso)])



    

    my_complex_stack = complex_stack(models=[gb_pipe, xgb_pipe, lasso_pipe, svr_pipe], meta_model=meta_model)

    evaluate_model(my_complex_stack, df_train, y_train, cv = 2)

    complex_models[name] = my_complex_stack

    

    create_submission('complex_stack_{}'.format(name), 'SalePrice', df_test, my_complex_stack, lambda x : x)
