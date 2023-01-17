# inital import
import pandas as pd 
import numpy as np 
import os
import sys
import operator
import pickle as pk

from scipy.stats.stats import pearsonr
from scipy.stats import skew, boxcox

import statsmodels.graphics.api as smg
from sklearn.preprocessing.imputation import stats
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.cross_validation import cross_val_score,cross_val_predict
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import accuracy_score, explained_variance_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib

from sklearn.linear_model import Lasso, LassoCV, LassoLarsCV
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score


import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
%matplotlib inline

os.chdir("C:/Users/212454979/JupyterProjects/Kaggle/HousePrices/")
dataset = pd.read_csv("train.csv", header=0, index_col=0)
# new_dataset = pd.read_csv("test.csv", header=0, index_col=0)
# randomize data
np.random.seed(0)
rd_ind = np.random.permutation(dataset.shape[0])
dataset = dataset.iloc[rd_ind]
medians = dataset.median()
medians.to_pickle('train_medians.pk')
# num_vars = [col for col in dataset.columns if dataset[col].dtypes=='int64']
# other_vars = [col for col in dataset.columns if col not in num_vars]
y = dataset['SalePrice'].apply(np.log)
X = dataset[[col for col in dataset.columns if col != 'SalePrice']]
X.MSSubClass = X.MSSubClass.astype('str')
types = X.dtypes.values
print("data types of variables:", np.unique(types))
# rank_vars = ['OverallQual', 'OverallCond'] #rank from 1-10
print(X.shape)
X.dtypes.value_counts()
num_df_raw = X.select_dtypes(exclude=['O'])
cat_df_raw = X.select_dtypes(include=['O'])
print("there are %d numerical variables and %d categorical variables" % (len(num_df_raw.columns), len(cat_df_raw.columns)))
# data quality check -- missing rate
num_summary = num_df_raw.describe()
missing_vars = num_summary.loc['count']/X.shape[0]
missing_vars = missing_vars[missing_vars<1]

# fill missing values with median values
for var in missing_vars.index:
    num_df_raw[var] = num_df_raw[var].fillna(num_df_raw[var].median())

num_summary[missing_vars.index].boxplot()
missing_vars
# pearson correlation
def pearson_calc(num_df, y, corr_threshold=0.05, pvalue_threshold=0.1):
    pearson_r = []
    # pearson_r['coeff'] = []
    # pearson_r['p_value'] = []
    for var in num_df.columns:
        pearson_r.append(pearsonr(num_df[var].values, y.values))
    pearson_r = np.array(pearson_r).round(3)
    # valid_vars = num_df.columns[filter_criteria]
    # valid_stats = pearson_r[filter_criteria]
    pearson_r = pd.DataFrame(dict(zip(num_df.columns, pearson_r)), index=['correff', 'p_value']).T
    filter_criteria = np.logical_and(np.abs(pearson_r['correff'])>=corr_threshold, pearson_r['p_value']<=pvalue_threshold)
    pearson_r['pass'] = filter_criteria
    print("there are %d out of total %d numerical variables have significant correlation with y" % 
          (pearson_r['pass'].sum(), pearson_r.shape[0]))
    pearson_r = pearson_r.sort_values('correff',ascending=False)
    return pearson_r
# grouping some levels and change types
cat_vars = ['GarageCars', 'Fireplaces', 'FullBath', 'HalfBath', 'BsmtFullBath', 'BedroomAbvGr',
                  'BsmtHalfBath', 'KitchenAbvGr', 'MoSold']
date_vars = ['YrSold', 'YearBuilt', 'YearRemodAdd','GarageYrBlt']

transform = False
num_df = num_df_raw.copy(deep=True)
num_df['GarageCars'] = np.where(num_df.GarageCars >=3, 3, num_df.GarageCars)
num_df['Fireplaces'] = np.where(num_df.Fireplaces >=2, 2, num_df.Fireplaces)
num_df['BsmtHalfBath'] = np.where(num_df.BsmtHalfBath >=1, 1, num_df.BsmtHalfBath)
num_df['KitchenAbvGr'] = np.where(num_df.KitchenAbvGr >=2, 2, num_df.KitchenAbvGr)
num_df['BedroomAbvGr'] = np.where(num_df.BedroomAbvGr >=5, 5, num_df.BedroomAbvGr)
# Create house' age to replace year of built
num_df['AgeBuilt'] = num_df.YrSold - num_df.YearBuilt
num_df['AgeBuilt'] = np.where(num_df.AgeBuilt <=0 , 0, num_df.AgeBuilt)
#     print(pearsonr(num_df['AgeBuilt'], y))
num_df['AgeRemodel'] = num_df.YrSold - num_df.YearRemodAdd
num_df['AgeRemodel'] = np.where(num_df.AgeRemodel <=0 , 0, num_df.AgeRemodel)
#     print(pearsonr(num_df['AgeRemodel'], y))

# calculate skewness
skew_cmp = {}
num_df_trans = pd.DataFrame()
for col in num_df.columns:
    if col not in cat_vars+date_vars:
        try:
            if num_df[col].min() <= 0:
                boxcox_trans = boxcox(num_df[col]+0.01)
            else:
                boxcox_trans = boxcox(num_df[col])
            diff = abs(skew(num_df[col])) - abs(skew(boxcox_trans[0]))
            skew_cmp[col] = [skew(num_df[col]), skew(boxcox_trans[0]), diff, boxcox_trans[1]]
            if diff>0 and transform:
                num_df_trans[col] = boxcox_trans[0]
            else:
#                 print("Don't transform %s" % col)
                num_df_trans[col] = num_df[col].values
#                 print("Pearson correlation: ", pearsonr(num_df_trans[col], y))
#                 print(pearsonr(num_df[col], y))
        except TypeError:
            pass
#                 print(col)
# only transfer those with change on skew - diff>0
skew_cmp = pd.DataFrame(skew_cmp).T
skew_cmp.columns = ['skew_before', 'skew_after', 'diff','lambda']
print(skew_cmp[skew_cmp['diff']<0])
# recalcuate the pearson correlation
raw_corr = pearson_calc(num_df, y, 0.1, 0.1)
trans_corr = pearson_calc(num_df_trans, y, 0.1, 0.1)
trans_corr.join(raw_corr, how='left', rsuffix='_raw')
# visualization by scatter plot and histograms
def plot_correlation(num_df, y):
    fig0 = plt.figure(figsize=(15, 10))
    row = 5; col = np.ceil((num_df.shape[1]+1)/row)
    fig = plt.figure(figsize=(15, 10))
    # print(row, col)
    pearson_r = pearson_calc(num_df, y, 0.1, 0.1)
    insig_vars = pearson_r[~pearson_r['pass']].index
    # y vs. x
    for i, var in enumerate(pearson_r.index):
        ax0 = fig0.add_subplot(row, col, i+1)
        if var in insig_vars:
            ax0.scatter(num_df[var], y, color='gray')
        else:
            ax0.scatter(num_df[var], y)
        ax0.set_title(pearson_r.loc[var, 'correff'])
        ax0.set_xlabel(var)

    # fig0.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.8, hspace=0.7)
    fig0.suptitle("y(x 1e5) vs. all numerical variables", fontsize=12)
    fig0.subplots_adjust(top=0.92, wspace = 0.3, hspace=0.7, left = 0.1, right = 0.95, bottom = 0.15)
    fig0.savefig('Pearson_Correlation.png')

    # x distribution
    for i, var in enumerate(pearson_r.index):
        ax = fig.add_subplot(row, col, i+1)
        if var in insig_vars:
            ax.hist(num_df[var], color='gray')
        else:
            ax.hist(num_df[var])
        ax.set_xlabel(var)

    ax2 = fig.add_subplot(row, col, i+2)
    ax2.hist(y, color='orange')
    ax2.set_xlabel(y.name)

    fig.suptitle("Blue: significant variables; Orange: y; Gray: non-significant variables", fontsize=12)
    fig.subplots_adjust(top=0.95, wspace = 0.3, hspace=0.5, left = 0.1, right = 0.95, bottom = 0.15)
    fig.savefig('Numerical_Variables_Distribution.png')
plot_correlation(num_df_trans, y)
# treat some int type variables as categorical ones
treat_rk = True
if treat_rk:
    for var in cat_vars:
        cat_df_raw[var] = num_df[var].astype(int).astype(str)
#         num_df = num_df.drop(rk_vars, axis=1)
    print("There are %d categorical variables after conversion" % cat_df_raw.shape[1])
    print("There are %d numerical variables after conversion" % num_df_trans.shape[1])
# fill missing with NA
cat_df_raw = cat_df_raw.fillna('NA')
# check the types
cat_df_raw[cat_vars].dtypes
# calculate missing rate
cat_summary = cat_df_raw.describe().T
pass_ratio = 0.95
high_freq_ratio = 0.95

cat_summary['missing_rate'] = 1-cat_summary['count']/cat_df_raw.shape[0]
cat_summary['missing_high'] = cat_summary['missing_rate']>pass_ratio
cat_summary['top_freq_high'] = cat_summary['freq']/cat_summary['count']>high_freq_ratio
# some variables with many levels
cat_summary[cat_summary['unique']>10]
# filter high missing rate variables and high freq variables
cat_summary[(cat_summary['top_freq_high']==0) & (cat_summary['missing_high']==0)].shape[0]
# correlation visualization by boxplot
row = 12; col = np.ceil((cat_df_raw.shape[1]+1)/row)
print(row, col)
fig = plt.figure(figsize=(16, 25))
for i, var in enumerate(sorted(cat_df_raw.columns)):
    ax = fig.add_subplot(row, col, i+1)
    _df = pd.concat([cat_df_raw[var], y.to_frame()], axis=1)
#     print(concat_df.head())
    _df.boxplot(ax=ax, by=var)
    ax.set_title(var)
    ax.set_xlabel('')
#     bp = concat_df.groupby(var).boxplot()
fig.subplots_adjust(top=0.95, wspace = 0.2, hspace=0.6, left = 0.1, right = 0.95, bottom = 0.15)
fig.savefig('Boxplot_Correlation.png')
import pickle as pk
# filter high missing or high freq variables
sig_vars_cat = cat_summary[(cat_summary['top_freq_high']==0) & (cat_summary['missing_high']==0)].index
trans_corr = pearson_calc(num_df_trans, y, 0.1, 0.1)
sig_vars_num = trans_corr[trans_corr['pass']].index

def filter_vars(cat_df_raw, num_df_trans, filter_cat=True, filter_num=True):
    global feature_names_raw, feature_names
    if filter_cat:
        cat_df_raw_ = cat_df_raw[[col for col in cat_df_raw.columns if col in sig_vars_cat]]
    else:
        cat_df_raw_ = cat_df_raw
    # make dummy variables
    cat_df = pd.get_dummies(cat_df_raw_, drop_first=False).reset_index(drop=True)
    if filter_num:
        num_df_trans = num_df_trans[[col for col in num_df_trans.columns if col in sig_vars_num]]
    print("categorical variables after selection: %d" % cat_df_raw_.shape[1])
    print("numerical variables after selection: %d" % num_df_trans.shape[1])
    all_df = pd.concat([num_df_trans, cat_df], axis=1)
    feature_names_raw = {'num': num_df_trans.columns.tolist(), 'cat': cat_df_raw_.columns.tolist()}
    feature_names = list(all_df.columns)
    print(all_df.shape)
    print(len(feature_names_raw['num']), len(feature_names_raw['cat']))
    return all_df, cat_df, num_df_trans, feature_names_raw

# save for future usage
all_df, cat_df, num_df_trans, feature_names_raw = filter_vars(cat_df_raw, num_df_trans)
all_df.to_pickle('train_processed.pk')
y.to_pickle('y_log.pk')
with open('feature_names_raw.pk', 'wb') as f:
    pk.dump(feature_names_raw, f)
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
def RMSE(actual, pred):
    error = actual.flatten() - pred.flatten()
    RMSE = np.sqrt(np.dot(error, error.T)/len(error)).round(6)
    return RMSE
def scale_choice(cat_df, num_df, y, scale_method, scale_y=True):
    cat_norm = cat_df.values
#     print(cat_norm.shape)
    if scale_method == 'default':
        num_norm = num_df.values
        all_norm = np.hstack((num_norm, cat_norm))
        y_norm = np.reshape(y, (-1, 1))
    else:
        scaler = norm_methods[scale_method]
        num_norm = scaler.fit_transform(num_df)
        if scale_y:
            y_norm = scaler.fit_transform(y.reshape(-1,1))
        else:
            y_norm = y.reshape(-1, 1)
        all_norm = np.hstack((num_norm, cat_norm))
    return num_norm, cat_norm, all_norm, y_norm
norm_methods = {'default': None, 'standard': StandardScaler(), 'minmax': MinMaxScaler(), 'robust':RobustScaler()}
# outlier detection
# PCA
decomp = PCA(n_components=10)
num_norm, cat_norm, all_norm, y_norm = scale_choice(cat_df, num_df_trans, y, 'standard')
print(all_norm.shape)
x_trans = decomp.fit_transform(all_norm)
print(decomp.explained_variance_ratio_.cumsum())
# PLS
decomp2 = PLSRegression(n_components=10, scale=True)
num_norm, cat_norm, all_norm, y_norm = scale_choice(cat_df, num_df_trans, y, 'default')
decomp2.fit(all_norm, y_norm)
# visualize the outliers
fig = plt.figure(figsize=(15, 10))
pc1 = 0; pc2 = 1; pc3 = 2
ax1 = fig.add_subplot(221)
ax1.scatter(x_trans[:, pc1], x_trans[:, pc2])
for i in range(all_norm.shape[0]):
    ax1.text(x_trans[i, pc1], x_trans[i, pc2], i)
ax1.set_title('PC1 vs. PC2')

ax2 = fig.add_subplot(222)
ax2.scatter(x_trans[:, pc1], x_trans[:, pc3])
for i in range(all_norm.shape[0]):
    ax2.text(x_trans[i, pc1], x_trans[i, pc3], i)
ax2.set_title('PC1 vs. PC3')

ax3 = fig.add_subplot(223)
ax3.scatter(decomp2.x_scores_[:, pc1], decomp2.x_scores_[:, pc2])
for i in range(all_norm.shape[0]):
    ax3.text(decomp2.x_scores_[i, pc1], decomp2.x_scores_[i, pc2], i)
ax3.set_title('PLS score 1 vs. score 2')

ax4 = fig.add_subplot(224)
ax4.scatter(decomp2.x_scores_[:, pc1], decomp2.y_scores_[:, pc1])
for i in range(all_norm.shape[0]):
    ax4.text(decomp2.x_scores_[i, pc1], decomp2.y_scores_[i, pc1], i)
ax4.set_title('PLS X score 1 vs. y score 1')
# all_df.iloc[[259, 548, 442, 1173, 875, 442]]
# outliers [259, 640]
# remove outliers -- those two points are influencing points, removing them reduced the RMSE
outliers_id = [259, 640]
print(all_df.shape)
print(num_df.shape, all_df.shape)
explained_var = pd.DataFrame({'num': [],  'all': [], 'cat': []})
# PLSRegression - scale selection has a big impact on the score

method_cm = {}
for method in norm_methods.keys():
    num_norm, cat_norm, all_norm, y_norm = scale_choice(method)
    print(method)
    explained_var = pd.DataFrame({'num': [],  'all': [], 'cat': []})
    for i in range(1, 16):
        if method == 'default':
            scale = True
        else:
            scale = False
        for prefix in explained_var.columns:
            pls = PLSRegression(n_components=i, scale=scale)
            pls.fit(eval('%s_norm' % prefix), y_norm)
            explained_var.loc[i, prefix]= pls.score(eval('%s_norm' % prefix), y_norm).round(3)
            del pls
    method_cm[method] = explained_var
for method in norm_methods.keys():
    print(method)
    if method == 'default':
        pass
    else:
        scaler = norm_methods[method]
        y_norm = scaler.fit_transform(y.reshape(-1,1))
        print(y_norm.mean(), y_norm.std())

# _center_scale_xy used by PLSRegression
# # center
# x_mean = X.mean(axis=0)
# X -= x_mean
# y_mean = Y.mean(axis=0)
# Y -= y_mean
# # scale
# if scale:
#     x_std = X.std(axis=0, ddof=1)
#     x_std[x_std == 0.0] = 1.0
#     X /= x_std
#     y_std = Y.std(axis=0, ddof=1)
#     y_std[y_std == 0.0] = 1.0
#     Y /= y_std
# else:
#     x_std = np.ones(X.shape[1])
#     y_std = np.ones(Y.shape[1])
# compare the results by different scaling methods
fig = plt.figure(figsize=(10, 8))
i = 1
for k, explained_var in method_cm.items():
    ax = fig.add_subplot(2,2,i)
    explained_var.plot(title=k, ylim=[0.7, 0.98],
                      grid=True, style=['o-', '*-', 'x-'], ax=ax
                      )
    i += 1
num_norm, cat_norm, all_norm, y_norm = scale_choice('default')
pls = PLSRegression(n_components=10, scale=True)
pls.fit(all_norm, y_norm)
pred = pls.predict(all_norm)
print("RMSE: %.3f" % RMSE(y_norm.flatten(), pred.flatten()))
# plt.plot(pls.x_scores_[:, 0], pls.y_scores_, 'o')
plt.plot(pls.x_scores_[:, 0], pls.y_scores_[:, 0], 'o')
# plt.plot(pls.x_scores_[:, 1], pls.y_scores_[:, 0], 'bo')
# plt.plot(pls.x_scores_[:, 2], pls.y_scores_[:, 0], 'ro')
for n_c in range(5):
    sorted_fid = np.argsort(np.abs(pls.x_weights_[:, n_c]))
    print(np.array(feature_names)[sorted_fid[-5:]])
outliers = np.where(pls.x_scores_[:,0]>10)[0]
outliers_df = dataset.iloc[outliers]
outliers_df['norm_y'] = outliers_df['SalePrice'].apply(np.log)
outliers_df['pred_y'] = pred[outliers].flatten()
outliers_df['error'] = outliers_df['norm_y'] - outliers_df['pred_y']

# LandContour (Bnk), YearBuilt (2008), Condition1 (Feedr)
# n_col = 0
# dataset.ix[dataset.GrLivArea>=4000, n_col:n_col+10]
plt.plot(pred[dataset.LandContour == 'Bnk'], y_norm[dataset.LandContour == 'Bnk'], 'o')
outliers_df[['1stFlrSF','TotalBsmtSF', 'GarageArea', 'GrLivArea', 
             'OverallQual', 'SaleCondition'] 
            + ['norm_y', 'pred_y', 'error']]
pls = PLSRegression(n_components=10, scale=True)
pls.fit(all_norm, y_norm)
# score = cross_val_score(pls, all_norm, y_norm, cv=4)
# print(score.round(3))
pred = cross_val_predict(pls, all_norm, y_norm, cv=4)
residuals = y_norm - pred
print("RMSE", RMSE(y_norm, pred).round(3))
print("R^2", explained_variance_score(y_norm, pred).round(3))
big_errors = np.where(np.abs(residuals)>=0.5)[0]
plt.boxplot(residuals, showmeans=True)
plt.title('Residual distribution')
y_norm.mean()
# cross validation
# cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
reg = RandomForestRegressor(n_estimators=10, random_state=0, criterion='mse')
pred = cross_val_predict(reg, all_df, y_norm, cv=5)
residuals = y_norm - pred
print("RMSE", RMSE(y_norm, pred).round(6))
print("R^2", explained_variance_score(y_norm, pred).round(6))
# train only by numerical variables
num_df_ = num_df.iloc[rd_ind]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.25)
print(X_test.shape, y_test.shape)
reg_num = RandomForestRegressor(n_estimators=10, random_state=0, criterion='mse')
reg_num.fit(X_train, y_train)
featureimp_num = dict(zip(X_train.columns, reg_num.feature_importances_.round(3)))
featureimp_num = pd.Series(featureimp_num).sort_values(ascending=False)
print("Train/test only by numerical variables...")
print("Train score: %.3f" % reg_num.score(X_train, y_train))
print("Test score: %.3f" % reg_num.score(X_test, y_test))
featureimp_num.to_frame('RF_importance').join(pearson_r).sort_values('correff', ascending=False)
# train only by categorical variables
X_train, X_test, y_train, y_test = train_test_split(cat_dummy_df, y, random_state=0, test_size=0.25)
reg_cat = RandomForestRegressor(n_estimators=100, random_state=0, criterion='mse')
reg_cat.fit(X_train, y_train)
print("Train/test only by categorical variables...")
print("Train score: %.3f" % reg_cat.score(X_train, y_train))
print("Test score: %.3f" % reg_cat.score(X_test, y_test))
featureimp_cat = dict(zip(X_train.columns, reg_cat.feature_importances_.round(3)))
featureimp_cat = pd.Series(featureimp_cat).sort_values(ascending=False)
featureimp_cat[:20]
def cv_rd_splitting(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.25)
#     print(X_test.shape, y_test.shape)
    reg = RandomForestRegressor(n_estimators=10, random_state=0, criterion='mse')
    reg.fit(X_train, y_train)
    featureimp = dict(zip(X.columns, reg.feature_importances_.round(3)))
    featureimp = pd.Series(featureimp).sort_values(ascending=False)
    train_score = RMSE(reg.predict(X_train), y_train.values)
    test_score = RMSE(reg.predict(X_test), y_test.values)
    return featureimp, train_score, test_score
# no lift after using dummy variables
reg = RandomForestRegressor(n_estimators=10, random_state=0, criterion='mse')
score = cross_val_score(reg, all_df, y, cv=4, scoring='neg_mean_squared_error')
score = np.sqrt(np.abs(score))
print("cv score: %.6f +/-%.6f" % (score.mean(), score.std()))

featureimp,train_score, test_score = cv_rd_splitting(all_df, y)
print("Train RMSE: %.6f" % train_score)
print("Test RMSE: %.6f" % test_score)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(range(len(featureimp)), featureimp.values.cumsum(), 'o', markersize=4)
ax.grid(True)
ax.set_title('Variable importance (cum.)')
featureimp[:30]
# join by index -- enable index params!
fig, axes = plt.subplots(2,1, sharex=True, sharey=False, figsize=(8, 8))
# fig.subplots_adjust(top=None, wspace = 0.5, hspace=0.5, left = None, right = None, bottom = None)
fig.tight_layout()
for i, var_type in enumerate(('num', 'cat')): 
    cm_imp = eval('featureimp_%s' % var_type).to_frame(name=var_type).merge(featureimp.to_frame(name='all'), how='inner', left_index=True, right_index=True)
#     cm_imp_cat = featureimp.to_frame('all').merge(featureimp_cat.to_frame(name='cat'), how='inner', left_index=True, right_index=True)
    cm_imp[var_type].fillna(0, inplace=True)
    cm_imp['diff'] = cm_imp['all'] - cm_imp[var_type]
    cm_imp.iloc[:20, :2].plot(kind='barh', ax=axes[i])
    # cm_imp_num.iloc[:, -1].plot(kind='line', ax=ax)
cm_results = []
max_features = 100
for n_var in range(1, max_features):
    selected_var_RF = featureimp[:n_var].index
    _, train_score, test_score = cv_rd_splitting(all_x_df[selected_var_RF], y)
    cm_results.append([train_score, test_score])
    if n_var%10 == 0:
        print(n_var)
#     print("Train score: %.3f" % reg.score(X_train[selected_var_RF], y_train))
#     print("Test score: %.3f" % reg.score(X_test[selected_var_RF], y_test))
fig = plt.figure()
ax = fig.add_subplot(111)
cm_results = np.array(cm_results)
ax.plot(range(1, max_features), cm_results[:, 0], 'o-', label='train_score')
ax.plot(range(1, max_features), cm_results[:, 1], 'o-', label='test_score')
# ax.set_label(['train_score', 'test_score'])
ax.grid()
ax.legend()
# Normalization after processing missing and change data type
method = 'default'
num_norm, cat_norm, all_norm, y_norm = scale_choice(cat_df, num_df_trans, y, method)
print(all_norm.shape, y_norm.shape)
remove_outliers = True
if remove_outliers:
    all_norm_ = np.delete(all_norm, outliers_id, 0)
    y_norm_ = np.delete(y_norm, outliers_id, 0)
else:
    all_norm_ = all_norm
    y_norm_ = y_norm
print(all_norm_.shape, y_norm_.shape)
# fit into LassoCV
def lasso(X_train, y_train, X_test, y_test, scale):
    if scale == 'default':
        scale = True
    else:
        scale = False
    model = LassoCV(cv=4, normalize= scale, max_iter=100
#                        , random_state=0
                       )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    residuals = y_test.flatten() - pred.flatten()
    rmse = RMSE(y_test, pred).round(6)
    r2 = explained_variance_score(y_test, pred).round(6)
#     print("RMSE: ", rmse)
#     print("R^2: ", r2)
    big_errors = np.where(np.abs(residuals)>=1)[0]
#     print(pred[big_errors], residuals[big_errors])
    return residuals, pred, rmse, r2, model
    
X_train, X_test, y_train, y_test = train_test_split(all_norm_, y_norm_, test_size=0.2, random_state=1)
residuals, pred, rmse, r2, model = lasso(X_train, y_train, X_test, y_test, method)
sorted_features = dict(zip(feature_names, model.coef_.round(6)))
sorted_features = sorted(sorted_features.items(), key=lambda x: x[1])
print(rmse, r2)

fig = plt.figure(figsize=(6, 8))
ax1 = fig.add_subplot(311)
ax1.boxplot(residuals, showmeans=True)
ax1.set_title('Residual distribution')
ax2 = fig.add_subplot(312)
ax2.plot(pred, y_test, 'o', [y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '-')
ax2.grid(True)
ax2.set_title('Actual (y) vs. Prediction (x)')
ax3 = fig.add_subplot(313)
ax3.bar(np.arange(len(feature_names)), pd.DataFrame(sorted_features).iloc[:,1])
ax3.grid(True)
plt.tight_layout()

feature_names = np.array(feature_names)

cm_results = []
for threshold in np.arange(0.00005, 0.01, 0.0005):
    selected_features = np.where(abs(model.coef_)>=threshold)[0]
    sig_features = dict(zip(feature_names[selected_features], model.coef_[selected_features].round(6)))
    sig_features = sorted(sig_features.items(), key=lambda x: x[1])
    cm_results.append({len(selected_features): lasso(X_train[:, selected_features], y_train,
                                                    X_test[:, selected_features], y_test, method)[2]})
cm_results[:3]
all_residuals = y_norm.ravel() - model.predict(all_norm).ravel()
for o in outliers_id:
    try:
        print(all_residuals[o])
    except IndexError:
        pass
from sklearn.model_selection import GridSearchCV
# select best number of features
model_sub = LassoCV(cv=5, normalize= True, max_iter=100, n_alphas=100)
model_sub.fit(all_norm_, y_norm_)
threshold = 0.00001
# search for the best threshold
results = []
thres_spaces = np.linspace(1e-6, 1e-4, 10)
for thre in thres_spaces:
    selected_features_ids = np.where(abs(model_sub.coef_)>=thre)[0]
    # refit the model with selected features
    model_ = LassoCV(cv=5, normalize= True, max_iter=100, n_alphas=100)
    model_.fit(all_norm_[:, selected_features_ids], y_norm_)
#     pred = model_sub.predict(X_test[:, selected_features_ids])
#     residuals = y_test.flatten() - pred.flatten()
    rmse = np.sqrt(model_.mse_path_[-1])
#     rmse = RMSE(y_test, pred).round(6)
#     r2 = explained_variance_score(y_test, pred).round(6)
    results.append([selected_features_ids.shape[0], rmse.mean(), rmse.std(), model_.alpha_])
results = np.array(results)
fig, ax1 = plt.subplots()
ax1.plot(thres_spaces, results[:, 1], 'go-', label='rmse')
ax1.set_ylabel('rmse')
ax1.grid(True)
ax1.set_xlabel('weights threshold')
ax1.legend(bbox_to_anchor=(1.08, 1), loc=2, borderaxespad=0)
ax2 = ax1.twinx()
ax2.set_ylabel('std')
ax2.plot(thres_spaces, results[:, 2], 'ro-', label='std')
for i in range(thres_spaces.shape[0]):
    ax1.text(thres_spaces[i], results[i, 1], np.int(results[i, 0]))
ax2.legend(loc='best')
ax2.set_ylim([0.005, 0.015])
ax2.legend(bbox_to_anchor=(1.08, 0.92), loc=2, borderaxespad=0)

best_id = np.argmin(results[:, 1])
best_thre = thres_spaces[best_id]
best_alpha = results[best_id, -1]
print(results[best_id])
# train with whole training dataset without cv
best_features_ids = np.where(abs(model_sub.coef_)>=best_thre)[0]
model_best = Lasso(alpha=best_alpha, normalize=True)
model_best.fit(X_train[:, best_features_ids], y_train)
pred = model_best.predict(X_test[:, best_features_ids])
residuals = y_test.flatten() - pred.flatten()
rmse = RMSE(y_test, pred).round(6)
r2 = explained_variance_score(y_test, pred).round(6)
print("RMSE: ", rmse)
print("R^2: ", r2)
# print("alpha: ", model.alpha_)
print(X_train[:, best_features_ids].shape)
# train with full dataset
print(all_norm_.shape, y_norm_.shape, len(best_features_ids))
model_best = Lasso(alpha=best_alpha, normalize=True)
model_best.fit(all_norm_[:, best_features_ids], y_norm_)
# save model
best_features_names = feature_names[best_features_ids]
joblib.dump(best_features_names, 'best_features.pk')
joblib.dump(model_best, 'lasso.pkl')
# select coefficient >= 0.005 to fit randomforest regressor
reg = RandomForestRegressor(n_estimators=10, random_state=0, criterion='mse')
reg.fit(X_train[:, best_features_ids], y_train)
pred = reg.predict(X_test[:, best_features_ids])
residuals = y_test.flatten() - pred.flatten()
rmse = RMSE(y_test, pred).round(6)
r2 = explained_variance_score(y_test, pred).round(6)
print(rmse, r2)
reg = KNeighborsRegressor(n_neighbors=100, weights='uniform', algorithm='auto', leaf_size=30, p=1, metric='minkowski')
# standardize data
num_norm, cat_norm, all_norm, y_norm = scale_choice(cat_df, num_df_trans, y, 'robust', scale_y=False)
all_norm_ = np.delete(all_norm, outliers_id, 0)
y_norm_ = np.delete(y_norm, outliers_id, 0)
print(all_norm_.shape, y_norm_.shape)
# reg.fit(all_norm_, y_norm_)
scores = cross_val_score(reg, all_norm_[:, best_features_ids], y_norm_, cv=5, scoring='neg_mean_squared_error')
rmse = np.sqrt(np.abs(scores))
print(rmse)
alpha_rg = {'learning_rate': np.linspace(0.1, 0.3, 10)}
reg = GradientBoostingRegressor(random_state=0)
# cross validation
grid = GridSearchCV(reg, param_grid=alpha_rg, scoring='neg_mean_squared_error', cv=5)
grid.fit(all_norm_[:, best_features_ids], y_norm_)
best_learning_rate
mean_scores = np.sqrt(np.abs(grid.cv_results_['mean_test_score']))
best_learning_rate = alpha_rg['learning_rate'][np.argmin(mean_scores)]
reg = GradientBoostingRegressor(learning_rate=best_learning_rate, min_samples_leaf=1, n_estimators=100, random_state=0)
score_1 = cross_val_score(reg, all_norm[:, selected_features_ids], y_norm, scoring='neg_mean_squared_error', cv=5)
print("score using features selected by Lasso: \n", 
      np.sqrt(np.abs(score_1)).mean().round(6),
     np.sqrt(np.abs(score_1)).std().round(6)) ## much better
# score_2 = cross_val_score(reg, all_norm, y_norm, scoring='neg_mean_squared_error', cv=5)
# print("score using all the features", np.sqrt(np.abs(score_2)).mean())
# test by full model
reg.fit(X_train[:, selected_features_ids], y_train)
rmse = RMSE(reg.predict(X_test[:, selected_features_ids]), y_test)
print(rmse)
pred = reg.predict(X_test[:, selected_features_ids])
residuals = y_test - pred
fig = plt.figure(figsize=(6, 4))
# ax1 = fig.add_subplot(111)
# ax1.boxplot(residuals, showmeans=True)
# ax1.set_title('Residual distribution')
ax2 = fig.add_subplot(111)
ax2.plot(pred, y_test, 'o', [y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '-')
ax2.grid(True)
ax2.set_title('Actual (y) vs. Prediction (x)')

cat_vars = ['GarageCars', 'Fireplaces', 'FullBath', 'HalfBath', 'BsmtFullBath', 'BedroomAbvGr',
                  'BsmtHalfBath', 'KitchenAbvGr', 'MoSold'] ## those variables will be converted to categorical types
date_vars = ['YrSold', 'YearBuilt', 'YearRemodAdd','GarageYrBlt']
def process_vars(num_df):
    num_df['GarageCars'] = np.where(num_df.GarageCars >=3, 3, num_df.GarageCars)
    num_df['Fireplaces'] = np.where(num_df.Fireplaces >=2, 2, num_df.Fireplaces)
    num_df['BsmtHalfBath'] = np.where(num_df.BsmtHalfBath >=1, 1, num_df.BsmtHalfBath)
    num_df['KitchenAbvGr'] = np.where(num_df.KitchenAbvGr >=2, 2, num_df.KitchenAbvGr)
    num_df['BedroomAbvGr'] = np.where(num_df.BedroomAbvGr >=5, 5, num_df.BedroomAbvGr)
    # Create house' age to replace year of built
    num_df['AgeBuilt'] = num_df.YrSold - num_df.YearBuilt
    num_df['AgeBuilt'] = np.where(num_df.AgeBuilt <=0 , 0, num_df.AgeBuilt)
    #     print(pearsonr(num_df['AgeBuilt'], y))
    num_df['AgeRemodel'] = num_df.YrSold - num_df.YearRemodAdd
    num_df['AgeRemodel'] = np.where(num_df.AgeRemodel <=0 , 0, num_df.AgeRemodel)
    #     print(pearsonr(num_df['AgeRemodel'], y))
    return num_df
# prepare new samples
# load new samples
new_dataset_raw = pd.read_csv('test.csv', header=0, index_col=0)
print("Raw test dataset shape: ", new_dataset_raw.shape)

# load subset variable names
with open('feature_names_raw.pk', 'rb') as f:
    feature_names_raw = pk.load(f)
    subset = []
    for v in feature_names_raw.values():
        subset.extend(v)
    print("there are %d features to subset" % len(subset))

# load best features
best_features_names = joblib.load('best_features.pk')

# load median of train set
medians = pd.read_pickle('train_medians.pk')

# fill missing values
new_dataset_raw.MSSubClass = new_dataset_raw.MSSubClass.astype(str)
for var in cat_vars:
    if var in new_dataset_raw.columns:
        new_dataset_raw[var] = new_dataset_raw[var].fillna(medians[var])
# preprocess some numerical variables
new_dataset = process_vars(new_dataset_raw)
print(new_dataset.shape)
# subset
new_dataset = new_dataset[subset]
print("Test dataset shape after subset: ", new_dataset.shape)

# fill missing values with median
for k, v in feature_names_raw.items():
    if k == 'num':
        for var in v:
            if new_dataset[var].isnull().sum()>0:
                new_dataset[var] = new_dataset[var].fillna(dataset[var].median()) 
    else:
        for var in v:
            if var in cat_vars:
                new_dataset[var] = new_dataset[var].astype(int).astype(str)
            new_dataset[var] = new_dataset[var].fillna('NA')
            
# make dummy variable
new_df = pd.get_dummies(new_dataset)
print(new_df.shape)
new_dataset.AgeBuilt.describe()
# dataset.GarageQual.value_counts()
# keep only matched features
kept_features = [f for f in new_df.columns if f in best_features_names]
print(len(kept_features))
out_features = [f for f in best_features_names if f not in kept_features]
print(out_features)
for f in out_features:
    new_df[f] = 0
new_df = new_df[best_features_names]
print(new_df.shape)
assert np.equal(new_df.columns.tolist(), best_features_names)
import datetime
# prediction
output = False
def output_prediction(new_df, output=False):
    pred_model = joblib.load('lasso.pkl')
    pred_new = np.exp(pred_model.predict(new_df))
    plt.hist(pred_new)
    # output
    print(np.median(pred_new))
    if output:
        datestamp = datetime.datetime.strftime(datetime.datetime.now(), '%m%d-%H')
        pd.DataFrame({'SalePrice':pred_new}, index=new_dataset.index).to_csv("submission_%s.csv" % datestamp, 
                                                                             index_label='Id')
    return pred_new
pred_new = output_prediction(new_df)
