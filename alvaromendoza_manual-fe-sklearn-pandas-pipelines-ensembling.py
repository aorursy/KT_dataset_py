# Fix random seeds for reproducibility

GLOBAL_SEED = 87216

from numpy.random import seed
seed(GLOBAL_SEED)

from tensorflow import set_random_seed
set_random_seed(GLOBAL_SEED)

import os
os.environ['PYTHONHASHSEED'] = '0'

import random as rn
rn.seed(GLOBAL_SEED)
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns; sns.set()

from pandas.api.types import CategoricalDtype

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.linear_model import Lasso, LassoCV
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

from xgboost import DMatrix
from xgboost import XGBRegressor
from xgboost import cv

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l1, l2, l1_l2
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

pd.options.display.max_columns = None
pd.options.display.max_rows = None

plt.rcParams['figure.figsize'] = [15, 5]
# Load train and test data
# DataFrames train and test are to be transformed in place,
# and ames I will keep mostly intact for future reference and EDA.
raw_data_path = r'../input/house-prices-advanced-regression-techniques'
metadata_path = r'../input/house-prices-metadata'
ames_dtypes = {'MSSubClass': 'str'}
ames = pd.read_csv(os.path.join(raw_data_path, 'train.csv'), dtype=ames_dtypes)
train = ames.copy()
target = train['SalePrice'].copy()
train.drop('SalePrice', axis=1, inplace=True)
test = pd.read_csv(os.path.join(raw_data_path, 'test.csv'), dtype=ames_dtypes)

# Load metadata
ordinal_features_recoding = pd.read_excel(os.path.join(metadata_path, 
                                                       'ordinal_features_recoding.xlsx'))
feature_description=pd.read_excel(os.path.join(metadata_path,'feature_description.xlsx'))
feature_categories=pd.read_excel(os.path.join(metadata_path,'feature_categories.xlsx'))
print('Train shape:', train.shape)
print('Test shape:', test.shape)
# Useful lists of feature names
NUMERIC_FEATURE_NAMES = ['LotFrontage', 'LotArea', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea',
                         'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF',
                         '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath',
                         'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',
                         'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF',
                         'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',
                         'MiscVal', 'MoSold', 'YrSold']

ORDINAL_FEATURE_NAMES = ['LotShape', 'Utilities', 'LandSlope', 'OverallQual', 'OverallCond',
                         'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
                         'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'Electrical', 'KitchenQual',
                         'Functional', 'FireplaceQu', 'GarageFinish', 'GarageQual', 'GarageCond',
                         'PavedDrive', 'PoolQC', 'Fence']

NOMINAL_FEATURE_NAMES = ['Id', 'MSSubClass', 'MSZoning', 'Street', 'Alley', 'LandContour',
                         'LotConfig', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
                         'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd',
                         'MasVnrType', 'Foundation', 'Heating', 'CentralAir', 'GarageType',
                         'MiscFeature', 'SaleType', 'SaleCondition']

FEATURES_WITH_MEANINGFUL_NAN = ['Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
                                'BsmtFinType2', 'FireplaceQu', 'GarageType', 'GarageFinish',
                                'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature']

# Check if all feature names correspond with source data
assert set(train.columns) - set(NUMERIC_FEATURE_NAMES) - \
    set(ORDINAL_FEATURE_NAMES) - set(NOMINAL_FEATURE_NAMES) == set()
def add_missing_0_to_mssubclass(df):
    """Zeros in 020-090 get cut off. This function prepends them back."""
    
    df['MSSubClass'] = df['MSSubClass'].apply(
        lambda x: '0' + str(x) if len(str(x)) == 2 else x)


add_missing_0_to_mssubclass(ames)
add_missing_0_to_mssubclass(train)
add_missing_0_to_mssubclass(test)
def replace_meaningful_nan(df, feature_names):
    """Replaces nan where 'NA' is an existing category."""
    df.loc[:, feature_names] = df.loc[:, feature_names].fillna('Missing')


replace_meaningful_nan(ames, FEATURES_WITH_MEANINGFUL_NAN)
train.head()
test.head()
ordinal_features_recoding.head()
feature_description.head()
feature_categories.head()
fig, ax = plt.subplots(2, 1, figsize=[15, 8])
sns.distplot(ames['SalePrice'], ax=ax[0])
sns.distplot(np.log(ames['SalePrice']), ax=ax[1], axlabel='Log SalePrice')
fig, ax = plt.subplots(1, 2)
stats.probplot(ames['SalePrice'], plot=ax[0])
stats.probplot(ames['SalePrice'].apply(np.log), plot=ax[1])
[a.get_lines()[0].set_markerfacecolor('C0') for a in ax]
ax[0].set_title('SalePrice')
ax[1].set_title('Log SalePrice')
plt.show()
fig = plt.figure(figsize=[15, 15])
feature_list = sorted(NUMERIC_FEATURE_NAMES)
num_subplots = len(feature_list)
ncols = 5
nrows = num_subplots // ncols + 1
for n, feature in enumerate(feature_list):
    ax = fig.add_subplot(nrows, ncols, n+1)
    sns.distplot(ames[feature].dropna(), ax=ax, bins=10,
                 kde=False, hist_kws={'alpha': 1, 'edgecolor': 'white'})
fig.suptitle('Distribution plots of numeric features', fontsize=18)
fig.tight_layout(rect=[0, 0.03, 1, 0.96])
fig = plt.figure(figsize=[15, 15])
feature_list = sorted(NUMERIC_FEATURE_NAMES)
num_subplots = len(feature_list)
ncols = 5
nrows = num_subplots // ncols + 1
for n, feature in enumerate(feature_list):
    ax = fig.add_subplot(nrows, ncols, n+1)
    ax.scatter(x=ames[feature], y=ames['SalePrice'],
               edgecolor='white', linewidth=0.4,)
    plt.xlabel(feature)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda y, pos: '%.0f' % (y * 1e-3)))
fig.suptitle('Scatter plots of numeric features vs SalePrice in 000 USD', fontsize=18)
fig.tight_layout(rect=[0, 0.03, 1, 0.96])
fig = plt.figure(figsize=[15, 20])
feature_list = sorted(ORDINAL_FEATURE_NAMES)
num_subplots = len(feature_list)
ncols = 4
nrows = num_subplots // ncols + 1
for n, feature in enumerate(feature_list):
    ax = fig.add_subplot(nrows, ncols, n+1)
    if ames[feature].dtype == np.object:
        order = ordinal_features_recoding.loc[ordinal_features_recoding['Name'] == feature, 'Code']
    else: 
        order = None
    sns.countplot(x=feature, data=ames, color='darkcyan', order=order, ax=ax)
    ax.set_ylabel('')
fig.suptitle('Count plots of ordinal features', fontsize=18)
fig.tight_layout(rect=[0, 0.03, 1, 0.96])
fig = plt.figure(figsize=[15, 15])
feature_list = sorted(ORDINAL_FEATURE_NAMES)
num_subplots = len(feature_list)
ncols = 5
nrows = num_subplots // ncols + 1
for n, feature in enumerate(feature_list):
    ax = fig.add_subplot(nrows, ncols, n+1)
    if ames[feature].dtype == np.object:
        order = ordinal_features_recoding.loc[ordinal_features_recoding['Name']
                                              == feature, 'Code']
    else:
        order = None
    sns.stripplot(x=ames[feature], y=ames['SalePrice'], color='darkcyan',
                  edgecolor='whitesmoke', linewidth=0.4, size=7, jitter=True, order=order, ax=ax)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda y, pos: '%.0f' % (y * 1e-3)))
    plt.ylabel('')
fig.suptitle('Stripplots of ordinal features vs SalePrice in 000 USD', fontsize=18)
fig.tight_layout(rect=[0, 0.03, 1, 0.96])
fig = plt.figure(figsize=[15, 20])
feature_list = sorted(ORDINAL_FEATURE_NAMES)
num_subplots = len(feature_list)
ncols = 4
nrows = num_subplots // ncols + 1
for n, feature in enumerate(feature_list):
    ax = fig.add_subplot(nrows, ncols, n+1)
    if ames[feature].dtype == np.object:
        order = ordinal_features_recoding.loc[ordinal_features_recoding['Name']
                                              == feature, 'Code']
    else:
        order = None
    sns.boxplot(x=ames[feature], y=ames['SalePrice'], color='darkcyan', order=order, ax=ax)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda y, pos: '%.0f' % (y * 1e-3)))
    plt.ylabel('')
fig.suptitle('Boxplots of ordinal features vs SalePrice in 000 USD', fontsize=18)
fig.tight_layout(rect=[0, 0.03, 1, 0.96])
fig = plt.figure(figsize=[15, 20])
feature_list = sorted(NOMINAL_FEATURE_NAMES)
feature_list.remove('Id')
num_subplots = len(feature_list)
ncols = 4
nrows = num_subplots // ncols + 1
for n, feature in enumerate(feature_list):
    ax = fig.add_subplot(nrows, ncols, n+1)
    sns.countplot(x=feature, data=ames, color='tomato', ax=ax)
    plt.xticks(rotation=90)
    ax.set_ylabel('')
fig.suptitle('Count plots of nominal features', fontsize=18)
fig.tight_layout(rect=[0, 0.03, 1, 0.96])
fig = plt.figure(figsize=[15, 20])
feature_list = sorted(NOMINAL_FEATURE_NAMES)
feature_list.remove('Id')
num_subplots = len(feature_list)
ncols = 4
nrows = num_subplots // ncols + 1
for n, feature in enumerate(feature_list):
    ax = fig.add_subplot(nrows, ncols, n+1)
    sns.stripplot(x=ames[feature], y=ames['SalePrice'], color='tomato',
                  edgecolor='whitesmoke', linewidth=0.4, size=7, jitter=True, ax=ax)
    plt.ylabel('')
    plt.xticks(rotation=90)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda y, pos: '%.0f' % (y * 1e-3)))
fig.suptitle('Stripplots of nominal features vs SalePrice in 000 USD', fontsize=18)
fig.tight_layout(rect=[0, 0.03, 1, 0.96])
fig = plt.figure(figsize=[15, 25])
feature_list = sorted(NOMINAL_FEATURE_NAMES)
feature_list.remove('Id')
num_subplots = len(feature_list)
ncols = 4
nrows = num_subplots // ncols + 1
for n, feature in enumerate(feature_list):
    ax = fig.add_subplot(nrows, ncols, n+1)
    sns.boxplot(x=ames[feature], y=ames['SalePrice'], color='tomato', ax=ax)
    plt.ylabel('')
    plt.xticks(rotation=90)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda y, pos: '%.0f' % (y * 1e-3)))
fig.suptitle('Stripplots of nominal features vs SalePrice in 000 USD', fontsize=18)
fig.tight_layout(rect=[0, 0.03, 1, 0.96])
# Overall correlation matrix
corr_matrix_pearson = ames.corr()
plt.figure(figsize=(11, 9))
sns.heatmap(corr_matrix_pearson, square=True)
plt.show()
# Features most correlated with SalePrice
top_correlated = corr_matrix_pearson.nlargest(10, 'SalePrice')[
    'SalePrice'].index
corr_matrix_pearson_top_correlated = corr_matrix_pearson.loc[top_correlated, top_correlated]
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix_pearson_top_correlated, cbar=True,
            annot=True, square=True, fmt='.2f', annot_kws={'size': 12})
plt.show()
def inspect_feature(feature, get_text=True, get_plots=True):
    """Prints general info on the selected feature (name, type, desctiprion) from ames,
    as well as results of describe for numeric features and results of value_counts 
    for ordinal and nominal features. Generates separate sets of discriptive plots 
    for numeric (distplot, scatterplot vs SalePrice) and other features (countplot, 
    barplot of median SalePrice for each category, stripplots vs SalePrice and 
    log SalePrice).
    """

    assert feature in ames.columns, \
    'feature must be name of an existing column in ames DataFrame!'
    
    f_type = feature_description.loc[feature_description['Name']
                                      == feature, 'Type2'].values[0]
    f_description = feature_description.loc[feature_description['Name']
                                             == feature, 'Description'].values[0]
    if get_text:

        # Print general info
        print('Name:', feature)
        print('Type:', f_type)
        print('Description:', f_description, '\n')

        # Print describe() or value_counts()
        if f_type == 'Numeric':
            f_stats = ames[feature].describe()
            f_stats['nan'] = ames[feature].isnull().sum()
            print('Stats:\n', pd.DataFrame(f_stats), sep='')
        else:
            val_counts = pd.DataFrame(ames[feature].value_counts(dropna=False))
            val_counts.rename({feature: 'Count'}, axis=1, inplace=True)
            val_counts['Fraction'] = val_counts['Count'] / \
                val_counts['Count'].sum()
            val_counts = pd.merge(val_counts, 
                                  (feature_categories
                                   .loc[feature_categories['Name'] == feature, :]
                                   .astype(str)),
                                  how='left', 
                                  left_on=val_counts.index.astype(str),
                                  right_on='Code')
            val_counts = val_counts[['Code', 'Category', 'Count', 'Fraction']]
            print('Value counts:\n', val_counts, sep='')

    if f_type == 'Numeric' and ames[feature].nunique() <= 10:
        f_type = 'Pseudo Ordinal'

    if get_plots:

        # Variables for use in plotting
        formatter000 = ticker.FuncFormatter(lambda y, pos: '%.0f' % (y * 1e-3))
        log_saleprice = ames['SalePrice'].apply(np.log)
        if ames[feature].dtype == np.object:
            if f_type == 'Ordinal':
                order = ordinal_features_recoding.loc[ordinal_features_recoding['Name']
                                                      == feature, 'Code']
            if f_type == 'Nominal':
                order = feature_categories.loc[feature_categories['Name'] == feature, 'Code']
        else:
            order = None

        # Generete plots for numeric features
        if f_type == 'Numeric':
            from warnings import filterwarnings
            filterwarnings("ignore", category=UserWarning)
            rc_dict = {'axes.titlesize': 17, 'axes.labelsize': 14,
                       'xtick.labelsize': 14, 'ytick.labelsize': 14}
            with plt.rc_context(rc=rc_dict):
                fig = plt.figure(figsize=[15, 4])
                layout = (1, 2)
                ax1 = plt.subplot2grid(layout, (0, 0), rowspan=1, colspan=1)
                ax2 = plt.subplot2grid(layout, (0, 1), rowspan=1, colspan=1)
                # Distplot
                sns.distplot(ames[feature].dropna(), ax=ax1, bins=10,
                             kde=False, hist_kws={'alpha': 1, 'edgecolor': 'white'})
                ax1.set_title('Distplot')
                # Scatter plot vs SalePrice
                ax2.scatter(x=ames[feature], y=ames['SalePrice'], s=70,
                            edgecolor='white', linewidth=0.5,)
                ax2.set_xlabel(feature)
                ax2.set_ylabel('SalePrice, 000 USD')
                ax2.yaxis.set_major_formatter(formatter000)
                ax2.set_title('Scatter plot vs SalePrice')
                fig.tight_layout()

        # Generete plots for ordinal and nominal features
        else:
            rc_dict = {'axes.titlesize': 17, 'axes.labelsize': 14,
                       'xtick.labelsize': 13, 'ytick.labelsize': 14}
            with plt.rc_context(rc=rc_dict):
                fig = plt.figure(figsize=[15, 10])
                layout = (2, 2)
                ax1 = plt.subplot2grid(layout, (0, 0), rowspan=1, colspan=1)
                ax2 = plt.subplot2grid(layout, (0, 1), rowspan=1, colspan=1)
                ax3 = plt.subplot2grid(layout, (1, 0), rowspan=1, colspan=1)
                ax4 = plt.subplot2grid(layout, (1, 1), rowspan=1, colspan=1)
                # Countplot
                sns.countplot(x=feature, data=ames,
                              color='darkcyan', order=order, ax=ax1)
                ax1.set_title('Countplot')
                # Barplot of median SalePrice for each category
                grouped = ames.groupby(feature)['SalePrice'].median()
                sns.barplot(x=grouped.index, y=grouped,
                            color='darkcyan', order=order, ax=ax2)
                ax2.yaxis.set_major_formatter(formatter000)
                ax2.set_title('Median SalePrice for each category')
                # Stripplot vs SalePrice
                sns.stripplot(x=ames[feature], y=ames['SalePrice'], color='darkcyan',
                              edgecolor='white', linewidth=0.5, size=9, jitter=True,
                              order=order, ax=ax3)
                ax3.yaxis.set_major_formatter(formatter000)
                ax3.set_title('Scatter plot vs SalePrice')
                # Stripplot plot vs log SalePrice
                sns.stripplot(x=ames[feature], y=log_saleprice, color='darkcyan',
                              edgecolor='white', linewidth=0.5, size=9, jitter=True,
                              order=order, ax=ax4)
                ax4.set_title('Scatter plot vs log SalePrice')
                for ax in fig.axes:
                    for tick in ax.get_xticklabels():
                        tick.set_rotation(90)
                    ax.set_xlabel('')
                    ax.set_ylabel('')
                fig.tight_layout()
def get_nan_counts(df):
    """Generates sorted nan counts and percentages for columns in a DataFrame."""
    nan_counts = pd.DataFrame(df.isnull().sum()).reset_index()
    nan_counts.columns = ['feature', 'is nan']
    nan_counts['is nan ratio'] = nan_counts['is nan'] / df.shape[0]
    nan_counts = nan_counts.sort_values('is nan ratio', ascending=False)
    return nan_counts
    

def association_test(X, y, sort_column=2):
    """Creates a DataFrame with measures of association between
    features in X, and y. Measures include 'Pearson's r', 
    'Spearman's rho', 'Root R^2', and 'p-value of F'.
    """
    # Regression
    output_regr = []
    X = pd.DataFrame(X)
    for feature in X.columns:
        if X[feature].dtype == np.object:
            dm = pd.get_dummies(X[feature])
        else:
            dm = X[feature]
        dm = sm.add_constant(dm)
        result = sm.OLS(y, dm.astype(float), missing='drop').fit()
        output_regr.append({'Feature': feature, 'Root R^2': np.sqrt(
            result.rsquared), 'p-value of F': result.f_pvalue})
    output_regr = pd.DataFrame(output_regr).set_index('Feature')
    output_regr.index.name = None
    
    # Correlation
    X = X.select_dtypes(exclude=np.object)
    pearson = X.apply(lambda col: col.corr(y, method='pearson'))
    spearman = X.apply(lambda col: col.corr(y, method='spearman'))
    output_correl = pd.concat([pearson, spearman], axis=1)
    output_correl.columns = ["Pearson's r", "Spearman's rho"]
    
    # Combined output
    sort = True if sort_column == 'index' else False
    output = pd.concat([output_correl, output_regr], sort=sort, axis=1)
    if sort_column != 'index':
        output = output.sort_values(output.columns[sort_column], ascending=False)
    return output
# Preliminaty regression modelling shows that this range contains two egregious outliers
outliers_index = train.loc[(train['GrLivArea'] > 4000)
                           & (target < 300000), :].index
train.drop(outliers_index, inplace=True)
target.drop(outliers_index, inplace=True)

# Id feature is useless without geolocation
train.drop('Id', axis=1, inplace=True)
test.drop('Id', axis=1, inplace=True)

# Log-transform SalePrice in order to make its distribution closer to normal
target = target.apply(np.log)
get_nan_counts(train).head(20)
get_nan_counts(test).head(20)
replace_meaningful_nan(train, FEATURES_WITH_MEANINGFUL_NAN)
get_nan_counts(train).head(10)
replace_meaningful_nan(test, FEATURES_WITH_MEANINGFUL_NAN)
get_nan_counts(test).head(20)
def impute_nan_manually(df):
    """Manual NaN imputation that does not cause data leakage.
    This must be done before recoding ordinal features.
    """
    # Correct typo
    test['GarageYrBlt'].replace(2207, 2007, inplace=True)
    # Replace wrong missing garage with nan
    for col in ['GarageQual', 'GarageCond']:
        wrong_missing_garage_filter = (
            df['GarageType'] != 'Missing') & (df[col] == 'Missing')
        df.loc[wrong_missing_garage_filter, col] = np.nan
    # Set year to 1900 for missing garages
    missing_garage_filter = df['GarageType'] == 'Missing'
    df.loc[missing_garage_filter, 'GarageYrBlt'] = 1900
    # Impute nan for MasVnr
    both_vnr_nan_filter = df['MasVnrType'].isnull() & df['MasVnrArea'].isnull()
    vnr_type_nan_filter = df['MasVnrType'].isnull() & (df['MasVnrArea'] > 0)
    df.loc[both_vnr_nan_filter, 'MasVnrType'] = 'None'
    df.loc[both_vnr_nan_filter, 'MasVnrArea'] = 0
    df.loc[vnr_type_nan_filter, 'MasVnrType'] = 'BrkFace'
    # Set nan to 0 for missing basements
    missing_bsmt_filter = (df['BsmtQual'] == 'Missing') & (
        df['BsmtCond'] == 'Missing')
    df.loc[missing_bsmt_filter, ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
                          'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath']] = 0
impute_nan_manually(train)
impute_nan_manually(test)
def recode_ordinal(df):
    """Replaces all string codes of ordinal features 
    with their numeric equivalents."""

    feature_names = set(ordinal_features_recoding['Name'])
    for col in feature_names:
        ser = ordinal_features_recoding.loc[ordinal_features_recoding['Name'] == col,
                               ['Code', 'NumericCode_1']].set_index('Code')
        ser = pd.Series(ser['NumericCode_1'])
        if df[col].dtype == 'O':
            df[col] = df[col].map(ser)
recode_ordinal(train)
train[ORDINAL_FEATURE_NAMES].dtypes
recode_ordinal(test)
test[ORDINAL_FEATURE_NAMES].dtypes
area_names_filter = ['area', 'square']
area_names = list(feature_description['Name'][feature_description['Description'].str.contains(
    '|'.join(area_names_filter))])
feature_description.loc[feature_description['Name'].isin(area_names), ['Name', 'Description']]
# Some area features are sums of other features:
# TotalBsmtSF = BsmtFinSF1 + BsmtFinSF2 + BsmtUnfSF
# GrLivArea = 1stFlrSF + 2ndFlrSF

floor_area_names = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
                    'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea']
train[floor_area_names].head(8)
association_test(train[area_names], target, sort_column=0)
def transform_area(df, names):
    
    # Porch features
    df['TotalPorchArea'] = df['EnclosedPorch'] + \
        df['3SsnPorch'] + df['OpenPorchSF']
    porch_type = df['TotalPorchArea'].apply(
        lambda x: 'Missing' if x == 0 else 'Multiple')
    porch_type[(df['TotalPorchArea'] == df['EnclosedPorch'])
               & (df['EnclosedPorch'] > 0)] = 'Enclosed'
    porch_type[(df['TotalPorchArea'] == df['3SsnPorch'])
               & (df['3SsnPorch'] > 0)] = '3Ssn'
    porch_type[(df['TotalPorchArea'] == df['OpenPorchSF'])
               & (df['OpenPorchSF'] > 0)] = 'Open'
    df['PorchType'] = porch_type
    porch_names = ['EnclosedPorch', '3SsnPorch', 'OpenPorchSF']
    df.drop(porch_names, axis=1, inplace=True)
    
    # Garage features
    df['GarageAreaPerCar'] = df['GarageArea'] / df['GarageCars']
    df.loc[~np.isfinite(df['GarageAreaPerCar']), 'GarageAreaPerCar'] = 0
    df.drop(['GarageArea'], axis=1, inplace=True)
    
    # Bathrooms
    df['TotalBath'] = df['BsmtFullBath'] + df['BsmtHalfBath'] + df['FullBath'] + df['HalfBath']
    df.drop(['BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath'], axis=1, inplace=True)
    
    # Express area features as relations to corresponding totals
    df['TotalFloorArea'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    by_tbsf_names = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF']
    by_gla_names = ['1stFlrSF', '2ndFlrSF']
    include = ['TotalPorchArea']
    exclude = ['TotalFloorArea', 'GarageArea']
    by_tfa_names = set(names + include) - \
        set(by_tbsf_names + by_gla_names + exclude + porch_names)
    divisors = ['TotalBsmtSF', 'GrLivArea', 'TotalFloorArea']
    tags = ['TBSF', 'GLA', 'TFA']
    for feats, div, tag in zip([by_tbsf_names, by_gla_names, by_tfa_names], divisors, tags):
        for f in feats:
            new_name = f + '/' + tag
            df[new_name] = df[f] / df[div]
#             df.drop(f, axis=1, inplace=True)
            df.loc[~np.isfinite(df[new_name]), new_name] = 0
    
#     # Total floor area per room, car, fireplace
#     tally_names = ['GarageCars', 'TotRmsAbvGrd', 'Fireplaces', 'BedroomAbvGr', 'KitchenAbvGr']
#     for f in tally_names:
#         new_name = 'TFA/' + f
#         df[new_name] = df['TotalFloorArea'] / df[f]
#         df.loc[~np.isfinite(df[new_name]), new_name] = 0
transform_area(train, area_names)
transform_area(test, area_names)
print('train.shape:', train.shape)
print('test.shape:', test.shape)
searchfor = ['/TBSF', '/GLA', '/TFA', 'TotalFloorArea']
new_area_names = list(train.columns[train.columns.str.contains('|'.join(searchfor))])
association_test(train[new_area_names], target, sort_column=0)
# searchfor = ['TFA/']
# new_tally_names = list(train.columns[train.columns.str.contains('|'.join(searchfor))])
# association_test(train[new_tally_names], target, sort_column=0)
# plt.scatter(x=train['TFA/KitchenAbvGr'], y=target)
bsmt_name_filter = ['Bsmt']
bsmt_names = list(train.columns[train.columns.str.contains('|'.join(bsmt_name_filter))])
feature_description.loc[feature_description['Name'].isin(bsmt_names), ['Name', 'Description']]
inspect_feature('BsmtFullBath')
# Try weighted basement finished area

bsmt_fin_sf_wghtd = train['BsmtFinType1'] * train['BsmtFinSF1/TBSF'] + \
    train['BsmtFinType2'] * train['BsmtFinSF2/TBSF']
plt.scatter(x=bsmt_fin_sf_wghtd, y=target, s=70,
            edgecolor='white', linewidth=0.5,)
plt.scatter(x=bsmt_fin_sf_wghtd[bsmt_fin_sf_wghtd < 0.4], 
            y=target[bsmt_fin_sf_wghtd < 0.4], s=70, c='red',
            edgecolor='white', linewidth=0.5)
def transform_bsmt(df):
    # Add weighted basement finished area, drop redundant area features
    df['BsmtFinSFWghtd'] = df['BsmtFinType1'] * df['BsmtFinSF1/TBSF'] + \
        df['BsmtFinType2'] * df['BsmtFinSF2/TBSF']
    to_drop = ['BsmtFinType1', 'BsmtFinSF1/TBSF', 'BsmtFinType2', 'BsmtFinSF2/TBSF']
    df.drop(to_drop, axis=1, inplace=True)
    
    # Add indicator features
    df['BsmtCond_TA'] = df['BsmtCond'].apply(lambda x: 1 if x == 4 else 0)
    df['BsmtFinSFWghtd_lessthan0.4'] = df['BsmtFinSFWghtd'].apply(lambda x: 1 if x < 4 else 0)
    
    # Add polynomial features
    df['BsmtQual_p2'] = df['BsmtQual'].apply(lambda x: np.power(x, 2))
transform_bsmt(train)
transform_bsmt(test)
print('train.shape:', train.shape)
print('test.shape:', test.shape)
qual_name_filter = ['Qual', 'Qu', 'QC', 'Fence']
qual_names = list(feature_description['Name'][feature_description['Name'].str.contains(
    '|'.join(qual_name_filter))])
qual_names.remove('LowQualFinSF')
feature_description.loc[feature_description['Name'].isin(qual_names), ['Name', 'Description']]
association_test(train[qual_names], target)
inspect_feature('GarageQual')
def transform_qual(df):
    # Absence of fence appear to add to SalePrice,
    # which does not make sense.
    df.drop(['Fence'], axis=1, inplace=True)
    
    # Aggregate all pool features into one
    df['HasPool'] = df['PoolQC'].apply(lambda x: 0 if x == 1 else 1)
    to_drop = ['PoolQC', 'PoolArea/TFA']
    df.drop(to_drop, axis=1, inplace=True)
    
    # Express all quality features as relation to OverallQual
    divide_by_OQ = ['ExterQual', 'BsmtQual', 'HeatingQC',
                    'KitchenQual', 'FireplaceQu', 'GarageQual']
    for f in divide_by_OQ:
        max_code = ordinal_features_recoding.loc[ordinal_features_recoding['Name'] == f, 
                                                 'NumericCode_1'].max()
        rescaled = (df[f] / max_code)*10
        new_name = f + '/OQ'
        # Dividing quality features by OverallQual produces new features
        # that show weird "sawtooth" relationship pattern with SalePrice. 
        # Below is an attempt to make it more linear.
        df[new_name] = (rescaled - df['OverallQual']) / df['OverallQual']
        df[new_name] = df[new_name].apply(lambda x: x - x if x < 0 else x)
        df.drop(f, axis=1, inplace=True)
transform_qual(train)
transform_qual(test)
print('train.shape:', train.shape)
print('test.shape:', test.shape)
searchfor = ['/OQ']
new_qual_names = list(train.columns[train.columns.str.contains('|'.join(searchfor))])
association_test(train[new_qual_names], target, sort_column=0)
cond_names = ['OverallCond', 'ExterCond', 'BsmtCond', 'GarageCond']
feature_description.loc[feature_description['Name'].isin(cond_names), ['Name', 'Description']]
association_test(train[cond_names], target)
inspect_feature('OverallCond')
def transform_cond(df):
    # Add indicator features
    df['OverallCond_5'] = df['OverallCond'].apply(lambda x: 1 if x == 5 else 0)
    df['ExterCond_TA'] = df['ExterCond'].apply(lambda x: 1 if x == 3 else 0)
    df['GarageCond_TA'] = df['GarageCond'].apply(lambda x: 1 if x == 4 else 0)
transform_cond(train)
transform_cond(test)
print('train.shape:', train.shape)
print('test.shape:', test.shape)
feature_description.loc[feature_description['Type2'] == 'Nominal', ['Name', 'Description']]
inspect_feature('MasVnrType', get_text=True)
association_test(train[list(set(NOMINAL_FEATURE_NAMES) & set(train.columns))], target)
def transfort_nominal(df):
    # Add indicator features
    df['Condition12_PosX'] = (df['Condition1'].isin(['PosN', 'PosA']) |
                              df['Condition1'].isin(['PosN', 'PosA'])) * 1
    df['HasShed'] = (df['MiscFeature'] == 'Shed') * 1
    # Merge categories
    df['BldgType'] = df['BldgType'].apply(
        lambda x: 'Other' if x not in ['1Fam', 'TwnhsE'] else x)
    df['RoofStyle'] = df['RoofStyle'].apply(
        lambda x: 'Other' if x not in ['Gable', 'Hip'] else x)
    df['RoofMatl'] = df['RoofMatl'].apply(lambda x: 'Other' if x not in [
                                          'CompShg', 'Tar&Grv', 'WdShngl', 'WdShake'] else x)
    df['Foundation'] = df['Foundation'].apply(
        lambda x: 'StoneWood' if x in ['Stone', 'Wood'] else x)
    df['Heating'] = df['Heating'].apply(
        lambda x: 'Other' if x in ['Floor', 'Grav', 'OthW', 'Wall'] else x)
    df['SaleType'] = df['SaleType'].apply(
        lambda x: 'Other' if x not in ['WD', 'New', 'COD'] else x)
    # Binarize
    df['CentralAir'] = df['CentralAir'].apply(
        lambda x: 1 if x == 'Y' else 0).astype(np.int)
    # Drop useless or redundant
    df.drop(['MiscFeature', 'MiscVal'], axis=1, inplace=True)
transfort_nominal(train)
transfort_nominal(test)
print('train.shape:', train.shape)
print('test.shape:', test.shape)
feature_description.loc[feature_description['Type2'] == 'Numeric', ['Name', 'Description']]
inspect_feature('GarageYrBlt')
association_test(train[list(set(NUMERIC_FEATURE_NAMES) & set(train.columns))], target)
def transform_age(df):
    df['AgeBuilt'] = df['YrSold'] - df['YearBuilt']
    df['AgeRemod'] = df['YrSold'] - df['YearRemodAdd']
    df['AgeGarage'] = df['YrSold'] - df['GarageYrBlt']
    df['EffectiveAge'] = (df['YearRemodAdd'] -
                          df['YearBuilt']) * 0.6 + df['AgeRemod']
    df.drop(['YearBuilt', 'YearRemodAdd', 'GarageYrBlt',
             'AgeBuilt', 'AgeRemod'], axis=1, inplace=True)
transform_age(train)
transform_age(test)
print('train.shape:', train.shape)
print('test.shape:', test.shape)
feature_description.loc[feature_description['Type2'] == 'Ordinal', ['Name', 'Description']]
association_test(train[list(set(ORDINAL_FEATURE_NAMES) & set(train.columns))], target)
inspect_feature('Functional')
# Top 30 features most correlated with SalePrice
pt_assoc_test = association_test(train, target)
pt_assoc_test.head(30)
# Top 20 features with largest positive skew
skewness = train.select_dtypes(include=np.number).apply(
    lambda x: stats.skew(x, nan_policy='propagate'))
skewness = pd.DataFrame(skewness.sort_values(ascending=False))
skewness.columns = ['Skew']
skewness.head(20)
# Show the effect of power transformation
transf = np.power(train['GarageCars'], 1)
print('Skew:', stats.skew(transf))
print('\nAssociation test:\n', association_test(transf, target))
fig, ax = plt.subplots(1, 1, figsize=[15, 5])
sns.distplot(transf, ax=ax)
plt.show()
def add_power_transformed(df):
    # Add root transformed features
    df['AgeGarage_p1/2'] = df['AgeGarage'].apply(lambda x: np.power(x, 1/2))
    df['TotRmsAbvGrd_p1/4'] = df['TotRmsAbvGrd'].apply(lambda x: np.power(x, 1/4))
    df['TotalFloorArea_p1/3'] = df['TotalFloorArea'].apply(lambda x: np.power(x, 1/3))
    df['GrLivArea_p1/6'] = df['GrLivArea'].apply(lambda x: np.power(x, 1/6))
    df['ExterQual/OQ_p1/3'] = df['ExterQual/OQ'].apply(lambda x: np.power(x, 1/3))
    df['BsmtQual/OQ_p1/2'] = df['BsmtQual/OQ'].apply(lambda x: np.power(x, 1/2))
    df['KitchenQual/OQ_p1/2.5'] = df['KitchenQual/OQ'].apply(lambda x: np.power(x, 1/2.5))
    df['LotArea/TFA_p1/7'] = df['LotArea/TFA'].apply(lambda x: np.power(x, 1/7))
    df['LotArea_p1/10'] = df['LotArea'].apply(lambda x: np.power(x, 1/10))
    df['KitchenAbvGr_p1/3'] = df['KitchenAbvGr'].apply(lambda x: np.power(x, 1/3))
    # Add interactions
    df['TotalFloorArea*OverallQual'] = df['TotalFloorArea'] * df['OverallQual']
    df['GrLivArea*OverallQual'] = df['GrLivArea'] * df['OverallQual']
    df['GarageCars*BsmtQual_p2'] = df['GarageCars'] * df['BsmtQual_p2']
    df['TotalBsmtSF*TotalBath'] = df['TotalBsmtSF'] * df['TotalBath']
    df['1stFlrSF*GarageFinish'] = df['1stFlrSF'] * df['GarageFinish']
    df['TotalBsmtSF*BsmtQual_p2'] = df['TotalBsmtSF'] * df['BsmtQual_p2']
    df['GarageCars*GarageFinish'] = df['GarageCars'] * df['GarageFinish']
    
add_power_transformed(train)
add_power_transformed(test)
modes = train.mode().iloc[0, :]
mode_counts = (train.values == modes.values).sum(axis=0)
mode_counts = pd.DataFrame(mode_counts, index=train.columns, columns=['Count'])
mode_counts['Mode'] = modes
mode_counts['Fraction'] = mode_counts['Count'] / train.shape[0]
mode_counts.sort_values('Fraction', inplace=True, ascending=False)
mode_counts = mode_counts[['Mode', 'Count', 'Fraction']]
mode_counts.head(10)
train.drop(['Utilities'], axis=1, inplace=True)
test.drop(['Utilities'], axis=1, inplace=True)
def trainsform_object_to_categorical(train_df, test_df):
    """Assigns a common exhaustive set of categories for each
    feature of type object in train and test. This elimates the
    need to align training and validatio/test datasets after
    dummification.
    """
    assert np.sum(train_df.columns != test_df.columns) == 0
    obj_cols = list(train_df.select_dtypes(include=np.object).columns)
    for df in [train_df, test_df]:
        for col in obj_cols:
            categories = (set(train_df[col].unique()) |
                          set(test_df[col].unique())) - set([np.nan])
            cat_type = CategoricalDtype(
                categories=categories, ordered=None)
            df[col] = df[col].astype(cat_type)
trainsform_object_to_categorical(train, test)
class MyImputer(BaseEstimator, TransformerMixin):
    def __init__(self, to_mode=None):
        self.to_mode = to_mode
        self.X_ft = None

    def fit(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        self.X_ft = X
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        X_tr = X.copy()
        nan_cols = X_tr.columns[X_tr.isnull().any()]
        
        # Fill via groupby
        to_groupby = ['MSZoning'] #['LotFrontage', 'MSZoning']
#         if 'LotFrontage' in nan_cols:
#             median_lotfront_per_hood = self.X_ft.groupby(
#                 'Neighborhood')['LotFrontage'].agg(np.median)
#             median_lotfront_mapped = X_tr['Neighborhood'].map(
#                 median_lotfront_per_hood)
#             X_tr['LotFrontage'].fillna(median_lotfront_mapped, inplace=True)
        if 'MSZoning' in nan_cols:
            mode_mszoning_per_hood = self.X_ft.groupby(
                'Neighborhood')['MSZoning'].agg(pd.Series.mode)
            mode_mszoning_mapped = X_tr['Neighborhood'].map(mode_mszoning_per_hood)
            X_tr['MSZoning'].fillna(mode_mszoning_mapped, inplace=True)
            
        # Fill with mode
        to_mode = list(set(self.to_mode) & set(nan_cols) - set(to_groupby))
        mode_ft = self.X_ft[self.to_mode].mode().iloc[0, :]
        X_tr.loc[:, self.to_mode] = X_tr.loc[:,
                                             self.to_mode].fillna(mode_ft)
        
        # Fill with median
        to_median = list(set(nan_cols) - set(self.to_mode) -
                         set(to_groupby))
        median_ft = self.X_ft[to_median].median()
        X_tr.loc[:, to_median] = X_tr.loc[:, to_median].fillna(median_ft)
        
        return X_tr
class MyDummifier(BaseEstimator, TransformerMixin):
    def __init__(self, params={}):
        self.params = params
        self.cols = None

    def fit(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        X = X.copy()
        X = pd.get_dummies(X, **self.params)
        return X
# Standard corss-validation KFold object
kfolds = KFold(n_splits=5, shuffle=True, random_state=238)
def rmse(y, y_hat):
    """Calculate RMSE based on actual and predicted values of dependent variable"""
    return np.sqrt(mean_squared_error(y, y_hat))


def rmse_cv(estimator, X, y, kf=None):
    """Calculate cross-validated RMSE when dependent variable is already log-transformed."""
    rmse_cv = np.sqrt(-cross_val_score(estimator, X, y,
                                       scoring="neg_mean_squared_error",
                                       cv=kf))
    return rmse_cv


def rmsle_cv(estimator, X, y, kf=None):
    """Calculate cross-validated RMSLE."""
    rmsle_cv = np.sqrt(np.log(-cross_val_score(estimator, X, y,
                                               scoring='neg_mean_squared_error',
                                               cv=kf)))
    return rmsle_cv


def run_rmse_cv(estimator, X, y, kf=None, record_name=None, num_results_to_show=2):
    """Wrapper function that runs rmse_cv, writes the results of all 
    runs into a dict of lists, and prints the results of a specified number of runs.
    """
    result = rmse_cv(estimator, X, y, kf).mean()
    if record_name != None:
        try:
            global estimator_result_record
            assert isinstance(estimator_result_record, dict)
        except:
            globals()['estimator_result_record'] = dict()
        try:
            estimator_result_record[record_name]
        except:
            estimator_result_record[record_name] = []
        estimator_result_record[record_name].append(result)
        num_results_recorded = len(estimator_result_record[record_name])
        for n in reversed(range(num_results_to_show)):
            try:
                print('RMSLE for run {}:'.format(num_results_recorded - n),
                      estimator_result_record[record_name][-(n+1)])
            except:
                continue
    return result


def run_gridsearch(estimators, param_grid=None, cv=kfolds):
    """Run standard grid-search and show results."""
    pipe = make_pipeline(preprocessing_pipe, *estimators)
    grid = GridSearchCV(pipe,
                        param_grid=param_grid,
                        cv=cv,
                        scoring='neg_mean_squared_error')
    grid.fit(train, target)
    return grid.best_params_, np.sqrt(-grid.best_score_)
# Standard preprocessing pipeline for all estimators
to_mode = list(train.select_dtypes(include='category').columns) + ORDINAL_FEATURE_NAMES
to_mode = list(set(to_mode) & set(train.columns))
preprocessing_pipe = make_pipeline(MyImputer(to_mode=to_mode), MyDummifier())
linreg_pipe = make_pipeline(preprocessing_pipe, LinearRegression())
linreg_rmse_cv = run_rmse_cv(linreg_pipe, train, target,
                             kf=kfolds, record_name='linreg', num_results_to_show=4)
# # Grid search for alpha

# param_grid = {'ridge__alpha': [0.0001, 0.0003, 0.0005, 0.0007, 0.0009,
#                                0.01, 0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 20, 30, 50]}
# run_gridsearch([RobustScaler(), Ridge()], param_grid=param_grid)

# # Results of the last run:
# # ({'ridge__alpha': 15}, 0.11028230638314304)
# Ridge with tweaked grid-searched alpha
ridge_pipe = make_pipeline(preprocessing_pipe, Ridge(**{'alpha': 17}))
ridge_rmse_cv = run_rmse_cv(ridge_pipe, train, target,
                            kf=kfolds, record_name='ridge', num_results_to_show=4)
# # Grid search for alpha

# # param_grid = {'lasso__alpha': [0.0001, 0.0003, 0.0005, 0.0007, 0.0009,
# #                                0.01, 0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 20, 30, 50]}
# param_grid = {'lasso__alpha': [0.00047, 0.00048, 0.00049, 0.00050, 0.00051, 
#                                0.00052, 0.00053, 0.00054, 0.00055, 0.00056, 0.00057]}
# run_gridsearch([RobustScaler(), Lasso()], param_grid=param_grid)

# # Results of the last run:
# # ({'lasso__alpha': 0.00053}, 0.10711245295695188)
# Lasso with tweaked grid-searched alpha
lasso_pipe = make_pipeline(
    preprocessing_pipe, RobustScaler(), Lasso(**{'alpha': 0.00055}))
lasso_rmse_cv = run_rmse_cv(lasso_pipe, train, target,
                            kf=kfolds, record_name='lasso', num_results_to_show=4)
# # Grid search for parameters

# param_grid = {'elasticnet__alpha': [0.0001, 0.0003, 0.0005, 0.0007, 0.0009,
#                                     0.01, 0.05, 0.1],
#               'elasticnet__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]}
# run_gridsearch([RobustScaler(), ElasticNet()], param_grid=param_grid)

# # Results of the last run:
# # ({'elasticnet__alpha': 0.0005, 'elasticnet__l1_ratio': 1.0},
# #  0.10712358737747567)
# ElasticNet with tweaked grid-searched parameters
enet_pipe = make_pipeline(preprocessing_pipe,
                          RobustScaler(),
                          ElasticNet(**{'alpha': 0.00055,
                                        'l1_ratio': 1}))
enet_rmse_cv = run_rmse_cv(enet_pipe, train, target,
                           kf=kfolds, record_name='enet', num_results_to_show=2)
# # Grid search for parameters

# param_grid = {'randomforestregressor__n_estimators': [5, 15, 30, 40, 100]
#               , 'randomforestregressor__max_features': ['auto', 'sqrt', 'log2']
#               , 'randomforestregressor__min_samples_leaf': [1, 3, 4, 5]
#               , 'randomforestregressor__min_samples_split': [2, 8, 10, 12]
#              }
# run_gridsearch([RandomForestRegressor()], param_grid=param_grid)

# # Results of the last run:
# # ({'randomforestregressor__max_features': 'sqrt',
# #   'randomforestregressor__min_samples_leaf': 1,
# #   'randomforestregressor__min_samples_split': 2,
# #   'randomforestregressor__n_estimators': 40},
# #  0.1350395546020487)
# RandomForestRegressor with tweaked grid-searched parameters

rf_pipe = make_pipeline(preprocessing_pipe,
                        RandomForestRegressor(
                            **{'max_features': 'auto',
                               'min_samples_leaf': 1,
                               'min_samples_split': 2,
                               'n_estimators': 40}
                        )
                        )
rf_rmse_cv = run_rmse_cv(rf_pipe, train, target, kf=kfolds,
                         record_name='rf', num_results_to_show=2)
# # Choose a relatively high learning rate, 
# # and find the optimal number of estimators for it

# param_grid = {'xgbregressor__learning_rate': [0.3],
#               'xgbregressor__n_estimators': [86, 88, 90, 92, 94]}

# run_gridsearch([XGBRegressor()], param_grid=param_grid)

# # Results of the last run:
# # ({'xgbregressor__learning_rate': 0.3, 'xgbregressor__n_estimators': 92},
# #  0.12494039958515651)
# # # Tune max_depth and min_child_weight 

# param_grid = {'xgbregressor__learning_rate': [0.3],
#               'xgbregressor__n_estimators': [92],
#               'xgbregressor__max_depth': [3],
#               'xgbregressor__min_child_weight': [2.8, 3.0, 3.1]
#              }

# run_gridsearch([XGBRegressor()], param_grid=param_grid)

# # Results of the last run:
# # ({'xgbregressor__learning_rate': 0.3,
# #   'xgbregressor__max_depth': 3,
# #   'xgbregressor__min_child_weight': 3.1,
# #   'xgbregressor__n_estimators': 92},
# #  0.12399834713926551)
# # Tune gamma

# param_grid = {'xgbregressor__learning_rate': [0.3],
#               'xgbregressor__n_estimators': [92],
#               'xgbregressor__max_depth': [3],
#               'xgbregressor__min_child_weight': [3.1],
#               'xgbregressor__gamma': [0.0, 0.1, 0.2, 0.3, 0.4]
#              }

# run_gridsearch([XGBRegressor()], param_grid=param_grid)

# # # Results of the last run:
# # ({'xgbregressor__gamma': 0.0,
# #   'xgbregressor__learning_rate': 0.3,
# #   'xgbregressor__max_depth': 3,
# #   'xgbregressor__min_child_weight': 3.1,
# #   'xgbregressor__n_estimators': 92},
# #  0.12399834713926551)
# # Recalibrate n_estimators

# param_grid = {'xgbregressor__learning_rate': [0.3],
#               'xgbregressor__n_estimators': [88, 90, 92, 94, 96, 100],
#               'xgbregressor__max_depth': [3],
#               'xgbregressor__min_child_weight': [3.1],
#               'xgbregressor__gamma': [0]
#              }

# run_gridsearch([XGBRegressor()], param_grid=param_grid)

# # # Results of the last run:
# # ({'xgbregressor__gamma': 0,
# #   'xgbregressor__learning_rate': 0.3,
# #   'xgbregressor__max_depth': 3,
# #   'xgbregressor__min_child_weight': 3.1,
# #   'xgbregressor__n_estimators': 96},
# #  0.12391391755598129)
# # Tune subsample and colsample_bytree

# param_grid = {'xgbregressor__learning_rate': [0.3],
#               'xgbregressor__n_estimators': [96],
#               'xgbregressor__max_depth': [3],
#               'xgbregressor__min_child_weight': [3.1],
#               'xgbregressor__gamma': [0],
#               'xgbregressor__subsample': [0.80, 0.85, 0.90, 1],
#               'xgbregressor__colsample_bytree': [0.80, 0.85, 0.90, 1]
#              }

# run_gridsearch([XGBRegressor()], param_grid=param_grid)

# # # Results of the last run:
# # ({'xgbregressor__colsample_bytree': 1,
# #   'xgbregressor__gamma': 0,
# #   'xgbregressor__learning_rate': 0.3,
# #   'xgbregressor__max_depth': 3,
# #   'xgbregressor__min_child_weight': 3.1,
# #   'xgbregressor__n_estimators': 96,
# #   'xgbregressor__subsample': 1},
# #  0.12391391755598129)
# # Tune regularization parameters

# param_grid = {'xgbregressor__learning_rate': [0.3],
#               'xgbregressor__n_estimators': [96],
#               'xgbregressor__max_depth': [3],
#               'xgbregressor__min_child_weight': [3.1],
#               'xgbregressor__gamma': [0],
#               'xgbregressor__subsample': [1],
#               'xgbregressor__colsample_bytree': [1],
#               'xgbregressor__reg_alpha': [0.19, 0.2, 0.21]
#              }

# run_gridsearch([XGBRegressor()], param_grid=param_grid)

# # # Results of the last run:
# # ({'xgbregressor__colsample_bytree': 1,
# #   'xgbregressor__gamma': 0,
# #   'xgbregressor__learning_rate': 0.3,
# #   'xgbregressor__max_depth': 3,
# #   'xgbregressor__min_child_weight': 3.1,
# #   'xgbregressor__n_estimators': 96,
# #   'xgbregressor__reg_alpha': 0.2,
# #   'xgbregressor__subsample': 1},
# #  0.12220976107917751)
# # Lower the learning rate and add more trees
# # Recalibrating subsample and colsample_bytree also appear to be necessary.

# param_grid = {'xgbregressor__learning_rate': [0.05],
#               'xgbregressor__n_estimators': [480],
#               'xgbregressor__max_depth': [3],
#               'xgbregressor__min_child_weight': [3.1],
#               'xgbregressor__gamma': [0],
#               'xgbregressor__subsample': [0.68, 0.7, 0.72],
#               'xgbregressor__colsample_bytree': [0.68, 0.7, 0.72],
#               'xgbregressor__reg_alpha': [0.2]
#              }

# run_gridsearch([XGBRegressor()], param_grid=param_grid)

# # # Results of the last run:
# # ({'xgbregressor__colsample_bytree': 0.72,
# #   'xgbregressor__gamma': 0,
# #   'xgbregressor__learning_rate': 0.05,
# #   'xgbregressor__max_depth': 3,
# #   'xgbregressor__min_child_weight': 3.1,
# #   'xgbregressor__n_estimators': 480,
# #   'xgbregressor__reg_alpha': 0.2,
# #   'xgbregressor__subsample': 0.68},
# #  0.11328134672815252)
gridsearched_params = {'learning_rate': 0.05,
                       'n_estimators': 600,
                       'max_depth': 3,
                       'min_child_weight': 3.1,
                       'gamma': 0,
                       'subsample': 0.68,
                       'colsample_bytree': 0.72,
                       'reg_alpha': 0.2}
xgb_model = XGBRegressor(**gridsearched_params, nthread=1, n_jobs=1)
xgb_pipe = make_pipeline(preprocessing_pipe, xgb_model)
xgb_rmse_cv = run_rmse_cv(xgb_pipe, train, target,
                          kf=kfolds, record_name='xgb', num_results_to_show=4)
def build_nn_model():
    model = Sequential()
    model.add(Dense(256, activation='relu', kernel_regularizer=l1(
        0.006), bias_regularizer=l1(0.006)))
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='relu', kernel_regularizer=l1(
        0.006), bias_regularizer=l1(0.006)))
    model.add(Dense(128, activation='relu', kernel_regularizer=l1(
        0.006), bias_regularizer=l1(0.006)))
    model.add(Dropout(0.1))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


nn_model = KerasRegressor(build_fn=build_nn_model,
                          epochs=100, batch_size=200, verbose=0)
nn_pipe = make_pipeline(preprocessing_pipe, StandardScaler(), nn_model)
nn_rmse_cv = run_rmse_cv(nn_pipe, train, target.values, kf=kfolds,
                     record_name='nn', num_results_to_show=4)
class BaggingRegressor(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models

    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        for model in self.models_:
            model.fit(X, y)
        return self

    def predict(self, X):
        predictions = np.column_stack([model.predict(X) for model in self.models_])
        return np.mean(predictions, axis=1)
bag_model = BaggingRegressor([ridge_pipe, lasso_pipe, xgb_pipe])
bag_rmse_cv = run_rmse_cv(bag_model, train, target, kf=kfolds,
                      record_name='bag', num_results_to_show=4)
class StackingRegressor(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds

    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X.iloc[train_index, :], y.iloc[train_index])
                y_pred = instance.predict(X.iloc[holdout_index, :])
                out_of_fold_predictions[holdout_index, i] = y_pred
        self.meta_model_.fit(out_of_fold_predictions, y)
        
        return self

    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X)
                             for model in base_models]).mean(axis=1)
            for base_models in self.base_models_])
        return self.meta_model_.predict(meta_features)
stack_model = StackingRegressor(base_models=(
    ridge_pipe, lasso_pipe, xgb_pipe), meta_model=Lasso(alpha=0.0006))
stack_rmse_cv = run_rmse_cv(stack_model, train, target, kf=kfolds,
                     record_name='stack', num_results_to_show=4)
scores = [ridge_rmse_cv, lasso_rmse_cv,
          enet_rmse_cv, xgb_rmse_cv, bag_rmse_cv, stack_rmse_cv]
model_names = ['Ridge', 'Lasso', 'ElasticNet', 'XGBoost', 'Bagging', 'Stacking']
fig = plt.figure(figsize=[15, 5])
ax = fig.add_subplot(111)
sns.barplot(model_names, scores, orient='v', ax=ax)
ax.set_ylim(bottom=0.104, top=0.117)
ax.set_title('Cross-validated RMSLE of candidate models')
for p in ax.patches:
    ax.annotate("%.6f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', fontsize=13, color='k', 
                xytext=(0, 20), textcoords='offset points')
plt.show()
final_model = BaggingRegressor([ridge_pipe, lasso_pipe, xgb_pipe])
final_model.fit(train, target)
final_prediction = np.exp(final_model.predict(test))
submission = pd.read_csv(os.path.join(raw_data_path, 'sample_submission.csv'))
print('submission.shape', submission.shape)
print('final_prediction.shape', final_prediction.shape)
submission['SalePrice'] = final_prediction
submission.to_csv('submission.csv', index=False)
submission.head()