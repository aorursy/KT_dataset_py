# Data analysis and wranging
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math
import sys
import os
from decimal import *
import warnings

# Set ipython's max row, max columns and display width display settings
pd.set_option('display.max_row', 1000)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 400)
pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x)) #Limiting floats output to 3 decimal points

# Visualisation
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
%matplotlib inline

# machine learning
from scipy.stats import norm, skew
import scipy.stats as stats
from scipy.stats import chi2_contingency
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline

# Feature selection
from sklearn.feature_selection import RFE

# Create the function to enable us to stop deprecated function warnings
def fxn():
    warnings.warn("deprecated", DeprecationWarning)
    
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)

def feature_null_analysis(df_desc, df, drop_theshold): 
    # Create lists of information we want to see
    feat_names = list(df)
    feat_dtype = list(df.dtypes)
    feat_default = ['Mode: ' + df[feat].mode()[0] if df[feat].dtype == 'object' else 'Median: ' + str(round(df[feat].median(),2)) for feat in list(df)]
    feat_default_perc = ['%/total: ' + str(round(((train[feat].value_counts().iloc[0] / train.shape[0]) * 100),2))
                             if train[feat].dtype == 'object' else '' for feat in list(train)]
    feat_nulls = df.isnull().sum()
    feat_nullperc = df.isnull().mean() * 100
    feat_dropind = ['Y' if val >= drop_theshold else 'N' for val in feat_nullperc]
    
    # Combine the info into one soreted list
    feat_analysis_all = sorted(list(zip(feat_names,feat_dtype,feat_default,feat_default_perc,feat_nulls,feat_nullperc, feat_dropind))
                               ,key=lambda x: x[4], reverse=True)
    feat_analysis_nulls = [feat for feat in feat_analysis_all if feat[5] > 0]  # features with nulls
    feat_droplist = [feat[0] for feat in feat_analysis_all if feat[5] >= 15]  # features recommended to drop
    
    # print the results
    print_feature_null_analysis('features', feat_analysis_nulls)
    
    # Pass back the list of features recommended to drop to make it easier to drop them
    # return feat_analysis_nulls, feat_droplist

def print_feature_null_analysis(df_desc, feat_analysis):
    # print the analysis
    print('\n{: >{width}}'.format(df_desc, width=2 * PRINT_WIDTH))
    print('{: >{width}}'.format('+++++++++', width=2 * PRINT_WIDTH))
    print('{: <{width}}{: <{width}}{: <{width}}{: <{width}}{: <{width}}{: <{width}}{: <{width}}'.
          format('Feature','DType','Mode/Median','Perc. Rows = Mode','No. Nulls','Perc. Nulls','Drop? (Y/N)', width=PRINT_WIDTH))
    print('{: <{width}}{: <{width}}{: <{width}}{: <{width}}{: <{width}}{: <{width}}{: <{width}}'.
          format('=======','=====','===========','=================','=========','===========','=====', width=PRINT_WIDTH))
    
    for feat in feat_analysis:
        print('{: <{width}}{: <{width}}{: <{width}}{: <{width}}{: <{width}}{: <{width}.2F}{: <{width}}'.
              format(feat[0],str(feat[1]),feat[2],feat[3],feat[4],feat[5],feat[6], width=PRINT_WIDTH))

def drop_features(df_desc, df, feats_to_drop):
    print('\nBefore shape for {}: {}'.format(df_desc, df.shape))
    df.drop(feats_to_drop, axis=1, inplace=True)
    print('\nAfter shape for {}: {}'.format(df_desc, df.shape))
    feat_analysis_nulls, feat_droplist = feature_null_analysis(df_desc, df, NULL_PERC_DROP_PERC)  # As we've dropped one or more features we need to recreate the analysis
    return feat_analysis_nulls, FEATURES_DROPPED + feats_to_drop

# def drop_target_feature(df):
#     target_feature_data = df[TARGET_FEATURE]
#     df.drop(TARGET_FEATURE, axis=1, inplace=True)
#     return target_feature_data

NULL_PERC_DROP_PERC = 15  # Set the threshold percentage of nulls in a column - will determine if it's recommended to drop
PRINT_WIDTH = 20  # Print parameter
# Check files in input directory - Windows
# input_dir = 'C:/Users/.........'
# Check files in input directory - Jupyter
input_dir = '../input/'
l = list(os.listdir(input_dir))
for f in l:
    print(f)
# Import Train and Test data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

train.head()
test.head()
# Save then drop the Id column from Train and Test - not needed for modelling/prediction

# Save Id
train_id = train['Id']
test_id = train['Id']

# Print shapes before drop
print('\nBefore shape for {}: {}'.format('train', train.shape))
print('\nBefore shape for {}: {}'.format('test', test.shape))

# Drop Id
train.drop('Id', axis=1, inplace=True)
test.drop('Id', axis=1, inplace=True)

# Print shapes after drop
print('\nAfter shape for {}: {}'.format('train', train.shape))
print('\nAfter shape for {}: {}'.format('test', test.shape))
# Analyse and print the features for Null values
feature_null_analysis('train', train, NULL_PERC_DROP_PERC)
train['SalePrice'].describe()
sns.set(style="whitegrid", palette='bright')
fig, ax = plt.subplots(figsize=(12,6))
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: '${:,.0F}'.format(x))) 
    p = sns.distplot(train['SalePrice'] , fit=stats.norm, ax=ax)
    ax.set(xlabel='Sale Price', ylabel='Density')
    sns.despine(trim=True)
    p.set(xlim=(0, None), ylim=(0,None))
# Get the fitted parameters used by the function plus skewness and kurtosis
mu, sigma = stats.norm.fit(train['SalePrice'])
sales_skew = train['SalePrice'].skew()
sales_kurtosis = train['SalePrice'].kurtosis()
print( '\n mu(Avg) = {:.2f}, sigma(SD) = {:.2f}, skew = {:.2F} and kurtosis = {:.2F}\n'.format(mu, sigma, sales_skew, sales_kurtosis))
numeric_features = train.select_dtypes(exclude=['object']).columns.tolist()

grid_cols = 2
pair_plots = 1
grid_rows = math.ceil(len(numeric_features) / ((grid_cols // pair_plots)))
fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(20,80))

axes = axes.ravel()
colix = 0
plt.subplots_adjust(top = 0.99, bottom=0.01, hspace=0.4, wspace=0.1)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()
    for i in range(len(numeric_features)):
        ax = axes[i]
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontname('Arial')
            label.set_fontsize(12)
        ax.set_ylabel('SalePrice', fontsize=12)    
        ax.set_xlabel(numeric_features[colix], fontsize=12)
        ax.tick_params(axis='x', rotation=70)
        
        g = sns.regplot(x=numeric_features[colix], y='SalePrice', ax=ax, data=train, scatter_kws={'s':3})
        colix += 1
categorical_features = train.select_dtypes(include=['object']).columns.tolist()

feat_cols = ['MSSubClass', 'MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
           'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 
           'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional',
           'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'SaleType', 'SaleCondition']

grid_cols = 4
pair_plots = 2
grid_rows = len(categorical_features) // ((grid_cols // pair_plots))
fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(20,80))
axes = axes.ravel()
dv = 'SalePrice'
colix = 0
plt.subplots_adjust(top = 0.99, bottom=0.01, hspace=0.4, wspace=0.2)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()
    for i in range(0,grid_rows * grid_cols,2):
        g = sns.boxplot(x=categorical_features[colix], y=dv, data=train, ax=axes[i])
        p = sns.countplot(x=categorical_features[colix], data=train, ax=axes[i+1])
        g.set_xticklabels(g.get_xticklabels(), rotation=70, fontsize = 12)
        p.set_xticklabels(g.get_xticklabels(), rotation=70, fontsize = 12)
        if categorical_features[colix] in ['YearBuilt', 'YearRemodAdd']:
            g.set_xticklabels(g.get_xticklabels(), visible=False)
            p.set_xticklabels(p.get_xticklabels(), visible=False)
        colix += 1
corrmat = train.corr()
fig, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=0.8, square=False);
# Saleprice correlation matrix
k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train[cols].values.T)
sns.set(font_scale=1.25)
fig, ax = plt.subplots(figsize=(10, 5))
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
iv_cols = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'TotRmsAbvGrd','YearBuilt']
dv = train['SalePrice']
grid_cols = 4
pair_plots = 1
colix = 0
grid_rows = math.ceil(len(iv_cols) / ((grid_cols // pair_plots)))
fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(25,8))
axes = axes.ravel()
plt.subplots_adjust(top = 0.99, bottom=0.01, hspace=0.4, wspace=0.5)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()
    for i in range(len(iv_cols)):
        g = sns.regplot(x=iv_cols[colix], y=dv, ax=axes[i], data=train, scatter_kws={'s':1}, order=2)
        colix += 1
#standardizing data
saleprice_scaled = StandardScaler().fit_transform(train['SalePrice'][:,np.newaxis]);
low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]
high_range= saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]
print('outer range (low) of the distribution:')
print(low_range)
print('\nouter range (high) of the distribution:')
print(high_range)
fig, ax = plt.subplots(figsize=(20,10))
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()
    g = sns.scatterplot(train['GrLivArea'], train['SalePrice'], palette="Set1", hue=train['Neighborhood'], legend='full', s=100)
    sns.regplot(train['GrLivArea'], train['SalePrice'], data=train, scatter=False)  # Add a regression line to the scatterplot
    # Put the legend out of the figure
    _ = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
outliers_df = train.query('GrLivArea > 4500 | SalePrice > 700000').sort_values('SalePrice')
outliers_df.sort_values('SalePrice', ascending=False)
# Deleting Outliers
print('\nBefore shape for {}: {}'.format('train', train.shape))
train = train.drop(train[(train['GrLivArea'] > 4500) & (train['SalePrice'] < 200000)].index)
# Check deletion
print('\nAfter shape for {}: {}'.format('train', train.shape))
train.sort_values(by = 'GrLivArea', ascending = False)[:2]
fig, ax = plt.subplots(figsize=(20,10))
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()
    g = sns.scatterplot(train['TotalBsmtSF'], train['SalePrice'], palette="Set1", hue=train['Neighborhood'], legend='full', s=100)
    sns.regplot(train['TotalBsmtSF'], train['SalePrice'], data=train, scatter=False)  # Add a regression line to the scatterplot
# Put the legend out of the figure
_ = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
sns.set(style="whitegrid", palette='bright')
fig, ax = plt.subplots(figsize=(12,6))
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: '${:,.0F}'.format(x))) 
    p = sns.distplot(train['SalePrice'] , fit=stats.norm, ax=ax)
    ax.set(xlabel='Sale Price', ylabel='Density')
    sns.despine(trim=True)
    p.set(xlim=(0, None), ylim=(0,None))
# Get the QQ-plot
fig = plt.figure(figsize=(12,6))
res = stats.probplot(train['SalePrice'], plot=plt)
# sns.despine(trim=True)
plt.show()
#We use the numpy fuction log1p which  applies log(1+x) to all elements of the column
train["SalePrice"] = np.log1p(train["SalePrice"])

#Check the new distribution 
fig, ax = plt.subplots(figsize=(12,6))
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()
    # ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: '${:,.0F}'.format(x))) 
    p = sns.distplot(train['SalePrice'] , fit=stats.norm, ax=ax)
    ax.set(xlabel='Sale Price', ylabel='Density')
    sns.despine(trim=True)
    # p.set(xlim=(0, None), ylim=(0,None))
# Get the QQ-plot
fig = plt.figure(figsize=(12,6))
res = stats.probplot(train['SalePrice'], plot=plt)
# sns.despine(trim=True)
plt.show()
y = train['SalePrice'].reset_index(drop=True)
train_features = train.drop(['SalePrice'], axis=1)
test_features = test
print('train_features shape: ', train_features.shape)
print('train_features shape: ', test_features.shape)
features = pd.concat([train_features, test_features]).reset_index(drop=True)
print('features shape: ', features.shape)
# Analyse and print the features for Null values
feature_null_analysis('features', features, NULL_PERC_DROP_PERC)
features['Functional'] = features['Functional'].fillna(features['Functional'].mode()[0])
features['Electrical'] = features['Electrical'].fillna(features['Electrical'].mode()[0])
features['KitchenQual'] = features['KitchenQual'].fillna(features['KitchenQual'].mode()[0])
features['Exterior1st'] = features['Exterior1st'].fillna(features['Exterior1st'].mode()[0])
features['Exterior2nd'] = features['Exterior2nd'].fillna(features['Exterior2nd'].mode()[0])
features['SaleType'] = features['SaleType'].fillna(features['SaleType'].mode()[0])
features.query('PoolArea > 0 & PoolQC.isnull()', engine='python')
features.loc[2418, 'PoolQC'] = 'Fa'
features.loc[2501, 'PoolQC'] = 'Gd'
features.loc[2597, 'PoolQC'] = 'Fa'
features.query('GarageType == "Detchd" & GarageYrBlt.isnull()', engine='python')
features.loc[2124, 'GarageYrBlt'] = features.loc[2124, 'YearRemodAdd']
features.loc[2574, 'GarageYrBlt'] = features.loc[2574, 'YearRemodAdd']

features.loc[2124, 'GarageFinish'] = features['GarageFinish'].mode()[0]
features.loc[2574, 'GarageFinish'] = features['GarageFinish'].mode()[0]

features.loc[2574, 'GarageCars'] = features['GarageCars'].median()

features.loc[2124, 'GarageArea'] = features['GarageArea'].median()
features.loc[2574, 'GarageArea'] = features['GarageArea'].median()

features.loc[2124, 'GarageQual'] = features['GarageQual'].mode()[0]
features.loc[2574, 'GarageQual'] = features['GarageQual'].mode()[0]

features.loc[2124, 'GarageCond'] = features['GarageCond'].mode()[0]
features.loc[2574, 'GarageCond'] = features['GarageCond'].mode()[0]
# Create a new df focusing on where any basement figure are null
basement_features = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF']
basement_df = features[basement_features]
basement_df_nulls = basement_df[basement_df.isnull().any(axis=1)]
#now select just the rows that have less then 5 NA's, meaning there is incongruency in the row
basement_df_nulls[(basement_df_nulls.isnull()).sum(axis=1) < 5]
features.loc[332, 'BsmtFinType2'] = 'ALQ' #since SF2 smaller than SF1
features.loc[947, 'BsmtExposure'] = 'No' 
features.loc[1485, 'BsmtExposure'] = 'No'
features.loc[2038, 'BsmtCond'] = 'TA'
features.loc[2183, 'BsmtCond'] = 'TA'
features.loc[2215, 'BsmtQual'] = 'Po' #v small basement so let's do Poor.
features.loc[2216, 'BsmtQual'] = 'Fa' #similar but a bit bigger.
features.loc[2346, 'BsmtExposure'] = 'No' #unfinished bsmt so prob not.
features.loc[2522, 'BsmtCond'] = 'Gd' #cause ALQ for bsmtfintype1
# Analyse and print the features for Null values
feature_null_analysis('features', features, NULL_PERC_DROP_PERC)
features.groupby('MSSubClass')['MSZoning'].apply(lambda x: x.mode()[0])
features['MSZoning'] = features.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))
# Analyse and print the features for Null values
feature_null_analysis('features', features, NULL_PERC_DROP_PERC)
features.loc[features['MasVnrType'].isnull() & features['MasVnrArea'].notnull()]
features.groupby('Neighborhood')['MasVnrType'].apply(lambda x: x.mode()[0])
features.loc[2608, 'MasVnrType'] = 'BrkFace'
# Analyse and print the features for Null values
feature_null_analysis('features', features, NULL_PERC_DROP_PERC)
categorical_features = list(features.select_dtypes(include='object'))
features.update(features[categorical_features].fillna('None'))
# Analyse and print the features for Null values
feature_null_analysis('features', features, NULL_PERC_DROP_PERC)
features.groupby('Neighborhood')['LotFrontage'].apply(lambda x: x.median())
features['LotFrontage'] = features.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.median())
# Analyse and print the features for Null values
feature_null_analysis('features', features, NULL_PERC_DROP_PERC)
garage_features = ['GarageType','GarageYrBlt','GarageFinish','GarageCars','GarageArea','GarageQual','GarageCond']
features.loc[features['GarageYrBlt'].isnull(), list(garage_features)]
# Check for any anomalies
features.loc[features['GarageYrBlt'].isnull() & features['GarageArea'] > 0]
features[(features['MasVnrArea'].isnull())]
numeric_features = list(features.select_dtypes(exclude='object'))
features.update(features[numeric_features].fillna(0))
# Analyse and print the features for Null values
feature_null_analysis('features', features, NULL_PERC_DROP_PERC)
features.describe()
features[features['GarageYrBlt'] > 2100]
features.loc[2590, 'GarageYrBlt'] = 2007  # Assuming the garage was added when the house was remodelled and someone made a typo entering 2207 instead of 2007
categorical_numeric_features = ['OverallQual', 'OverallCond', 'MSSubClass']
all_numeric_features = list(features.select_dtypes(exclude='object'))

# Remove them from the list of numerics
gen_numeric_features = [feat for feat in all_numeric_features if feat not in categorical_numeric_features]

# get the skew measures
skew_features = features[gen_numeric_features].apply(lambda x: skew(x)).sort_values(ascending=False)
skew_features
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax

high_skew = skew_features[skew_features > 0.5]

# Log1p doesn't give as good results - try it!
# for feat in high_skew.index:
#     features[feat] = np.log1p(features[feat])

for feat in high_skew.index:
    features[feat] = boxcox1p(features[feat], boxcox_normmax(features[feat]+1))

skew_features2 = features[gen_numeric_features].apply(lambda x: skew(x)).sort_values(ascending=False)
skew_features2
# Convert numeric categorical features to Object dtype
features_to_convert = ['MSSubClass']
features[features_to_convert].dtypes

for feat in features_to_convert:
    print('{} before - {}'.format(feat,features[feat].dtype), end=' -> ')
    features[feat] = features[feat].astype('object')
    print('{} after - {}'.format(feat,features[feat].dtype))
# Convert ordinal categorical 
from sklearn.preprocessing import LabelEncoder

ordinal_catorgorical_features = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'KitchenQual', 
                                 'Functional', 'FireplaceQu', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'LotShape', 'LandSlope', 'PavedDrive']
print('Shape all_data: {}'.format(features.shape))
features.head()
# process columns, apply LabelEncoder to categorical features
for feat in ordinal_catorgorical_features:
    lbl = LabelEncoder() 
    lbl.fit(list(features[feat].values)) 
    features[feat] = lbl.transform(list(features[feat].values))
print('Shape all_data: {}'.format(features.shape))
print(features[ordinal_catorgorical_features].dtypes)
features.head()
categorical_features = list(features.select_dtypes(include='object'))
cat_value_counts = [(features[x].value_counts().iloc[0:3] / features.shape[0]) * 100 for x in categorical_features if (features[x].value_counts().iloc[0] / features.shape[0]) * 100 > 98]

for i,x in enumerate(cat_value_counts):
    print(x)
cols_dropped = ['Street', 'Utilities', 'Condition2','RoofMatl', 'Heating', 'PoolQC']
features.drop(cols_dropped, axis=1, inplace=True)
features.head()
features['Total_sqr_footage'] = (features['BsmtFinSF1'] + features['BsmtFinSF2'] +
                                 features['1stFlrSF'] + features['2ndFlrSF'])

features['Total_Bathrooms'] = (features['FullBath'] + (0.5*features['HalfBath']) + 
                               features['BsmtFullBath'] + (0.5*features['BsmtHalfBath']))

features['Total_porch_sf'] = (features['OpenPorchSF'] + features['3SsnPorch'] +
                              features['EnclosedPorch'] + features['ScreenPorch'] +
                             features['WoodDeckSF'])


#simplified features
features['haspool'] = features['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
features['has2ndfloor'] = features['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
features['hasgarage'] = features['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
features['hasbsmt'] = features['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
features['hasfireplace'] = features['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
features.shape
final_features = pd.get_dummies(features).reset_index(drop=True)
final_features.shape
final_features.head()
y.shape
X = final_features.iloc[:len(y),:]
testing_features = final_features.iloc[len(X):,:]
print(X.shape)
print(testing_features.shape)
# Imports
import statsmodels.api as sm

model = sm.OLS(endog = y, exog = X)
results = model.fit()
test = results.outlier_test()
# Change the last parameter of the plot to choose another feature
import statsmodels.graphics as smgraphics
figure = smgraphics.regressionplots.plot_fit(results, 1)
pd.set_option('display.float_format', lambda x: '{:.9f}'.format(x)) #Limiting floats output to 3 decimal points
print('Outliers:')
test.loc[test['bonf(p)'] < 0.001]
# Check shape before outlier drop
print('X shape before: {}'.format(X.shape))
print('y shape before: {}'.format(y.shape))
# Drop the outliers
outlier_indexes = list(test.loc[test['bonf(p)'] < 0.001].index)
X = X.drop(X.index[outlier_indexes])
y = y.drop(y.index[outlier_indexes])
print('X shape after: {}'.format(X.shape))
print('y shape after: {}'.format(y.shape))
# Create the ovefit list
df_overfit = pd.DataFrame(columns=['feature', 'feature mode %'])
for feat in X.columns:
    if (X[feat].value_counts().iloc[0] / X.shape[0] * 100) > 97.5:
        df_overfit = df_overfit.append({'feature': feat, 'feature mode %': (X[feat].value_counts().iloc[0] / X.shape[0] * 100)}, ignore_index=True)

# Print the overfit list and shape of train and test sets
for i, row in df_overfit.iterrows():
    print('Feature: {: <{WIDTH}}   >   % of Values: {: <{WIDTH}}'.format(row[0], row[1], WIDTH=PRINT_WIDTH))
print('\nX shape before: {}'.format(X.shape))
print('testing_features shape before: {}'.format(testing_features.shape))

# Create list of overfit features

overfit_features = list(df_overfit['feature'])
print('No. of Features: {}'.format(len(overfit_features)))
X = X.drop(overfit_features, axis=1)
testing_features = testing_features.drop(overfit_features, axis=1)
print('X shape after: {}'.format(X.shape))
print('testing_features shape after: {}'.format(testing_features.shape))
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline

#Build our model method
lm = LinearRegression()

#Build our cross validation method
kfolds = KFold(n_splits=10, shuffle=True, random_state=23)

#build our model scoring function
def cv_rmse(model):
    rmse = np.sqrt(-cross_val_score(model, X, y, 
                                   scoring="neg_mean_squared_error", 
                                   cv = kfolds))
    return(rmse)


#second scoring metric
def cv_rmsle(model):
    rmsle = np.sqrt(np.log(-cross_val_score(model, X, y, scoring = 'neg_mean_squared_error', cv=kfolds)))
    return(rmsle)
benchmark_model = make_pipeline(RobustScaler(),lm).fit(X=X, y=y)
cv_rmse(benchmark_model).mean()
coeffs = pd.DataFrame(list(zip(X.columns, benchmark_model.steps[1][1].coef_)), columns=['Predictors', 'Coefficients'])

coeffs.sort_values(by='Coefficients', ascending=False)
from sklearn.linear_model import RidgeCV

def ridge_selector(k):
    ridge_model = make_pipeline(RobustScaler(),
                                RidgeCV(alphas = [k],
                                        cv=kfolds)).fit(X, y)
    
    ridge_rmse = cv_rmse(ridge_model).mean()
    return(ridge_rmse)
r_alphas = [.0001, .0003, .0005, .0007, .0009, 
          .01, 0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 20, 30, 50, 60, 70, 80]

ridge_scores = []
for alpha in r_alphas:
    score = ridge_selector(alpha)
    ridge_scores.append(score)
plt.plot(r_alphas, ridge_scores, label='Ridge')
plt.legend('center')
plt.xlabel('alpha')
plt.ylabel('score')

ridge_score_table = pd.DataFrame(ridge_scores, r_alphas, columns=['RMSE'])
ridge_score_table
alphas_alt = [7.5, 7.6, 7.7, 7.8, 7.9, 8, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9, 9, 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8 , 
              9.9, 10]
# alphas_alt = [14, 14.1, 14.2, 14.3, 14.4, 14.5, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5, 15.6, 15.7, 15.8 , 15.9,
#              16, 16.1, 16.2, 16.3, 16.4, 16.5]

ridge_model2 = make_pipeline(RobustScaler(),
                            RidgeCV(alphas = alphas_alt,
                                    cv=kfolds)).fit(X, y)

cv_rmse(ridge_model2).mean()
ridge_model2.steps[1][1].alpha_
from sklearn.linear_model import LassoCV


alphas = [0.00005, 0.0001, 0.0003, 0.0005, 0.0007, 
          0.0009, 0.01]
alphas2 = [0.00005, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005,
           0.0006, 0.0007, 0.0008]


lasso_model2 = make_pipeline(RobustScaler(),
                             LassoCV(max_iter=1e7,
                                    alphas = alphas2,
                                    random_state = 42)).fit(X, y)
scores = lasso_model2.steps[1][1].mse_path_

plt.plot(alphas2, scores, label='Lasso')
plt.legend(loc='center')
plt.xlabel('alpha')
plt.ylabel('RMSE')
plt.tight_layout()
plt.show()
lasso_model2.steps[1][1].alpha_
cv_rmse(lasso_model2).mean()
coeffs = pd.DataFrame(list(zip(X.columns, lasso_model2.steps[1][1].coef_)), columns=['Predictors', 'Coefficients'])
used_coeffs = coeffs[coeffs['Coefficients'] != 0].sort_values(by='Coefficients', ascending=False)
print(used_coeffs.shape)
print(used_coeffs)
used_coeffs_values = X[used_coeffs['Predictors']]
used_coeffs_values.shape
overfit_test2 = []
for i in used_coeffs_values.columns:
    counts2 = used_coeffs_values[i].value_counts()
    zeros2 = counts2.iloc[0]
    if zeros2 / len(used_coeffs_values) * 100 > 99.5:
        overfit_test2.append(i)
        
overfit_test2
from sklearn.linear_model import ElasticNetCV

e_alphas = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]
e_l1ratio = [0.75, 0.8, 0.85, 0.9, 0.95, 0.99, 1]

elastic_cv = make_pipeline(RobustScaler(), 
                           ElasticNetCV(max_iter=1e7, alphas=e_alphas, 
                                        cv=kfolds, l1_ratio=e_l1ratio))

elastic_model3 = elastic_cv.fit(X, y)
cv_rmse(elastic_model3).mean()
print(elastic_model3.steps[1][1].l1_ratio_)
print(elastic_model3.steps[1][1].alpha_)
from sklearn.model_selection import GridSearchCV
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4
%matplotlib inline
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

def modelfit(alg, dtrain, target, useTrainCV=True, 
             cv_folds=5, early_stopping_rounds=50):
    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain.values, 
                              label=y.values)
        
        print("\nGetting Cross-validation result..")
        cvresult = xgb.cv(xgb_param, xgtrain, 
                          num_boost_round=alg.get_params()['n_estimators'], 
                          nfold=cv_folds,metrics='rmse', 
                          early_stopping_rounds=early_stopping_rounds,
                          verbose_eval = True)
        alg.set_params(n_estimators=cvresult.shape[0])
    
    #Fit the algorithm on the data
    print("\nFitting algorithm to data...")
    alg.fit(dtrain, target, eval_metric='rmse')
        
    #Predict training set:
    print("\nPredicting from training data...")
    dtrain_predictions = alg.predict(dtrain)
        
    #Print model report:
    print("\nModel Report")
    print("RMSE : %.4g" % np.sqrt(mean_squared_error(target.values,
                                             dtrain_predictions)))
xgb3 = XGBRegressor(learning_rate =0.01, n_estimators=3460, max_depth=3,
                     min_child_weight=0 ,gamma=0, subsample=0.7,
                     colsample_bytree=0.7,objective= 'reg:linear',
                     nthread=4,scale_pos_weight=1,seed=27, reg_alpha=0.00006)

xgb_fit = xgb3.fit(X, y)
cv_rmse(xgb_fit).mean()
from sklearn import svm
svr_opt = svm.SVR(C = 100000, gamma = 1e-08)

svr_fit = svr_opt.fit(X, y)
cv_rmse(svr_fit).mean()
from lightgbm import LGBMRegressor

lgbm_model = LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
cv_rmse(lgbm_model).mean()
lgbm_fit = lgbm_model.fit(X, y)
from mlxtend.regressor import StackingCVRegressor
from sklearn.pipeline import make_pipeline

#setup models
ridge = make_pipeline(RobustScaler(), 
                      RidgeCV(alphas = alphas_alt, cv=kfolds))

lasso = make_pipeline(RobustScaler(),
                      LassoCV(max_iter=1e7, alphas = alphas2,
                              random_state = 42, cv=kfolds))

elasticnet = make_pipeline(RobustScaler(), 
                           ElasticNetCV(max_iter=1e7, alphas=e_alphas, 
                                        cv=kfolds, l1_ratio=e_l1ratio))

lightgbm = make_pipeline(RobustScaler(),
                        LGBMRegressor(objective='regression',num_leaves=5,
                                      learning_rate=0.05, n_estimators=720,
                                      max_bin = 55, bagging_fraction = 0.8,
                                      bagging_freq = 5, feature_fraction = 0.2319,
                                      feature_fraction_seed=9, bagging_seed=9,
                                      min_data_in_leaf =6, 
                                      min_sum_hessian_in_leaf = 11))

xgboost = make_pipeline(RobustScaler(),
                        XGBRegressor(learning_rate =0.01, n_estimators=3460, 
                                     max_depth=3,min_child_weight=0 ,
                                     gamma=0, subsample=0.7,
                                     colsample_bytree=0.7,
                                     objective= 'reg:linear',nthread=4,
                                     scale_pos_weight=1,seed=27, 
                                     reg_alpha=0.00006))


#stack
stack_gen = StackingCVRegressor(regressors=(ridge, lasso, elasticnet, 
                                            xgboost, lightgbm), 
                               meta_regressor=xgboost,
                               use_features_in_secondary=True)

#prepare dataframes
stackX = np.array(X)
stacky = np.array(y)
#scoring 

print("cross validated scores")

for model, label in zip([ridge, lasso, elasticnet, xgboost, lightgbm, stack_gen],
                     ['RidgeCV', 'LassoCV', 'ElasticNetCV', 'xgboost', 'lightgbm',
                      'StackingCVRegressor']):
    
    SG_scores = cross_val_score(model, stackX, stacky, cv=kfolds,
                               scoring='neg_mean_squared_error')
    print("RMSE", np.sqrt(-SG_scores.mean()), "SD", scores.std(), label)
stack_gen_model = stack_gen.fit(stackX, stacky)
em_preds = elastic_model3.predict(testing_features)
lasso_preds = lasso_model2.predict(testing_features)
ridge_preds = ridge_model2.predict(testing_features)
stack_gen_preds = stack_gen_model.predict(testing_features)
xgb_preds = xgb_fit.predict(testing_features)
svr_preds = svr_fit.predict(testing_features)
lgbm_preds = lgbm_fit.predict(testing_features)
stack_preds = ((0.2*em_preds) + (0.1*lasso_preds) + (0.2*ridge_preds) + 
               (0.2*xgb_preds) + (0.1*lgbm_preds) + (0.2*stack_gen_preds))
submission = pd.read_csv('../input/sample_submission.csv')
submission.iloc[:,1] = np.expm1(stack_preds)
submission.to_csv("final_submission.csv", index=False)
