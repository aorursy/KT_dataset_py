import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
plt.style.use('fivethirtyeight')

from datetime import datetime
from category_encoders import TargetEncoder

from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from scipy import stats
from scipy.stats import skew, boxcox_normmax, norm
from scipy.special import boxcox1p
from lightgbm import LGBMRegressor

import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator

import warnings
warnings.simplefilter('ignore')
print('Setup complete')
# Read data
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
print("Train set size:", train.shape)
print("Test set size:", test.shape)
train.head()
# Helper Functions

def plot_numerical(col, discrete=False):
    if discrete:
        fig, ax = plt.subplots(1,2,figsize=(12,6))
        sns.stripplot(x=col, y='SalePrice', data=train, ax=ax[0])
        sns.countplot(train[col], ax=ax[1])
        fig.suptitle(str(col) + ' analysis')
    else:
        fig, ax = plt.subplots(1,2,figsize=(12,6))
        sns.scatterplot(x=col, y='SalePrice', data=train, ax=ax[0])
        sns.distplot(train[col], kde=False, ax=ax[1])
        fig.suptitle(str(col) + ' analysis')

def plot_categorical(col):
    fig, ax = plt.subplots(1,2,figsize=(12,6), sharey=True)
    sns.stripplot(x=col, y='SalePrice', data=train, ax=ax[0])
    sns.boxplot(x=col, y='SalePrice', data=train, ax=ax[1])
    fig.suptitle(str(col) + ' analysis')
    
print('Plot functions are ready to use')
plt.figure(figsize=(8,5))
a = sns.distplot(train.SalePrice, kde=False)
plt.title('SalePrice distribution')
a = plt.axvline(train.SalePrice.describe()['25%'], color='b')
a = plt.axvline(train.SalePrice.describe()['75%'], color='b')
print('SalePrice description:')
print(train.SalePrice.describe().to_string())
# Select numerical features only
num_features = [col for col in train.columns if train[col].dtype in ['int64', 'float64']]
# Remove Id & SalePrice 
num_features.remove('Id')
num_features.remove('SalePrice')
# Create a numerical columns only dataframe
num_analysis = train[num_features].copy()
# Impute missing values with the median just for the moment
for col in num_features:
    if num_analysis[col].isnull().sum() > 0:
        num_analysis[col] = SimpleImputer(strategy='median').fit_transform(num_analysis[col].values.reshape(-1,1))
# Train a model   
clf = ExtraTreesRegressor(random_state=42)
h = clf.fit(num_analysis, train.SalePrice)
feature_imp = pd.DataFrame(sorted(zip(clf.feature_importances_,num_features)), columns=['Value','Feature'])
plt.figure(figsize=(16,10))
sns.barplot(x='Value', y='Feature', data=feature_imp.sort_values(by='Value', ascending=False))
plt.title('Most important numerical features with ExtraTreesRegressor')
del clf, h;
plt.figure(figsize=(8,8))
plt.title('Correlation matrix with SalePrice')
selected_columns = ['OverallQual', 'GarageCars', 'GrLivArea', 'YearBuilt', 'FullBath', '1stFlrSF', 'TotalBsmtSF', 'GarageArea']
a = sns.heatmap(train[selected_columns + ['SalePrice']].corr(), annot=True, square=True)
plot_numerical('OverallQual', True)
plot_numerical('GarageCars', True)
plot_numerical('GrLivArea')
plot_numerical('YearBuilt')
plot_numerical('FullBath', True)
plot_numerical('1stFlrSF')
plot_numerical('TotalBsmtSF')
plot_numerical('GarageArea')
# Select categorical features only
cat_features = [col for col in train.columns if train[col].dtype =='object']
# Create a categorical columns only dataframe
cat_analysis = train[cat_features].copy()
# Impute missing values with NA just for the moment
for col in cat_analysis:
    if cat_analysis[col].isnull().sum() > 0:
        cat_analysis[col] = SimpleImputer(strategy='constant').fit_transform(cat_analysis[col].values.reshape(-1,1))
# Target encoding
target_enc = TargetEncoder(cols=cat_features)
cat_analysis = target_enc.fit_transform(cat_analysis, train.SalePrice) 
# Train a model 
clf = ExtraTreesRegressor(random_state=42)
h = clf.fit(cat_analysis, train.SalePrice)
feature_imp = pd.DataFrame(sorted(zip(clf.feature_importances_,cat_features)), columns=['Value','Feature'])
plt.figure(figsize=(16,10))
sns.barplot(x='Value', y='Feature', data=feature_imp.sort_values(by='Value', ascending=False))
plt.title('Most important categorical features with ExtraTreesRegressor')
del clf, h;
fig, ax = plt.subplots(1,2,figsize=(16,6), sharey=True)
sns.stripplot(x='Neighborhood', y='SalePrice', data=train, ax=ax[0])
sns.boxplot(x='Neighborhood', y='SalePrice', data=train, ax=ax[1])
ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=90)
ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=90)
fig.suptitle('Neighborhood analysis')
plt.show()
plot_categorical('ExterQual')
plot_categorical('BsmtQual')
plot_categorical('KitchenQual')
fig, ax = plt.subplots(1,2,figsize=(16,6), sharey=True)
train_missing = round(train.isnull().mean()*100, 2)
train_missing = train_missing[train_missing > 0]
train_missing.sort_values(inplace=True)
sns.barplot(train_missing.index, train_missing, ax=ax[0], color='orange')
ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=90)
ax[0].set_ylabel('Percentage of missing values')
ax[0].set_title('Train set')
test_missing = round(test.isnull().mean()*100, 2)
test_missing = test_missing[test_missing > 0]
test_missing.sort_values(inplace=True)
sns.barplot(test_missing.index, test_missing, ax=ax[1], color='orange')
ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=90)
ax[1].set_title('Test set')
plt.show()
plot_numerical('LotFrontage')
print('LotFrontage minimum:', train.LotFrontage.min())
plot_categorical('FireplaceQu')
fig, ax = plt.subplots(2,2,figsize=(12,10), sharey=True)
sns.stripplot(x='Fence', y='SalePrice', data=train, ax=ax[0][0])
sns.stripplot(x='Alley', y='SalePrice', data=train, ax=ax[0][1])
sns.stripplot(x='MiscFeature', y='SalePrice', data=train, ax=ax[1][0])
sns.stripplot(x='PoolQC', y='SalePrice', data=train, ax=ax[1][1])
fig.suptitle('Analysis of columns with more than 80% of missing values')
plt.show()
# Gereksiz olan ID sütununu çıkarıyorum.

train.drop('Id', axis=1, inplace=True)
test.drop('Id', axis=1, inplace=True)

train_features = train
test_features = test
# Ön işleme işlemleri için train ve test veri setini birleştirme.

features = pd.concat([train, test]).reset_index(drop=True)
print(features.shape)
def missing_values_table(dataframe):
    variables_with_na = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[variables_with_na].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[variables_with_na].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    dtypes = dataframe.dtypes
    dtypesna = dtypes.loc[(np.sum(features.isnull()) != 0)]
    missing_df = pd.concat([n_miss, np.round(ratio, 2), dtypesna], axis=1, keys=['n_miss', 'ratio', 'type'])
    if len(missing_df)>0:
        print(missing_df)
        missing = train.isnull().sum()
        missing = missing[missing > 0]
        missing.sort_values(inplace=True)
        missing.plot.bar()
        print("\nThere are {} columns with missing values\n".format(len(missing_df)))
    else:
        print("\nThere is no missing value")
missing_values_table(features)
# Buradaki eksik gözlemler o özelliğin olmadığı anlamına gelmekte. Bu yüzden None atayacağım.

none_cols = ['Alley', 'PoolQC', 'MiscFeature', 'Fence', 'FireplaceQu', 'GarageType',
             'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtQual', 'BsmtCond',
             'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'MasVnrType']

# Sayısal değişkenlerdeki eksik gözlemler de aynı şekilde, bunlara 0 atıyorum.

zero_cols = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath','BsmtHalfBath', 'GarageYrBlt', 'GarageArea', 'GarageCars', 'MasVnrArea']

# Bu değişkenlerdeki eksik gözlemlere mod atayacağım.

freq_cols = ['Electrical', 'Exterior1st', 'Exterior2nd', 'Functional', 'KitchenQual','SaleType', 'Utilities']


for col in zero_cols:
    features[col].replace(np.nan, 0, inplace=True)

for col in none_cols:
    features[col].replace(np.nan, 'None', inplace=True)

for col in freq_cols:
    features[col].replace(np.nan, features[col].mode()[0], inplace=True)
# MsZoning değişkenindeki boş değerleri MSSubClassa göre doldurma.

features['MSZoning'] = features.groupby('MSSubClass')['MSZoning'].apply(
    lambda x: x.fillna(x.mode()[0]))

# LotFrontage mülkiyetin cadde ile bağlantısını gösteren bir değişken, her mahallenin cadde bağlantısının birbirine benzeyebileceğinden bunu Neighborhood'a a göre doldurdum.

features['LotFrontage'] = features.groupby(
    ['Neighborhood'])['LotFrontage'].apply(lambda x: x.fillna(x.median()))

# Sayısal değişken olup aslında kategorik değişken olması gerekenleri düzeltme

features['MSSubClass'] = features['MSSubClass'].astype(str)
features['YrSold'] = features['YrSold'].astype(str)
features['MoSold'] = features['MoSold'].astype(str)
missing_values_table(features)
def stalk(dataframe, var, target="SalePrice"):
    print("{}  | type: {}\n".format(var, dataframe[var].dtype))
    print(pd.DataFrame({"n": dataframe[var].value_counts(),
                                "Ratio": 100 * dataframe[var].value_counts() / len(dataframe),
                                "TARGET_MEDIAN": dataframe.groupby(var)[target].median(),
                                "Target_MEAN": dataframe.groupby(var)[target].mean()}), end="\n\n\n")
    
    plt.figure(figsize=(15,5))
    chart = sns.countplot(
    data=features,
    x=features[var])
    chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment='right')
    plt.show();
stalk(features, "Neighborhood")
# Neighboor içerisindeki benzer değerde olanları birbiri ile grupladım.

neigh_map = {'MeadowV': 1,'IDOTRR': 1,'BrDale': 1,'BrkSide': 2,'OldTown': 2,'Edwards': 2,
             'Sawyer': 3,'Blueste': 3,'SWISU': 3,'NPkVill': 3,'NAmes': 3,'Mitchel': 4,
             'SawyerW': 5,'NWAmes': 5,'Gilbert': 5,'Blmngtn': 5,'CollgCr': 5,
             'ClearCr': 6,'Crawfor': 6,'Veenker': 7,'Somerst': 7,'Timber': 8,
             'StoneBr': 9,'NridgHt': 10,'NoRidge': 10}

features['Neighborhood'] = features['Neighborhood'].map(neigh_map).astype('int')
# Derecelendirme içeren değişkenleri ordinal yapıya getirdim.

ext_map = {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
features['ExterQual'] = features['ExterQual'].map(ext_map).astype('int')

ext_map = {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
features['ExterCond'] = features['ExterCond'].map(ext_map).astype('int')

bsm_map = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
features['BsmtQual'] = features['BsmtQual'].map(bsm_map).astype('int')
features['BsmtCond'] = features['BsmtCond'].map(bsm_map).astype('int')

bsmf_map = {'None': 0,'Unf': 1,'LwQ': 2,'Rec': 3,'BLQ': 4,'ALQ': 5,'GLQ': 6}
features['BsmtFinType1'] = features['BsmtFinType1'].map(bsmf_map).astype('int')
features['BsmtFinType2'] = features['BsmtFinType2'].map(bsmf_map).astype('int')

heat_map = {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
features['HeatingQC'] = features['HeatingQC'].map(heat_map).astype('int')
features['KitchenQual'] = features['KitchenQual'].map(heat_map).astype('int')
features['FireplaceQu'] = features['FireplaceQu'].map(bsm_map).astype('int')
features['GarageCond'] = features['GarageCond'].map(bsm_map).astype('int')
features['GarageQual'] = features['GarageQual'].map(bsm_map).astype('int')
# RARE ANALYZER
def rare_analyser(dataframe, target, rare_perc):
    rare_columns = [col for col in dataframe.columns if len(dataframe[col].value_counts()) <= 20
                    and (dataframe[col].value_counts() / len(dataframe) < rare_perc).any(axis=None)]
    for var in rare_columns:
        print(var, ":", len(dataframe[var].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[var].value_counts(),
                            "RATIO": dataframe[var].value_counts() / len(dataframe),
                            "TARGET_MEDIAN": dataframe.groupby(var)[target].median()}), end="\n\n\n")


rare_analyser(features, "SalePrice", 0.01)
def stalk(dataframe, var, target="SalePrice"):
    print("{}  | type: {}\n".format(var, dataframe[var].dtype))
    print(pd.DataFrame({"n": dataframe[var].value_counts(),
                                "Ratio": 100 * dataframe[var].value_counts() / len(dataframe),
                                "TARGET_MEDIAN": dataframe.groupby(var)[target].median(),
                                "Target_MEAN": dataframe.groupby(var)[target].mean()}), end="\n\n\n")
    
    plt.figure(figsize=(10,5))
    chart = sns.countplot(
    data=features,
    x=features[var])
    chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment='right')
    plt.show();
stalk(features,"LotShape")
features.loc[(features["LotShape"] == "Reg"), "LotShape"] = 1
features.loc[(features["LotShape"] == "IR1"), "LotShape"] = 2
features.loc[(features["LotShape"] == "IR2"), "LotShape"] = 3 
features.loc[(features["LotShape"] == "IR3"), "LotShape"] = 3 

features["LotShape"] = features["LotShape"].astype("int")
stalk(features,"GarageCars")
features.loc[(features["GarageCars"] == "4"), "GarageCars"] = 3
stalk(features,"LotConfig")
features.loc[(features["LotConfig"]=="Inside"),"LotConfig"] = 1
features.loc[(features["LotConfig"]=="FR2"),"LotConfig"] = 1
features.loc[(features["LotConfig"]=="Corner"),"LotConfig"] = 1

features.loc[(features["LotConfig"]=="FR3"),"LotConfig"] = 2
features.loc[(features["LotConfig"]=="CulDSac"),"LotConfig"] = 2
stalk(features, "LandSlope")
features.loc[features["LandSlope"] == "Gtl", "LandSlope"] = 1

features.loc[features["LandSlope"] == "Sev", "LandSlope"] = 2
features.loc[features["LandSlope"] == "Mod", "LandSlope"] = 2
features["LandSlope"]= features["LandSlope"].astype("int")
stalk(features,"OverallQual")
features.loc[features["OverallQual"] == 1, "OverallQual"] = 1
features.loc[features["OverallQual"] == 2, "OverallQual"] = 1
features.loc[features["OverallQual"] == 3, "OverallQual"] = 1
features.loc[features["OverallQual"] == 4, "OverallQual"] = 2
features.loc[features["OverallQual"] == 5, "OverallQual"] = 3
features.loc[features["OverallQual"] == 6, "OverallQual"] = 4
features.loc[features["OverallQual"] == 7, "OverallQual"] = 5
features.loc[features["OverallQual"] == 8, "OverallQual"] = 6
features.loc[features["OverallQual"] == 9, "OverallQual"] = 7
features.loc[features["OverallQual"] == 10, "OverallQual"] = 8
stalk(features,"Exterior1st")
stalk(features,"MasVnrType")
features.loc[features["MasVnrType"] == "BrkCmn" , "MasVnrType"] = "None" 
stalk(features,"Foundation")
features.loc[features["Foundation"] == "Stone", "Foundation"] = "BrkTil"
features.loc[features["Foundation"] == "Wood", "Foundation"] = "CBlock"
stalk(features,"Fence")
features.loc[features["Fence"] == "MnWw", "Fence"] = "MnPrv"
features.loc[features["Fence"] == "GdWo", "Fence"] = "MnPrv"
# RARE ANALYZER
def rare_analyser(dataframe, target, rare_perc):
    rare_columns = [col for col in dataframe.columns if len(dataframe[col].value_counts()) <= 20
                    and (dataframe[col].value_counts() / len(dataframe) < rare_perc).any(axis=None)]
    for var in rare_columns:
        print(var, ":", len(dataframe[var].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[var].value_counts(),
                            "RATIO": dataframe[var].value_counts() / len(dataframe),
                            "TARGET_MEDIAN": dataframe.groupby(var)[target].median()}), end="\n\n\n")


rare_analyser(features, "SalePrice", 0.01)
def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()
    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]
    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])
    return temp_df

features = rare_encoder(features, 0.01)

# Plotting numerical features with polynomial order to detect outliers by eye.

def srt_reg(y, df):
    fig, axes = plt.subplots(12, 3, figsize=(25, 80))
    axes = axes.flatten()

    for i, j in zip(df.select_dtypes(include=['number']).columns, axes):

        sns.regplot(x=i,
                    y=y,
                    data=df,
                    ax=j,
                    order=3,
                    ci=None,
                    color='#25B89B',
                    line_kws={'color': 'grey'},
                    scatter_kws={'alpha':0.4})
        j.tick_params(labelrotation=45)
        j.yaxis.set_major_locator(MaxNLocator(nbins=8))

        plt.tight_layout()
srt_reg('SalePrice', train)
features.shape
# Dropping outliers after detecting them by eye.
features.loc[2590, 'GarageYrBlt'] = 2007 # missing value it was 2207

features = features.drop(features[(features['OverallQual'] < 5)
                                  & (features['SalePrice'] > 200000)].index)
features = features.drop(features[(features['GrLivArea'] > 4000)
                                  & (features['SalePrice'] < 200000)].index)
features = features.drop(features[(features['GarageArea'] > 1200)
                                  & (features['SalePrice'] < 200000)].index)
features = features.drop(features[(features['TotalBsmtSF'] > 3000)
                                  & (features['SalePrice'] > 320000)].index)
features = features.drop(features[(features['1stFlrSF'] < 3000)
                                  & (features['SalePrice'] > 600000)].index)
features = features.drop(features[(features['1stFlrSF'] > 3000)
                                  & (features['SalePrice'] < 200000)].index)

# Dropping target value
y = features['SalePrice']
y.dropna(inplace=True)
features.drop(columns='SalePrice', inplace=True)
features.shape
# Creating new features  based on previous observations. There might be some highly correlated features now. You cab drop them if you want to...

features['TotalSF'] = (features['BsmtFinSF1'] + features['BsmtFinSF2'] +
                       features['1stFlrSF'] + features['2ndFlrSF'])
features['TotalBathrooms'] = (features['FullBath'] +
                              (0.5 * features['HalfBath']) +
                              features['BsmtFullBath'] +
                              (0.5 * features['BsmtHalfBath']))

features['TotalPorchSF'] = (features['OpenPorchSF'] + features['3SsnPorch'] +
                            features['EnclosedPorch'] +
                            features['ScreenPorch'] + features['WoodDeckSF'])

features['YearBlRm'] = (features['YearBuilt'] + features['YearRemodAdd'])

# Merging quality and conditions.

features['TotalExtQual'] = (features['ExterQual'] + features['ExterCond'])
features['TotalBsmQual'] = (features['BsmtQual'] + features['BsmtCond'] +
                            features['BsmtFinType1'] +
                            features['BsmtFinType2'])
features['TotalGrgQual'] = (features['GarageQual'] + features['GarageCond'])
features['TotalQual'] = features['OverallQual'] + features[
    'TotalExtQual'] + features['TotalBsmQual'] + features[
        'TotalGrgQual'] + features['KitchenQual'] + features['HeatingQC']

# Creating new features by using new quality indicators.

features['QualGr'] = features['TotalQual'] * features['GrLivArea']
features['QualBsm'] = features['TotalBsmQual'] * (features['BsmtFinSF1'] +
                                                  features['BsmtFinSF2'])
features['QualPorch'] = features['TotalExtQual'] * features['TotalPorchSF']
features['QualExt'] = features['TotalExtQual'] * features['MasVnrArea']
features['QualGrg'] = features['TotalGrgQual'] * features['GarageArea']
features['QlLivArea'] = (features['GrLivArea'] -
                         features['LowQualFinSF']) * (features['TotalQual'])
features['QualSFNg'] = features['QualGr'] * features['Neighborhood']

features["new_home"] = features["YearBuilt"]
features.loc[features["new_home"] == features["YearRemodAdd"], "new_home"] = 0
features.loc[features["new_home"] != features["YearRemodAdd"], "new_home"] = 1

# Creating some simple features.

features['HasPool'] = features['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
features['Has2ndFloor'] = features['2ndFlrSF'].apply(lambda x: 1
                                                     if x > 0 else 0)
features['HasGarage'] = features['QualGrg'].apply(lambda x: 1 if x > 0 else 0)
features['HasBsmt'] = features['QualBsm'].apply(lambda x: 1 if x > 0 else 0)
features['HasFireplace'] = features['Fireplaces'].apply(lambda x: 1
                                                        if x > 0 else 0)
features['HasPorch'] = features['QualPorch'].apply(lambda x: 1 if x > 0 else 0)
# Observing the effects of newly created features on sale price.

def srt_reg(feature):
    merged = features.join(y)
    fig, axes = plt.subplots(5, 3, figsize=(25, 40))
    axes = axes.flatten()

    new_features = [
        'TotalSF', 'TotalBathrooms', 'TotalPorchSF', 'YearBlRm',
        'TotalExtQual', 'TotalBsmQual', 'TotalGrgQual', 'TotalQual', 'QualGr',
        'QualBsm', 'QualPorch', 'QualExt', 'QualGrg', 'QlLivArea', 'QualSFNg'
    ]

    for i, j in zip(new_features, axes):

        sns.regplot(x=i,
                    y=feature,
                    data=merged,
                    ax=j,
                    order=3,
                    ci=None,
                    color='#25B89B',
                    line_kws={'color': 'grey'},
                    scatter_kws={'alpha':0.4})
        j.tick_params(labelrotation=45)
        j.yaxis.set_major_locator(MaxNLocator(nbins=10))

        plt.tight_layout()



srt_reg('SalePrice')

skewed = [
    'LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2',
    'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea',
    'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
    'ScreenPorch', 'PoolArea', 'LowQualFinSF', 'MiscVal'
]
# Skewnesslık derecesini bulma

skew_features = np.abs(features[skewed].apply(lambda x: skew(x)).sort_values(
    ascending=False))

# Skewnesslık derecesine göre filtreleme

high_skew = skew_features[skew_features > 0.3]

# Yüksek skewnesslığa sahip olanların indexini alma

skew_index = high_skew.index

# Yüksek skewnessa sahip değişkenlere Box-Cox Transformation uygulanması

for i in skew_index:
    features[i] = boxcox1p(features[i], boxcox_normmax(features[i] + 1))
# Atılacaklar listesi

to_drop = ['Utilities','PoolQC','YrSold','MoSold','ExterQual','BsmtQual','GarageQual','KitchenQual','HeatingQC',]

features.drop(columns=to_drop, inplace=True)
# Kategorik değişkenlere O.H.E uyguluyorum.
# Normalde regresyon modellerinde First-Drop = True yapılması gerekiyor fakat, bazı ağaç modelleri de kullanacağım için ilk dummy'leri atmıyorum. 
# Hatta bunun biraz puanımı arttırdığımı da söyleyebilirim.

features = pd.get_dummies(data=features)
print(f'Toplam eksik gözlem sayısı: {features.isna().sum().sum()}')
features.shape
# train ve test ayırma
submission_model = features.copy()
train = features.iloc[:len(y), :]
test = features.iloc[len(train):, :]
correlations = train.join(y).corrwith(train.join(y)['SalePrice']).iloc[:-1].to_frame()
correlations['Abs Corr'] = correlations[0].abs()
sorted_correlations = correlations.sort_values('Abs Corr', ascending=False)['Abs Corr']
fig, ax = plt.subplots(figsize=(12,12))
sns.heatmap(sorted_correlations.to_frame()[sorted_correlations>=.5], cmap='coolwarm', annot=True, vmin=-1, vmax=1, ax=ax);
def plot_dist3(df, feature, title):
    
    # Creating a customized chart. and giving in figsize and everything.
    
    fig = plt.figure(constrained_layout=True, figsize=(12, 8))
    
    # creating a grid of 3 cols and 3 rows.
    
    grid = gridspec.GridSpec(ncols=3, nrows=3, figure=fig)

    # Customizing the histogram grid.
    
    ax1 = fig.add_subplot(grid[0, :2])
    
    # Set the title.
    
    ax1.set_title('Histogram')
    
    # plot the histogram.
    
    sns.distplot(df.loc[:, feature],
                 hist=True,
                 kde=True,
                 fit=norm,
                 ax=ax1,
                 color='#e74c3c')
    ax1.legend(labels=['Normal', 'Actual'])

    # customizing the QQ_plot.
    
    ax2 = fig.add_subplot(grid[1, :2])
    
    # Set the title.
    
    ax2.set_title('Probability Plot')
    
    # Plotting the QQ_Plot.
    stats.probplot(df.loc[:, feature].fillna(np.mean(df.loc[:, feature])),
                   plot=ax2)
    ax2.get_lines()[0].set_markerfacecolor('#e74c3c')
    ax2.get_lines()[0].set_markersize(12.0)

    # Customizing the Box Plot:
    
    ax3 = fig.add_subplot(grid[:, 2])
    # Set title.
    
    ax3.set_title('Box Plot')
    
    # Plotting the box plot.
    
    sns.boxplot(df.loc[:, feature], orient='v', ax=ax3, color='#e74c3c')
    ax3.yaxis.set_major_locator(MaxNLocator(nbins=24))

    plt.suptitle(f'{title}', fontsize=24)
# Checking target variable.

plot_dist3(train.join(y), 'SalePrice', 'Log Dönüşüm Öncesi Sale Price')
# Setting model data.

X_my = train
X_test_my = test
y_ = y
y_log =y
y_log = np.log1p(y_log)
plot_dist3(train.join(y_log), 'SalePrice', 'Log Dönüşüm Sonrası Sale Price')
X_my = RobustScaler().fit_transform(X_my)
X_test_my = RobustScaler().fit_transform(X_test_my)
X_train, X_test, y_train, y_test = train_test_split(X_my, y_log, test_size=0.20, random_state=46)
### NON LINEAR MODELS
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import pickle
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)

models = [('LR', LinearRegression()),
          ("Ridge", Ridge()),
          ("Lasso", Lasso()),
          ("ElasticNet", ElasticNet()),
          ('RF', RandomForestRegressor()),
          ('GBM', GradientBoostingRegressor()),
          ("XGBoost", XGBRegressor()),
          ("LightGBM", LGBMRegressor()),
          ("CatBoost", CatBoostRegressor(verbose=False))]

# Base modellerin test hataları
print("Base Test RMSE")
for name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmsLe = np.sqrt(mean_squared_error(y_test, y_pred))
    msg = "%s: %f" % (name, rmsLe)
    print(msg)
# Base modellerin test hataları
print("Base Train RMSE")
for name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_train)
    rmsLe = np.sqrt(mean_squared_error(y_train, y_pred))
    msg = "%s: %f" % (name, rmsLe)
    print(msg)
X_train, X_test, y_train, y_test = train_test_split(X_my, y_, test_size=0.20, random_state=46)
models = [('LR', LinearRegression()),
          ("Ridge", Ridge()),
          ("Lasso", Lasso()),
          ("ElasticNet", ElasticNet()),
          ('RF', RandomForestRegressor()),
          ('GBM', GradientBoostingRegressor()),
          ("XGBoost", XGBRegressor()),
          ("LightGBM", LGBMRegressor()),
          ("CatBoost", CatBoostRegressor(verbose=False))]

# Base modellerin test hataları
print("Test RMSE")
for name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmsLe = np.sqrt(mean_squared_error(y_test, y_pred))
    msg = "%s: %f" % (name, rmsLe)
    print(msg)
# Base modellerin train hataları
print("Train RMSE")
for name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_train)
    rmsLe = np.sqrt(mean_squared_error(y_train, y_pred))
    msg = "%s: %f" % (name, rmsLe)
    print(msg)
X = train
X_test = test
y = np.log1p(y)
# Loading neccesary packages for modelling.

from sklearn.model_selection import cross_val_score, KFold, cross_validate
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV, TweedieRegressor
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from mlxtend.regressor import StackingCVRegressor # This is for stacking part, works well with sklearn and others...
# Setting kfold for future use.

kf = KFold(10, random_state=42)
# Some parameters for ridge, lasso and elasticnet.

alphas_alt = [15.5, 15.6, 15.7, 15.8, 15.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5]
alphas2 = [
    5e-05, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008
]
e_alphas = [
    0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007
]
e_l1ratio = [0.8, 0.85, 0.9, 0.95, 0.99, 1]

# ridge_cv

ridge = make_pipeline(RobustScaler(), RidgeCV(
    alphas=alphas_alt,
    cv=kf,
))

# lasso_cv:

lasso = make_pipeline(
    RobustScaler(),
    LassoCV(max_iter=1e7, alphas=alphas2, random_state=42, cv=kf))

# elasticnet_cv:

elasticnet = make_pipeline(
    RobustScaler(),
    ElasticNetCV(max_iter=1e7,
                 alphas=e_alphas,
                 cv=kf,
                 random_state=42,
                 l1_ratio=e_l1ratio))

# svr:

svr = make_pipeline(RobustScaler(),
                    SVR(C=21, epsilon=0.0099, gamma=0.00017, tol=0.000121))

# gradientboosting:

gbr = GradientBoostingRegressor(n_estimators=2900,
                                learning_rate=0.0161,
                                max_depth=4,
                                max_features='sqrt',
                                min_samples_leaf=17,
                                loss='huber',
                                random_state=42)

# lightgbm:

lightgbm = LGBMRegressor(objective='regression',
                         n_estimators=3500,
                         num_leaves=5,
                         learning_rate=0.00721,
                         max_bin=163,
                         bagging_fraction=0.35711,
                         n_jobs=-1,
                         bagging_seed=42,
                         feature_fraction_seed=42,
                         bagging_freq=7,
                         feature_fraction=0.1294,
                         min_data_in_leaf=8)

# xgboost:

xgboost = XGBRegressor(
    learning_rate=0.0139,
    n_estimators=4500,
    max_depth=4,
    min_child_weight=0,
    subsample=0.7968,
    colsample_bytree=0.4064,
    nthread=-1,
    scale_pos_weight=2,
    seed=42,
)


# hist gradient boosting regressor:

hgrd= HistGradientBoostingRegressor(    loss= 'least_squares',
    max_depth= 2,
    min_samples_leaf= 40,
    max_leaf_nodes= 29,
    learning_rate= 0.15,
    max_iter= 225,
                                    random_state=42)

# tweedie regressor:
 
tweed = make_pipeline(RobustScaler(),TweedieRegressor(alpha=0.005))


# stacking regressor:

stack_gen = StackingCVRegressor(regressors=(ridge, lasso, elasticnet, gbr,
                                            xgboost, lightgbm,hgrd, tweed),
                                meta_regressor=xgboost,
                                use_features_in_secondary=True)
def model_check(X, y, estimators, cv):
    
    ''' A function for testing multiple estimators.'''
    
    model_table = pd.DataFrame()

    row_index = 0
    for est, label in zip(estimators, labels):

        MLA_name = label
        model_table.loc[row_index, 'Model Name'] = MLA_name

        cv_results = cross_validate(est,
                                    X,
                                    y,
                                    cv=cv,
                                    scoring='neg_root_mean_squared_error',
                                    return_train_score=True,
                                    n_jobs=-1)

        model_table.loc[row_index, 'Train RMSE'] = -cv_results[
            'train_score'].mean()
        model_table.loc[row_index, 'Test RMSE'] = -cv_results[
            'test_score'].mean()
        model_table.loc[row_index, 'Test Std'] = cv_results['test_score'].std()
        model_table.loc[row_index, 'Time'] = cv_results['fit_time'].mean()

        row_index += 1

    model_table.sort_values(by=['Test RMSE'],
                            ascending=True,
                            inplace=True)

    return model_table
# Setting list of estimators and labels for them:

estimators = [ridge, lasso, elasticnet, gbr, xgboost, lightgbm, svr, hgrd, tweed]
labels = [
    'Ridge', 'Lasso', 'Elasticnet', 'GradientBoostingRegressor',
    'XGBRegressor', 'LGBMRegressor', 'SVR', 'HistGradientBoostingRegressor','TweedieRegressor'
]
# Executing cross validation.

raw_models = model_check(X, y, estimators, kf)
display(raw_models.style.background_gradient(cmap='summer'))
# Fitting the models on train data.

print('=' * 20, 'START Fitting', '=' * 20)
print('=' * 55)

print(datetime.now(), 'StackingCVRegressor')
stack_gen_model = stack_gen.fit(X.values, y.values)
print(datetime.now(), 'Elasticnet')
elastic_model_full_data = elasticnet.fit(X, y)
print(datetime.now(), 'Lasso')
lasso_model_full_data = lasso.fit(X, y)
print(datetime.now(), 'Ridge')
ridge_model_full_data = ridge.fit(X, y)
print(datetime.now(), 'SVR')
svr_model_full_data = svr.fit(X, y)
print(datetime.now(), 'GradientBoosting')
gbr_model_full_data = gbr.fit(X, y)
print(datetime.now(), 'XGboost')
xgb_model_full_data = xgboost.fit(X, y)
print(datetime.now(), 'Lightgbm')
lgb_model_full_data = lightgbm.fit(X, y)
print(datetime.now(), 'Hist')
hist_full_data = hgrd.fit(X, y)
print(datetime.now(), 'Tweed')
tweed_full_data = tweed.fit(X, y)
print('=' * 20, 'FINISHED Fitting', '=' * 20)
print('=' * 58)
# Blending models by assigning weights:

def blend_models_predict(X):
    return ((0.1 * elastic_model_full_data.predict(X)) +
            (0.1 * lasso_model_full_data.predict(X)) +
            (0.1 * ridge_model_full_data.predict(X)) +
            (0.1 * svr_model_full_data.predict(X)) +
            (0.05 * gbr_model_full_data.predict(X)) +
            (0.1 * xgb_model_full_data.predict(X)) +
            (0.05 * lgb_model_full_data.predict(X)) +
            (0.05 * hist_full_data.predict(X)) +
            (0.1 * tweed_full_data.predict(X)) +
            (0.25 * stack_gen_model.predict(X.values)))
submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
# Inversing and flooring log scaled sale price predictions
submission['SalePrice'] = np.floor(np.expm1(blend_models_predict(X_test)))
# Defining outlier quartile ranges
q1 = submission['SalePrice'].quantile(0.0050)
q2 = submission['SalePrice'].quantile(0.99)

# Applying weights to outlier ranges to smooth them
submission['SalePrice'] = submission['SalePrice'].apply(
    lambda x: x if x > q1 else x * 0.77)
submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x
                                                        if x < q2 else x * 1.1)
submission = submission[['Id', 'SalePrice']]
# Saving submission file

submission.to_csv('mysubmission.csv', index=False)
print(
    'Save submission',
    datetime.now(),
)
submission.head()





