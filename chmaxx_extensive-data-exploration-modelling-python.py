%matplotlib inline

%config InlineBackend.figure_format = 'retina'

import matplotlib.pyplot as plt



plt.style.use('ggplot')

import matplotlib.cm as cm

import seaborn as sns



import pandas as pd

import numpy as np

from numpy import percentile

from scipy import stats

from scipy.stats import skew

from scipy.special import boxcox1p



import os, sys

import calendar



from sklearn.preprocessing import StandardScaler, RobustScaler

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.impute import SimpleImputer

from sklearn.dummy import DummyRegressor

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline, make_pipeline

from sklearn.model_selection import cross_val_score, cross_validate

from sklearn.model_selection import GridSearchCV, KFold

from sklearn.decomposition import PCA

from sklearn.metrics import mean_squared_log_error, make_scorer

from sklearn.metrics.scorer import neg_mean_squared_error_scorer



from sklearn.linear_model import LinearRegression, ElasticNet, Lasso, Ridge, RidgeCV

from sklearn.svm import SVR

from sklearn.kernel_ridge import KernelRidge

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor



import xgboost as xgb

import lightgbm as lgb





import warnings

warnings.filterwarnings('ignore')



plt.rc('font', size=18)        

plt.rc('axes', titlesize=22)      

plt.rc('axes', labelsize=18)      

plt.rc('xtick', labelsize=12)     

plt.rc('ytick', labelsize=12)     

plt.rc('legend', fontsize=12)   



plt.rcParams['font.sans-serif'] = ['Verdana']



# function that converts to thousands

# optimizes visual consistence if we plot several graphs on top of each other

def format_1000(value, tick_number):

    return int(value / 1_000)



pd.options.mode.chained_assignment = None

pd.options.display.max_seq_items = 500

pd.options.display.max_rows = 500

pd.set_option('display.float_format', lambda x: '%.5f' % x)



BASE_PATH = "/kaggle/input/house-prices-advanced-regression-techniques/"
df = pd.read_csv(f"{BASE_PATH}train.csv")

df_test = pd.read_csv(f"{BASE_PATH}test.csv")

df.head()
df.info(verbose=True, null_counts=True)
missing = [(c, df[c].isna().mean()*100) for c in df]

missing = pd.DataFrame(missing, columns=["column_name", "percentage"])

missing = missing[missing.percentage > 0]

display(missing.sort_values("percentage", ascending=False))
plt.figure(figsize=(16,5))

df.SalePrice.plot(kind="hist", bins=100, rwidth=0.9)

plt.title("Sales Price value distribution")

plt.xlabel("Sales Price")

plt.tight_layout()

plt.show()



plt.figure(figsize=(16,5))

df.SalePrice.plot(kind="box", vert=False)

plt.title("Sales Price value distribution")

plt.xlabel("Sales Price")

plt.yticks([0], [''])

plt.ylabel("SalePrice\n", rotation=90)

plt.tight_layout()

plt.show()
# calculate percentiles and IQR

q25 = percentile(df.SalePrice, 25)

q75 = percentile(df.SalePrice, 75)

iqr = q75 - q25



# calculate normal and extreme upper and lower cut off

cut_off = iqr * 3

lower = q25 - cut_off 

upper = q75 + cut_off



print(f'Percentiles:\n25th  =  {q25}\n75th  =  {q75}\n\nIQR   =   {iqr}\nlower = {lower}\nupper =  {upper}')
df[df.SalePrice > upper]
print(f"The skewness of SalePrice is: {df.SalePrice.skew():.2f}")



plt.figure(figsize=(16,9))

_ = stats.probplot(df['SalePrice'], plot=plt)

plt.title("Probability plot: SalePrice")

plt.show()
display(df.select_dtypes("number").skew().sort_values(ascending=False))
plt.figure(figsize=(16, 9))

_ = stats.probplot(df['LotArea'], plot=plt)

plt.title("Probability plot: LotArea")

plt.show()
logSalePrice = np.log1p(df.SalePrice.values)

print(f"Skewness of log transformed sale prices: {pd.DataFrame(logSalePrice).skew().values[0]:.2f}")



plt.figure(figsize=(16,5));

plt.hist(logSalePrice, bins=100, rwidth=0.9)

plt.title("SalePrice log transformed")

plt.xlabel("SalePrice distribution – log transformed")

plt.show()
display(df.SalePrice.describe())
# we drop Id (not relevant)

corr = df.drop(["Id"], axis=1).select_dtypes(include="number").corr()



plt.figure(figsize=(16,16));

corr["SalePrice"].sort_values(ascending=True)[:-1].plot(kind="barh")

plt.title("Correlation of numerical features to SalePrice")

plt.xlabel("Correlation to SalePrice")

plt.tight_layout()

plt.show()
# sort by SalePrice to make more sense in row order

plt.subplots(figsize=(16,16));

sns.heatmap(corr, cmap="RdBu", square=True, cbar_kws={"shrink": .7})

plt.title("Correlation matrix of all numerical features\n")

plt.tight_layout()

plt.show()
# OverallQual: Rates the overall material and finish of the house

plt.figure(figsize=(16,5));

df.groupby("OverallQual")["SalePrice"].count().plot(kind="bar")

plt.title("Count of observations in overall quality categories («OverallQual»)")

plt.ylabel("Count")

plt.tight_layout()

plt.show()



plt.figure(figsize=(16,10));

ax = sns.boxplot(x="OverallQual", y="SalePrice", data=df)

plt.title("Distribution of sale prices in overall quality categories («OverallQual»)")

ax.yaxis.set_major_formatter(plt.FuncFormatter(format_1000))

plt.ylabel("SalePrice (in 1000)")

plt.tight_layout()

plt.show()
# OverallCond: Rates the overall condition of the house

plt.figure(figsize=(16,5));

df.groupby("OverallCond")["SalePrice"].count().plot(kind="bar")

plt.title("Count of observations in overall condition categories («OverallCond»)")

plt.ylabel("Count")

plt.tight_layout()

plt.show()



plt.figure(figsize=(16,9));

ax = sns.boxplot(x="OverallCond", y="SalePrice", data=df)

plt.title("Distribution of sale prices in overall condition categories («OverallCond»)")

ax.yaxis.set_major_formatter(plt.FuncFormatter(format_1000))

plt.ylabel("SalePrice (in 1000)")

plt.tight_layout()

plt.show()
plt.figure(figsize=(16,9));

sns.swarmplot(x="OverallQual", y="SalePrice", data=df)

plt.title("Swarm plot: Distribution & count of sale prices in overall *quality* categories")

plt.tight_layout()

plt.show()



plt.figure(figsize=(16,9));

sns.swarmplot(x="OverallCond", y="SalePrice", data=df)

plt.title("Swarm plot: Distribution & count of sale prices in overall *condition* categories")

plt.tight_layout()

plt.show()
plt.figure(figsize=(16,5));

sns.scatterplot(x="GrLivArea", y="SalePrice", data=df, linewidth=0.2, alpha=0.9)

plt.title(f"SalePrice vs. GrLivArea")

plt.tight_layout()

plt.show()
plt.figure(figsize=(16,5));

sns.scatterplot(x="GrLivArea", y="SalePrice", hue="OverallQual", data=df, 

                legend="full", linewidth=0.2, alpha=0.9)

plt.title(f"SalePrice vs. GrLivArea")

plt.legend(bbox_to_anchor=(1, 1), loc=2)

plt.tight_layout()

plt.show()
df_cut = pd.DataFrame(pd.cut(np.log(df.GrLivArea), bins=10, labels=np.arange(0,10)))

df_comb = pd.concat([df_cut, df.SalePrice], axis=1)



plt.figure(figsize=(16,5));

df_comb.groupby("GrLivArea").SalePrice.count().plot(kind="bar")

plt.title("Count of observations in living area (binned values)")

plt.ylabel("Count")

plt.tight_layout()

plt.show()



plt.figure(figsize=(16,9));

ax = sns.boxplot(x="GrLivArea", y="SalePrice", data=df_comb)

plt.title("Distribution of observations in living area (binned values)")

ax.yaxis.set_major_formatter(plt.FuncFormatter(format_1000))

plt.ylabel("SalePrice (in 1000)")

plt.tight_layout()

plt.show()
features = ["GarageArea", "TotalBsmtSF", "1stFlrSF", "LotArea"]



for feature in features:

    plt.figure(figsize=(16,5));

    sns.scatterplot(x=feature, y="SalePrice", hue="OverallQual", data=df, 

                legend="full", linewidth=0.2, alpha=0.9)

    plt.legend(bbox_to_anchor=(1, 1), loc=2)

    plt.title(f"SalePrice vs {feature}")

    plt.show()
plt.figure(figsize=(16,5));

df.groupby("YearBuilt").SalePrice.count().plot(kind="bar")

plt.title("Count of observations in build year")

plt.ylabel("Count")

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()



plt.figure(figsize=(16,8));

ax = sns.boxplot(x="YearBuilt", y="SalePrice", data=df)

plt.axis(ymin=0, ymax=800000)

plt.title("Distribution of SalePrice in build year")

ax.yaxis.set_major_formatter(plt.FuncFormatter(format_1000))

plt.ylabel("SalePrice (in 1000)")

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()
# bin years to decades since plotting every single year would clutter the plot

decades = np.arange(1870, 2015, 10)

df_cut = pd.cut(df.YearBuilt, bins=decades, labels=decades[:-1])

df_comb = pd.concat([df_cut, df.SalePrice], axis=1)



plt.figure(figsize=(16,5));

df_comb.groupby("YearBuilt").SalePrice.count().plot(kind="bar")

plt.title("Count of observations in build year (binned values to decades)")

plt.ylabel("Count")

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()



plt.figure(figsize=(16,8));

ax = sns.boxplot(x="YearBuilt", y="SalePrice", data=df_comb)

plt.axis(ymin=0, ymax=800000)

plt.title("Distribution of SalePrice in build year (binned values to decades)")

ax.yaxis.set_major_formatter(plt.FuncFormatter(format_1000))

plt.ylabel("SalePrice (in 1000)")

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()
plt.figure(figsize=(16,5));

df.groupby("YrSold").SalePrice.count().plot(kind="bar")

plt.title("Count of observations in years of sale")

plt.ylabel("Count")

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()



plt.figure(figsize=(16,8));

ax = sns.boxplot(x="YrSold", y="SalePrice", data=df)

plt.axis(ymin=0, ymax=800000)

plt.title("Distribution of SalePrice in years of sale")

ax.yaxis.set_major_formatter(plt.FuncFormatter(format_1000))

plt.ylabel("SalePrice (in 1000)")

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()



months_names = calendar.month_name[1:13]



plt.figure(figsize=(16,5));

df.groupby("MoSold").SalePrice.count().plot(kind="bar")

plt.title("Count of observations in months of sale")

plt.ylabel("Count")

plt.xticks(ticks=np.arange(0, 12), labels=months_names, rotation=45)

plt.tight_layout()

plt.show()



plt.figure(figsize=(16,8));

ax = sns.boxplot(x="MoSold", y="SalePrice", data=df)

plt.axis(ymin=0, ymax=800000)

plt.title("Distribution of SalePrice in months of sale")

ax.yaxis.set_major_formatter(plt.FuncFormatter(format_1000))

plt.ylabel("SalePrice (in 1000)")

plt.xticks(ticks=np.arange(0, 12), labels=months_names, rotation=45)

plt.tight_layout()

plt.show()
df["Age"] = df.YrSold - df.YearBuilt

print(df.Age.describe())
decades = np.arange(0, 136, 10)

df_cut = pd.cut(df.Age, bins=decades, labels=decades[:-1])

df_comb = pd.concat([df_cut, df.SalePrice], axis=1)



plt.figure(figsize=(16,5));

df_comb.groupby("Age").SalePrice.count().plot(kind="bar")

plt.title("Count of observations of property age (binned to decades)")

plt.ylabel("Count")

plt.xticks(rotation=45)

plt.gca().invert_xaxis()

plt.tight_layout()

plt.show()



plt.figure(figsize=(16,8));

ax = sns.boxplot(x="Age", y="SalePrice", data=df_comb)

plt.axis(ymin=0, ymax=800000)

plt.title("Distribution of SalePrice respective property age (binned to decades)")

ax.yaxis.set_major_formatter(plt.FuncFormatter(format_1000))

plt.ylabel("SalePrice (in 1000)")

plt.xticks(rotation=45)

plt.gca().invert_xaxis()

plt.tight_layout()

plt.show()
# MSSubClass: Identifies the type of dwelling involved in the sale.

order = df.groupby("MSSubClass").SalePrice.mean().sort_values(ascending=False).index



plt.figure(figsize=(16,5));

df_g = df.groupby("MSSubClass").SalePrice.count()

df_g = df_g.reindex(order)

df_g.plot(kind="bar")

plt.title("Count of observations for type of dwelling")

plt.ylabel("Count")

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()



plt.figure(figsize=(16,8));

ax = sns.boxplot(x="MSSubClass", y="SalePrice", data=df, order=order)

plt.axis(ymin=0, ymax=800000)

plt.title(f"Distribution of SalePrice for type of dwelling")

ax.yaxis.set_major_formatter(plt.FuncFormatter(format_1000))

plt.ylabel("SalePrice (in 1000)")

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()
# BldgType: Type of dwelling

order = df.groupby("BldgType").SalePrice.mean().sort_values(ascending=False).index



plt.figure(figsize=(16,5));

df_g = df.groupby("BldgType").SalePrice.count()

df_g = df_g.reindex(order)

df_g.plot(kind="bar")

plt.title("Count of observations for type of dwelling («BldgType»)")

plt.ylabel("Count")

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()



plt.figure(figsize=(16,8));

ax = sns.boxplot(x="BldgType", y="SalePrice", data=df, order=order)

plt.axis(ymin=0, ymax=800000)

plt.title(f"Distribution of SalePrice for type of dwelling («BldgType»)")

ax.yaxis.set_major_formatter(plt.FuncFormatter(format_1000))

plt.ylabel("SalePrice (in 1000)")

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()
# create descending list of neighborhood names according to mean of sale price

# used to sort columns in plots

order = df.groupby("Neighborhood").SalePrice.mean().sort_values(ascending=False).index



plt.figure(figsize=(16,5));

df_g = df.groupby("Neighborhood").SalePrice.count().sort_values(ascending=False)

df_g = df_g.reindex(order)

df_g.plot(kind="bar")

plt.title("Neighborhood, count of observations")

plt.ylabel("Count")

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()



plt.figure(figsize=(16,8));

ax = sns.boxplot(x="Neighborhood", y="SalePrice", data=df, order=order)

plt.axis(ymin=0, ymax=800000)

plt.title(f"Neighborhood, distribution of SalePrice")

ax.yaxis.set_major_formatter(plt.FuncFormatter(format_1000))

plt.ylabel("SalePrice (in 1000)")

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()
order = df.groupby("SaleType").SalePrice.mean().sort_values(ascending=False).index



plt.figure(figsize=(16,5));

df_g = df.groupby("SaleType").SalePrice.count().sort_values(ascending=False)

df_g = df_g.reindex(order)

df_g.plot(kind="bar")

plt.title("SaleType, count of observations")

plt.ylabel("Count")

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()



plt.figure(figsize=(16,8));

ax = sns.boxplot(x="SaleType", y="SalePrice", data=df, order=order)

plt.axis(ymin=0, ymax=800000)

plt.title(f"SaleType, distribution of SalePrice")

ax.yaxis.set_major_formatter(plt.FuncFormatter(format_1000))

plt.ylabel("SalePrice (in 1000)")

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()
order = df.groupby("SaleCondition").SalePrice.mean().sort_values(ascending=False).index



plt.figure(figsize=(16,5));

df_g = df.groupby("SaleCondition").SalePrice.count().sort_values(ascending=False)

df_g = df_g.reindex(order)

df_g.plot(kind="bar")

plt.title("SaleCondition, count of observations")

plt.ylabel("Count")

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()



plt.figure(figsize=(16,8));

ax = sns.boxplot(x="SaleCondition", y="SalePrice", data=df, order=order)

plt.axis(ymin=0, ymax=800000)

plt.title(f"SaleCondition, distribution of SalePrice")

ax.yaxis.set_major_formatter(plt.FuncFormatter(format_1000))

plt.ylabel("SalePrice (in 1000)")

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()
# start fresh with again reading in the data

df = pd.read_csv(f"{BASE_PATH}train.csv")

df_test = pd.read_csv(f"{BASE_PATH}test.csv")



# concat all samples to one dataframe for cleaning

# need to be careful not to leak data from test to training set! 

# e.g. by filling missing data with mean of *all* samples rather than training samples only

feat = pd.concat([df, df_test]).reset_index(drop=True).copy()
fig, axes = plt.subplots(nrows=18, ncols=2, figsize=(16,36))

num = feat.drop(["Id", "SalePrice"], axis=1).select_dtypes("number")



for idx, column in enumerate(num.columns[1:]):

    num[column].plot(kind="hist", bins=100, rwidth=.9, title=column, ax=axes[idx//2, idx%2])

    ax=axes[idx//2, idx%2].yaxis.label.set_visible(False)



plt.tight_layout()

plt.show()
# get columns with NaN values

missing = feat.columns[feat.isna().any()]

print(missing)
# fix missing values in features



# Alley: NA means no alley acces so we fill with string «None»

feat.Alley = feat.Alley.fillna("None")



# BsmtQual et al – NA for features means "no basement", filling with string "None"

bsmt_cols = ['BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'BsmtQual']

for col in bsmt_cols:

    feat[col] = feat[col].fillna("None")



# Basement sizes: NaN likely means 0, can be set to int

for col in ['BsmtFullBath', 'BsmtHalfBath', 'BsmtUnfSF', 'BsmtFinSF1', 'BsmtFinSF2', 'TotalBsmtSF']:

    feat[col] = feat[col].fillna(0).astype(int)

    

# Electrical: NA likely means unknown, filling with most frequent value SBrkr

feat.Electrical = feat.Electrical.fillna("SBrkr")



# Exterior1st: NA likely means unknown, filling with most frequent value VinylSd

feat.Exterior1st = feat.Exterior1st.fillna("VinylSd")



# Exterior2nd: NA likely means no 2nd material, filling with «None»

feat.Exterior2nd = feat.Exterior2nd.fillna("None")



# Fence: NA means «No Fence» filling with «None»

feat.Fence = feat.Fence.fillna("None")



# FireplaceQu: NA means «No Fireplace» filling with «None»

feat.FireplaceQu = feat.FireplaceQu.fillna("None")



# Functional: NA means «typical» filling with «Typ»

feat.Functional = feat.Functional.fillna("Typ")



# GarageType et al – NA means "no garage", filling with string "None"

grg_cols = ['GarageCond', 'GarageFinish', 'GarageQual', 'GarageType']

for col in grg_cols:

    feat[col] = feat[col].fillna("None")



# Garage sizes: NaN means «no garage» == 0, unsure if build year should be 0?

for col in ['GarageArea', 'GarageCars', 'GarageYrBlt']:

    feat[col] = feat[col].fillna(0).astype(int)



# fix one outlier GarageYrBlt == 2207

to_fix = feat[feat.GarageYrBlt == 2207].index

feat.loc[to_fix, "GarageYrBlt"] = int(feat.GarageYrBlt.mean())

    

# KitchenQual: filling NaNs with most frequent value «Typical/Average» («TA»)

feat.KitchenQual = feat.KitchenQual.fillna("TA")



# LotFrontage can be set to integer, filling missing values with 0

feat.LotFrontage = feat.LotFrontage.fillna(0).astype(int)



# MSZoning filling NaNs with most frequent value «RL» (residental low density)

feat.MSZoning = feat.MSZoning.fillna("RL")



# MSSubClass is encoded numerical but actually categorical

feat.MSSubClass = feat.MSSubClass.astype(str)



# Masonry: NA very likely means no masonry so we fill with string «None» or 0 for size

feat.MasVnrType = feat.MasVnrType.fillna("None")

feat.MasVnrArea = feat.MasVnrArea.fillna(0).astype(int)



# MiscFeature means likely no feature, filling with None

feat.MiscFeature = feat.MiscFeature.fillna("None")



# PoolQC means likely no pool, filling with None

feat.PoolQC = feat.PoolQC.fillna("None")



# SaleType: NaNs likely mean unknown, filling with most frequent value «WD»

feat.SaleType = feat.SaleType.fillna("WD")



# Utilities: NaNs likely mean unknown, filling with most frequent value «AllPub»

feat.Utilities = feat.Utilities.fillna("AllPub")
# label encode ordinal features where there is order in categories

# unfortunately can't use LabelEncoder or pd.factorize() since strings do not contain order of values



feat = feat.replace({  "Alley":        {"None" : 0, "Grvl" : 1, "Pave" : 2},

                       "BsmtCond":     {"None" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, 

                                        "Gd" : 4, "Ex" : 5},

                       "BsmtExposure": {"None" : 0, "No" : 2, "Mn" : 2, "Av": 3, 

                                        "Gd" : 4},

                       "BsmtFinType1": {"None" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, 

                                        "BLQ" : 4, "ALQ" : 5, "GLQ" : 6},

                       "BsmtFinType2": {"None" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, 

                                        "BLQ" : 4, 

                                         "ALQ" : 5, "GLQ" : 6},

                       "BsmtQual":     {"None" : 0, "Po" : 1, "Fa" : 2, "TA": 3, 

                                        "Gd" : 4, "Ex" : 5},

                       "CentralAir":   {"None" : 0, "N" : 1, "Y" : 2},

                       "ExterCond":    {"None" : 0, "Po" : 1, "Fa" : 2, "TA": 3, 

                                        "Gd": 4, "Ex" : 5},

                       "ExterQual":    {"None" : 0, "Po" : 1, "Fa" : 2, "TA": 3, 

                                        "Gd": 4, "Ex" : 5},

                       "Fence":        {"None" : 0, "MnWw" : 1, "GdWo" : 2, "MnPrv": 3, 

                                        "GdPrv" : 4},

                       "FireplaceQu":  {"None" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, 

                                        "Gd" : 4, "Ex" : 5},

                       "Functional":   {"None" : 0, "Sal" : 1, "Sev" : 2, "Maj2" : 3, 

                                        "Maj1" : 4, "Mod": 5, "Min2" : 6, "Min1" : 7, 

                                        "Typ" : 8},

                       "GarageCond":   {"None" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, 

                                        "Gd" : 4, "Ex" : 5},

                       "GarageQual":   {"None" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, 

                                        "Gd" : 4, "Ex" : 5},

                       "GarageFinish": {"None" : 0, "Unf" : 1, "RFn" : 2, "Fin" : 3},

                       "HeatingQC":    {"None" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, 

                                        "Gd" : 4, "Ex" : 5},

                       "KitchenQual":  {"None" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, 

                                        "Gd" : 4, "Ex" : 5},

                       "LandContour":  {"None" : 0, "Low" : 1, "HLS" : 2, "Bnk" : 3, 

                                        "Lvl" : 4},

                       "LandSlope":    {"None" : 0, "Sev" : 1, "Mod" : 2, "Gtl" : 3},

                       "LotShape":     {"None" : 0, "IR3" : 1, "IR2" : 2, "IR1" : 3, 

                                        "Reg" : 4},

                       "PavedDrive":   {"None" : 0, "N" : 0, "P" : 1, "Y" : 2},

                       "PoolQC":       {"None" : 0, "Fa" : 1, "TA" : 2, "Gd" : 3, 

                                        "Ex" : 4},

                       "Street":       {"None" : 0, "Grvl" : 1, "Pave" : 2},

                       "Utilities":    {"None" : 0, "ELO" : 1, "NoSeWa" : 2, "NoSewr" : 3, 

                                        "AllPub" : 4}}

                     )



feat.BsmtCond = feat.BsmtCond.astype(int)
# only one hot encode «true» categoricals...  

# ... rather than ordinals, where order matters and we already label encoded in the previous cells



def onehot_encode(data):

    df_numeric = data.select_dtypes(exclude=['object'])

    df_obj = data.select_dtypes(include=['object']).copy()



    cols = []

    for c in df_obj:

        dummies = pd.get_dummies(df_obj[c])

        dummies.columns = [c + "_" + str(x) for x in dummies.columns]

        cols.append(dummies)

    df_obj = pd.concat(cols, axis=1)



    data = pd.concat([df_numeric, df_obj], axis=1)

    data.reset_index(inplace=True, drop=True)

    return data



feat = onehot_encode(feat)
# map months to seasons: 0 == winter, 1 == spring etc.

seasons = {12 : 0, 1 : 0, 2 : 0, 

           3 : 1, 4 : 1, 5 : 1,

           6 : 2, 7 : 2, 8 : 2, 

           9 : 3, 10 : 3, 11 : 3}



feat["SeasonSold"]     = feat["MoSold"].map(seasons)

feat["YrActualAge"]    = feat["YrSold"] - feat["YearBuilt"]



feat['TotalSF1']       = feat['TotalBsmtSF'] + feat['1stFlrSF'] + feat['2ndFlrSF']

feat['TotalSF2']       = (feat['BsmtFinSF1'] + feat['BsmtFinSF2'] + feat['1stFlrSF'] + feat['2ndFlrSF'])

feat["AllSF"]          = feat["GrLivArea"] + feat["TotalBsmtSF"]

feat["AllFlrsSF"]      = feat["1stFlrSF"] + feat["2ndFlrSF"]

feat["AllPorchSF"]     = feat["OpenPorchSF"] + feat["EnclosedPorch"] + feat["3SsnPorch"] + feat["ScreenPorch"]



feat['TotalBath']      = 2 * (feat['FullBath'] + (0.5 * feat['HalfBath']) + feat['BsmtFullBath'] + (0.5 * feat['BsmtHalfBath']))

feat["TotalBath"]      = feat["TotalBath"].astype(int)

feat['TotalPorch']     = (feat['OpenPorchSF'] + feat['3SsnPorch'] + feat['EnclosedPorch'] + feat['ScreenPorch'] + feat['WoodDeckSF'])

feat["OverallScore"]   = feat["OverallQual"] * feat["OverallCond"]

feat["GarageScore"]    = feat["GarageQual"] * feat["GarageCond"]

feat["ExterScore"]     = feat["ExterQual"] * feat["ExterCond"]

feat["KitchenScore"]   = feat["KitchenAbvGr"] * feat["KitchenQual"]

feat["FireplaceScore"] = feat["Fireplaces"] * feat["FireplaceQu"]

feat["GarageScore"]    = feat["GarageArea"] * feat["GarageQual"]

feat["PoolScore"]      = feat["PoolArea"] * feat["PoolQC"]



feat['hasPool']        = feat['PoolArea'].apply(lambda x: 1 if x > 0 else 0)

feat['has2ndFloor']    = feat['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)

feat['hasGarage']      = feat['GarageArea'].apply(lambda x: 1 if x > 0 else 0)

feat['hasBsmt']        = feat['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)

feat['hasFireplace']   = feat['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
# create new ordinal features by binning continuous features

# log transform values before binning taking into account skewed distributions



cut_cols = ["LotArea", "YearBuilt", "YearRemodAdd", "MasVnrArea", 'BsmtFinSF1',

            'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF',

            'LowQualFinSF', 'GrLivArea', "GarageYrBlt", 'GarageArea', 'WoodDeckSF', 'OpenPorchSF']

frames = []

for cut_col in cut_cols:

    tmp = pd.DataFrame(pd.cut(np.log1p(feat[cut_col]), bins=10, labels=np.arange(0,10)))

    tmp.columns = [cut_col + "_binned"]

    frames.append(tmp)

    

binned = pd.concat(frames, axis=1).astype(int)

feat = pd.concat([feat, binned], axis=1)
df = pd.read_csv(f"{BASE_PATH}train.csv")

df_test = pd.read_csv(f"{BASE_PATH}test.csv")



dtrain = feat[feat.SalePrice.notnull()].copy()

dtest  = feat[feat.SalePrice.isnull()].copy()

dtest  = dtest.drop("SalePrice", axis=1).reset_index(drop=True)

print(f"Raw data shape   : {df.shape}  {df_test.shape}")

print(f"Clean data shape : {dtrain.shape} {dtest.shape}")
X = df

y = df.SalePrice

metric = 'neg_mean_squared_log_error'

clf = DummyRegressor("median")

kfold = KFold(n_splits=10, shuffle=True, random_state=1)

print(f"{np.sqrt(-cross_val_score(clf, X, y, cv=kfold, scoring=metric)).mean():.4f} Log Error")
X = df[["GrLivArea"]]

y = df.SalePrice

metric = 'neg_mean_squared_log_error'

clf = LinearRegression()

kfold = KFold(n_splits=10, shuffle=True, random_state=1)

print(f"{np.sqrt(-cross_val_score(clf, X, y, cv=kfold, scoring=metric)).mean():.4f} Log Error")
X = df[["OverallCond"]]

y = df.SalePrice

metric = 'neg_mean_squared_log_error'

clf = LinearRegression()

kfold = KFold(n_splits=10, shuffle=True, random_state=1)

print(f"{np.sqrt(-cross_val_score(clf, X, y, cv=kfold, scoring=metric)).mean():.4f} Log Error")
try:

    X = df[["OverallQual"]]

    y = df.SalePrice

    metric = 'neg_mean_squared_log_error'

    clf = LinearRegression()

    kfold = KFold(n_splits=10, shuffle=True, random_state=1)

    print(f"{np.sqrt(-cross_val_score(clf, X, y, cv=kfold, scoring=metric)).mean():.4f} Log Error")

except Exception as e:

    print("Oh no... An error has occured...")

    print(e)
feature = "OverallCond"

plt.figure(figsize=(16,5))

sns.regplot(x=feature, y="SalePrice", data=df)

plt.title(f"{feature}")

plt.show()



feature = "OverallQual"

plt.figure(figsize=(16,5))

ax = sns.regplot(x=feature, y="SalePrice", data=df)

plt.title(f"{feature}")

ax.annotate('Negative values!', xy=(1.5, 5_000), xytext=(3, 500_000), 

            arrowprops=dict(facecolor='darkred'))

plt.show()
feature = "OverallQual"

plt.figure(figsize=(16,5))

sns.regplot(x=feature, y="SalePrice", data=df[df[feature] >= 3])

plt.title(f"{feature}")

plt.show()
X = df[df.OverallQual >= 3][["OverallQual"]]

y = df[df.OverallQual >= 3].SalePrice

metric = 'neg_mean_squared_log_error'

clf = LinearRegression()

kfold = KFold(n_splits=10, shuffle=True, random_state=1)

print(f"{np.sqrt(-cross_val_score(clf, X, y, cv=kfold, scoring=metric)).mean():.4f} Log Error")
# we select all the categorical data for crossvalidation

X = df.select_dtypes("object")

y = df.SalePrice

metric = 'neg_mean_squared_log_error'



# use make_pipeline to automatically fill missing values and one hot encode

clf = make_pipeline(SimpleImputer(strategy='most_frequent', fill_value='missing'), 

                    OneHotEncoder(handle_unknown="ignore"), LinearRegression())

kfold = KFold(n_splits=10, shuffle=True, random_state=1)

print(f"{np.sqrt(-cross_val_score(clf, X, y, cv=kfold, scoring=metric)).mean():.4f} Log Error")
# again fix OverallQual in order to not crash crossvalidation with our metric

df_fixed = df[df.OverallQual >= 3]



# we select all the numerical data for crossvalidation

X = df_fixed.select_dtypes("number").drop(["SalePrice"], axis=1)

y = df_fixed.SalePrice

metric = 'neg_mean_squared_log_error'

clf = make_pipeline(SimpleImputer(), StandardScaler(), LinearRegression())

kfold = KFold(n_splits=10, shuffle=True, random_state=1)

print(f"{np.sqrt(-cross_val_score(clf, X, y, cv=kfold, scoring=metric)).mean():.4f} Log Error")
X = df.select_dtypes("number").drop(["SalePrice"], axis=1)



# log transform SalePrice to fix skewed distribution

# we also now can skip removing the samples with OverallQual < 3

y = np.log1p(df.SalePrice)

metric = 'neg_mean_squared_error'



clf = make_pipeline(SimpleImputer(), StandardScaler(), LinearRegression())

kfold = KFold(n_splits=10, shuffle=True, random_state=1)

print(f"{np.sqrt(-cross_val_score(clf, X, y, cv=kfold, scoring=metric)).mean():.4f} Log Error")
classifiers = [

               Ridge(), 

               Lasso(), 

               ElasticNet(),

               KernelRidge(),

               SVR(),

               RandomForestRegressor(),

               GradientBoostingRegressor(),

               lgb.LGBMRegressor(),

               xgb.XGBRegressor(objective="reg:squarederror"),

]



clf_names = [

            "ridge      ",

            "lasso      ",

            "elastic    ",

            "kernlrdg   ",

            "svr        ",

            "rndmforest ", 

            "gbm        ", 

            "lgbm       ", 

            "xgboost    ",

]
X = dtrain.drop(["SalePrice"], axis=1)

y = np.log1p(dtrain.SalePrice)

metric = 'neg_mean_squared_error'



for clf_name, clf in zip(clf_names, classifiers):

    kfold = KFold(n_splits=5, shuffle=True, random_state=1)

    print(f"{clf_name} {np.sqrt(-cross_val_score(clf, X, y, cv=kfold, scoring=metric)).mean():.4f}")
X = dtrain[dtrain.GrLivArea < 4000].drop(["SalePrice"], axis=1)

y = np.log1p(dtrain[dtrain.GrLivArea < 4000].SalePrice)

metric = 'neg_mean_squared_error'



for clf_name, clf in zip(clf_names, classifiers):

    kfold = KFold(n_splits=5, shuffle=True, random_state=1)

    print(f"{clf_name} {np.sqrt(-cross_val_score(clf, X, y, cv=kfold, scoring=metric)).mean():.4f}")
X = dtrain[dtrain.GrLivArea < 4000].drop(["SalePrice"], axis=1)

# we carefully reduce dimensionality from 271 features to 250 dimensions

pca = PCA(n_components=250)

X_pca = pca.fit_transform(X)

y = np.log1p(dtrain[dtrain.GrLivArea < 4000].SalePrice)

metric = 'neg_mean_squared_error'



for clf_name, clf in zip(clf_names, classifiers):

    kfold = KFold(n_splits=5, shuffle=True, random_state=1)

    print(f"{clf_name} {np.sqrt(-cross_val_score(clf, X_pca, y, cv=kfold, scoring=metric)).mean():.4f}")
X = dtrain[dtrain.GrLivArea < 4000].drop(["SalePrice"], axis=1).copy()



sk = pd.DataFrame(X.iloc[:, :60].skew(), columns=["skewness"])

sk = sk[sk.skewness > .75]

for feature_ in sk.index:

    X[feature_] = boxcox1p(X[feature_], 0.15)



y = np.log1p(dtrain[dtrain.GrLivArea < 4000].SalePrice)

metric = 'neg_mean_squared_error'



for clf_name, clf in zip(clf_names, classifiers):

    kfold = KFold(n_splits=5, shuffle=True, random_state=1)

    print(f"{clf_name} {np.sqrt(-cross_val_score(clf, X, y, cv=kfold, scoring=metric)).mean():.4f}")
X = dtrain[dtrain.GrLivArea < 4000].drop(["SalePrice"], axis=1).copy()

y = np.log1p(dtrain[dtrain.GrLivArea < 4000].SalePrice)



# again we apply the boxcox transformation

sk = pd.DataFrame(X.iloc[:, :60].skew(), columns=["skewness"])

sk = sk[sk.skewness > .75]

for feature_ in sk.index:

    X[feature_] = boxcox1p(X[feature_], 0.15)



clf = xgb.XGBRegressor(objective="reg:squarederror", n_jobs=-1)

coeffs = clf.fit(X, y).feature_importances_

df_co = pd.DataFrame(coeffs, columns=["importance_"])

df_co.index = X.columns

df_co.sort_values("importance_", ascending=True, inplace=True)



plt.figure(figsize=(16,9))

df_co.iloc[250:, :].importance_.plot(kind="barh")

plt.title(f"XGBoost feature importance")

plt.show()
# use only the most promising classifiers

classifiers = [

               Ridge(), 

               Lasso(), 

               ElasticNet(),

               KernelRidge(),

               GradientBoostingRegressor(),

               lgb.LGBMRegressor(),

               xgb.XGBRegressor(objective="reg:squarederror"),

]



clf_names = [

            "ridge      ",

            "lasso      ",

            "elastic    ",

            "kernlrdg   ",

            "gbm        ", 

            "lgbm       ", 

            "xgboost    ",

]
X = dtrain[dtrain.GrLivArea < 4000].drop(["SalePrice"], axis=1).copy()

y = np.log1p(dtrain[dtrain.GrLivArea < 4000].SalePrice)

X_test = dtest.copy()



sk = pd.DataFrame(X.iloc[:, :60].skew(), columns=["skewness"])

sk = sk[sk.skewness > .75]

for feature_ in sk.index:

    X[feature_] = boxcox1p(X[feature_], 0.15)

    X_test[feature_] = boxcox1p(X_test[feature_], 0.15)
# for clf_name, clf in zip(clf_names, classifiers):

#     kfold = KFold(n_splits=5, shuffle=True, random_state=1)

#     print(f"{clf_name} {np.sqrt(-cross_val_score(clf, X, y, cv=kfold, scoring=metric)).mean():.4f}")
predictions = []



for clf_name, clf in zip(clf_names, classifiers):

    clf.fit(X, y)

    preds = clf.predict(X_test)

    # reverse log transform predicted sale prices with np.expm1()

    predictions.append(np.expm1(preds))
print(df.SalePrice.describe())
p_stats = [df.SalePrice.describe()]



for idx, clf_name in zip(range(0,7), clf_names):

    plt.figure(figsize=(16,5))

    p_tmp = pd.DataFrame(predictions[idx], columns=["preds"])

    sns.distplot(df.SalePrice)

    sns.distplot(p_tmp)

    plt.legend(["sales prices: trained", "sales price: predicted"])

    plt.xlim(0, 400_000)

    plt.title(f"{clf_name.strip()}: distributions of trained and predicted values")

    plt.tight_layout()

    plt.show()

    print(f"{clf_name.strip()} min/max of predicted sales prices")

    print(f"{p_tmp.min().values[0]:.0f} {p_tmp.max().values[0]:.0f}")

    p_stats.append(p_tmp.describe())
for idx, clf_name in enumerate(clf_names):

    p = pd.DataFrame(predictions[idx])

    # filter all values beyond max sale price in training data

    p_out = p[p[0] > 755_000].astype(int) 

    if len(p_out) > 0:

        p_out.columns = [f"{clf_name.strip().capitalize()} _ predicted SalePrice"]

        display(p_out)
dtest.loc[1089][["YearBuilt", "YrSold", "YrActualAge"]]
X = dtrain[dtrain.GrLivArea < 4000].drop(["SalePrice"], axis=1)



sk = pd.DataFrame(X.iloc[:, :60].skew(), columns=["skewness"])

sk = sk[sk.skewness > .75]

for feature_ in sk.index:

    X[feature_] = boxcox1p(X[feature_], 0.15)



y = np.log1p(dtrain[dtrain.GrLivArea < 4000].SalePrice)

metric = 'neg_mean_squared_error'



predictions = []

# create an array for stats, set stats of training set as first column

pred_stats = [df.SalePrice.describe()]



for clf_name, clf in zip(clf_names, classifiers):

    clf.fit(X, y)

    preds = clf.predict(X_test)

    # reverse log transform predicted sale prices with np.expm1()

    predictions.append(np.expm1(preds))

    pred_stats.append(pd.DataFrame(np.expm1(preds)).describe())



# calculate correlations of predicted values between each of the models

pr_corr = pd.DataFrame(predictions).T.corr()

pr_corr.columns = [x.strip() for x in clf_names]

pr_corr.index = [x.strip() for x in clf_names]

plt.figure(figsize=(16,9))

sns.heatmap(pr_corr, cmap="RdBu", square=True, cbar_kws={"shrink": .8})

plt.title("Correlation of predictions of trained models")

plt.tight_layout()

plt.show()
p = pd.concat(pred_stats, axis=1)

cols = ["training data"]

cols.extend(clf_names)

p.columns = cols

plt.figure(figsize=(16,9))

sns.heatmap(p.corr(), cmap="RdBu", square=True, cbar_kws={"shrink": .8})

plt.title("Correlation of statistics of trained models")

plt.tight_layout()

plt.show()
# X = dtrain[dtrain.GrLivArea < 4000].drop(["SalePrice"], axis=1)

# y = np.log1p(dtrain[dtrain.GrLivArea < 4000].SalePrice)

# metric = 'neg_mean_squared_error'



# sk = pd.DataFrame(X.iloc[:, :60].skew(), columns=["skewness"])

# sk = sk[sk.skewness > .75]

# for feature_ in sk.index:

#     X[feature_] = boxcox1p(X[feature_], 0.15)
# # GridSearchCV Ridge

# ridge = make_pipeline(RobustScaler(), Ridge(alpha=15, random_state=1))

# param_grid = {

#     'ridge__alpha' : np.linspace(12, 18, 10),

#     'ridge__max_iter' : np.linspace(10, 200, 5),

# }

# search = GridSearchCV(ridge, param_grid, cv=5, scoring=metric, n_jobs=-1)

# search.fit(X, y)

# print(f"{search.best_params_}")

# print(f"{np.sqrt(-search.best_score_):.4}")
# # GridSearchCV Lasso

# lasso = make_pipeline(RobustScaler(), Lasso(alpha=0.00044, random_state=1))

# param_grid = {'lasso__alpha' : np.linspace(0.00005, 0.001, 30)}

# search = GridSearchCV(lasso, param_grid, cv=5, scoring=metric, n_jobs=-1)

# search.fit(X, y)

# print(f"{search.best_params_}")

# print(f"{np.sqrt(-search.best_score_):.4}")
# # GridSearchCV ElasticNet

# elastic = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=1, random_state=1))

# param_grid = {

#     'elasticnet__alpha' : np.linspace(0.0001, 0.001, 10),

#     'elasticnet__l1_ratio' : np.linspace(0.5, 1, 10),

# }

# search = GridSearchCV(elastic, param_grid, cv=5, scoring=metric, n_jobs=-1)

# search.fit(X, y)

# print(f"{search.best_params_}")

# print(f"{np.sqrt(-search.best_score_):.4}")
# # GridSearchCV KernelRidge

# kernel = KernelRidge(alpha=1)

# param_grid = {'alpha' : np.linspace(0.001, 1, 30)}

# search = GridSearchCV(kernel, param_grid, cv=5, scoring=metric, n_jobs=-1)

# search.fit(X, y)

# print(f"{search.best_params_}")

# print(f"{np.sqrt(-search.best_score_):.4}")
# # GridSearchCV GBM

# # huber loss is considered less sensitive to outliers

# gbm = GradientBoostingRegressor(n_estimators=2500, learning_rate=0.04,

#                                    max_depth=2, max_features='sqrt',

#                                    min_samples_leaf=15, min_samples_split=10, 

#                                    loss='huber', random_state=1)

# param_grid = {

#     'n_estimators' : [2500],

#     'learning_rate' : [0.03, 0.04, 0.05],

#     'max_depth' : [2],

# }

# search = GridSearchCV(gbm, param_grid, cv=5, scoring=metric, n_jobs=-1)

# search.fit(X, y)

# print(f"{search.best_params_}")

# print(f"{np.sqrt(-search.best_score_):.4}")
# # GridSearchCV LightGBM

# lgbm = lgb.LGBMRegressor(objective='regression', num_leaves=5,

#                         learning_rate=0.03, n_estimators=8000,

#                         max_bin=55, bagging_fraction=0.8,

#                         bagging_freq=5, feature_fraction=0.23,

#                         feature_fraction_seed=9, bagging_seed=9,

#                         min_data_in_leaf=6, min_sum_hessian_in_leaf=11)

# param_grid = {

#     'n_estimators' : [8000],

#     'learning_rate' : [0.03],

# }

# search = GridSearchCV(clf, param_grid, cv=5, scoring=metric, n_jobs=-1)

# search.fit(X, y)

# print(f"{search.best_params_}")

# print(f"{np.sqrt(-search.best_score_):.4}")
# # GridSearchCV XGBoost

# xgbreg = xgb.XGBRegressor(objective="reg:squarederror",

#                              colsample_bytree=0.46, gamma=0.047, 

#                              learning_rate=0.04, max_depth=2, 

#                              min_child_weight=0.5, n_estimators=2000,

#                              reg_alpha=0.46, reg_lambda=0.86,

#                              subsample=0.52, random_state=1, n_jobs=-1)



# param_grid = {

#     'xgbregressor__max_depth' : [2],

#     'xgbregressor__estimators' : [1600, 1800, 2000],

#     "xgbregressor__learning_rate" : [0.02, 0.03, 0.04],

#     "xgbregressor__min_child_weight" : [0.2, 0.3, 0.4],

#     }

# search = GridSearchCV(clf, param_grid, cv=3, scoring=metric, n_jobs=-1)

# search.fit(X, y)

# print(f"{search.best_params_}")

# print(f"{np.sqrt(-search.best_score_):.4}")
# # try a stacked regressor on top of the seven tuned classifiers 

# # leaving out xgboost in the stack for now since it seems to crash the stacked regressor

# clf_to_stack = [lasso, ridge, elastic, kernel, gbm, lgbm]



# stack = StackingCVRegressor(regressors=(clf_to_stack),

#                             meta_regressor=xgb.XGBRegressor(objective="reg:squarederror", n_jobs=-1), 

#                             use_features_in_secondary=True)



# print(f"{np.sqrt(-cross_val_score(stack, X, y, scoring=metric)).mean():.4f} Log Error")
X = dtrain[dtrain.GrLivArea < 4000].drop(["SalePrice"], axis=1)

y = np.log1p(dtrain[dtrain.GrLivArea < 4000].SalePrice)



X_test = dtest

# fixing the outlier where YrSold is earlier than YrBuilt

X_test.loc[1089]["YrSold"] = 2009

X_test.loc[1089]["YrActualAge"] = 0



metric = 'neg_mean_squared_error'
# apply box cox transformation on numerical features

# skipping the one hot encoded features as well as engineered ones

sk = pd.DataFrame(X.iloc[:, :60].skew(), columns=["skewness"])

sk = sk[sk.skewness > .75]

for feature_ in sk.index:

    X[feature_] = boxcox1p(X[feature_], 0.15)

    X_test[feature_] = boxcox1p(X_test[feature_], 0.15)
ridge   = make_pipeline(RobustScaler(), Ridge(alpha=15, random_state=1))

lasso   = make_pipeline(RobustScaler(), Lasso(alpha=0.0005, random_state=1))

elastic = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, 

                                                   l1_ratio=1, random_state=1))

kernel  = KernelRidge(alpha=1.0)



gbm = GradientBoostingRegressor(n_estimators=2500, learning_rate=0.04,

                                   max_depth=2, max_features='sqrt',

                                   min_samples_leaf=15, min_samples_split=10, 

                                   loss='huber', random_state=1)



lgbm = lgb.LGBMRegressor(objective='regression', num_leaves=5,

                        learning_rate=0.03, n_estimators=8000,

                        max_bin=55, bagging_fraction=0.8,

                        bagging_freq=5, feature_fraction=0.23,

                        feature_fraction_seed=9, bagging_seed=9,

                        min_data_in_leaf=6, min_sum_hessian_in_leaf=11)



xgbreg = xgb.XGBRegressor(objective="reg:squarederror",

                             colsample_bytree=0.46, gamma=0.047, 

                             learning_rate=0.04, max_depth=2, 

                             min_child_weight=0.5, n_estimators=2000,

                             reg_alpha=0.46, reg_lambda=0.86,

                             subsample=0.52, random_state=1, n_jobs=-1)
classifiers = [ridge, lasso, elastic, kernel, gbm, lgbm, xgbreg]

clf_names   = ["ridge  ", "lasso  ", "elastic", "kernel ", "gbm    ", "lgbm   ", "xgbreg "]



predictions_exp = []



for clf_name, clf in zip(clf_names, classifiers):

    print(f"{clf_name} {np.sqrt(-cross_val_score(clf, X, y, scoring=metric).mean()):.5f}")

    clf.fit(X, y)

    preds = clf.predict(X_test)

    predictions_exp.append(np.expm1(preds))
prediction_final = pd.DataFrame(predictions_exp).mean().T.values

submission = pd.DataFrame({'Id': df_test.Id.values, 'SalePrice': prediction_final})

submission.to_csv(f"submission.csv", index=False)