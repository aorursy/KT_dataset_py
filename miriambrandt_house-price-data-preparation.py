import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, skew
import warnings
warnings.filterwarnings('ignore')
df_train = pd.read_csv("../input/train.csv")
df_test = pd.read_csv("../input/test.csv")
df_train.shape
df_train.columns
df_train.info()
unique_ids = len(set(df_train.Id))
total_ids = df_train.shape[0]
print("There are " + str(total_ids - unique_ids) + " duplicate Ids in the train dataset.")
df_train["SalePrice"].describe()
sns.distplot(df_train["SalePrice"], bins=30, fit=norm)
plt.show()
stats.probplot(df_train["SalePrice"], plot=plt)
plt.show()
numeric_features = df_train.dtypes[df_train.dtypes != "object"].index
numeric_features = numeric_features.drop("SalePrice")
categorical_features = df_train.dtypes[df_train.dtypes == "object"].index
f = pd.melt(df_train, value_vars=numeric_features)
g = sns.FacetGrid(f, col="variable",  col_wrap=3, sharex=False, sharey=False, size = 5)
g = g.map(sns.distplot, "value")
plt.show()
f = pd.melt(df_train, value_vars=categorical_features)
g = sns.FacetGrid(f, col="variable", col_wrap=3, sharex=False, sharey=False, size = 5)
g = g.map(sns.countplot, "value")
plt.show()
df_train["Alley"].value_counts()
df_train["Utilities"].value_counts()
def regplot(x, y, **kwargs):
    sns.regplot(x=x, y=y)
    x=plt.xticks(rotation=90)
f = pd.melt(df_train, id_vars=['SalePrice'], value_vars=numeric_features)
g = sns.FacetGrid(f, col="variable",  col_wrap=3, sharex=False, sharey=False, size=5)
g = g.map(regplot, "value", "SalePrice")
plt.show()
def boxplot(x, y, **kwargs):
    sns.boxplot(x=x, y=y)
    x=plt.xticks(rotation=90)
f = pd.melt(df_train, id_vars=['SalePrice'], value_vars=categorical_features)
g = sns.FacetGrid(f, col="variable",  col_wrap=3, sharex=False, sharey=False, size=5)
g = g.map(boxplot, "value", "SalePrice")
plt.show()
corr = df_train.corr()
fig, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corr, vmax=.8, square=True);
plt.show()
print(corr.iloc[-1].sort_values(ascending=False).drop("SalePrice").head(10))
print(corr.loc["GarageCars"].sort_values(ascending=False).drop("GarageCars").head(1))
df_train["SalePrice"] = np.log1p(df_train["SalePrice"])
sns.distplot(df_train["SalePrice"], bins=30, fit=norm)
plt.show()
stats.probplot(df_train["SalePrice"], plot=plt)
plt.show()
print(df_train["SalePrice"].sort_values().head())
print(df_train["SalePrice"].sort_values().tail())
df_train = df_train[df_train["SalePrice"] < 13.5]
df_train = df_train[df_train["SalePrice"] > 10.5]
stats.probplot(df_train["SalePrice"], plot=plt)
plt.show()
df_train.shape
sns.regplot(x="GrLivArea", y="SalePrice", data=df_train);
plt.show()
df_train = df_train[df_train["GrLivArea"] < 4000]
sns.regplot(x="GrLivArea", y="SalePrice", data=df_train);
plt.show()
df_train.shape
df_test.shape
df = pd.concat([df_train, df_test])
df.shape
df_na = pd.DataFrame()
df_na["Feature"] = df.columns
missing = ((df.isnull().sum() / len(df)) * 100).values
df_na["Missing"] = missing
df_na = df_na[df_na["Feature"] != "SalePrice"]
df_na = df_na[df_na["Missing"] != 0]
df_na=df_na.sort_values(by="Missing", ascending=False)
print(df_na)
sns.barplot(x="Feature", y="Missing", data=df_na)
plt.xticks(rotation=90)
plt.show()
df['Alley'] = df['Alley'].fillna('None')
df['MasVnrType'] = df['MasVnrType'].fillna('None')
df['BsmtQual'] = df['BsmtQual'].fillna('None')
df['BsmtCond'] = df['BsmtCond'].fillna('None')
df['BsmtExposure'] = df['BsmtExposure'].fillna('None')
df['BsmtFinType1'] = df['BsmtFinType1'].fillna('None')
df['BsmtFinType2'] = df['BsmtFinType2'].fillna('None')
df['FireplaceQu'] = df['FireplaceQu'].fillna('None')
df['GarageType'] = df['GarageType'].fillna('None')
df['GarageFinish'] = df['GarageFinish'].fillna('None')
df['GarageQual'] = df['GarageQual'].fillna('None')
df['GarageCond'] = df['GarageCond'].fillna('None')
df['Fence'] = df['Fence'].fillna('None')
df['MiscFeature'] = df['MiscFeature'].fillna('None')
df['PoolQC'] = df['PoolQC'].fillna('None')
df['MSZoning'] = df['MSZoning'].fillna('RL')
df["Exterior1st"] = df["Exterior1st"].fillna('VinylSd')
df["Exterior2nd"] = df["Exterior2nd"].fillna('VinylSd')
df['Electrical'] = df['Electrical'].fillna('SBrkr')
df['KitchenQual'] = df['KitchenQual'].fillna('TA')
df['Functional'] = df['Functional'].fillna('Typ')
df['SaleType'] = df['SaleType'].fillna('WD')
df['Utilities'] = df['Utilities'].fillna('AllPub')
df['MasVnrArea'] = df['MasVnrArea'].fillna(0)
df['BsmtFinSF1'] = df['BsmtFinSF1'].fillna(0)
df['BsmtFinSF2'] = df['BsmtFinSF2'].fillna(0)
df['BsmtUnfSF'] = df['BsmtUnfSF'].fillna(0)
df['TotalBsmtSF'] = df['TotalBsmtSF'].fillna(0)
df['BsmtFullBath'] = df['BsmtFullBath'].fillna(0)
df['BsmtHalfBath'] = df['BsmtHalfBath'].fillna(0)
df['GarageCars'] = df['GarageCars'].fillna(0)
df['GarageArea'] = df['GarageArea'].fillna(0)
df['GarageYrBlt'] = df['GarageYrBlt'].fillna(0)
df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
df_na = pd.DataFrame()
df_na["Feature"] = df.columns
missing = ((df.isnull().sum() / len(df)) * 100).values
df_na["Missing"] = missing
df_na = df_na[df_na["Feature"] != "SalePrice"]
df_na = df_na[df_na["Missing"] != 0]
df_na=df_na.sort_values(by="Missing", ascending=False)
print(df_na)
df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
num_to_cat=["BedroomAbvGr", "BsmtFullBath", "BsmtHalfBath", "Fireplaces", "FullBath",
            "GarageCars", "HalfBath", "KitchenAbvGr", "MoSold", "MSSubClass", "OverallCond", 
            "OverallQual", "TotRmsAbvGrd", "YrSold"]

df[num_to_cat] = df[num_to_cat].apply(lambda x: x.astype("str"))
def boxplot(x, y, **kwargs):
    sns.boxplot(x=x, y=y)
f = pd.melt(df_train, id_vars=['SalePrice'], value_vars=num_to_cat)
g = sns.FacetGrid(f, col="variable",  col_wrap=3, sharex=False, sharey=False, size=5)
g = g.map(boxplot, "value", "SalePrice")
plt.show()
numeric_features = df.dtypes[df.dtypes != "object"].index
numeric_features = numeric_features.drop("SalePrice")

skew_before = df[numeric_features].apply(lambda x: skew(x.dropna()))
print(skew_before)
df_log = np.log1p(df[numeric_features])
skew_after = df_log[numeric_features].apply(lambda x: skew(x.dropna()))
skew_diff = (abs(skew_before)-abs(skew_after)).sort_values(ascending=False)
df[skew_diff[skew_diff>0].index] = np.log1p(df[skew_diff[skew_diff>0].index])
skew_new = df[numeric_features].apply(lambda x: skew(x.dropna()))

print(skew_new)
df = pd.get_dummies(df, drop_first=True)
df.info()
train = df.iloc[:1454]
test = df.iloc[1454:].drop("SalePrice", axis=1)
train.info()
test.info()