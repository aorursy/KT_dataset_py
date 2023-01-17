"""
Create by Karl Bina, Computer vision and data science engineer (Telitem Company)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})
import warnings
warnings.filterwarnings('ignore')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

        
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
train_set = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
test_set = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
print(train_set.shape)
print(test_set.shape)
train_set.head()
test_set.head()
train = train_set.copy()
test = test_set.copy()
train.shape
train.dtypes.value_counts()
train.dtypes.value_counts().plot.pie()
(train.isna().sum() / train.shape[0]).sort_values(ascending=False)
import seaborn as sns
plt.figure(figsize=(16,8))
sns.heatmap(train.isna(), cbar=False)
train['SalePrice'].value_counts(normalize=True)
train['SalePrice'].mean()
for col in train.select_dtypes('float64'):
    if col == 'Id':
        pass
    else:
        plt.figure(figsize=(10,8))
        sns.distplot(train[col])
for col in train.select_dtypes('int'):
    try:
        plt.figure(figsize=(8,8))
        sns.distplot(train[col])
    except RuntimeError as re:
        if str(re).startswith("Selected KDE bandwidth is 0. Cannot estimate density."):
            plt.figure(figsize=(8,8))
            sns.distplot(train[col], kde_kws={'bw': 0.1})
        else:
            raise re
train.head(3)
plt.figure(figsize=(18,8))
train.corr()['SalePrice'].plot.bar()
plt.xlabel('Variable')
plt.ylabel('Correlation')
plt.title('Correlation SalePrice')
correlation_SalePrice = pd.DataFrame(train.corr()['SalePrice'])
plt.figure(figsize=(12,12))
sns.heatmap(correlation_SalePrice, annot=True)
train['SalePrice'].describe()
plt.figure(figsize=(10,8))
sns.distplot(train['SalePrice'])
train.corr()['SalePrice'][train.corr()['SalePrice'] > 0.5]
train.corr()['SalePrice'][(train.corr()['SalePrice'] > 0) & (train.corr()['SalePrice'] < 0.5)]
for col in train.drop(['SalePrice'], axis=1).select_dtypes('int64'):
    plt.figure(figsize=(10,4))
    plt.scatter(train[col],train['SalePrice'])
    plt.title('Relation between SalePrice and {}'.format(col))
    plt.xlabel(col)
    plt.ylabel('SalePrice')
    plt.show()
plt.figure(figsize=(12,8))
sns.boxplot(x="OverallQual", y="SalePrice",palette=["m", "g"], data=train)
plt.figure(figsize=(12,8))
sns.boxplot(x="GarageQual", y="SalePrice",palette=["m", "g"], data=train)
plt.figure(figsize=(12,8))
sns.boxplot(x="GarageCars", y="SalePrice",palette=["m", "g"], data=train)
plt.figure(figsize=(18,8))
sns.boxplot(x="MoSold", y="SalePrice", data=train)
plt.figure(figsize=(18,8))
sns.boxplot(x="YearBuilt", y="SalePrice", data=train)
plt.xticks(rotation=90)
plt.show()
train.corr()['SalePrice'][train.corr()['SalePrice'] > 0.50]
display_pair = train[["SalePrice","OverallQual","YearBuilt","TotalBsmtSF","GrLivArea","FullBath","GarageCars","GarageArea"]]
display_pair.shape
#plt.figure()
#sns.pairplot(display_pair)
#plt.show()
plt.figure(figsize=(20,12))
sns.heatmap(train.corr()[(train.corr()['SalePrice'] > 0)])
from scipy.stats import norm
from scipy import stats
plt.figure(figsize=(14,8))
plt.subplot(2,2,1)
sns.distplot(train['SalePrice'], fit=norm)
plt.subplot(2,2,2)
res = stats.probplot(train['SalePrice'], plot=plt)
train['SalePrice'] = np.log1p(train['SalePrice'])
plt.figure(figsize=(14,8))
plt.subplot(2,2,1)
sns.distplot(train['SalePrice'], fit=norm)
plt.subplot(2,2,2)
res = stats.probplot(train['SalePrice'], plot=plt)
plt.figure(figsize=(14,8))
plt.subplot(2,2,1)
sns.distplot(train['GrLivArea'], fit=norm)
plt.subplot(2,2,2)
res = stats.probplot(train['GrLivArea'], plot=plt)
train['GrLivArea'] = np.log1p(train['GrLivArea'])
test['GrLivArea'] = np.log1p(test['GrLivArea'])
plt.figure(figsize=(14,8))
plt.subplot(2,2,1)
sns.distplot(train['GrLivArea'], fit=norm)
plt.subplot(2,2,2)
res = stats.probplot(train['GrLivArea'], plot=plt)
plt.figure(figsize=(14,8))
plt.subplot(2,2,1)
sns.distplot(train['TotalBsmtSF'], fit=norm)
plt.subplot(2,2,2)
res = stats.probplot(train['TotalBsmtSF'], plot=plt)
train['logHasBsmt'] = pd.Series(len(train['TotalBsmtSF']), index=train.index)
train['logHasBsmt'] = 0 
train.loc[train['TotalBsmtSF']>0,'logHasBsmt'] = 1
test['logHasBsmt'] = pd.Series(len(test['TotalBsmtSF']), index=test.index)
test['logHasBsmt'] = 0 
test.loc[test['TotalBsmtSF']>0,'logHasBsmt'] = 1
train.loc[train['logHasBsmt']==1,'TotalBsmtSF'] = np.log1p(train['TotalBsmtSF'])
test.loc[test['logHasBsmt']==1,'TotalBsmtSF'] = np.log1p(test['TotalBsmtSF'])
train.head()
plt.figure(figsize=(14,8))
plt.subplot(2,2,1)
sns.distplot(train[train['TotalBsmtSF']>0]['TotalBsmtSF'], fit=norm)
plt.subplot(2,2,2)
res = stats.probplot(train[train['TotalBsmtSF']>0]['TotalBsmtSF'], plot=plt)
# Inspired by juliencs's script https://www.kaggle.com/juliencs/a-study-on-regression-applied-to-the-ames-dataset
def Encodage(df):
    df.loc[:,"Alley"] = df.loc[:, "Alley"].fillna("None")
    df.loc[:, "BedroomAbvGr"] = df.loc[:, "BedroomAbvGr"].fillna(0)
    df.loc[:, "BsmtQual"] = df.loc[:, "BsmtQual"].fillna("No")
    df.loc[:, "BsmtCond"] = df.loc[:, "BsmtCond"].fillna("No")
    df.loc[:, "BsmtExposure"] = df.loc[:, "BsmtExposure"].fillna("No")
    df.loc[:, "BsmtFinType1"] = df.loc[:, "BsmtFinType1"].fillna("No")
    df.loc[:, "BsmtFinType2"] = df.loc[:, "BsmtFinType2"].fillna("No")
    df.loc[:, "BsmtFullBath"] = df.loc[:, "BsmtFullBath"].fillna(0)
    df.loc[:, "BsmtHalfBath"] = df.loc[:, "BsmtHalfBath"].fillna(0)
    df.loc[:, "BsmtUnfSF"] = df.loc[:, "BsmtUnfSF"].fillna(0)
    df.loc[:, "CentralAir"] = df.loc[:, "CentralAir"].fillna("N")
    df.loc[:, "Condition1"] = df.loc[:, "Condition1"].fillna("Norm")
    df.loc[:, "Condition2"] = df.loc[:, "Condition2"].fillna("Norm")
    df.loc[:, "EnclosedPorch"] = df.loc[:, "EnclosedPorch"].fillna(0)
    df.loc[:, "ExterCond"] = df.loc[:, "ExterCond"].fillna("TA")
    df.loc[:, "ExterQual"] = df.loc[:, "ExterQual"].fillna("TA")
    df.loc[:, "Fence"] = df.loc[:, "Fence"].fillna("No")
    df.loc[:, "FireplaceQu"] = df.loc[:, "FireplaceQu"].fillna("No")
    df.loc[:, "Fireplaces"] = df.loc[:, "Fireplaces"].fillna(0)
    df.loc[:, "Functional"] = df.loc[:, "Functional"].fillna("Typ")
    df.loc[:, "GarageType"] = df.loc[:, "GarageType"].fillna("No")
    df.loc[:, "GarageFinish"] = df.loc[:, "GarageFinish"].fillna("No")
    df.loc[:, "GarageQual"] = df.loc[:, "GarageQual"].fillna("No")
    df.loc[:, "GarageCond"] = df.loc[:, "GarageCond"].fillna("No")
    df.loc[:, "GarageArea"] = df.loc[:, "GarageArea"].fillna(0)
    df.loc[:, "GarageCars"] = df.loc[:, "GarageCars"].fillna(0)
    df.loc[:, "HalfBath"] = df.loc[:, "HalfBath"].fillna(0)
    df.loc[:, "HeatingQC"] = df.loc[:, "HeatingQC"].fillna("TA")
    df.loc[:, "KitchenAbvGr"] = df.loc[:, "KitchenAbvGr"].fillna(0)
    df.loc[:, "KitchenQual"] = df.loc[:, "KitchenQual"].fillna("TA")
    df.loc[:, "LotFrontage"] = df.loc[:, "LotFrontage"].fillna(0)
    df.loc[:, "LotShape"] = df.loc[:, "LotShape"].fillna("Reg")
    df.loc[:, "MasVnrType"] = df.loc[:, "MasVnrType"].fillna("None")
    df.loc[:, "MasVnrArea"] = df.loc[:, "MasVnrArea"].fillna(0)
    df.loc[:, "MiscFeature"] = df.loc[:, "MiscFeature"].fillna("No")
    df.loc[:, "MiscVal"] = df.loc[:, "MiscVal"].fillna(0)
    df.loc[:, "OpenPorchSF"] = df.loc[:, "OpenPorchSF"].fillna(0)
    df.loc[:, "PavedDrive"] = df.loc[:, "PavedDrive"].fillna("N")
    df.loc[:, "PoolQC"] = df.loc[:, "PoolQC"].fillna("No")
    df.loc[:, "PoolArea"] = df.loc[:, "PoolArea"].fillna(0)
    df.loc[:, "SaleCondition"] = df.loc[:, "SaleCondition"].fillna("Normal")
    df.loc[:, "ScreenPorch"] = df.loc[:, "ScreenPorch"].fillna(0)
    df.loc[:, "TotRmsAbvGrd"] = df.loc[:, "TotRmsAbvGrd"].fillna(0)
    df.loc[:, "Utilities"] = df.loc[:, "Utilities"].fillna("AllPub")
    df.loc[:, "WoodDeckSF"] = df.loc[:, "WoodDeckSF"].fillna(0)
    return df
def Replace_numerica_features(df):
    df = df.replace({"MSSubClass" : {20 : "SC20", 30 : "SC30", 40 : "SC40", 45 : "SC45", 
                                       50 : "SC50", 60 : "SC60", 70 : "SC70", 75 : "SC75", 
                                       80 : "SC80", 85 : "SC85", 90 : "SC90", 120 : "SC120", 
                                       150 : "SC150", 160 : "SC160", 180 : "SC180", 190 : "SC190"},
                       "MoSold" : {1 : "Jan", 2 : "Feb", 3 : "Mar", 4 : "Apr", 5 : "May", 6 : "Jun",
                                   7 : "Jul", 8 : "Aug", 9 : "Sep", 10 : "Oct", 11 : "Nov", 12 : "Dec"}
                      })
    return df
def Replace_category_features(df):
    df = df.replace({"Alley" : {"Grvl" : 1, "Pave" : 2},
                       "BsmtCond" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "BsmtExposure" : {"No" : 0, "Mn" : 1, "Av": 2, "Gd" : 3},
                       "BsmtFinType1" : {"No" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4, 
                                         "ALQ" : 5, "GLQ" : 6},
                       "BsmtFinType2" : {"No" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4, 
                                         "ALQ" : 5, "GLQ" : 6},
                       "BsmtQual" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA": 3, "Gd" : 4, "Ex" : 5},
                       "ExterCond" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5},
                       "ExterQual" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5},
                       "FireplaceQu" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "Functional" : {"Sal" : 1, "Sev" : 2, "Maj2" : 3, "Maj1" : 4, "Mod": 5, 
                                       "Min2" : 6, "Min1" : 7, "Typ" : 8},
                       "GarageCond" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "GarageQual" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "HeatingQC" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "KitchenQual" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "LandSlope" : {"Sev" : 1, "Mod" : 2, "Gtl" : 3},
                       "LotShape" : {"IR3" : 1, "IR2" : 2, "IR1" : 3, "Reg" : 4},
                       "PavedDrive" : {"N" : 0, "P" : 1, "Y" : 2},
                       "PoolQC" : {"No" : 0, "Fa" : 1, "TA" : 2, "Gd" : 3, "Ex" : 4},
                       "Street" : {"Grvl" : 1, "Pave" : 2},
                       "Utilities" : {"ELO" : 1, "NoSeWa" : 2, "NoSewr" : 3, "AllPub" : 4}}
                     )
    return df
def Simplifications_of_existing_features(df):
    df["SimplOverallQual"] = df.OverallQual.replace({1 : 1, 2 : 1, 3 : 1, 
                                                       4 : 2, 5 : 2, 6 : 2, 
                                                       7 : 3, 8 : 3, 9 : 3, 10 : 3 
                                                      })
    df["SimplOverallCond"] = df.OverallCond.replace({1 : 1, 2 : 1, 3 : 1, 
                                                           4 : 2, 5 : 2, 6 : 2, 
                                                           7 : 3, 8 : 3, 9 : 3, 10 : 3 
                                                          })
    df["SimplPoolQC"] = df.PoolQC.replace({1 : 1, 2 : 1, 
                                                 3 : 2, 4 : 2 
                                                })
    df["SimplGarageCond"] = df.GarageCond.replace({1 : 1, 
                                                         2 : 1, 3 : 1, 
                                                         4 : 2, 5 : 2 
                                                        })
    df["SimplGarageQual"] = df.GarageQual.replace({1 : 1, 
                                                         2 : 1, 3 : 1, 
                                                         4 : 2, 5 : 2 
                                                        })
    df["SimplFireplaceQu"] = df.FireplaceQu.replace({1 : 1, 
                                                           2 : 1, 3 : 1, 
                                                           4 : 2, 5 : 2 
                                                          })
    df["SimplFireplaceQu"] = df.FireplaceQu.replace({1 : 1, 
                                                           2 : 1, 3 : 1, 
                                                           4 : 2, 5 : 2 
                                                          })
    df["SimplFunctional"] = df.Functional.replace({1 : 1, 2 : 1, 
                                                         3 : 2, 4 : 2, 
                                                         5 : 3, 6 : 3, 7 : 3, 
                                                         8 : 4 
                                                        })
    df["SimplKitchenQual"] = df.KitchenQual.replace({1 : 1, 
                                                           2 : 1, 3 : 1, 
                                                           4 : 2, 5 : 2 
                                                          })
    df["SimplHeatingQC"] = df.HeatingQC.replace({1 : 1, 
                                                       2 : 1, 3 : 1, 
                                                       4 : 2, 5 : 2 
                                                      })
    df["SimplBsmtFinType1"] = df.BsmtFinType1.replace({1 : 1, 
                                                             2 : 1, 3 : 1, 
                                                             4 : 2, 5 : 2, 6 : 2 
                                                            })
    df["SimplBsmtFinType2"] = df.BsmtFinType2.replace({1 : 1, 
                                                             2 : 1, 3 : 1, 
                                                             4 : 2, 5 : 2, 6 : 2 
                                                            })
    df["SimplBsmtCond"] = df.BsmtCond.replace({1 : 1, 
                                                     2 : 1, 3 : 1, 
                                                     4 : 2, 5 : 2 
                                                    })
    df["SimplBsmtQual"] = df.BsmtQual.replace({1 : 1, 
                                                     2 : 1, 3 : 1, 
                                                     4 : 2, 5 : 2 
                                                    })
    df["SimplExterCond"] = df.ExterCond.replace({1 : 1, 
                                                       2 : 1, 3 : 1, 
                                                       4 : 2, 5 : 2 
                                                      })
    df["SimplExterQual"] = df.ExterQual.replace({1 : 1, 
                                                       2 : 1, 3 : 1, 
                                                       4 : 2, 5 : 2 
                                                      })
    df["OverallGrade"] = df["OverallQual"] * df["OverallCond"]
    df["GarageGrade"] = df["GarageQual"] * df["GarageCond"]
    df["ExterGrade"] = df["ExterQual"] * df["ExterCond"]
    df["KitchenScore"] = df["KitchenAbvGr"] * df["KitchenQual"]
    df["FireplaceScore"] = df["Fireplaces"] * df["FireplaceQu"]
    df["GarageScore"] = df["GarageArea"] * df["GarageQual"]
    df["PoolScore"] = df["PoolArea"] * df["PoolQC"]
    df["SimplOverallGrade"] = df["SimplOverallQual"] * df["SimplOverallCond"]
    df["SimplExterGrade"] = df["SimplExterQual"] * df["SimplExterCond"]
    df["SimplPoolScore"] = df["PoolArea"] * df["SimplPoolQC"]
    df["SimplGarageScore"] = df["GarageArea"] * df["SimplGarageQual"]
    df["SimplFireplaceScore"] = df["Fireplaces"] * df["SimplFireplaceQu"]
    df["SimplKitchenScore"] = df["KitchenAbvGr"] * df["SimplKitchenQual"]
    df["TotalBath"] = df["BsmtFullBath"] + (0.5 * df["BsmtHalfBath"]) + \
    df["FullBath"] + (0.5 * df["HalfBath"])
    df["AllSF"] = df["GrLivArea"] + df["TotalBsmtSF"]
    df["AllFlrsSF"] = df["1stFlrSF"] + df["2ndFlrSF"]
    df["AllPorchSF"] = df["OpenPorchSF"] + df["EnclosedPorch"] + \
    df["3SsnPorch"] + df["ScreenPorch"]
    df["HasMasVnr"] = df.MasVnrType.replace({"BrkCmn" : 1, "BrkFace" : 1, "CBlock" : 1, 
                                                   "Stone" : 1, "None" : 0})
    df["BoughtOffPlan"] = df.SaleCondition.replace({"Abnorml" : 0, "Alloca" : 0, "AdjLand" : 0, 
                                                          "Family" : 0, "Normal" : 0, "Partial" : 1})
    return df
def Polynomials_features(df):
    df["OverallQual-s2"] = df["OverallQual"] ** 2
    df["OverallQual-s3"] = df["OverallQual"] ** 3
    df["OverallQual-Sq"] = np.sqrt(df["OverallQual"])
    df["AllSF-2"] = df["AllSF"] ** 2
    df["AllSF-3"] = df["AllSF"] ** 3
    df["AllSF-Sq"] = np.sqrt(df["AllSF"])
    df["AllFlrsSF-2"] = df["AllFlrsSF"] ** 2
    df["AllFlrsSF-3"] = df["AllFlrsSF"] ** 3
    df["AllFlrsSF-Sq"] = np.sqrt(df["AllFlrsSF"])
    df["GrLivArea-2"] = df["GrLivArea"] ** 2
    df["GrLivArea-3"] = df["GrLivArea"] ** 3
    df["GrLivArea-Sq"] = np.sqrt(df["GrLivArea"])
    df["SimplOverallQual-s2"] = df["SimplOverallQual"] ** 2
    df["SimplOverallQual-s3"] = df["SimplOverallQual"] ** 3
    df["SimplOverallQual-Sq"] = np.sqrt(df["SimplOverallQual"])
    df["ExterQual-2"] = df["ExterQual"] ** 2
    df["ExterQual-3"] = df["ExterQual"] ** 3
    df["ExterQual-Sq"] = np.sqrt(df["ExterQual"])
    df["GarageCars-2"] = df["GarageCars"] ** 2
    df["GarageCars-3"] = df["GarageCars"] ** 3
    df["GarageCars-Sq"] = np.sqrt(df["GarageCars"])
    df["TotalBath-2"] = df["TotalBath"] ** 2
    df["TotalBath-3"] = df["TotalBath"] ** 3
    df["TotalBath-Sq"] = np.sqrt(df["TotalBath"])
    df["KitchenQual-2"] = df["KitchenQual"] ** 2
    df["KitchenQual-3"] = df["KitchenQual"] ** 3
    df["KitchenQual-Sq"] = np.sqrt(df["KitchenQual"])
    df["GarageScore-2"] = df["GarageScore"] ** 2
    df["GarageScore-3"] = df["GarageScore"] ** 3
    df["GarageScore-Sq"] = np.sqrt(df["GarageScore"])
    return df
train = Encodage(train)
train = Replace_numerica_features(train)
train = Replace_category_features(train)
train = Simplifications_of_existing_features(train)
train = Polynomials_features(train)
train.head()
test_data = test_set.copy()
df_test = Encodage(test)
df_test = Replace_numerica_features(df_test)
df_test = Replace_category_features(df_test)
df_test = Simplifications_of_existing_features(df_test)
df_test = Polynomials_features(df_test)
target = train["SalePrice"]
categorical_features = train.select_dtypes(include = ["object"]).columns
numerical_features = train.select_dtypes(exclude = ["object"]).columns
correlation_SalePrice = pd.DataFrame(train.corr()['SalePrice']).sort_values(by="SalePrice",ascending=False)
plt.figure(figsize=(18,18))
sns.heatmap(correlation_SalePrice, annot=True)
numerical_features = numerical_features.drop("SalePrice")
categorical_features_test = df_test.select_dtypes(include = ["object"]).columns
numerical_features_test = df_test.select_dtypes(exclude = ["object"]).columns
train_num = train[numerical_features]
train_cat = train[categorical_features]
test_num = df_test[numerical_features_test]
test_cat = df_test[categorical_features_test]
train_num = train_num.fillna(train_num.median())
test_num = test_num.fillna(test_num.median())
train_num.head()
train_cat.head()
test_num.head()
test_cat.head()
from scipy.stats import skew 
"""
In statistics, skewness is a measure of the asymmetry of the probability distribution of a random variable
about its mean. In other words, skewness tells you the amount and direction of skew . The skewness value 
can be positive or negative, or even undefined. If skewness is 0, the data are perfectly symmetrical,
although it is quite unlikely for real-world data. As a general rule of thumb:
- If skewness is less than -1 or greater than 1, the distribution is highly skewed.
- If skewness is between -1 and -0.5 or between 0.5 and 1, the distribution is moderately skewed.
- If skewness is between -0.5 and 0.5, the distribution is approximately symmetric.
"""
def skewness(dFrame):
    skewness = dFrame.apply(lambda x: skew(x))
    skewness = skewness[abs(skewness) > 0.5]
    print(str(skewness.shape[0]) + " skewed numerical features to log transform")
    skewed_features = skewness.index
    dFrame[skewed_features] = np.log1p(dFrame[skewed_features])
    return dFrame
train_num = skewness(train_num)
test_num = skewness(test_num)
train_num.head()
test_num.head()
train_cat = pd.get_dummies(train_cat)
test_cat = pd.get_dummies(test_cat)
train_cat.head()
test_cat.head()
train_data_set = pd.concat([train_num, train_cat], axis=1)
test_data_set = pd.concat([test_num, test_cat], axis=1)
#del best_coor[0]
#best_coor.insert(0,'Id')
#train_data_set = train_data_set[best_coor]
#test_data_set = test_data_set[best_coor]
# we can combine categorical and numerical features
train_data_set.head()
test_data_set.head()
from sklearn.model_selection import train_test_split
print(train_data_set.shape)
print(target.shape)
X_train, X_test, y_train, y_test = train_test_split(train_data_set, target, test_size=0.2, random_state=0)
print(X_train.shape)
print(X_test.shape)
train_data_set.head()
test_data_set.head()
print(test_data_set.shape)
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso, LassoCV, ElasticNet
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from math import sqrt
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import cross_val_score, KFold
List_of_model = []
preprocessor = make_pipeline(SelectKBest(f_classif, k='all'))
score = make_scorer(mean_squared_error, greater_is_better=False)
alphas = 10**np.linspace(10,-2,100)*0.5
Linear_Regression_pipe = make_pipeline(preprocessor, StandardScaler(), LinearRegression())
RidgeCV_Regression_pipe = make_pipeline(preprocessor, StandardScaler(), RidgeCV(alphas = alphas))
Ridge_Regression_pipe = make_pipeline(preprocessor, StandardScaler(), Ridge(alpha=0.01))
LassoCV_Regression_pipe = make_pipeline(preprocessor, StandardScaler(), LassoCV(alphas = None, cv = 10, max_iter = 100000))
Lasso_Regression_pipe = make_pipeline(preprocessor, StandardScaler(), Lasso(alpha = 0.01))
ElasticNet_Regression_pipe = make_pipeline(preprocessor, StandardScaler(), ElasticNet(alpha=0.01, l1_ratio=0.5))
RandomForestRegressor_pipe = make_pipeline(preprocessor, StandardScaler(), RandomForestRegressor(max_depth=2, random_state=0))
AdaBoostRegressor_pipe = make_pipeline(preprocessor, StandardScaler(), AdaBoostRegressor(random_state=0, n_estimators=100))
SVR_pipe = make_pipeline(preprocessor, StandardScaler(), SVR(C=1.0, epsilon=0.2))
DecisionTreeRegressor_pipe = make_pipeline(preprocessor, StandardScaler(), DecisionTreeRegressor(max_depth=3,random_state=0))
KNeighborsRegressor_pipe = make_pipeline(preprocessor, StandardScaler(), KNeighborsRegressor())
List_of_model = {'Linear_Regression_pipe': Linear_Regression_pipe,
                'Ridge_Regression_pipe': Ridge_Regression_pipe,
                'RidgeCV_Regression_pipe': RidgeCV_Regression_pipe,
                'Lasso_Regression_pipe': Lasso_Regression_pipe,
                'LassoCV_Regression_pipe': LassoCV_Regression_pipe,
                'ElasticNet_Regression_pipe': ElasticNet_Regression_pipe,
                'RandomForestRegressor_pipe': RandomForestRegressor_pipe,
                'AdaBoostRegressor_pipe': AdaBoostRegressor_pipe,
                'SVR_pipe': SVR_pipe,
                'DecisionTreeRegressor_pipe': DecisionTreeRegressor_pipe,
                'KNeighborsRegressor_pipe': KNeighborsRegressor_pipe}
def RMSE_Cross_validation(model,X,y,score):
    rmse = np.sqrt(-cross_val_score(model, X, y, cv=5, scoring=score))
    return (rmse)
dict_rmse = {}
def Model_Evaluation(model, key):
    model.fit(X_train, y_train)
    predic_train = model.predict(X_train)
    y_pred = model.predict(X_test) 
      
    print("{} Performance for training and testing set".format(key))
    print("----------------------------------------------------------------------------")
    rmse_train = RMSE_Cross_validation(model,X_train, y_train,"neg_mean_squared_error").mean()
    rmse_test = RMSE_Cross_validation(model,X_test, y_test,"neg_mean_squared_error").mean()
    print("Train Mean Squared Error:",rmse_train )
    print("Test MSE:", rmse_test)
    dict_rmse.update({key:rmse_train})
    dict_rmse.update({key+'test': rmse_test})
    plt.figure()
    plt.title(key)
    plt.scatter(predic_train, predic_train - y_train, c="green", marker="o", label="Training")
    plt.scatter(y_pred, y_pred - y_test, c="blue", marker="s", label="Testing")
    plt.xlabel("Predict value")
    plt.ylabel("Residuals")
    plt.legend()
    plt.show()
    
    plt.figure()
    plt.title(key)
    plt.scatter(predic_train, y_train, c="green", marker="o", label="Training")
    plt.scatter(y_pred, y_test, c="blue", marker="s", label="Testing")
    plt.xlabel("Predict value")
    plt.ylabel("Real value")
    plt.legend()
    plt.show()
    predictors = X_train.columns
    if (key == "Ridge_Regression_pipe"):
        sortval = pd.Series(model.named_steps['ridge'].coef_, predictors).sort_values()
        coef = pd.concat([sortval.head(10),sortval.tail(10)
                         ])
        coef.plot(kind='bar', title='Modal Coefficients')
    elif (key == "RidgeCV_Regression_pipe"):
        sortval = pd.Series(model.named_steps['ridgecv'].coef_, predictors).sort_values()
        coef = pd.concat([sortval.head(10), sortval.tail(10)])
        coef.plot(kind='bar', title='Modal Coefficients')
    elif (key == "Lasso_Regression_pipe"):
        sortval = pd.Series(model.named_steps['lasso'].coef_, predictors).sort_values()
        coef = pd.concat([sortval.head(10), sortval.tail(10)])
        coef.plot(kind='bar', title='Modal Coefficients')
    elif (key == "LassoCV_Regression_pipe"):
        sortval = pd.Series(model.named_steps['lassocv'].coef_, predictors).sort_values()
        coef = pd.concat([sortval.head(10), sortval.tail(10)])
        coef.plot(kind='bar', title='Modal Coefficients')
    elif (key == "ElasticNet_Regression_pipe" ):
        sortval = pd.Series(model.named_steps['elasticnet'].coef_, predictors).sort_values()
        coef = pd.concat([sortval.head(10), sortval.tail(10)])
        coef.plot(kind='bar', title='Modal Coefficients')
for key, model  in List_of_model.items():
    Model_Evaluation(model, key)
dict_rmse
list_model_optimize = []
alphas = 10**np.linspace(10,-2,50)*0.5
preprocessor = make_pipeline(SelectKBest(f_classif, k='all'))
LassoCV_Regression_pipe = make_pipeline(preprocessor, StandardScaler(), LassoCV(alphas=alphas, cv=5, n_jobs=-1, max_iter = 105000))
ElasticNet_Regression_pipe = make_pipeline(preprocessor, StandardScaler(), ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter = 105000))
list_model_optimize = {'LassoCV_Regression_pipe': LassoCV_Regression_pipe,
                        'ElasticNet_Regression_pipe': ElasticNet_Regression_pipe}
list_columns = test_data_set.columns.tolist()
list_columns_train = train_data_set.columns.tolist()
list_final = []
for col in list_columns:
    if col in list_columns_train:
        list_final.append(col)
train_final_set = train_data_set[list_final]
test_final_set = test_data_set[list_final]
print(train_final_set.shape)
print(test_final_set.shape)
def Model_Evaluation_submission(model, key):
    model.fit(train_final_set, target)
    predic_train = model.predict(train_final_set)
    y_pred = model.predict(test_final_set) 
      
    print("{} Performance for training and testing set".format(key))
    print("----------------------------------------------------------------------------")
    rmse_train = RMSE_Cross_validation(model,train_final_set, target,"neg_mean_squared_error").mean()
    print("Train Mean Squared Error:",rmse_train )
    print("Score:",model.score(train_final_set, target))
    
    plt.figure()
    plt.title(key)
    plt.scatter(predic_train, target, c="green", marker="o", label="Training")
    plt.xlabel("Predict value")
    plt.ylabel("Real value")
    plt.legend()
    plt.show()
    return y_pred
for key, model  in list_model_optimize.items():
    Model_Evaluation(model, key)
lasso_pred = Model_Evaluation_submission(LassoCV_Regression_pipe,"lasso")
lasso_pred
ElasticNet_pred = Model_Evaluation_submission(ElasticNet_Regression_pipe,"ElasticNet")
ElasticNet_pred
from sklearn.ensemble import BaggingRegressor
import xgboost as xgb
import lightgbm as lgbm
regrElasticNet = BaggingRegressor(base_estimator=ElasticNet(alpha=0.01, l1_ratio=0.5), n_estimators=2400,n_jobs=1, random_state=0).fit(train_final_set, target)
y_baggin_elasticNet = regrElasticNet.predict(test_final_set)
y_pred_bagging_elasticNet = np.expm1(y_baggin_elasticNet);
y_pred_bagging_elasticNet
regrElasticNet.score(train_final_set,target)
xgb_model = xgb.XGBRegressor( booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=0.6, gamma=0,
             importance_type='gain', learning_rate=0.01, max_delta_step=0,
             max_depth=100, min_child_weight=1.7817, n_estimators=2400,
             n_jobs=1, nthread=None, objective='reg:linear',
             reg_alpha=0.6, reg_lambda=0.6, scale_pos_weight=1, 
             silent=None, subsample=0.8, verbosity=0)
xgb_model_pipe = make_pipeline(preprocessor, StandardScaler(), xgb_model)
xgb_model_pipe.fit(train_final_set,target)
xgb_predict = xgb_model_pipe.predict(test_final_set)
xgb_score = xgb_model_pipe.score(train_final_set, target)
xgb_train_predict = xgb_model_pipe.predict(train_final_set)
xgb_predict = np.expm1(xgb_predict)
print(xgb_predict)
xgb_score
print(RMSLE(target, xgb_train_predict))
lgbm_model = lgbm.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
lgbm_model_pipe = make_pipeline(preprocessor, StandardScaler(), lgbm_model)
lgbm_model_pipe.fit(train_final_set,target)
lgbm_predict = lgbm_model_pipe.predict(test_final_set)
lgbm_score = lgbm_model_pipe.score(train_final_set, target)
lgbm_train_predict = lgbm_model_pipe.predict(train_final_set)
lgbm_predict = np.expm1(lgbm_predict)
print(lgbm_predict)

lgbm_score
print(RMSLE(target, lgbm_train_predict))
def RMSLE(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.preprocessing import RobustScaler
class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
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
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred
                
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self
   
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_ ])
        return self.meta_model_.predict(meta_features)
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
stacked_averaged_models = StackingAveragedModels(base_models = (xgb_model, lgbm_model, regrElasticNet),
                                                 meta_model = lasso)
n_folds = 5
def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train_final_set.values)
    rmse= np.sqrt(-cross_val_score(model, train_final_set.values, target, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)
score = rmsle_cv(lasso)
print("lasso score mean :{} score std {}".format(score.mean(),score.std()))
stacked_averaged_models.fit(train_final_set.values, target)
stacked_train_pred = stacked_averaged_models.predict(train_final_set.values)
stacked_pred = np.expm1(stacked_averaged_models.predict(test_final_set.values))
print(RMSLE(target, stacked_train_pred))
print('RMSLE score train :',RMSLE(target,stacked_train_pred*0.80 +xgb_train_predict*0.1 + lgbm_train_predict*0.1 ))
predict_ens_final = stacked_pred*0.80 + xgb_predict*0.1 + lgbm_predict*0.1
predict_ens_final
submission_id = pd.DataFrame(test_final_set["Id"])
submission_salePrice = pd.DataFrame({"SalePrice":predict_ens_final})
submission = pd.concat([submission_id,submission_salePrice],axis=1)
submission.to_csv('house_submission.csv', header= True, index= False) 