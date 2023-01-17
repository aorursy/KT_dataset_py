# Default imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline
train = pd.read_csv('../input/train.csv', index_col='Id')
test = pd.read_csv('../input/test.csv', index_col='Id')
train.head()
test.head()
# all feature columns
train.columns[:-1]
plt.scatter(train.GrLivArea, train.SalePrice)
plt.xlabel('GrLivArea')
plt.ylabel('SalePrice')
train = train.drop(train[(train.GrLivArea>4000) & (train.SalePrice<300000)].index)
plt.scatter(train.GrLivArea, train.SalePrice)
plt.xlabel('GrLivArea')
plt.ylabel('SalePrice')
plt.scatter(train.LotFrontage, train.SalePrice)
plt.xlabel('LotFrontage')
plt.ylabel('SalePrice')
train = train.drop(train[train.LotFrontage > 300].index)
plt.scatter(train.LotFrontage, train.SalePrice)
plt.xlabel('LotFrontage')
plt.ylabel('SalePrice')
plt.scatter(train[train.MasVnrArea != 0].MasVnrArea, train[train.MasVnrArea != 0].SalePrice)
plt.xlabel('MasVnrArea')
plt.ylabel('SalePrice')
train = train.drop(train[train.MasVnrArea > 1500].index)
plt.scatter(train[train.MasVnrArea != 0].MasVnrArea, train[train.MasVnrArea != 0].SalePrice)
plt.xlabel('MasVnrArea')
plt.ylabel('SalePrice')
train_last_id = train.shape[0]
y = train.SalePrice
df = pd.concat((train, test))
df.drop(['SalePrice'], axis=1, inplace=True)
df.shape
print('{} NaN values'.format(df.isnull().sum().sum()))
df.columns[df.isnull().any()]
cols = ['Alley', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
        'BsmtQual', 'Fence', 'FireplaceQu', 'GarageCond', 'GarageFinish',
        'GarageQual', 'GarageType', 'MiscFeature', 'PoolQC']
df.loc[:, cols] = df.loc[:, cols].fillna('None')
print('{} NaN values'.format(df[cols].isnull().sum().sum()))
cols = ['Electrical', 'Exterior1st', 'Exterior2nd', 'Functional', 'KitchenQual', 'MasVnrType', 'MSZoning', 'SaleType']
for col in cols:
    df.loc[:, col].fillna(df[col].mode()[0], inplace=True)
print('{} NaN values'.format(df[cols].isnull().sum().sum()))
cols = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'BsmtFullBath', 'BsmtHalfBath',
        'GarageYrBlt', 'GarageArea', 'GarageCars', 'MasVnrArea', 'TotalBsmtSF', 'LotFrontage']
for col in cols:
    print('{}: {} rows'.format(col,df[df[col].isnull()].shape[0]))
bsmt_cols = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2',
             'BsmtFinSF2', 'BsmtUnfSF', 'BsmtFullBath', 'BsmtHalfBath','TotalBsmtSF']
null_cols = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'BsmtFullBath', 'BsmtHalfBath', 'TotalBsmtSF']
df.loc[df[null_cols].isnull().any(axis=1), bsmt_cols]
df.loc[df[null_cols].isnull().any(axis=1), bsmt_cols] = df.loc[df[null_cols].isnull().any(axis=1), bsmt_cols].fillna(0.0)
df.loc[(df.index == 2121) | (df.index == 2189), bsmt_cols]
garage_cols = ['GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond']
null_cols = ['GarageArea', 'GarageCars']
df.loc[df[null_cols].isnull().any(axis=1), garage_cols]
cols = ['GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond']
for col in cols:
    df.loc[df.index == 2577, col] = df.loc[df.GarageType == 'Detchd', col].mode().iloc[0]
df.loc[df.index == 2577, garage_cols]
df.loc[df.GarageYrBlt.isnull(), garage_cols]
df.loc[df.GarageYrBlt.isnull(), garage_cols]
cols = ['GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond']
for col in cols:
    df.loc[df.index == 2127, col] = df.loc[df.GarageType == 'Detchd', col].mode().iloc[0]
df.loc[df.index == 2127, garage_cols]
df.loc[df.GarageYrBlt.isnull(), 'GarageYrBlt'] = 0.0
cols = ['MasVnrType', 'MasVnrArea']
df.loc[df.MasVnrArea.isnull(), cols]
df.loc[df.MasVnrArea.isnull(), 'MasVnrArea'] = df.loc[:, 'MasVnrArea'].mode().iloc[0]
df.loc[:, 'LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(
    lambda x: x.fillna(np.random.randint(x.median() - x.std(), x.mean() + x.std())))
df.loc[:, 'LotFrontage'] = df.LotFrontage.astype(int)
df.drop('Utilities', axis=1, inplace=True)
df = df.replace({"LotShape" : {"IR3" : 1, "IR2" : 2, "IR1" : 3, "Reg" : 4},
                 "LandSlope" : {"Sev" : 1, "Mod" : 2, "Gtl" : 3},
                 "ExterQual" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5},
                 "ExterCond" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5},
                 "BsmtQual" : {"None" : 0, "Po" : 1, "Fa" : 2, "TA": 3, "Gd" : 4, "Ex" : 5},
                 "BsmtCond" : {"None" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                 "BsmtExposure" : {"None" : 0, "Mn" : 1, "Av": 2, "Gd" : 3},
                 "BsmtFinType1" : {"None" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4, "ALQ" : 5, "GLQ" : 6},
                 "BsmtFinType2" : {"None" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4, "ALQ" : 5, "GLQ" : 6},
                 "HeatingQC" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                 "KitchenQual" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                 "Functional" : {"Sal" : 1, "Sev" : 2, "Maj2" : 3, "Maj1" : 4, "Mod": 5, "Min2" : 6, "Min1" : 7, "Typ" : 8},
                 "FireplaceQu" : {"None" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},                       
                 "GarageQual" : {"None" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                 "GarageCond" : {"None" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                 "PavedDrive" : {"N" : 0, "P" : 1, "Y" : 2},
                 "PoolQC" : {"None" : 0, "Fa" : 1, "TA" : 2, "Gd" : 3, "Ex" : 4},
                 "Fence" : {'None' : 0, "MnWw" : 1, "GdWo" : 2, "MnPrv" : 3, "GdPrv" : 4}
                 })
df["SimplOverallQual"] = df.OverallQual.replace({1 : 1, 2 : 1, 3 : 1, # bad
                                                       4 : 2, 5 : 2, 6 : 2, # average
                                                       7 : 3, 8 : 3, 9 : 3, 10 : 3 # good
                                                      })
df["SimplOverallCond"] = df.OverallCond.replace({1 : 1, 2 : 1, 3 : 1, # bad
                                                       4 : 2, 5 : 2, 6 : 2, # average
                                                       7 : 3, 8 : 3, 9 : 3, 10 : 3 # good
                                                      })
df["SimplPoolQC"] = df.PoolQC.replace({1 : 1, 2 : 1, # average
                                             3 : 2, 4 : 2 # good
                                            })
df["SimplGarageCond"] = df.GarageCond.replace({1 : 1, # bad
                                                     2 : 1, 3 : 1, # average
                                                     4 : 2, 5 : 2 # good
                                                    })
df["SimplGarageQual"] = df.GarageQual.replace({1 : 1, # bad
                                                     2 : 1, 3 : 1, # average
                                                     4 : 2, 5 : 2 # good
                                                    })
df["SimplFireplaceQu"] = df.FireplaceQu.replace({1 : 1, # bad
                                                       2 : 1, 3 : 1, # average
                                                       4 : 2, 5 : 2 # good
                                                      })
df["SimplFireplaceQu"] = df.FireplaceQu.replace({1 : 1, # bad
                                                       2 : 1, 3 : 1, # average
                                                       4 : 2, 5 : 2 # good
                                                      })
df["SimplFunctional"] = df.Functional.replace({1 : 1, 2 : 1, # bad
                                                     3 : 2, 4 : 2, # major
                                                     5 : 3, 6 : 3, 7 : 3, # minor
                                                     8 : 4 # typical
                                                    })
df["SimplKitchenQual"] = df.KitchenQual.replace({1 : 1, # bad
                                                       2 : 1, 3 : 1, # average
                                                       4 : 2, 5 : 2 # good
                                                      })
df["SimplHeatingQC"] = df.HeatingQC.replace({1 : 1, # bad
                                                   2 : 1, 3 : 1, # average
                                                   4 : 2, 5 : 2 # good
                                                  })
df["SimplBsmtFinType1"] = df.BsmtFinType1.replace({1 : 1, # unfinished
                                                         2 : 1, 3 : 1, # rec room
                                                         4 : 2, 5 : 2, 6 : 2 # living quarters
                                                        })
df["SimplBsmtFinType2"] = df.BsmtFinType2.replace({1 : 1, # unfinished
                                                         2 : 1, 3 : 1, # rec room
                                                         4 : 2, 5 : 2, 6 : 2 # living quarters
                                                        })
df["SimplBsmtCond"] = df.BsmtCond.replace({1 : 1, # bad
                                                 2 : 1, 3 : 1, # average
                                                 4 : 2, 5 : 2 # good
                                                })
df["SimplBsmtQual"] = df.BsmtQual.replace({1 : 1, # bad
                                                 2 : 1, 3 : 1, # average
                                                 4 : 2, 5 : 2 # good
                                                })
df["SimplExterCond"] = df.ExterCond.replace({1 : 1, # bad
                                                   2 : 1, 3 : 1, # average
                                                   4 : 2, 5 : 2 # good
                                                  })
df["SimplExterQual"] = df.ExterQual.replace({1 : 1, # bad
                                                   2 : 1, 3 : 1, # average
                                                   4 : 2, 5 : 2 # good
                                                  })
df["OverallQC"] = df["OverallQual"] * df["OverallCond"]
# Overall quality of the garage
df["GarageQC"] = df["GarageQual"] * df["GarageCond"]
# Overall quality of the exterior
df["ExterQC"] = df["ExterQual"] * df["ExterCond"]
# Overall kitchen score
df["KitchenScore"] = df["KitchenAbvGr"] * df["KitchenQual"]
# Overall fireplace score
df["FireplaceScore"] = df["Fireplaces"] * df["FireplaceQu"]
# Overall garage score
df["GarageScore"] = df["GarageArea"] * df["GarageQual"]
# Overall pool score
df["PoolScore"] = df["PoolArea"] * df["PoolQC"]
# Simplified overall quality of the house
df["SimplOverallQC"] = df["SimplOverallQual"] * df["SimplOverallCond"]
# Simplified overall quality of the exterior
df["SimplExterQC"] = df["SimplExterQual"] * df["SimplExterCond"]
# Simplified overall pool score
df["SimplPoolScore"] = df["PoolArea"] * df["SimplPoolQC"]
# Simplified overall garage score
df["SimplGarageScore"] = df["GarageArea"] * df["SimplGarageQual"]
# Simplified overall fireplace score
df["SimplFireplaceScore"] = df["Fireplaces"] * df["SimplFireplaceQu"]
# Simplified overall kitchen score
df["SimplKitchenScore"] = df["KitchenAbvGr"] * df["SimplKitchenQual"]
# Total number of bathrooms
df["TotalBath"] = df["BsmtFullBath"] + 0.5 * df["BsmtHalfBath"] + df["FullBath"] + 0.5 * df["HalfBath"]
# Total SF for house (incl. basement)
df["AllSF"] = df["GrLivArea"] + df["TotalBsmtSF"]
# Total SF for 1st + 2nd floors
df["AllFlrsSF"] = df["1stFlrSF"] + df["2ndFlrSF"]
# Total SF for porch
df["AllPorchSF"] = df["OpenPorchSF"] + df["EnclosedPorch"] + df["3SsnPorch"] + df["ScreenPorch"]
# Has alley access or not
df['HasAlley'] = 1
df.loc[df.Alley == 'None', 'HasAlley'] = 0
# Has basement or not
df['HasBsmt'] = 1
df.loc[df.BsmtQual == 0, 'HasBsmt'] = 0
# Has atleast 1 fireplace or not
df['HasFireplace'] = 1
df.loc[df.Fireplaces == 0, 'HasFireplace'] = 0
# Has garage or not
df['HasGarage'] = 1
df.loc[df.GarageType == 'None', 'HasGarage'] = 0
# Has pool or not
df['HasPool'] = 1
df.loc[df.PoolArea == 0.0, 'HasBsmt'] = 0
# Has masonry veneer or not
df["HasMasVnr"] = 1
df.loc[df.MasVnrType == 'None', 'HasMasVnr'] = 0
# House completed before sale or not
df["BoughtOffPlan"] = 0
df.loc[df.SaleCondition == 'Partial', 'BoughtOffPlan'] = 1
# get our new train data
train = df[:train_last_id]
train['SalePrice'] = y
#Find most important features relative to target
print("Find most important features relative to target")
corr = train.corr()
corr.sort_values(["SalePrice"], ascending = False, inplace = True)
print(corr.SalePrice)
# Create new features
# 3* Polynomials on the top 10 existing features
df["AllSF-2"] = df["AllSF"] ** 2
df["AllSF-3"] = df["AllSF"] ** 3
df["AllSF-Sq"] = np.sqrt(df["AllSF"])
df["OverallQual-s2"] = df["OverallQual"] ** 2
df["OverallQual-s3"] = df["OverallQual"] ** 3
df["OverallQual-Sq"] = np.sqrt(df["OverallQual"])
df["AllFlrsSF-2"] = df["AllFlrsSF"] ** 2
df["AllFlrsSF-3"] = df["AllFlrsSF"] ** 3
df["AllFlrsSF-Sq"] = np.sqrt(df["AllFlrsSF"])
df["GrLivArea-2"] = df["GrLivArea"] ** 2
df["GrLivArea-3"] = df["GrLivArea"] ** 3
df["GrLivArea-Sq"] = np.sqrt(df["GrLivArea"])
df["ExterQual-2"] = df["ExterQual"] ** 2
df["ExterQual-3"] = df["ExterQual"] ** 3
df["ExterQual-Sq"] = np.sqrt(df["ExterQual"])
df["SimplOverallQual-s2"] = df["SimplOverallQual"] ** 2
df["SimplOverallQual-s3"] = df["SimplOverallQual"] ** 3
df["SimplOverallQual-Sq"] = np.sqrt(df["SimplOverallQual"])
df["KitchenQual-2"] = df["KitchenQual"] ** 2
df["KitchenQual-3"] = df["KitchenQual"] ** 3
df["KitchenQual-Sq"] = np.sqrt(df["KitchenQual"])
df["TotalBsmtSF-2"] = df["TotalBsmtSF"] ** 2
df["TotalBsmtSF-3"] = df["TotalBsmtSF"] ** 3
df["TotalBsmtSF-Sq"] = np.sqrt(df["TotalBsmtSF"])
df["GarageCars-2"] = df["GarageCars"] ** 2
df["GarageCars-3"] = df["GarageCars"] ** 3
df["GarageCars-Sq"] = np.sqrt(df["GarageCars"])
df["TotalBath-2"] = df["TotalBath"] ** 2
df["TotalBath-3"] = df["TotalBath"] ** 3
df["TotalBath-Sq"] = np.sqrt(df["TotalBath"])
from scipy import stats
#histogram and normal probability plot
sns.distplot(y, fit=stats.norm)
fig = plt.figure()
res = stats.probplot(y, plot=plt)
#applying log transformation
y = np.log1p(y)
#transformed histogram and normal probability plot
sns.distplot(y, fit=stats.norm)
fig = plt.figure()
res = stats.probplot(y, plot=plt)
skewness = df[df.select_dtypes(exclude=['object']).columns].apply(lambda x: stats.skew(x))
skewness = skewness[abs(skewness) > 0.5]
print(str(skewness.shape[0]) + " skewed numerical features to log transform")
skewed_features = skewness.index
df[skewed_features] = np.log1p(df[skewed_features])
df.BoughtOffPlan
df = pd.get_dummies(df, columns=df.select_dtypes(include=['object']).columns)
df.shape
train = df[:train_last_id]
test = df[train_last_id:]
X = train
from sklearn.model_selection import KFold, cross_val_score
def rmsle_cv(model):
    kf = KFold(5, shuffle=True, random_state=42).get_n_splits(X.values)
    rmse = np.sqrt(-cross_val_score(model, X.values, y.values, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
   
    # We again fit the data on clones of the original models
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)
        
        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred
                
        # Now train the cloned  meta-model using the out-of-fold predictions as new feature
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self
   
    #Do the predictions of all base models on the test data and use the averaged predictions as 
    #meta-features for the final prediction which is done by the meta-model
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_ ])
        return self.meta_model_.predict(meta_features)
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
import xgboost
import lightgbm
linreg = LinearRegression()
score = rmsle_cv(linreg)
print("Linear Regression score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
ada = AdaBoostRegressor(n_estimators=402,random_state=42)
score = rmsle_cv(ada)
print("Ada Boost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
GBoost = GradientBoostingRegressor(n_estimators=402,random_state=42)
score = rmsle_cv(GBoost)
print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
forest = RandomForestRegressor(n_estimators=402,random_state=42)
score = rmsle_cv(forest)
print("Random Forest score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
xgb = xgboost.XGBRegressor(n_estimators=402, random_state =42)
score = rmsle_cv(xgb)
print("XGBoost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
lgb = lightgbm.LGBMRegressor(objective='regression', n_estimators=402, random_state =42)
score = rmsle_cv(lgb)
print("LightGBM score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
stacked_averaged_models = StackingAveragedModels(base_models = (GBoost, xgb, lgb),
                                                 meta_model = linreg)

score = rmsle_cv(stacked_averaged_models)
print("Stacking Averaged models score: {:.4f} ({:.4f})".format(score.mean(), score.std()))
stacked_averaged_models.fit(X.values, y.values)
res = pd.DataFrame()
res['Id'] = test.index
res['SalePrice'] = np.expm1(stacked_averaged_models.predict(test.values))
res.to_csv('submission.csv',index=False)