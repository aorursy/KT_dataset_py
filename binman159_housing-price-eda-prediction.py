import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

%matplotlib inline
%config InlineBackend.figure_format = 'retina'
df_train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv", index_col="Id")
df_test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv", index_col="Id")
msno.matrix(df_train, figsize=(75, 30), labels=True) # so much null (white space)
df = pd.concat([df_train, df_test], axis=0)
df.head()
drop_col = ["Neighborhood", "BsmtFinSF2", "MiscFeature"]
val_col = ["LotFrontage", "MasVnrArea", "GarageYrBlt", "LotFrontage", "BsmtFinSF1", "BsmtUnfSF", "TotalBsmtSF", "BsmtFullBath", "BsmtHalfBath", "GarageCars", "GarageArea"]
cat_col = ["BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "Electrical", "GarageType", "GarageFinish", 
           "GarageQual", "GarageCond", "MasVnrType", "FireplaceQu", "Alley"]
binary_col = ["PoolQC", "Fence"]
# Add: Alley, FireplaceQu
df = df.drop(columns=drop_col)
for i in val_col: 
    df[i].fillna(0, inplace=True)
for i in cat_col: 
    df[i] = df[i].fillna("No")
for i in binary_col: 
    df[i].fillna(0, inplace=True)
    df[i] = df[i].apply(lambda x: 0 if x == 0 else 1)
cor_df = df.select_dtypes(np.number).corr()
plt.figure(figsize=(8, 6))
sns.heatmap(cor_df)
df = pd.get_dummies(df.iloc[:, df.columns != "SalePrice"])
X = df.iloc[:df_train.shape[0], :]
y = df_train["SalePrice"]
X_test = df.iloc[df_train.shape[0] :, :]
msno.matrix(X, figsize=(75, 30), labels=True) #NA checked
msno.matrix(X_test, figsize=(75, 30), labels=True)
sns.distplot(y)
fig = plt.figure(figsize=(12, 5))
fig.add_subplot(121)
sns.boxplot(x=X["PoolQC"], y=y)
fig.add_subplot(122)
sns.boxplot(x=X["Fence"], y=y)
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=.1, random_state=101)
print(X_train.shape, X_val.shape, X_test.shape)
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
alpha_L2 = [15, 20, 30, 35, 40, 45, 50, 55]
parameters = {'alpha':alpha_L2, 'solver':('svd', 'saga')}
clf = GridSearchCV(Ridge(random_state=101), parameters)
clf.fit(X_train.select_dtypes(np.number), y_train)
clf.best_params_
print(f'train score: {clf.best_score_}')
print(f'validation score: {clf.score(X_val, y_val)}')
y_pred_ridge = clf.predict(X_test)

sub_1 = pd.concat([pd.Series(X_test.index), pd.Series(y_pred_ridge)], axis=1).rename(columns={0:"SalePrice"})
sub_1.head()
# sub_1.to_csv("submission_Ridge.csv", index=False)
from sklearn.linear_model import Lasso
alpha_L1 = [200, 220, 250, 280, 300]
parameters = {'alpha':alpha_L1}
clf = GridSearchCV(Lasso(random_state=101), parameters)
clf.fit(X_train.select_dtypes(np.number), y_train)
clf.best_params_
print(f'train score: {clf.best_score_}')
print(f'validation score: {clf.score(X_val, y_val)}')
y_pred_lasso = clf.predict(X_test)

sub_2 = pd.concat([pd.Series(X_test.index), pd.Series(y_pred_lasso)], axis=1).rename(columns={0:"SalePrice"})
sub_2.head()
# sub_2.to_csv("submission_Lasso.csv", index=False)
from xgboost import XGBRegressor
xgboost = XGBRegressor(learning_rate=0.005,n_estimators=4500,
                                     max_depth=5, min_child_weight=0,
                                     gamma=0, subsample=0.7,
                                     colsample_bytree=0.7,
                                     objective='reg:linear', nthread=-1,
                                     scale_pos_weight=1, seed=27,
                                     reg_alpha=0.005)
xgboost.fit(X_train, y_train)
print(f'train score: {xgboost.score(X_train, y_train)}')
print(f'validation score: {xgboost.score(X_val, y_val)}')
y_pred_xgboost = xgboost.predict(X_test)

sub_3 = pd.concat([pd.Series(X_test.index), pd.Series(y_pred_xgboost)], axis=1).rename(columns={0:"SalePrice"})
sub_3.head()
# sub_3.to_csv("submission_XGBoost.csv", index=False)
# parameters = {'learning_rate': [0.005], 'n_estimators':[4500], 'max_depth':[5], 'reg_alpha':[0.005]}
# clf = GridSearchCV(XGBRegressor(min_child_weight=0, gamma=0, 
#                                         colsample_bytree=0.7, objective='reg:linear', nthread=-1,
#                                         scale_pos_weight=1, subsample=.7, seed=27), parameters)
# clf.fit(X_train.select_dtypes(np.number), y_train)
# clf.best_params_

# # {'learning_rate': 0.005,
# #  'max_depth': 5,
# #  'n_estimators': 4500,
# #  'reg_alpha': 0.005}
# print(f'train score: {clf.best_score_}')
# print(f'validation score: {clf.score(X_val, y_val)}')

# # train score: 0.8856452700799474
# # validation score: 0.9125119162731631
# y_pred = clf.predict(X_test)

# sub_3 = pd.concat([pd.Series(X_test.index), pd.Series(y_pred)], axis=1).rename(columns={0:"SalePrice"})
# sub_3.head()
# sub_3.to_csv("submission_XGBoost_Optimized.csv", index=False)
from lightgbm import LGBMRegressor
lightgbm = LGBMRegressor(objective='regression', 
                                       num_leaves=5,
                                       learning_rate=0.01, 
                                       n_estimators=5500,
                                       max_bin=200, 
                                       bagging_fraction=0.75,
                                       bagging_freq=5, 
                                       bagging_seed=7,
                                       feature_fraction=0.2,
                                       feature_fraction_seed=7,
                                       verbose=-1,
                                       )
lightgbm.fit(X_train, y_train)
print(f'train score: {lightgbm.score(X_train, y_train)}')
print(f'validation score: {lightgbm.score(X_val, y_val)}')
y_pred_lgbm = lightgbm.predict(X_test)

sub_4 = pd.concat([pd.Series(X_test.index), pd.Series(y_pred_lgbm)], axis=1).rename(columns={0:"SalePrice"})
sub_4.head()
# sub_4.to_csv("submission_LightGBM.csv", index=False)
# parameters = {'learning_rate': [0.01], 'n_estimators':[5500, 6000, 7500], 'num_leaves':[3, 4, 5]}
# clf = GridSearchCV(LGBMRegressor(objective='regression', 
#                                        max_bin=200, 
#                                        bagging_fraction=0.75,
#                                        bagging_freq=5, 
#                                        bagging_seed=7,
#                                        feature_fraction=0.2,
#                                        feature_fraction_seed=7,
#                                        verbose=-1,), parameters)
# clf.fit(X_train.select_dtypes(np.number), y_train)
# clf.best_params_

# #{'learning_rate': 0.01, 'n_estimators': 5500, 'num_leaves': 5}
# print(f'train score: {clf.best_score_}')
# print(f'validation score: {clf.score(X_val, y_val)}')

# #train score: 0.8856209897695116
# # validation score: 0.9069389402737372
# y_pred = clf.predict(X_test)

# sub_5 = pd.concat([pd.Series(X_test.index), pd.Series(y_pred)], axis=1).rename(columns={0:"SalePrice"})
# sub_5.head()
# sub_5.to_csv("submission_LightGBM_Optimized.csv", index=False)
mul_lgbm = 0.30
mul_xgboost = 0.47
mul_ridge = 0.03
mul_lasso = 0.2
print(mul_lgbm + mul_xgboost + mul_ridge + mul_lasso)
y_pred_final_1 = (y_pred_lgbm * mul_lgbm) + (y_pred_xgboost * mul_xgboost) + (y_pred_ridge * mul_ridge) + (y_pred_lasso * mul_lasso)

sub_6 = pd.concat([pd.Series(X_test.index), pd.Series(y_pred_final_1)], axis=1).rename(columns={0:"SalePrice"})
sub_6.tail()
sub_6.to_csv("./submission_Merge_1.csv", index=False) #0.12077