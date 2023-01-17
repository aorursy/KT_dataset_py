import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from IPython.display import Image



import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

warnings.filterwarnings("ignore", category=DeprecationWarning)
train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
train.head(3)
test.head(3)
fig, ax = plt.subplots()

ax.scatter(x = train['GrLivArea'], y = train['SalePrice'])

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('GrLivArea', fontsize=13)

plt.show()
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)
fig, ax = plt.subplots()

ax.scatter(train['GrLivArea'], train['SalePrice'])

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('GrLivArea', fontsize=13)

plt.show()
cor = train.corr()

cor_fe = cor.index[abs(cor['SalePrice']) >= 0.3]

cor_fe
plt.figure(figsize=(15,10))

sns.heatmap(train[cor_fe].corr(),annot=True)
fe_name = list(test)

df_train = train[fe_name]

df = pd.concat((df_train,test))
print(train.shape, test.shape, df.shape)
from scipy import stats

from scipy.stats import norm
sns.distplot(train['SalePrice'],fit = norm)
stats.probplot(train['SalePrice'], plot=plt)
train['SalePrice'] = np.log1p(train["SalePrice"])

sns.distplot(train['SalePrice'],fit=norm)
stats.probplot(train['SalePrice'], plot=plt)
target = train['SalePrice']
null_df = (df.isna().sum() / len(df)) *100

null_df = null_df.drop(null_df[null_df == 0].index).sort_values(ascending=False)[:30]

missing_data = pd.DataFrame({'Missing Ratio' :null_df})

missing_data.head(20)
f, ax = plt.subplots(figsize=(15, 12))

plt.xticks(rotation='90')

sns.barplot(x=null_df.index, y=null_df)
df["PoolQC"] = df["PoolQC"].fillna("None")
df["MiscFeature"] = df["MiscFeature"].fillna("None")
df["Alley"] = df["Alley"].fillna("None")
df["Fence"] = df["Fence"].fillna("None")
df["FireplaceQu"] = df["FireplaceQu"].fillna("None")
df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):

    df[col] = df[col].fillna('None')
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):

    df[col] = df[col].fillna(0)
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):

    df[col] = df[col].fillna(0)
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):

    df[col] = df[col].fillna('None')
df["MasVnrType"] = df["MasVnrType"].fillna("None")

df["MasVnrArea"] = df["MasVnrArea"].fillna(0)
df['MSZoning'] = df['MSZoning'].fillna(df['MSZoning'].mode()[0])
df = df.drop(['Utilities'], axis=1)
df["Functional"] = df["Functional"].fillna("Typ")
df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])
df['KitchenQual'] = df['KitchenQual'].fillna(df['KitchenQual'].mode()[0])
df['Exterior1st'] = df['Exterior1st'].fillna(df['Exterior1st'].mode()[0])

df['Exterior2nd'] = df['Exterior2nd'].fillna(df['Exterior2nd'].mode()[0])
df['SaleType'] = df['SaleType'].fillna(df['SaleType'].mode()[0])
df['MSSubClass'] = df['MSSubClass'].fillna("None")
null_df = (df.isna().sum() / len(df)) *100

null_df = null_df.drop(null_df[null_df == 0].index).sort_values(ascending=False)[:30]

missing_data = pd.DataFrame({'Missing Ratio' :null_df})

missing_data.head(20)
#MSSubClass

df['MSSubClass'] = df['MSSubClass'].apply(str)

#OverallCond

df['OverallCond'] = df['OverallCond'].astype(str)

#YrSold,MoSold

df['YrSold'] = df['YrSold'].astype(str)

df['MoSold'] = df['MoSold'].astype(str)
df_obj = df.select_dtypes(include='object')

df_obj.head(3)
li_obj = list(df_obj.columns)
df_num = df.select_dtypes(exclude = 'object')

df_num.head(3)
li_num = list(df_num.columns)
from sklearn.preprocessing import LabelEncoder

cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 

        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 

        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',

        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 

        'YrSold', 'MoSold')



for c in cols:

    lb = LabelEncoder() 

    lb.fit(list(df[c].values)) 

    df[c] = lb.transform(list(df[c].values))
df['TotalSF'] = (df['TotalBsmtSF'] 

                       + df['1stFlrSF'] 

                       + df['2ndFlrSF'])



df['YrBltAndRemod'] = df['YearBuilt'] + df['YearRemodAdd']



df['Total_sqr_footage'] = (df['BsmtFinSF1'] 

                                 + df['BsmtFinSF2'] 

                                 + df['1stFlrSF'] 

                                 + df['2ndFlrSF']

                                )

                                 



df['Total_Bathrooms'] = (df['FullBath'] 

                               + (0.5 * df['HalfBath']) 

                               + df['BsmtFullBath'] 

                               + (0.5 * df['BsmtHalfBath'])

                              )

                               



df['Total_porch_sf'] = (df['OpenPorchSF'] 

                              + df['3SsnPorch'] 

                              + df['EnclosedPorch'] 

                              + df['ScreenPorch'] 

                              + df['WoodDeckSF']

                             )
df['haspool'] = df['PoolArea'].apply(lambda x: 1 if x > 0 else 0)

df['has2ndfloor'] = df['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)

df['hasgarage'] = df['GarageArea'].apply(lambda x: 1 if x > 0 else 0)

df['hasbsmt'] = df['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)

df['hasfireplace'] = df['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
row = 11

col = 3 



fig, axs = plt.subplots(row,col, figsize = (col*3,row*4))



for r in range(0,row):

    for c in range(0,col):

        i = r*col + c

        if i < len(li_num):

            sns.regplot(train[li_num[i]],target , ax = axs[r][c])
stats.pearsonr(train[li_num[11]],target)
strong_num = ['OverallQual','YearBuilt','YearRemodAdd','TotalBsmtSF','1stFlrSF',

             'FullBath','TotRmsAbvGrd','GarageYrBlt','GarageCars','GrLivArea']
row = 12

col = 4 



fig, axs = plt.subplots(row,col, figsize = (col*4,row*3))



for r in range(0,row):

    for c in range(0,col):

        i = r*col + c

        if i < len(li_obj):

            sns.boxplot(train[li_obj[i]],target , ax = axs[r][c])
strong_obj = [ 'MSZoning', 'Neighborhood', 'Condition2', 'MasVnrType', 'ExterQual', 

                'BsmtQual','CentralAir', 'Electrical', 'KitchenQual', 'SaleType']
for li in strong_obj:

    sns.violinplot(x= li, y = target, data=train)

    plt.show()
numeric_features = df.dtypes[df.dtypes != "object"].index
from scipy.stats import skew 

skewness = df[numeric_features].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
high_skewness = skewness[abs(skewness) > 0.5]

skew_feat = high_skewness.index
print(high_skewness)

print(skew_feat)
df[['MiscVal', 'PoolArea', 'haspool', 'LotArea', 'LowQualFinSF',

       '3SsnPorch', 'LandSlope', 'KitchenAbvGr', 'BsmtFinSF2', 'EnclosedPorch',

       'ScreenPorch', 'BsmtHalfBath', 'MasVnrArea', 'OpenPorchSF',

       'WoodDeckSF', 'Total_porch_sf', '1stFlrSF', 'Total_sqr_footage',

       'LotFrontage', 'GrLivArea', 'TotalSF', 'BsmtFinSF1', 'BsmtUnfSF',

       '2ndFlrSF', 'TotRmsAbvGrd', 'Fireplaces', 'HalfBath', 'TotalBsmtSF',

       'BsmtFullBath', 'OverallCond', 'YearBuilt', 'GarageFinish', 'LotShape',

       'MoSold', 'Alley', 'BsmtExposure', 'KitchenQual', 'ExterQual', 'Fence',

       'ExterCond', 'BsmtCond', 'PavedDrive', 'BsmtFinType2', 'GarageQual',

       'CentralAir', 'GarageCond', 'GarageYrBlt', 'hasgarage', 'Functional',

       'hasbsmt', 'Street', 'PoolQC']].head(3)
from scipy.special import boxcox1p

lam = 0.15

for feat in skew_feat:

    df[feat] = boxcox1p(df[feat], lam)
df[['MiscVal', 'PoolArea', 'haspool', 'LotArea', 'LowQualFinSF',

       '3SsnPorch', 'LandSlope', 'KitchenAbvGr', 'BsmtFinSF2', 'EnclosedPorch',

       'ScreenPorch', 'BsmtHalfBath', 'MasVnrArea', 'OpenPorchSF',

       'WoodDeckSF', 'Total_porch_sf', '1stFlrSF', 'Total_sqr_footage',

       'LotFrontage', 'GrLivArea', 'TotalSF', 'BsmtFinSF1', 'BsmtUnfSF',

       '2ndFlrSF', 'TotRmsAbvGrd', 'Fireplaces', 'HalfBath', 'TotalBsmtSF',

       'BsmtFullBath', 'OverallCond', 'YearBuilt', 'GarageFinish', 'LotShape',

       'MoSold', 'Alley', 'BsmtExposure', 'KitchenQual', 'ExterQual', 'Fence',

       'ExterCond', 'BsmtCond', 'PavedDrive', 'BsmtFinType2', 'GarageQual',

       'CentralAir', 'GarageCond', 'GarageYrBlt', 'hasgarage', 'Functional',

       'hasbsmt', 'Street', 'PoolQC']].head(3)
df = pd.get_dummies(df)

print(df.shape)
new_train = df[:train.shape[0]]

new_test = df[train.shape[0]:]
new_train = pd.concat([new_train,target], axis=1, sort=False)
corr_new_train = new_train.corr()

plt.figure(figsize=(5,15))

sns.heatmap(corr_new_train[['SalePrice']].sort_values(by=['SalePrice'],

                                ascending=False).head(30),annot=True)
col_corr_dict = corr_new_train['SalePrice'].sort_values(ascending=False).to_dict()
best_columns=[]

for key,value in col_corr_dict.items():

    if ((value>=0.33) & (value<0.9)) | (value<=-0.325):

        best_columns.append(key)

print(len(best_columns))
new_train = new_train.drop(['SalePrice'], axis=1)

new_train = new_train.drop(['Id'], axis=1)

new_test = new_test.drop(['Id'], axis=1)
final_train = new_train[best_columns]

final_test = new_test[best_columns]

final_num = list(final_train.columns)
row = 19

col = 2



fig, axs = plt.subplots(row,col, figsize = (20,60))

fig.subplots_adjust(hspace=0.8)



for r in range(0,row):

    for c in range(0,col):

        i = r*col + c

        if i < len(best_columns):

            sns.regplot(final_train[final_num[i]],target,fit_reg=True,marker='o', ax = axs[r][c])
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV

from sklearn.metrics import mean_squared_error

from sklearn.pipeline import make_pipeline

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from xgboost import XGBRegressor

from lightgbm import LGBMRegressor

from sklearn.linear_model import ElasticNet, Lasso, LinearRegression
from sklearn.preprocessing import RobustScaler
rbst_scaler=RobustScaler()

X_rbst=rbst_scaler.fit_transform(new_train)

test_rbst=rbst_scaler.transform(new_test)
import statsmodels.api as sm
model = sm.OLS(target.values, new_train)
re = model.fit()
re.summary()
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()

vif['Features'] = new_train.columns

vif['vif'] = [variance_inflation_factor(

    new_train.values, i) for i in range(new_train.shape[1])]
vif.sort_values(by='vif',ascending=False)[165:190]
from sklearn.preprocessing import RobustScaler

rbst_scaler=RobustScaler()

X_rbst=rbst_scaler.fit_transform(new_train)

test_rbst=rbst_scaler.transform(new_test)
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
kfold = KFold(n_splits=4)
random_state = 1

reg = []



reg.append(Lasso(random_state = random_state))

reg.append(ElasticNet(random_state = random_state))

reg.append(RandomForestRegressor(random_state=random_state))

reg.append(GradientBoostingRegressor(random_state=random_state))

reg.append(XGBRegressor(silent=True,random_state=random_state))

reg.append(LGBMRegressor(verbose_eval=False,random_state = random_state))
reg_results = []



for regre in reg :

    reg_results.append(np.mean(np.sqrt(-cross_val_score(regre, X_rbst, y = target,scoring = 'neg_mean_squared_error',

                                       cv = kfold, n_jobs=-4))))
reg_means = []

reg_std = []

for reg_result in reg_results:

    reg_means.append(reg_result.mean())

    reg_std.append(reg_result.std())
reg_re = pd.DataFrame({"CrossValMeans":reg_means,"CrossValerrors": reg_std})

reg_re
# Gradient boosting 파라미터 튜닝

GBC = GradientBoostingRegressor()

gb_param_grid = {'n_estimators' : [100,200,300],

              'learning_rate': [0.1, 0.05, 0.01],

              'max_depth': [4, 8],

              'min_samples_leaf': [100,150],

              'max_features': [0.3, 0.1] 

              }

gsGBC = GridSearchCV(GBC,param_grid = gb_param_grid, cv=kfold, scoring="neg_mean_squared_error", n_jobs= 4, verbose = 1)

gsGBC.fit(X_rbst,target)

GBC_best = gsGBC.best_estimator_



# 최고 점수

gsGBC.best_score_
# XGBoost 파라미터 튜닝 

#xgb_param_grid = {'learning_rate': [1,0.1,0.01,0.001],

#              'n_estimators': [50, 100, 200, 500, 1000,3000],

#              'max_depth' : [1,3,5,10,50],

#              'subsample': [0.6, 0.7, 1.0],

#              'colsample_bytree' : [0.3,0.5,0.7,1],

#              'scale_pos_weight' : [0.5,1],

#              'reg_alpha': [0,0.05,0.0005,0.000005]

#               }
# XGBoost 파라미터 튜닝 



XGB = XGBRegressor()

xgb_param_grid = {'learning_rate': [1,0.1,0.01,0.001],

              'n_estimators': [50, 100, 200, 500, 1000],

              'max_depth' : [1,3,5,10,50]}

gsXGB = GridSearchCV(XGB,param_grid = xgb_param_grid, cv=kfold, scoring="neg_mean_squared_error", n_jobs= 4, verbose = 1)

gsXGB.fit(X_rbst,target)

XGB_best = gsXGB.best_estimator_



# 최고 점수

gsXGB.best_score_
#LGBMClassifier 파라미터 튜닝

LGB = LGBMRegressor()

lgb_param_grid = {

    'num_leaves' : [1,5,10],

    'learning_rate': [1,0.1,0.01,0.001],

    'n_estimators': [50, 100, 200, 500, 1000,5000], 

    'max_depth': [15,20,25],

    'num_leaves': [50, 100, 200],

    'min_split_gain': [0.3, 0.4],

}

gsLGB = GridSearchCV(LGB,param_grid = lgb_param_grid, cv=kfold, scoring="neg_mean_squared_error", n_jobs= 4, verbose = 1)

gsLGB.fit(X_rbst,target)

LGB_best = gsLGB.best_estimator_



# 최고 점수

gsLGB.best_score_
test_Survived_GBC = pd.Series(GBC_best.predict(test_rbst), name="GBC")

test_Survived_XGB = pd.Series(XGB_best.predict(test_rbst), name="XGB")

test_Survived_LGB = pd.Series(LGB_best.predict(test_rbst), name="LGB")



ensemble_results = pd.concat([test_Survived_XGB,test_Survived_LGB,

                              test_Survived_GBC],axis=1)

g= sns.heatmap(ensemble_results.corr(),annot=True)
ensemble = np.expm1(0.1*test_Survived_GBC + 0.8*test_Survived_XGB + 0.1*test_Survived_LGB)

submission = pd.DataFrame({

    "Id" :test['Id'],

    "SalePrice": ensemble

})

submission.head()
#submission.to_csv('ensemblesubmission.csv', index=False)
from sklearn.ensemble import VotingRegressor
votingC = VotingRegressor(estimators=[('XGB', XGB_best), ('LGB', LGB_best), ('GBC',GBC_best)], n_jobs=4)

votingC = votingC.fit(X_rbst, target)  
test_SalePrice = pd.Series(votingC.predict(test_rbst), name="SalePrice")
submission = pd.DataFrame({

    "Id" :test['Id'],

    "SalePrice": np.expm1(test_SalePrice)

})

submission.head()
#submission.to_csv('votingsubmission.csv', index=False)
from mlxtend.regressor import StackingRegressor

from sklearn.linear_model import LogisticRegression

from sklearn.utils.testing import ignore_warnings
params = {'meta_regressor__C': [0.1, 1.0, 10.0, 100.0],

          'use_features_in_secondary' : [True, False]}
clf1 = XGB_best

clf2 = LGB_best

clf3 = GBC_best



lr = LogisticRegression()

st_re= StackingRegressor(regressors=[clf1, clf2, clf3], meta_regressor=RandomForestRegressor())

st_mod = st_re.fit(X_rbst, target)

st_pred = st_mod.predict(test_rbst)
submission = pd.DataFrame({

    "Id" :test['Id'],

    "SalePrice": np.expm1(st_pred)

})

submission.head()
kfolds = KFold(n_splits=10, shuffle=True, random_state=42)
def rmsle(y, y_pred):

    return np.sqrt(mean_squared_error(y, y_pred))



def cv_rmse(model, X=new_train):

    rmse = np.sqrt(-cross_val_score(model, X_rbst, target,

                                    scoring="neg_mean_squared_error",

                                    cv=kfolds))

    return (rmse)
alphas_ridge = [14.5, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5]

alphas_lasso = [5e-05, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008]

alphas_enect = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]

enect_l1ratio = [0.8, 0.85, 0.9, 0.95, 0.99, 1]
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV

from sklearn.pipeline import make_pipeline

from sklearn.svm import SVR

from mlxtend.regressor import StackingCVRegressor
ridge = make_pipeline(RobustScaler(),

                      RidgeCV(alphas=alphas_ridge, cv=kfolds))
lasso = make_pipeline(RobustScaler(),

                      LassoCV(max_iter=1e7, alphas=alphas_lasso,

                              random_state=2, cv=kfolds))
enet = make_pipeline(RobustScaler(),

                           ElasticNetCV(max_iter=1e7, alphas=alphas_enect,

                                        cv=kfolds, l1_ratio=enect_l1ratio))
svr = make_pipeline(RobustScaler(),

                      SVR(C= 20, epsilon= 0.008, gamma=0.0003,))
stack_gen = StackingCVRegressor(regressors=(ridge, lasso, enet,

                                            GBC_best, XGB_best, LGB_best),

                                meta_regressor=XGB_best,

                                use_features_in_secondary=True)
score = cv_rmse(ridge)

print("Kernel Ridge score: {:.4f} ({:.4f})\n".format(score.mean(),score.std()))



score = cv_rmse(lasso)

print("Lasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))



score = cv_rmse(enet)

print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))



score = cv_rmse(svr)

print("SVR score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))



score = cv_rmse(GBC_best)

print("Lightgbm score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))



score = cv_rmse(XGB_best)

print("GradientBoosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))



score = cv_rmse(LGB_best)

print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()) )
stack_gen_model = stack_gen.fit(np.array(X_rbst), np.array(target))

elastic_model_full_data = enet.fit(X_rbst, target)

lasso_model_full_data = lasso.fit(X_rbst, target)

ridge_model_full_data = ridge.fit(X_rbst, target)

svr_model_full_data = svr.fit(X_rbst, target)

gbr_model_full_data = GBC_best.fit(X_rbst, target)

xgb_model_full_data = XGB_best.fit(X_rbst, target)

lgb_model_full_data = LGB_best.fit(X_rbst, target)
def blend_models_predict(X):

    return ((0.1 * elastic_model_full_data.predict(X)) + \

            (0.1 * lasso_model_full_data.predict(X)) + \

            (0.1 * ridge_model_full_data.predict(X)) + \

            (0.1 * svr_model_full_data.predict(X)) + \

            (0.1 * gbr_model_full_data.predict(X)) + \

            (0.15 * xgb_model_full_data.predict(X)) + \

            (0.1 * lgb_model_full_data.predict(X)) + \

            (0.25 * stack_gen_model.predict(np.array(X))))
pred = np.floor(np.expm1(blend_models_predict(test_rbst)))

submission = pd.DataFrame({

    "Id" :test['Id'],

    "SalePrice": pred

})

submission.head()
submission.to_csv('final_submission.csv', index=False)