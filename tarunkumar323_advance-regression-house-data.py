import numpy as np 

import pandas as pd 

import seaborn as sns

from matplotlib import pyplot as plt

import matplotlib.style as style

from scipy import stats



style.use('ggplot')

sns.set_style('whitegrid')



import warnings

warnings.filterwarnings('ignore')
train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')



print('test shape: ',test.shape)

print('train shape:',train.shape)

train.drop("Id", axis=1,inplace=True)

test.drop("Id", axis=1,inplace=True)

print(train.corr()['SalePrice'].sort_values(ascending=False))
style.use('fivethirtyeight')

plt.scatter(train['GrLivArea'], train['SalePrice'])

plt.ylabel('SalePrice')

plt.xlabel('GrLivArea')
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<200000)].index)
style.use('ggplot')

fig,ax = plt.subplots()

stats.probplot(train['SalePrice'],plot=ax)
train.SalePrice = np.log1p(train.SalePrice)

sns.distplot(train['SalePrice'], norm_hist=True)


# fig,ax = plt.subplots(nrows=int(len(numerical)/3)+1,ncols=3,figsize=(20,60))

# i=0

# j=0

# for col in numerical:

#     ax[i,j].scatter(train[col],train['SalePrice'])

#     ax[i,j].set_xlabel(col,fontsize=15)

#     ax[i,j].set_ylabel('SalePrice',fontsize=15)

#     j+=1

#     if j>2:

#         j=0

#         i+=1





# plt.subplots(figsize=(30,20))

# sns.heatmap(train.corr(), cmap=sns.diverging_palette(20, 220, n=200), annot=True,center=0)
print(train.isnull().sum().sort_values(ascending = False)[train.isnull().sum().sort_values(ascending = False) != 0])

print(test.isnull().sum().sort_values(ascending = False)[test.isnull().sum().sort_values(ascending = False) != 0])
y = train['SalePrice']

train.drop('SalePrice', axis=1,inplace=True)
all_data = pd.concat((train,test)).reset_index(drop = True)

all_data.isnull().sum().sort_values(ascending = False)[all_data.isnull().sum().sort_values(ascending = False) != 0]
# missing values with type none

miss_val_1 = ["Alley",

              "PoolQC", 

              "MiscFeature",

              "Fence",

              "FireplaceQu",

              "GarageType",

              "GarageFinish",

              "GarageQual",

              "GarageCond",

              'BsmtQual',

              'BsmtCond',

              'BsmtExposure',

              'BsmtFinType1',

              'BsmtFinType2',

              'MasVnrType']





#missing values with type 0

miss_val_2=['BsmtFinSF1',

            'BsmtFinSF2',

            'BsmtUnfSF',

            'TotalBsmtSF',

            'BsmtFullBath', 

            'BsmtHalfBath', 

            'GarageYrBlt',

            'GarageArea',

            'GarageCars',

            'MasVnrArea']



for i  in miss_val_1:

    all_data[i]=all_data[i].fillna('None')

    

for i in miss_val_2:

    all_data[i]=all_data[i].fillna(0)
all_data.isnull().sum().sort_values(ascending = False)[all_data.isnull().sum().sort_values(ascending = False) != 0]
all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].transform( lambda x: x.fillna(x.mean()))

all_data['MSSubClass'] = all_data['MSSubClass'].astype(str)

all_data['MSZoning'] = all_data.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))

all_data['YrSold'] = all_data['YrSold'].astype(str)

all_data['MoSold'] = all_data['MoSold'].astype(str)

all_data['Functional'] = all_data['Functional'].fillna('Typ') 

all_data['Utilities'] = all_data['Utilities'].fillna('AllPub') 

all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0]) 

all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])

all_data['KitchenQual'] = all_data['KitchenQual'].fillna("TA") 

all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])

all_data['Electrical'] = all_data['Electrical'].fillna("SBrkr")
all_data.isnull().sum().sort_values(ascending = False)[all_data.isnull().sum().sort_values(ascending = False) != 0]
numerical = all_data.dtypes[all_data.dtypes != "object"].index
from scipy.stats import skew

from scipy.special import boxcox1p

from scipy.stats import boxcox_normmax



skew = all_data[numerical].apply(lambda x: skew(x)).sort_values(ascending=False)



skew_index = skew[abs(skew)>0.5].index



for feat in skew_index:

        all_data[feat] = boxcox1p(all_data[feat], boxcox_normmax(all_data[feat] + 1))
from sklearn.preprocessing import LabelEncoder

cat = all_data.dtypes[all_data.dtypes == "object"].index

for i in cat:

    all_data[i].dtype

    le = LabelEncoder()

    all_data[i] = le.fit_transform(all_data[i])

    all_data[i].dtype
X = all_data.iloc[:len(y), :]



X_sub = all_data.iloc[len(y):, :]
def cv_rmse(model, X=X):

    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=kfolds))

    return (rmse)
from xgboost import XGBRegressor

from lightgbm import LGBMRegressor

from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV

from sklearn.svm import SVR

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.model_selection import cross_val_score, KFold



kfolds = KFold(n_splits=10, shuffle=True, random_state=42)



alphas_alt = [14.5, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5]

alphas2 = [5e-05, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008]

e_alphas = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]

e_l1ratio = [0.8, 0.85, 0.9, 0.95, 0.99, 1]



ridge = make_pipeline(RobustScaler(), RidgeCV(alphas=alphas_alt, cv=kfolds))

lasso = make_pipeline(RobustScaler(), LassoCV(max_iter=1e7, alphas=alphas2, random_state=42, cv=kfolds))

elasticnet = make_pipeline(RobustScaler(), ElasticNetCV(max_iter=1e7, alphas=e_alphas, cv=kfolds, l1_ratio=e_l1ratio))                                

svr = make_pipeline(RobustScaler(), SVR(C= 20, epsilon= 0.008, gamma=0.0003,))





xgboost = XGBRegressor(learning_rate=0.01,n_estimators=3460,

                                     max_depth=3, min_child_weight=0,

                                     gamma=0, subsample=0.7,

                                     colsample_bytree=0.7,

                                     nthread=-1,

                                     scale_pos_weight=1, seed=27,

                                     reg_alpha=0.00006)



lightgbm = LGBMRegressor(objective='regression', 

                                       num_leaves=4,

                                       learning_rate=0.01, 

                                       n_estimators=5000,

                                       max_bin=200, 

                                       bagging_fraction=0.75,

                                       bagging_freq=5, 

                                       bagging_seed=7,

                                       feature_fraction=0.2,

                                       feature_fraction_seed=7,

                                       verbose=-1,

                                       )



score = cv_rmse(ridge)

print("Ridge: {:.4f} \n".format(score.mean()))



score = cv_rmse(lasso)

print("LASSO: {:.4f} \n".format(score.mean()))



score = cv_rmse(elasticnet)

print("elastic net: {:.4f} \n".format(score.mean()))



score = cv_rmse(svr)

print("SVR: {:.4f} \n".format(score.mean()))



score = cv_rmse(lightgbm)

print("lightgbm: {:.4f} \n".format(score.mean()))



score = cv_rmse(xgboost)

print("xgboost: {:.4f} \n".format(score.mean()))
elastic = elasticnet.fit(X, y)



lasso = lasso.fit(X, y)



ridge = ridge.fit(X, y)



svr = svr.fit(X, y)



xgb = xgboost.fit(X, y)



lgb = lightgbm.fit(X, y)
def models_predict(X):

    return ((0.15 * elastic.predict(X)) + \

            (0.15 * lasso.predict(X)) + \

            (0.15 * ridge.predict(X)) + \

            (0.15 * svr.predict(X)) + \

            (0.20 * xgb.predict(X)) + \

            (0.20 * lgb.predict(X)))
submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")

submission.iloc[:,1] = np.expm1(models_predict(X_sub))

submission.to_csv("submission111.csv", index=False)
# sub_1 = pd.read_csv('../input/lasso-model-for-regression-problem/lasso_sol22_Median.csv')

# sub_2 = pd.read_csv('../input/hybrid-svm-benchmark-approach-0-11180-lb-top-2/hybrid_solution.csv')



# submission.iloc[:,1] = np.floor((0.50 * models_predict(X_sub)) + 

#                                 (0.25 * sub_1.iloc[:,1]) +

#                                 (0.25 * sub_2.iloc[:,1]))

# submission.to_csv("submission222.csv", index=False)