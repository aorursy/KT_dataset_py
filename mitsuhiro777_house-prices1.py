# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import copy
train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

print(len(train),len(test))



train_ID = train['Id']

test_ID = test['Id']
train.head(5)
test.head(5)
print(train['SalePrice'].describe())

train['SalePrice'].hist(bins=20, range=(0, 800000))
train_copy = copy.deepcopy(train["SalePrice"])

train["SalePrice"] = np.log1p(train_copy) # 元に戻すときはexpm1

train['SalePrice'].hist(bins=20, range=(10, 15))
fig, ax = plt.subplots()

ax.scatter(x = train['GrLivArea'], y = train['SalePrice'])

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('GrLivArea', fontsize=13)

plt.show()
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)



#Check the graphic again

fig, ax = plt.subplots()

ax.scatter(train['GrLivArea'], train['SalePrice'])

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('GrLivArea', fontsize=13)

plt.show()
ntrain = train.shape[0]

ntest = test.shape[0]

y_train = train.SalePrice.values

all_data = pd.concat((train, test),sort=True).reset_index(drop=True)

all_data.drop(['SalePrice'], axis=1, inplace=True)

print("all_data size is : {}".format(all_data.shape))
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100

all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]

missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})

print(missing_data)
print(all_data["PoolQC"].value_counts())

all_data['PoolQC'] = all_data['PoolQC'].replace({"Ex":3,"Gd":2,"Fa":1})

all_data['PoolQC'] = all_data['PoolQC'].fillna(0)

print(all_data["PoolQC"].value_counts())
print(all_data["Alley"].value_counts())

all_data['Alley'] = all_data['Alley'].replace({"Pave":2,"Grvl":1})

all_data['Alley'] = all_data['Alley'].fillna(0)

print(all_data["Alley"].value_counts())
print(all_data["MiscFeature"].value_counts())

all_data['MiscFeature'] = all_data['MiscFeature'].replace({"TenC":3, "Gar2":1, "Shed":2, "Othr":1})

all_data['MiscFeature'] = all_data['MiscFeature'].fillna(0)

print(all_data["MiscFeature"].value_counts())
# Fenceを五段階評価で置き換え

print("Before","\n",all_data['Fence'].value_counts(),"\n")

all_data['Fence'] = all_data['Fence'].replace({"GdPrv":4,"MnPrv":3,"GdWo":2,"MnWw":1})

all_data['Fence'] = all_data['Fence'].fillna(0)

print("After","\n",all_data['Fence'].value_counts())
# Fenceを五段階評価で置き換え

print("Before","\n",all_data['FireplaceQu'].value_counts(),"\n")

all_data['FireplaceQu'] = all_data['FireplaceQu'].replace({"Ex":5,"Gd":4,"TA":3,"Fa":2,"Po":1})

all_data['FireplaceQu'] = all_data['FireplaceQu'].fillna(0)

print("After","\n",all_data['FireplaceQu'].value_counts())
# 相関の高いLotArea、1stFlrSFで穴埋め

from sklearn import linear_model



# print(all_data.corr()['LotFrontage'])

df_LotFrontage_yes = all_data[all_data['LotFrontage'].isnull()==False]

clf_LotFrontage = linear_model.LinearRegression()

clf_LotFrontage.fit(df_LotFrontage_yes[['LotArea', '1stFlrSF']],df_LotFrontage_yes['LotFrontage'])



pd.options.mode.chained_assignment = None

for i in range(len(all_data['LotFrontage'])):

    if all_data.isnull().iloc[i]['LotFrontage']==True:

        all_data['LotFrontage'][i] = int(clf_LotFrontage.predict([all_data.iloc[i][['LotArea', '1stFlrSF']].values]))
print("Before","\n",all_data['GarageCond'].value_counts(),"\n")

all_data['GarageCond'] = all_data['GarageCond'].replace({"Ex":5,"Gd":4,"TA":3,"Fa":2,"Po":1})

all_data['GarageCond'] = all_data['GarageCond'].fillna(0)

print("After","\n",all_data['GarageCond'].value_counts(),"\n")



print("Before","\n",all_data['GarageQual'].value_counts(),"\n")

all_data['GarageQual'] = all_data['GarageQual'].replace({"Ex":5,"Gd":4,"TA":3,"Fa":2,"Po":1})

all_data['GarageQual'] = all_data['GarageQual'].fillna(0)

print("After","\n",all_data['GarageQual'].value_counts(),"\n")



print("Before","\n",all_data['GarageFinish'].value_counts(),"\n")

all_data['GarageFinish'] = all_data['GarageFinish'].replace({"Fin":3,"RFn":2,"Unf":1})

all_data['GarageFinish'] = all_data['GarageFinish'].fillna(0)

print("After","\n",all_data['GarageFinish'].value_counts())



print("Before","\n",all_data['GarageArea'].value_counts(),"\n")

all_data['GarageArea'] = all_data['GarageArea'].fillna(0)

print("After","\n",all_data['GarageArea'].value_counts())



print("Before","\n",all_data['GarageCars'].value_counts(),"\n")

all_data['GarageCars'] = all_data['GarageCars'].fillna(0)

print("After","\n",all_data['GarageCars'].value_counts())

# print(all_data.corr()['GarageYrBlt'])

# YearBuilt、YearRemodAddと相関高いし削除     

all_data = all_data.drop("GarageYrBlt", axis=1)
all_data['GarageType'] = all_data['GarageType'].fillna(all_data['GarageType'].mode()[0])

all_data['GarageType'] = all_data['GarageType'].fillna(all_data['GarageType'].mode()[0])
# BsmQual、BsmCondは6段階、BsmtExposureは5段階、BsmtFinType1、BsmtFinType2は7段階評価



all_data['BsmtCond'] = all_data['BsmtCond'].replace({"Ex":5,"Gd":4,"TA":3,"Fa":2,"Po":1})

all_data['BsmtCond'] = all_data['BsmtCond'].fillna(0)



all_data['BsmtQual'] = all_data['BsmtQual'].replace({"Ex":5,"Gd":4,"TA":3,"Fa":2,"Po":1})

all_data['BsmtQual'] = all_data['BsmtQual'].fillna(0)



all_data['BsmtExposure'] = all_data['BsmtExposure'].replace({"Gd":4,"Av":3,"Mn":2,"No":1})

all_data['BsmtExposure'] = all_data['BsmtExposure'].fillna(0)



all_data['BsmtFinType1'] = all_data['BsmtFinType1'].replace({"GLQ":6,"ALQ":5,"BLQ":4,"Rec":3,"LwQ":2, "Unf":1})

all_data['BsmtFinType1'] = all_data['BsmtFinType1'].fillna(0)



all_data['BsmtFinType2'] = all_data['BsmtFinType2'].replace({"GLQ":6,"ALQ":5,"BLQ":4,"Rec":3,"LwQ":2, "Unf":1})

all_data['BsmtFinType2'] = all_data['BsmtFinType2'].fillna(0)

all_data['BsmtFullBath'] = all_data['BsmtFullBath'].fillna(0)

all_data['BsmtHalfBath'] = all_data['BsmtHalfBath'].fillna(0)

all_data['BsmtFinSF1'] = all_data['BsmtFinSF1'].fillna(0)

all_data['BsmtFinSF2'] = all_data['BsmtFinSF2'].fillna(0)

all_data['BsmtUnfSF'] = all_data['BsmtUnfSF'].fillna(0)

all_data['TotalBsmtSF'] = all_data['TotalBsmtSF'].fillna(0)
all_data['MasVnrType'] = all_data['MasVnrType'].fillna("None")

all_data['MasVnrArea'] = all_data['MasVnrArea'].fillna(0)
all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])

all_data['Functional'] = all_data['Functional'].fillna(all_data['Functional'].mode()[0])

all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])

all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])

all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])

all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])
all_data['Utilities'] = all_data['Utilities'].replace({"AllPub":1,"NoSeWa":0})

all_data['Utilities'] = all_data['Utilities'].fillna(0)

print("After","\n",all_data['Utilities'].value_counts())
# kitchenの項目はNAがないので、kitchenがないわけではないので0ではない

# Poはない

all_data['KitchenQual'] = all_data['KitchenQual'].replace({"Ex":4,"Gd":3,"TA":2,"Fa":1})

all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100

all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)

missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})

missing_data.head()
all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)

all_data['OverallCond'] = all_data['OverallCond'].astype(str)

all_data['YrSold'] = all_data['YrSold'].astype(str)

all_data['MoSold'] = all_data['MoSold'].astype(str)
all_data['ExterCond'] = all_data['ExterCond'].replace({"Ex":5,"Gd":4,"TA":3,"Fa":2,"Po":1})

all_data['ExterQual'] = all_data['ExterQual'].replace({"Ex":4,"Gd":3,"TA":2,"Fa":1})

all_data['HeatingQC'] = all_data['HeatingQC'].replace({"Ex":5,"Gd":4,"TA":3,"Fa":2,"Po":1})
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
# from scipy import stats

# from scipy.stats import norm, skew



# numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index



# # Check the skew of all numerical features

# skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

# print("\nSkew in numerical features: \n")

# skewness = pd.DataFrame({'Skew' :skewed_feats})

# skewness.head(10)

# skewness = skewness[abs(skewness) > 0.75]

# print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))



# from scipy.special import boxcox1p

# skewed_features = skewness.index

# lam = 0.15

# for feat in skewed_features:

#     all_data[feat] = boxcox1p(all_data[feat], lam)
all_data = all_data.drop("Id", axis = 1)
all_data = pd.get_dummies(all_data)
all_data = pd.get_dummies(all_data)

print(all_data.shape)



x_train = all_data[:ntrain]

x_test = all_data[ntrain:]
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC

from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor

from sklearn.kernel_ridge import KernelRidge

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.model_selection import KFold, cross_val_score, train_test_split

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

from sklearn.metrics import mean_squared_error

import xgboost as xgb

import lightgbm as lgb

import numpy as np
def kf_score(model):

    kfold = KFold(n_splits=5, shuffle=True, random_state=0)

    scores_msq = cross_val_score(model, x_train, y_train, cv = kfold, scoring='neg_mean_squared_error')



    scores = []

    for score_msq in scores_msq:

        scores.append(np.sqrt(-score_msq))

        

    return scores
Lasso_model = make_pipeline(RobustScaler(), Lasso(alpha =0.001, random_state=26))



# 交差検証

scores = kf_score(Lasso_model)

print('Cross-Validation scores: {}'.format(scores))

print('Average score: {}'.format(np.mean(scores)))
Elastic_model = make_pipeline(RobustScaler(), ElasticNet(alpha =0.0044,l1_ratio=0.038  ,random_state=17))



# 交差検証

scores = kf_score(Elastic_model)

print('Cross-Validation scores: {}'.format(scores))

print('Average score: {}'.format(np.mean(scores)))
Random_Forest_model = make_pipeline(

RandomForestRegressor(

max_depth=18,

n_estimators=2566,

max_features='sqrt',

n_jobs=4,

verbose=0)

)



# # 交差検証

# scores = kf_score(Random_Forest_model)

# print('Cross-Validation scores: {}'.format(scores))

# print('Average score: {}'.format(np.mean(scores)))
GBoost_model = make_pipeline(

GradientBoostingRegressor(

n_estimators= 1175, 

learning_rate=0.05,

max_depth=4, 

max_features='sqrt',

min_samples_leaf=13, 

min_samples_split=31, 

loss='huber', 

random_state =3)

)



# # 交差検証

# scores = kf_score(GBoost_model)



# print('Cross-Validation scores: {}'.format(scores))

# print('Average score: {}'.format(np.mean(scores)))
Xgboost_model = make_pipeline(

xgb.XGBRegressor(

n_estimators= 4089, 

max_depth=14, 

max_features='sqrt',

colsample_bytree = 0.5,

gamma=0,

min_child_weight=9,

reg_alpha=0.0048, 

subsample = 0.5,

learning_rate = 0.05,

random_state =9)

)



# # 交差検証

# scores = kf_score(Xgboost_model)



# print('Cross-Validation scores: {}'.format(scores))

# print('Average score: {}'.format(np.mean(scores)))
LGB_model = make_pipeline(

lgb.LGBMRegressor(objective='regression', 

num_leaves=18,

n_estimators=1380,               

max_bin = 28,                                    

bagging_fraction =0.5,

bagging_freq = 1,

feature_fraction = 0.4,

feature_fraction_seed=1, 

bagging_seed=4,

min_data_in_leaf =1,

min_sum_hessian_in_leaf = 1,

learning_rate=0.05 )

)



# # 交差検証

# scores = kf_score(LGB_model)



# print('Cross-Validation scores: {}'.format(scores))

# print('Average score: {}'.format(np.mean(scores)))
class StackingModels(BaseEstimator, RegressorMixin, TransformerMixin):

    def __init__(self, base_models, meta_model = None, n_folds=5):

        self.base_models = base_models

        self.n_folds = n_folds

   

    def fit_each_model(self, x, y):

        print("fit each models", "\n") 

        self.base_models_ = [list() for x in self.base_models]

        kfold = KFold(n_splits=self.n_folds, shuffle=True)

        out_of_fold_predictions = np.zeros((x.shape[0], len(self.base_models)))      

        

        for i, model in enumerate(self.base_models):

            for train_index, holdout_index in kfold.split(x, y):

                instance = clone(model)

                self.base_models_[i].append(instance)

                instance.fit(x[train_index], y[train_index])

                y_pred = instance.predict(x[holdout_index])

                out_of_fold_predictions[holdout_index, i] = y_pred

                

            print(model)

            print(np.sqrt(mean_squared_error(out_of_fold_predictions[:, i] , y)),  "\n")          

            

        return out_of_fold_predictions

                

    def predict_each_model(self, x, y=None): 

        print("predict each models")

        each_predict = []

        for base_models in self.base_models_:

            _predict = np.column_stack([model.predict(x) for model in base_models]).mean(axis=1)

            each_predict.append(_predict)

            

            if y is not None:

                print(np.sqrt(mean_squared_error(_predict, y))) 

                

        return each_predict

            

    def fit(self, x, y, meta_model):

        print("fit stack models", "\n") 

        self.meta_models_ = []

        kfold = KFold(n_splits=self.n_folds, shuffle=True)

        out_of_fold_predictions = np.zeros((x.shape[0]))

              

        for train_index, holdout_index in kfold.split(x, y):

            instance = clone(meta_model)

            self.meta_models_.append(instance)

            instance.fit(x[train_index], y[train_index])

            y_pred = instance.predict(x[holdout_index])

            out_of_fold_predictions[holdout_index] = y_pred

            

        print(np.sqrt(mean_squared_error(out_of_fold_predictions, y)),  "\n")          

        return out_of_fold_predictions

   

    def predict(self, x, y= None):

        print("predict stack models")

        stack_predict = np.column_stack([model.predict(x) for model in self.meta_models_]).mean(axis=1)



        if y is not None:

            print(np.sqrt(mean_squared_error(stack_predict, y))) 

                

        return stack_predict
stacked_models = StackingModels(base_models = (Lasso_model, Elastic_model, 

                                               GBoost_model, Xgboost_model, LGB_model))

#stacked_models = StackingModels(base_models = (LGB, Lasso))

out_of_fold_predictions = stacked_models.fit_each_model(x_train.values,y_train)
each_predict = stacked_models.predict_each_model(x_train.values, y_train)
n_folds = 5



def kf_score_stack(model, x, y):

    kf = KFold(n_folds, shuffle=True).get_n_splits(x)

    rmse= np.sqrt(-cross_val_score(model, x, y, scoring="neg_mean_squared_error", cv = kf))

    return(rmse)
Lasso_model_stack = make_pipeline(RobustScaler(), Lasso(alpha =0.0004, random_state=26))



stacked_models.fit(out_of_fold_predictions, y_train, meta_model = Lasso_model_stack)

stack_predict = stacked_models.predict(out_of_fold_predictions, y_train)
Elastic_model_stack = make_pipeline(RobustScaler(), ElasticNet(alpha =0.0044,l1_ratio=0.038  ,random_state=17))



stacked_models.fit(out_of_fold_predictions, y_train, meta_model = Elastic_model_stack)

stack_predict = stacked_models.predict(out_of_fold_predictions, y_train)
import optuna



def objective(trial):

 

    max_depth = trial.suggest_int('max_depth', 3, 7) # 深すぎると過学習になるかも...

    n_estimators = trial.suggest_int('n_estimators', 10, 400) # しっかりやるなら100以上

    max_features = trial.suggest_categorical('max_features', ['sqrt', 'auto'])

    min_child_weight = trial.suggest_discrete_uniform(' min_child_weight', 0.2, 1, 0.1)

    gamma = trial.suggest_discrete_uniform('gamma',  0, 3, 0.3)

    random_state =  trial.suggest_int('random_state', 1, 30)

 

    # model

    model = xgb.XGBRegressor(

                max_features=max_features,

                n_estimators=n_estimators,

                min_child_weight=min_child_weight,

                max_depth=max_depth,

                gamma=gamma,

                criterion='rmse',

                random_state=random_state,           

    )

    scores = kf_score_stack(model, out_of_fold_predictions, y_train)

        

    return np.mean(scores)
### commit するときにはコメントアウトする ####

# study = optuna.create_study()



# study.optimize(func=objective, # 実行する関数

#                n_trials=500, # 試行回数

#                timeout=None, # 与えられた秒数後に学習を中止します。default=None

#                n_jobs=5 # 並列実行するjob数

#               )
Xgboost_stack = xgb.XGBRegressor(

                max_features='sqrt',

                n_estimators=85,

                min_child_weight=0.5,

                max_depth=3,

                gamma=0.0,

                criterion='rmse',

                random_state=29,           

    )



stacked_models.fit(out_of_fold_predictions, y_train, meta_model = Xgboost_stack)

stack_predict_stack = stacked_models.predict(out_of_fold_predictions, y_train)
# import optuna



# def objective(trial):

 

#     max_depth = trial.suggest_int('max_depth', 3, 10) # 深すぎると過学習になるかも...

#     n_estimators = trial.suggest_int('n_estimators', 10, 400) # しっかりやるなら100以上

#     max_features = trial.suggest_categorical('max_features', ['sqrt', 'auto'])

#     min_child_weight = trial.suggest_discrete_uniform(' min_child_weight', 0.2, 1, 0.1)

#     gamma = trial.suggest_discrete_uniform('gamma',  0, 3, 0.3)

#     random_state =  trial.suggest_int('random_state', 1, 30)

#     rate_drop =  trial.suggest_discrete_uniform('rate_drop', 0.1, 0.8, 0.1)

#     skip_drop =  trial.suggest_discrete_uniform('skip_drop', 0.1, 0.8, 0.1)

    

#     # model

#     model  =  xgb.XGBRegressor(

#                 booster = "dart",

#                 n_estimators= n_estimators, 

#                 max_depth=max_depth, 

#                 max_features=max_features,

#                 gamma=gamma,

#                 min_child_weight=min_child_weight,

#                 learning_rate = 0.05,

#                 random_state =random_state,

#                 rate_drop = rate_drop,

#                 skip_drop = skip_drop

#         )

    

#     scores = kf_score_stack(model, out_of_fold_predictions, y_train)

        

#     return np.mean(scores)
# ### commit するときにはコメントアウトする ####



# study = optuna.create_study()



# study.optimize(func=objective, # 実行する関数

#                n_trials=500, # 試行回数

#                timeout=None, # 与えられた秒数後に学習を中止します。default=None

#                n_jobs=5 # 並列実行するjob数

#               )
# Xgboost_dart_stack = xgb.XGBRegressor(

#                 booster = "dart",

#                 max_features='sqrt',

#                 n_estimators=217,

#                 min_child_weight=0.4,

#                 max_depth=3,

#                 gamma=0.0,

#                 criterion='rmse',

#                 random_state=21,       

#                 rate_drop=0.3,

#                 skip_drop=0.7 

#     )



# stacked_models.fit(out_of_fold_predictions, y_train, meta_model = Xgboost_dart_stack)

# stack_predict = stacked_models.predict(out_of_fold_predictions, y_train)
each_predict_test = stacked_models.predict_each_model(x_test.values)



# stack xgboost

stacked_models.fit(out_of_fold_predictions, y_train, meta_model = Xgboost_stack)

stack_predict_test_xgboost = stacked_models.predict(np.column_stack(each_predict_test))



# stack xgboost_dart

# stacked_models.fit(out_of_fold_predictions, y_train, meta_model = Xgboost_dart_stack)

# stack_predict_test_xgboost_dart = stacked_models.predict(np.column_stack(each_predict_test))



# stack lasso

stacked_models.fit(out_of_fold_predictions, y_train, meta_model = Lasso_model)

stack_predict_test_lasso = stacked_models.predict(np.column_stack(each_predict_test))



# each model

# predict_test_randomforest = each_predict_test[0]

# predict_test_gboost = each_predict_test[1]

# predict_test_xgboost = each_predict_test[2]

predict_test_lgb = each_predict_test[3]

# stack xgboost

submit_stack_xgboost = pd.DataFrame()

submit_stack_xgboost['Id'] = test_ID

submit_stack_xgboost['SalePrice'] = np.expm1(stack_predict_test_xgboost)

submit_stack_xgboost.to_csv('submission_stack_xgboost.csv',index=False)



# stack xgboost dart

# submit_stack_xgboost_dart = pd.DataFrame()

# submit_stack_xgboost_dart['Id'] = test_ID

# submit_stack_xgboost_dart['SalePrice'] = np.expm1(stack_predict_test_xgboost_dart)

# submit_stack_xgboost_dart.to_csv('submission_stack_xgboost_dart.csv',index=False)



# stack lasso

submit_stack_lasso = pd.DataFrame()

submit_stack_lasso['Id'] = test_ID

submit_stack_lasso['SalePrice'] = np.expm1(stack_predict_test_lasso)

submit_stack_lasso.to_csv('submission_stack_lasso.csv',index=False)



# # stack randomforest

# submit_randomforest = pd.DataFrame()

# submit_randomforest['Id'] = test_ID

# submit_randomforest['SalePrice'] = np.expm1(predict_test_randomforest)

# submit_randomforest.to_csv('submission_randomforest.csv',index=False)



# # stack gboost

# submit_gboost = pd.DataFrame()

# submit_gboost['Id'] = test_ID

# submit_gboost['SalePrice'] = np.expm1(predict_test_gboost)

# submit_gboost.to_csv('submission_gboost.csv',index=False)



# # stack xgboost

# submit_xgboost = pd.DataFrame()

# submit_xgboost['Id'] = test_ID

# submit_xgboost['SalePrice'] = np.expm1(predict_test_xgboost)

# submit_xgboost.to_csv('submission_xgboost.csv',index=False)



# # stack lgb

# submit_lgb = pd.DataFrame()

# submit_lgb['Id'] = test_ID

# submit_lgb['SalePrice'] = np.expm1(predict_test_lgb)

# submit_lgb.to_csv('submission_lgb.csv',index=False)