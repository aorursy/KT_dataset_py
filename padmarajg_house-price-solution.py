

from datetime import datetime

from scipy.stats import skew  # for some statistics

from scipy.special import boxcox1p

from scipy.stats import boxcox_normmax

from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.svm import SVR

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.model_selection import KFold, cross_val_score

from sklearn.metrics import mean_squared_error

from mlxtend.regressor import StackingCVRegressor

from xgboost import XGBRegressor

from lightgbm import LGBMRegressor

import matplotlib.pyplot as plt

import scipy.stats as stats

import sklearn.linear_model as linear_model

import seaborn as sns

from sklearn.manifold import TSNE

from sklearn.cluster import KMeans

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler





import pandas as pd

import numpy as np

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import LabelEncoder

from xgboost import XGBRegressor

import category_encoders as ce

import lightgbm as lgb

from sklearn import metrics

import itertools

from matplotlib import pyplot as plt

%matplotlib inline

import seaborn as sns



import os

print(os.listdir("../input"))

print(os.listdir("../input/house-prices-advanced-regression-techniques"))
import warnings

warnings.filterwarnings('ignore')
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

print ("Data is loaded!")
train.drop(['Id'], axis=1, inplace=True)

test.drop(['Id'], axis=1, inplace=True)
# Remove rows with missing target, separate target from predictors.

train.dropna(axis=0, subset=['SalePrice'], inplace=True)

#removing two outliers ie: two rows with GrLivArea > 4500

train = train[train.GrLivArea < 4500]

train.reset_index(drop=True, inplace=True)

#store target in y and all other indepenent variables in X

y = train.SalePrice

train.drop(['SalePrice'], axis=1, inplace=True)
sns.set_style("whitegrid")

missing = train.isnull().sum()

missing = missing[missing > 0]

missing.sort_values(inplace=True)

missing.plot.bar()
def missing_percentage(df):

    

    total = df.isnull().sum().sort_values(ascending = False)[df.isnull().sum().sort_values(ascending = False) != 0]

    percent = round(df.isnull().sum().sort_values(ascending = False)/len(df)*100,2)[round(df.isnull().sum().sort_values(ascending = False)/len(df)*100,2) != 0]

    return pd.concat([total, percent], axis=1, keys=['Total','Percent'])



missing_percentage(train)
#transforming target variable to normal distribution using log transformation

y=np.log1p(y)

y=y.reset_index(drop=True)



#concatenate train and test data 

train_test=pd.concat((train,test)).reset_index(drop=True)
#Separate Numerical and categorical columns.

train_test_num = train_test.select_dtypes(exclude=['object'])

train_test_obj= train_test.select_dtypes(exclude=['int','float'])



my_imputer=SimpleImputer()

my_obj_imputer=SimpleImputer(strategy="most_frequent")

#Impute categorical data

imputed_train_test_num=pd.DataFrame(my_imputer.fit_transform(train_test_num))

#Impute numerical data

imputed_train_test_obj = pd.DataFrame(my_obj_imputer.fit_transform(train_test_obj))

#imputation remove indexes so put column indexes back

imputed_train_test_num.columns = train_test_num.columns

imputed_train_test_obj.columns = train_test_obj.columns

#Concatenate imputed categorical and numerical columns

imputed_train_test=pd.concat([imputed_train_test_obj,imputed_train_test_num],axis=1)
imputed_train_test = pd.get_dummies(imputed_train_test).reset_index(drop=True)
label_encoder = LabelEncoder()

for col in set(train_test_obj):

    imputed_train_test[col]=label_encoder.fit_transform (imputed_train_test[col])

    

for col in set(train_test_obj): 

    count_enc = ce.CountEncoder(cols=col)

    #imputed_train_test[col +'_labels']=count_enc.fit_transform(imputed_train_test[col])

    imputed_train_test[col]=count_enc.fit_transform(imputed_train_test[col])

    
X_train = imputed_train_test.iloc[:len(y), :]

X_test = imputed_train_test.iloc[len(y):, :]
for col in set(train_test_obj):

    target_enc = ce.TargetEncoder(cols=col)

    target_enc.fit(X_train[col],y)

    X_train[col+'_target' ]=target_enc.transform (X_train[col])

    X_test[col+'_target' ]=target_enc.transform(X_test[col])
outliers = [30, 88, 462, 631, 1322,691,934,297,322,185,53,495]

X_train = X_train.drop(X_train.index[outliers])

y = y.drop(y.index[outliers])



overfit = []

for i in X_train.columns:

    counts = X_train[i].value_counts()

    zeros = counts.iloc[0]

    if zeros / len(X_train) * 100 > 99.94:

        overfit.append(i)



overfit = list(overfit)

X_train = X_train.drop(overfit, axis=1)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y, train_size=0.8, test_size=0.2,

random_state=0)
kfolds = KFold(n_splits=10, shuffle=True, random_state=42)



def rmsle(y, y_pred):

    return np.sqrt(mean_squared_error(y, y_pred))



def cv_rmse(model, X_train=X_train):

    rmse = np.sqrt(-cross_val_score(model, X_train,y_train, scoring="neg_mean_squared_error", cv=kfolds))

    return (rmse)
alphas_alt = [14.5, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5]

alphas2 = [5e-05, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008]

e_alphas = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]

e_l1ratio = [0.8, 0.85, 0.9, 0.95, 0.99, 1]
ridge = make_pipeline(RobustScaler(), RidgeCV(alphas=alphas_alt, cv=kfolds))

lasso = make_pipeline(RobustScaler(), LassoCV(max_iter=1e7, alphas=alphas2, random_state=42, cv=kfolds))

elasticnet = make_pipeline(RobustScaler(), ElasticNetCV(max_iter=1e7, alphas=e_alphas, cv=kfolds, l1_ratio=e_l1ratio))                                

svr = make_pipeline(RobustScaler(), SVR(C= 20, epsilon= 0.008, gamma=0.0003,))
gbr = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=4, max_features='sqrt', min_samples_leaf=15, min_samples_split=10, loss='huber', random_state =42)
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
xgboost = XGBRegressor(learning_rate=0.01,n_estimators=3460,

                                     max_depth=3, min_child_weight=0,

                                     gamma=0, subsample=0.7,

                                     colsample_bytree=0.7,

                                     objective='reg:linear', nthread=-1,

                                     scale_pos_weight=1, seed=27,

                                     reg_alpha=0.00006)
stack_gen = StackingCVRegressor(regressors=(ridge, lasso, elasticnet, gbr, xgboost, lightgbm),

                                meta_regressor=xgboost,

                                use_features_in_secondary=True)
score = cv_rmse(ridge)

score = cv_rmse(lasso)

print("LASSO: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )



score = cv_rmse(elasticnet)

print("elastic net: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )



score = cv_rmse(svr)

print("SVR: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )



score = cv_rmse(lightgbm)

print("lightgbm: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )



score = cv_rmse(gbr)

print("gbr: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )



score = cv_rmse(xgboost)

print("xgboost: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )
print('START Fit')



print('stack_gen')

stack_gen_model = stack_gen.fit(np.array(X_train), np.array(y_train))



print('elasticnet')

elastic_model_full_data = elasticnet.fit(X_train, y_train)



print('Lasso')

lasso_model_full_data = lasso.fit(X_train,y_train)



print('Ridge')

ridge_model_full_data = ridge.fit(X_train, y_train)



print('Svr')

svr_model_full_data = svr.fit(X_train, y_train)



print('GradientBoosting')

gbr_model_full_data = gbr.fit(X_train, y_train)



print('xgboost')

xgb_model_full_data = xgboost.fit(X_train, y_train)



print('lightgbm')

lgb_model_full_data = lightgbm.fit(X_train, y_train)
def blend_models_predict(X):

    return ((0.1 * elastic_model_full_data.predict(X)) + \

            (0.05 * lasso_model_full_data.predict(X)) + \

            (0.1 * ridge_model_full_data.predict(X)) + \

            (0.1 * svr_model_full_data.predict(X)) + \

            (0.1 * gbr_model_full_data.predict(X)) + \

            (0.15 * xgb_model_full_data.predict(X)) + \

            (0.1 * lgb_model_full_data.predict(X)) + \

            (0.3 * stack_gen_model.predict(np.array(X))))
print('RMSLE score on train data:')

print(rmsle(y_train, blend_models_predict(X_train)))
print('Predict submission')

submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")

submission.iloc[:,1] = np.floor(np.expm1(blend_models_predict(X_test)))   
submission.to_csv('submission.csv', index=False)

submission.to_csv