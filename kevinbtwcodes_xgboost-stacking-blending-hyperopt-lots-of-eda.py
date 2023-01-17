import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import os

%matplotlib inline
# '../input/house-prices-advanced-regression-techniques' for Kaggle

def load_housing_data(data_name):

    csv_path = os.path.join('../input/house-prices-advanced-regression-techniques', data_name)

    return pd.read_csv(csv_path)
train, test = load_housing_data("train.csv"), load_housing_data("test.csv")
train.head()
train.shape
train.select_dtypes(include='number').columns
train.select_dtypes(include='object').columns
train['YrSold'].value_counts()
def to_str(feature, data=train):

    data[feature] = data[feature].astype(str)
to_str('YrSold')

to_str('MoSold')

to_str('MSSubClass')

to_str('YrSold',test)

to_str('MoSold',test)

to_str('MSSubClass',test)
train.dtypes.value_counts()
sns.set_style("darkgrid")
def plot_SalePrice(data=train['SalePrice']):    

    sns.distplot(data)

    plt.ylabel("Frequency")

    plt.show()
plot_SalePrice()
#skewness and kurtosis

print("Skewness: " + str(train['SalePrice'].skew()))

print("Kurtosis: " + str(train['SalePrice'].kurt()))
corr_matrix = train.corr()

(corr_matrix["SalePrice"]**2).sort_values(ascending=False)
corr = train.corr()**2

plt.subplots(figsize=(12,12))

sns.heatmap(corr, vmax=0.9, square=True)

plt.show()
def scatterplot(x, y='SalePrice', data=train):

#     plt.subplots(figsize=(12,8))

    sns.scatterplot(x=x, y=y, data=data)

    plt.show()
sns.boxplot(x='OverallQual', y='SalePrice', data=train)

plt.show()
scatterplot('GrLivArea')
scatterplot('GarageCars')
scatterplot('GarageArea')
scatterplot('TotalBsmtSF')
scatterplot('1stFlrSF')
scatterplot('TotRmsAbvGrd')
scatterplot('YearBuilt')
scatterplot('YearRemodAdd')
scatterplot('MasVnrArea')
train.sort_values('GrLivArea', ascending=False).head(2)['GrLivArea']
train = train[train['GrLivArea'] < 4676]
train.shape
train.sort_values('GrLivArea', ascending=False).head(2)['GrLivArea']
plot_SalePrice()
train['SalePrice'] = np.log1p(train['SalePrice'])
plot_SalePrice(train['SalePrice'])
test_dummy = test.copy()

test_dummy['SalePrice'] = np.zeros(len(test))

combined = pd.concat([train,test_dummy])
combined.shape
def missing_percent(data, n=35):

    num_of_nulls = data.isnull().sum().sort_values(ascending=False)

    percent = (data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)

    result = pd.concat([num_of_nulls, percent], axis=1, keys=['Number', 'Percent'])

    return result.head(n)
missing_percent(train)
missing_percent(test)
missing_percent(combined)
import warnings

warnings.filterwarnings("ignore")
def fill_missing(df):

    

    # Filling 'Functional' according to the data description

    df['Functional'] = df['Functional'].fillna('Typ')

    

    # Filling some categorical values with mode

    cats = ['Electrical','SaleType','Exterior2nd','KitchenQual','Exterior1st','MSZoning']

    for cat in cats:

        df[cat] = df[cat].fillna(df[cat].mode()[0])

    

    # Filling LotFrontage by grouping by neighborhood and taking the median

    df['LotFrontage'] = df['LotFrontage'].fillna(

        df.groupby('Neighborhood')['LotFrontage'].transform('median'))

    

    # Filling the rest of the categorical value with 'None'

    # (For some features like those of basement and garage, NA means None.

    # But for some features we don't know so let's just use None)

    df_cat = df[list(df.select_dtypes(include='object').columns)]

    df.update(df_cat.fillna('None'))

    

    # Filling the rest of the numerical values with 0

    # (For some features like alley and LotFrontage, NA means 0.

    # But for some features we don't know so let's just use 0)

    df_num = df[list(df.select_dtypes(include='number').columns)]

    df.update(df_num.fillna(0))

    

    return df
fill_missing(train).isnull().sum().sum()
fill_missing(test).isnull().sum().sum()
test_dummy = test.copy()

test_dummy['SalePrice'] = np.zeros(len(test))

combined = pd.concat([train,test_dummy])
missing_percent(combined,1)
from scipy.special import boxcox1p

from scipy.stats import boxcox_normmax



def fix_skew(df):

    

    skew_index = []



    # A function to show the top 10 skewed features

    def show_skew(df):

        df_num = df[list(df.select_dtypes(include='number').columns)]



        skew_features = df_num.apply(lambda x: x.skew()).sort_values(ascending=False)



        high_skew = skew_features[skew_features > 0.6]

        nonlocal skew_index

        skew_index = high_skew.index



        print("{} features have skew > 0.6 :".format(high_skew.shape[0]))

        skewness = pd.DataFrame({'Skew' :high_skew})

        print(skew_features.head(10))

        

    # Before transformation

    show_skew(df)



    # Transformation

    for i in skew_index:

        df[i] = boxcox1p(df[i], boxcox_normmax(df[i] + 1))

        

    # After transformation

    show_skew(df)
fix_skew(combined)
combined.select_dtypes(include='number').columns
combined.select_dtypes(include='object').columns
combined['Has2ndFlr'] = combined['2ndFlrSF'].apply(lambda x : 1 if x > 0 else 0)



combined['HasWoodDeck'] = combined['WoodDeckSF'].apply(lambda x : 1 if x > 0 else 0)

combined['HasOpenPorch'] = combined['OpenPorchSF'].apply(lambda x : 1 if x > 0 else 0)

combined['HasEnclosedPorch'] = combined['EnclosedPorch'].apply(lambda x : 1 if x > 0 else 0)

combined['Has3SsnPorch'] = combined['3SsnPorch'].apply(lambda x : 1 if x > 0 else 0)

combined['HasScreenPorch'] = combined['ScreenPorch'].apply(lambda x : 1 if x > 0 else 0)
def binarize(column, data=combined):

    print(combined[column].value_counts())

    combined[column] = combined[column].apply(lambda x : 0 if x == 'None' else 1)

    print(combined[column].value_counts())
binarize('PoolQC')
combined['HouseQualAdd'] = combined['OverallQual'] + combined['OverallCond']

combined['HouseQualProd'] = combined['OverallQual'] * combined['OverallCond']
combined['TotalSqrFt'] = (combined['BsmtFinSF1'] + combined['BsmtFinSF2']

                    + combined['1stFlrSF'] + combined['2ndFlrSF'])

combined['TotalSF'] = (combined['TotalBsmtSF'] + combined['1stFlrSF'] + combined['2ndFlrSF'])

combined['TotalPorchSF'] = (combined['WoodDeckSF'] + combined['OpenPorchSF']

                         + combined['EnclosedPorch'] + combined['3SsnPorch'] 

                         + combined['ScreenPorch'])

combined['TotalBath'] = (combined['FullBath'] 

                      + (0.5 * combined['HalfBath']) 

                      + combined['BsmtFullBath'] 

                      + (0.5 * combined['BsmtHalfBath']))
combined.drop(['Street', 'Utilities'], axis=1, inplace=True)
combined.shape
combined_encoded = pd.get_dummies(combined).reset_index(drop=True)
combined_encoded.shape
combined_encoded = combined_encoded.loc[:,~combined_encoded.columns.duplicated()]
combined_encoded.shape
len(train)
y_test, X_test = (combined_encoded[len(train):]["SalePrice"], 

                 combined_encoded.drop(["SalePrice"], axis=1)[len(train):])
y_test.shape, X_test.shape
y_train_full, X_train_full = (combined_encoded[:len(train)]["SalePrice"], 

                             combined_encoded.drop(["SalePrice"], axis=1)[:len(train)])
y_train_full.shape, X_train_full.shape
from sklearn.model_selection import train_test_split



X_train, X_valid, y_train, y_valid = train_test_split(

    X_train_full, y_train_full, test_size=0.20, random_state=42)
X_train.shape, X_valid.shape
from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor

from lightgbm import LGBMRegressor

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.svm import SVR

from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV

from mlxtend.regressor import StackingCVRegressor

from sklearn.model_selection import KFold, cross_val_score

from sklearn.pipeline import make_pipeline

from sklearn.metrics import mean_squared_error

import hyperopt

from hyperopt import hp

from hyperopt import fmin, tpe, hp, anneal, Trials

from sklearn.preprocessing import RobustScaler
def rmsle(y, y_pred):

    # y and y_pred are already logarithmic as we've used the log1p transform

    return np.sqrt(mean_squared_error(y, y_pred))
kfolds = KFold(n_splits=10, random_state=42, shuffle=True)



def cv_rmse(model, X=X_train_full):

    rmse = np.sqrt(-cross_val_score(model, X, y_train_full, 

                                    scoring='neg_mean_squared_error', cv=kfolds))

    return (rmse)
def mse_cv(params, cv=kfolds, X=X_train_full, y=y_train_full):

    # the function gets a set of variable parameters in "params"

    params = {'max_depth': int(params['max_depth']),

              'learning_rate': params['learning_rate'],

              'gamma': params['gamma'],

              'colsample_bytree': params['colsample_bytree'], 

              'subsample': params['subsample']

             }

    

    # we use this params to create a new LGBM Regressor

    model = XGBRegressor(objective='reg:linear', n_estimators=200,

                          random_state=42, 

                          **params)

    

    # and then conduct the cross validation with the same folds as before

    score = np.sqrt(-cross_val_score(model, X, y, cv=cv, scoring="neg_mean_squared_error", n_jobs=-1).mean())



    return score
# possible values of parameters

space={'max_depth' : hp.quniform('max_depth', 2, 10, 1),

       'learning_rate': hp.loguniform('learning_rate', -5, 0), 

       'gamma': hp.loguniform('gamma', -1, 0), 

       'colsample_bytree': hp.loguniform('colsample_bytree', -1, 0), 

       'subsample': hp.loguniform('subsample', -1, 0)

      }



# trials will contain logging information

trials = Trials()



best=fmin(fn=mse_cv, # function to optimize

          space=space, 

          algo=tpe.suggest, # optimization algorithm, hyperotp will select its parameters automatically

          max_evals=200, # maximum number of iterations

          trials=trials, # logging

          rstate=np.random.RandomState(42) # fixing random state for the reproducibility

         )
# computing the score on the test set

xgb_reg = XGBRegressor(random_state=42, n_estimators=5000,

                     max_depth=int(best['max_depth']),learning_rate=best['learning_rate'], 

                     gamma=best['gamma'], colsample_bytree=best['colsample_bytree'], 

                     subsample=best['subsample'], objective='reg:linear', 

                     nthread=-1, scale_pos_weight=1)



print(np.mean(cv_rmse(xgb_reg))) 



print("Best MSE {:.3f} params {}".format( mse_cv(best), best)) 
xgb_reg = XGBRegressor(random_state=42,

                       n_estimators=7000, 

                       max_depth=2, 

                       learning_rate=0.138170657, 

                       gamma=0.38498075, 

                       colsample_bytree=0.599437614, 

                       objective='reg:linear', 

                       nthread=-1,

                       scale_pos_weight=1,

                       subsample=0.534382267)
lgbm_reg = LGBMRegressor(random_state=42, 

                         n_estimators=7000, 

                         num_leaves=3, 

                         learning_rate=0.099384860, 

                         bagging_seed=5, 

                         feature_fraction_seed=5, 

                         bagging_fraction=0.407679661, 

                         feature_fraction=0.563479974, 

                         min_sum_hessian_in_leaf=20)
svr_reg = make_pipeline(RobustScaler(), SVR(C=11,

                                            gamma=0.006742321288,

                                            epsilon=0.010861719298))
gb_reg = GradientBoostingRegressor(max_depth=21,

                                   min_samples_leaf=11,

                                   min_samples_split=8,

                                   learning_rate=0.038954801112873964, 

                                   loss='huber', max_features='sqrt',

                                   n_estimators=7000, random_state=42)
ridge_alphas = [1e-10, 1e-8, 1e-5, 9e-4, 7e-4, 5e-4, 3e-4, 1e-4, 1e-3, 5e-2, 1e-2, 

                0.1, 0.3, 1, 3, 5, 10, 15, 18, 20, 30, 50, 75, 100]



lasso_alphas = [5e-5, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008]



elastic_alphas = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001]

elastic_l1ratio = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.97, 0.99, 1]
ridge_reg = make_pipeline(RobustScaler(), RidgeCV(alphas=ridge_alphas, cv=kfolds))

lasso_reg = make_pipeline(RobustScaler(), LassoCV(max_iter=1e7, alphas=lasso_alphas, 

                                              random_state=42, cv=kfolds))

elasticnet_reg = make_pipeline(RobustScaler(), ElasticNetCV(max_iter=1e7, 

                                                            alphas=elastic_alphas, 

                                                            cv=kfolds, 

                                                            l1_ratio=elastic_l1ratio))                                
stack_gen = StackingCVRegressor(regressors=(ridge_reg, lasso_reg, elasticnet_reg, xgb_reg, lgbm_reg, gb_reg, svr_reg),

                                meta_regressor=xgb_reg,

                                use_features_in_secondary=True)
scores = {}



score = cv_rmse(lgbm_reg)

print("lgbm_reg: {:.4f} ({:.4f})".format(score.mean(), score.std()))

scores['lgbm'] = (score.mean(), score.std())



score = cv_rmse(xgb_reg)

print("xgb_reg: {:.4f} ({:.4f})".format(score.mean(), score.std()))

scores['xgb'] = (score.mean(), score.std())



score = cv_rmse(svr_reg)

print("SVR: {:.4f} ({:.4f})".format(score.mean(), score.std()))

scores['svr'] = (score.mean(), score.std())



score = cv_rmse(ridge_reg)

print("ridge_reg: {:.4f} ({:.4f})".format(score.mean(), score.std()))

scores['ridge'] = (score.mean(), score.std())



score = cv_rmse(lasso_reg)

print("lasso_reg: {:.4f} ({:.4f})".format(score.mean(), score.std()))

scores['lasso'] = (score.mean(), score.std())



score = cv_rmse(elasticnet_reg)

print("elasticnet_reg: {:.4f} ({:.4f})".format(score.mean(), score.std()))

scores['elasticnet'] = (score.mean(), score.std())



score = cv_rmse(gb_reg)

print("gb_reg: {:.4f} ({:.4f})".format(score.mean(), score.std()))

scores['gb'] = (score.mean(), score.std())
%%time

print('START Fit')



print('stack_gen')

stack_gen_model = stack_gen.fit(np.array(X_train_full), np.array(y_train_full))



print('elasticnet')

elastic_reg_model = elasticnet_reg.fit(X_train_full, y_train_full)



print('Lasso')

lasso_reg_model = lasso_reg.fit(X_train_full, y_train_full)



print('Ridge') 

ridge_reg_model = ridge_reg.fit(X_train_full, y_train_full)



print('Svr')

svr_reg_model = svr_reg.fit(X_train_full, y_train_full)



print('GradientBoosting')

gb_reg_model = gb_reg.fit(X_train_full, y_train_full)



print('xgboost')

xgb_reg_model = xgb_reg.fit(X_train_full, y_train_full)



print('lightgbm')

lgbm_reg_model = lgbm_reg.fit(X_train_full, y_train_full)



print('END FIT')
def blender(X):

    return ((0.1 * elastic_reg_model.predict(X)) + \

            (0.1 * lasso_reg_model.predict(X)) + \

            (0.1 * ridge_reg_model.predict(X)) + \

            (0.1 * svr_reg_model.predict(X)) + \

            (0.1 * gb_reg_model.predict(X)) + \

            (0.1 * xgb_reg_model.predict(X)) + \

            (0.1 * lgbm_reg_model.predict(X)) + \

            (0.3 * stack_gen_model.predict(np.array(X))))
print('RMSLE score on train data:')

print(rmsle(y_train_full, blender(X_train_full)))
print('Predict submission')

submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")

submission.iloc[:,1] = np.floor(np.expm1(blender(X_test)))
submission.to_csv("submission.csv", index=False)