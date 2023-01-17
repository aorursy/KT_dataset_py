# imports

import numpy as np

import pandas as pd

import seaborn as sns

color = sns.color_palette()

sns.set_style('darkgrid')

from matplotlib import pyplot as plt



from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import RobustScaler



from sklearn.tree import DecisionTreeRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.svm import SVR

from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC

from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor,  ExtraTreesRegressor,  AdaBoostRegressor

from sklearn.kernel_ridge import KernelRidge

import xgboost as xgb

import lightgbm as lgb



from sklearn.pipeline import make_pipeline

from sklearn.model_selection import KFold, cross_val_score, train_test_split

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import GridSearchCV



from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone



import warnings

warnings.filterwarnings('ignore')

%matplotlib inline



pd.set_option('display.float_format', lambda x: '{:.4f}'.format(x)) # limit floats
# data already preprocessed, engineered, scaled, etc

df_train = pd.read_csv('/kaggle/input/housing-df-train/df_train.csv')

df_test = pd.read_csv('/kaggle/input/housing-df-test/df_test.csv')
print(df_train.shape)

print(df_test.shape)
X = df_train.iloc[:,0:220]

y = df_train.iloc[:,220]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 50)
n_folds = 10



def rmsle_cv(model):

    kf = KFold(n_folds, shuffle=True, random_state=50).get_n_splits(X_train)

    rmse = np.sqrt(-cross_val_score(model, X_train, y_train, scoring="neg_mean_squared_error", cv = kf))

    return(rmse)
## All models are made robust to outliers, good practice



# LASSO Regression

lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))



# Elastic Net Regression

ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))



# Kernel Ridge Regression

KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)



# K-Neighbors (needs tuning)

KNN = make_pipeline(RobustScaler(), KNeighborsRegressor())



# Decision Tree Regressor

CART = make_pipeline(RobustScaler(), DecisionTreeRegressor())



# Support Vector Regression

SVReg = make_pipeline(RobustScaler(), SVR()) # could have better settings



# Gradient Boosting Regression

GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,

                                   max_depth=4, max_features='sqrt',

                                   min_samples_leaf=15, min_samples_split=10, 

                                   loss='huber', random_state =5)



# XGBoost

model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 

                             learning_rate=0.05, max_depth=3, 

                             min_child_weight=1.7817, n_estimators=2200,

                             reg_alpha=0.4640, reg_lambda=0.8571,

                             subsample=0.5213, silent=1,

                             random_state =7, nthread = -1)



# LightGBM

model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,

                              learning_rate=0.05, n_estimators=720,

                              max_bin = 55, bagging_fraction = 0.8,

                              bagging_freq = 5, feature_fraction = 0.2319,

                              feature_fraction_seed=9, bagging_seed=9,

                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)



names = ['lasso', 'ENet', 'KRR', 'KNN', 'CART', 'SVReg', 'GBoost', 'model_xgb', 'model_lgb']

models = [lasso, ENet, KRR, KNN, CART, SVReg, GBoost, model_xgb, model_lgb]



# # non-ensemble (faster)

# names = ['lasso', 'ENet', 'KRR', 'KNN', 'CART', 'SVReg', 'GBoost']

# models = [lasso, ENet, KRR, KNN, CART, SVReg, GBoost]
results = []



for model, name in zip(models, names):

    score = rmsle_cv(model)

    print('{} score: {:.4f} ({:.4f})'.format(name, score.mean(), score.std()))



    results.append(score)
# boxplot algorithm comparison

fig = plt.figure()

fig.suptitle('Algorithm Comparison')

ax = fig.add_subplot(111)

plt.boxplot(results)

ax.set_xticklabels(names)

plt.show()
def rmsle(y, y_pred):

    return np.sqrt(mean_squared_error(y, y_pred))



fittedmodel = ENet.fit(X_train, y_train)

predictions = fittedmodel.predict(X_test)



rmsle(y_test, predictions)
class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):

    def __init__(self, models):

        self.models = models

        

    # we define clones of the original models to fit the data in

    def fit(self, X, y):

        self.models_ = [clone(x) for x in self.models]

        

        # Train cloned base models

        for model in self.models_:

            model.fit(X, y)



        return self

    

    #Now we do the predictions for cloned models and average them

    def predict(self, X):

        predictions = np.column_stack([

            model.predict(X) for model in self.models_

        ])

        return np.mean(predictions, axis=1) 
averaged_models = AveragingModels(models = (ENet, GBoost, KRR, lasso))



score = rmsle_cv(averaged_models)

print('{} score: {:.4f} ({:.4f})'.format('Averaged models', score.mean(), score.std()))
fittedmodel = averaged_models.fit(X_train, y_train)

predictions = fittedmodel.predict(X_test)



rmsle(y_test, predictions)
sub = pd.DataFrame()

sub['Id'] = test_ID

sub['SalePrice'] = np.expm1(predictions)

sub.to_csv('submission.csv',index=False)



# keep in mind this will give error since we don't have test_ID in this book