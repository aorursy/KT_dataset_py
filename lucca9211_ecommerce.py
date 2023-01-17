import math

import pandas as pd

import numpy as np

from scipy.stats import kurtosis, skew



from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score

from catboost import CatBoostRegressor

from xgboost import XGBRegressor

from sklearn.metrics import mean_squared_log_error

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec

from scipy import stats

import matplotlib.style as style

style.use('fivethirtyeight')

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV

from sklearn.svm import SVR

from sklearn.ensemble import GradientBoostingRegressor

from lightgbm import LGBMRegressor

from mlxtend.regressor import StackingCVRegressor

from datetime import datetime

import warnings

warnings.filterwarnings("ignore")
train = pd.read_csv("../input/challenge8/train.csv")

test = pd.read_csv("../input/challenge8/test.csv")
train.describe()
train.head()
print (f"Train has {train.shape[0]} rows and {train.shape[1]} columns")

print (f"Test has {test.shape[0]} rows and {test.shape[1]} columns")
train["Product"] = train["Product"].apply(lambda word: word.replace('P-',''))

train["Product_Brand"] = train["Product_Brand"].apply(lambda word: word.replace('B-',''))



test["Product"] = test["Product"].apply(lambda word: word.replace('P-',''))

test["Product_Brand"] = test["Product_Brand"].apply(lambda word: word.replace('B-',''))

def plot_fn(df, feature):

    

    fig = plt.figure(constrained_layout=True, figsize=(12,8))

    # grid of 3 cols and 3 rows. 

    grid = gridspec.GridSpec(ncols=3, nrows=3, figure=fig)

    



    # histogram grid customizing. 

    ax1 = fig.add_subplot(grid[0, :2])

    ## Set the title. 

    ax1.set_title('Histogram')

    ## plot the histogram. 

    sns.distplot(df.loc[:,feature], norm_hist=True, ax = ax1)



    # customizing the QQ_plot. 

    ax2 = fig.add_subplot(grid[1, :2])

    ## Set the title. 

    ax2.set_title('QQ_plot')

    ## Plotting the QQ_Plot. 

    stats.probplot(df.loc[:,feature], plot = ax2)



    ## Customizing the Box Plot. 

    ax3 = fig.add_subplot(grid[:, 2])

    ## Set title. 

    ax3.set_title('Box Plot')

    ## Plotting the box plot. 

    sns.boxplot(df.loc[:,feature], orient='v', ax = ax3 );

    

plot_fn(train, 'Selling_Price')
train['Product'] = train['Product'].astype(int)

train['Product_Brand'] = train['Product_Brand'].astype(int)



test['Product'] = test['Product'].astype(int)

test['Product_Brand'] = test['Product_Brand'].astype(int)
# skewness and kurtosis

print("Skewness: " + str(train['Selling_Price'].skew()))

print("Kurtosis: " + str(train['Selling_Price'].kurt()))
# correlation of all the features with target variable. 

(train.corr()**2)["Selling_Price"].sort_values(ascending = False)[1:]

# trainsforming target variable using numpy.log1p, 

train["SalePrice"] = np.log1p(train["Selling_Price"])



# Plotting the newly transformed response variable

plot_fn(train, 'SalePrice')
train.drop(columns=['Selling_Price'],axis=1, inplace=True)



## Saving the target values in "y_train". 

y = train['SalePrice'].reset_index(drop=True)

## Combining train and test datasets together so that we can do all the work at once. 

all_data = pd.concat((train, test)).reset_index(drop = True)

## Dropping the target variable. 

all_data.drop(['SalePrice'], axis = 1, inplace = True)
all_data.head()
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index



skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x)).sort_values(ascending=False)



skewed_feats

final_features = pd.get_dummies(all_data).reset_index(drop=True)

final_features.shape
X = final_features.iloc[:len(y), :]



X_sub = final_features.iloc[len(y):, :]

def overfit_reducer(df):

    """

    This function takes in a dataframe and returns a list of features that are overfitted.

    """

    overfit = []

    for i in df.columns:

        counts = df[i].value_counts()

        zeros = counts.iloc[0]

        if zeros / len(df) * 100 > 99.94:

            overfit.append(i)

    overfit = list(overfit)

    return overfit





overfitted_features = overfit_reducer(X)

X = X.drop(overfitted_features, axis=1)

X_sub = X_sub.drop(overfitted_features, axis=1)
X.shape,y.shape, X_sub.shape

kfolds = KFold(n_splits=10, shuffle=True, random_state=42)



def rmsle(y, y_pred):

    return np.sqrt(mean_squared_log_error(y, y_pred))



def cv_rmse(model, X=X):

    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=kfolds))

    return (rmse)

alphas_alt = [14.5, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5]

alphas2 = [5e-05, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008]

e_alphas = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]

e_l1ratio = [0.8, 0.85, 0.9, 0.95, 0.99, 1]

ridge = make_pipeline(RobustScaler(), RidgeCV(alphas=alphas_alt, cv=kfolds))

lasso = make_pipeline(RobustScaler(), LassoCV(max_iter=1e7, 

                                              alphas=alphas2, 

                                              random_state=42, 

                                              cv=kfolds))

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
stack_gen = StackingCVRegressor(regressors=(ridge, lasso, elasticnet, xgboost, lightgbm),

                                meta_regressor=xgboost,

                                use_features_in_secondary=True)
score = cv_rmse(ridge)

print("Ridge: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )



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

stack_gen_model = stack_gen.fit(np.array(X), np.array(y))



print('elasticnet')

elastic_model_full_data = elasticnet.fit(X, y)



print('Lasso')

lasso_model_full_data = lasso.fit(X, y)



print('Ridge') 

ridge_model_full_data = ridge.fit(X, y)



print('Svr')

svr_model_full_data = svr.fit(X, y)



print('GradientBoosting')

gbr_model_full_data = gbr.fit(X, y)



print('xgboost')

xgb_model_full_data = xgboost.fit(X, y)



print('lightgbm')

lgb_model_full_data = lightgbm.fit(X, y)
def blend_models_predict(X):

    return ((0.1 * elastic_model_full_data.predict(X)) + \

            (0.05 * lasso_model_full_data.predict(X)) + \

            (0.2 * ridge_model_full_data.predict(X)) + \

            (0.1 * svr_model_full_data.predict(X)) + \

            (0.1 * gbr_model_full_data.predict(X)) + \

            (0.15 * xgb_model_full_data.predict(X)) + \

            (0.1 * lgb_model_full_data.predict(X)) + \

            (0.3 * stack_gen_model.predict(np.array(X))))

print('RMSLE score on train data:')

print(rmsle(y, blend_models_predict(X)))

print('Predict submission')

submission = pd.read_excel("../input/challenge8/sample_submission.xlsx")

submit = np.floor(np.expm1(blend_models_predict(X_sub)))
submit
submit1 = np.expm1(blend_models_predict(X_sub))
submit1
s = pd.DataFrame({'Selling_Price': submit1})
s.head()
s.to_excel("submission.xlsx", index=False)