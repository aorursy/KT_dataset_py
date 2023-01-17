#import some necessary librairies

# https://www.kaggle.com/edumagalhaes/quality-prediction-in-a-mining-process

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

%matplotlib inline

import matplotlib.pyplot as plt  # Matlab-style plotting

import seaborn as sns

color = sns.color_palette()

sns.set_style('darkgrid')

import warnings

def ignore_warn(*args, **kwargs):

    pass

warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)



from scipy import stats

from scipy.stats import norm, skew #for some statistics
#Now let's import and put the train and test datasets in  pandas dataframe

path1 = '../input/quality-prediction-in-a-mining-process/'

path2 = '../input/defect_per_model/'

train = pd.read_csv(path1+'MiningProcess_Flotation_Plant_Database.csv',decimal=",",parse_dates=["date"],infer_datetime_format=True)
train.describe()
print ("Size of train data : {}" .format(train.shape))
pd.set_option('display.max_columns', None)

train.head(100)
train.info()

# d = train.date.value_counts()

# d.index

# i = 0

# for ind in d.index:

#     if '2017-03-13' in ind:

#         print(ind)

#         i+=1

# print('total unique timestamp',i)

# 2017-03-13 01:00:00
#corelation

corrmat = train.corr()

corrmat
plt.figure(figsize=(10,10))

g = sns.heatmap(train.corr(),annot=True,cmap="RdYlGn")
train['Flotation Column 01 Level'].hist()
train['% Iron Feed'].hist()

# from scipy.stats import kstest

# for item in train.columns.drop('date'):

#     print(kstest(train[item],'norm'))

    
train['Flotation Column 01 Air Flow'].hist()
# train.skew(axis=0).sort_values(ascending=False)
fig, ax = plt.subplots()

ax.scatter(x = train['% Iron Feed'], y = train['% Silica Concentrate'])

plt.ylabel('% Silica Concentrate', fontsize=13)

plt.xlabel('% Iron Feed', fontsize=13)

plt.show()
def check_skewness(col):

    sns.distplot(train[col] , fit=norm);

    fig = plt.figure()

    res = stats.probplot(train[col], plot=plt)

    # Get the fitted parameters used by the function

    (mu, sigma) = norm.fit(train[col])

    print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

    

check_skewness('% Silica Concentrate')
check_skewness('% Iron Feed')
check_skewness('Flotation Column 01 Level')
#We use the numpy fuction log1p which  applies log(1+x) to all elements of the column

train["% Silica Concentrate"] = np.log1p(train["% Silica Concentrate"])



check_skewness('% Silica Concentrate')
# Check the skew of all numerical features

skewed_feats = train.drop('date',axis=1).apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

print("\nSkew in numerical features: \n")

skewness = pd.DataFrame({'Skew' :skewed_feats})

skewness.head(15)
skewness = skewness[abs(skewness) > 0.75]

print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))



from scipy.special import boxcox1p

skewed_features = skewness.index

lam = 0.15

for feat in skewed_features:

    #all_data[feat] += 1

    train[feat] = boxcox1p(train[feat], lam)
train.head()
train.describe()
y = train['% Silica Concentrate']

X = train.iloc[:,1:-2]

X.head()
from sklearn.model_selection import KFold, cross_val_score, train_test_split

X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.2, random_state=3)

# gc.collect()  

print(X_train.shape)

print(X_test.shape)

print(y_train.shape)
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC

from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor

from sklearn.kernel_ridge import KernelRidge

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

from sklearn.model_selection import KFold, cross_val_score, train_test_split

from sklearn.metrics import mean_squared_error

import xgboost as xgb

import lightgbm as lgb
#Validation function

n_folds = 5



def rmsle_cv(model):

    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)

    rmse= np.sqrt(-cross_val_score(model, X_train.values, y_train, scoring="neg_mean_squared_error", cv = kf))

    return(rmse)
#http://sklearn.lzjqsdd.com/modules/linear_model.html#ridge-regression

#Ridge not sucessful

'''KRR = KernelRidge(alpha=0.5,kernel='polynomial,  degree=2, coef0=2.5)#

score = rmsle_cv(KRR)

print("Kernel Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))'''
'''lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))

score = rmsle_cv(lasso)

print("Lasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))'''

# Lasso score: 0.1477 (0.0003)
'''ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))

score = rmsle_cv(ENet)

print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))'''

# ElasticNet score: 0.1477 (0.0003)
#https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html

'''GBoost = GradientBoostingRegressor(verbose=1, random_state =5)#n_estimators=3000, learning_rate=0.05,

#                                    max_depth=4, max_features='sqrt',

#                                    min_samples_leaf=15, min_samples_split=10, 

#                                    loss='huber', random_state =5)

score = rmsle_cv(GBoost)

print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))'''

# Gradient Boosting score: 0.1290 (0.0003)
from joblib import dump, load



'''GBoostMd = GBoost.fit(X_train.values,y_train)

dump(GBoost, 'GBoost_fitted.joblib')'''
import lightgbm as lgb

'''myLGB = lgb.LGBMRegressor(objective='regression')

myLGB.fit(X_train, y_train)

score = rmsle_cv(myLGB)

print("Lasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))'''

# myLGB.booster_.save_model('lgbmodel.txt')

# Lasso score: 0.1033 (0.0003)
from xgboost import XGBRegressor

'''myXGB = XGBRegressor(n_estimators=1000, learning_rate=0.05,verbose=True)



# myXGB.fit(X_train, X_train, early_stopping_rounds=50, verbose=True)#eval_set=[(X_test, y_test)],

score = rmsle_cv(myXGB)

print("Lasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

#0.11

myXGB.fit(X_train, y_train)'''

# myXGB.save_model('myXGB_save.joblib')
#following 3 not good

# LassoMd = lasso.fit(train.values,y_train)

# ENetMd = ENet.fit(train.values,y_train)

# KRRMd = KRR.fit(train.values,y_train)

myLGB = lgb.Booster(model_file=path2+'lgbmodel.txt')

myXGB = xgb.Booster({'nthread':-1}) # init model

myXGB.load_model(path2+'myXGB_save.joblib')      # load data



GBoost = load(path2+'GBoost_fitted.joblib')


Silica_pred = np.expm1(myLGB.predict(X_test))*0.6+np.expm1(myXGB.predict(xgb.DMatrix(X_test)))*0.3+np.expm1(GBoost.predict(X_test.values))*0.1
                                                                                                       

# from sklearn.metrics import mean_squared_error

print ("The RMSE of the model ensemble is %0.3f" %(mean_squared_error(y_test, Silica_pred)))

# The RMSE of the model ensemble is 0.214
print ("The RMSE of the lgb model  is %0.3f" %(mean_squared_error(y_test, np.expm1(myLGB.predict(X_test)))))

# The RMSE of the lgb model  is 0.215