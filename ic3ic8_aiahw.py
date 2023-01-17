import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
plt.style.use('seaborn')
from scipy.stats import norm, skew
import numpy as np

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
train = pd.read_csv("../input/train.csv",index_col = 0)
test = pd.read_csv("../input/test.csv",index_col = 0)

lable_train = train
plt.subplot(1, 2, 1)
sns.distplot(train.SalePrice, kde=False, fit = norm)

plt.subplot(1, 2, 2)
sns.distplot(np.log1p(train.SalePrice), kde=False, fit = norm)
plt.xlabel('Log SalePrice')
plt.show()
train.SalePrice = np.log1p(train.SalePrice)
corrmat = train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat,  vmin=-0.3, vmax=0.8,square=True)
plt.show()
for column in corrmat[corrmat.SalePrice>0.6].index:
    plt.subplot(2, 2, 1)
    sns.distplot(train[column], fit=norm)

    plt.subplot(2, 2, 2)
    plt.scatter(train[column], train['SalePrice'])
    
    plt.show()
train[train.GrLivArea > 4500]
train = train[train.GrLivArea <= 4500]
train.shape
for column in corrmat[corrmat.SalePrice>0.6].index:
    plt.subplot(2, 2, 1)
    sns.distplot(train[column], fit=norm)

    plt.subplot(2, 2, 2)
    plt.scatter(train[column], train['SalePrice'])
    
    plt.show()
y = train.SalePrice.reset_index(drop=True)
train = train.drop(["SalePrice"], axis=1)
X = pd.concat([train, test]).reset_index(drop=True)
def Check_null():
    nulls = X.isnull().sum()
    nullcols = nulls.loc[(nulls != 0)]
    dtypes = X.dtypes.loc[(nulls != 0)]
    info = pd.concat([nullcols, dtypes], axis=1).sort_values(by=0, ascending=False)
    print(info)
    print("There are", len(nullcols), "columns with missing values")
Check_null()
Drop_index = []
for columns,value in X.isnull().sum().iteritems():
    if value > len(X.index)*0.9:
        Drop_index.append(columns)
X = X.drop(Drop_index, axis=1)
print(X.shape)
print(Drop_index)
No_Thing_ob = ["Fence", "FireplaceQu", "GarageFinish", "GarageQual", 
               "GarageCond", "GarageType", "BsmtExposure", "BsmtCond",
              "BsmtQual", "BsmtFinType2", "BsmtFinType1", "MasVnrType",]

for columns in No_Thing_ob:
    X[columns] = X[columns].fillna("NO_")
    

X["BsmtQual"] = X["BsmtQual"].map({"EX":100, 
                                   "Gd":90, 
                                   "TA":80, 
                                   "Fa":70, 
                                   "Po":60, 
                                   "NA":0})
Check_null()
for columns in ['MSSubClass']:
    X.update(X[columns].astype('str'))


not_object_columns =  X.select_dtypes(exclude=["object"]).columns
object_columns = X.select_dtypes(include=["object"]).columns
X['LotFrontage'] = X.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))

for columns in object_columns:
    X[columns] = X[columns].fillna(X[columns].mode()[0])
 
from sklearn.preprocessing import Imputer
not_object_imp = Imputer(missing_values='NaN', strategy='median', axis=0)
np_X = not_object_imp.fit_transform(X[not_object_columns])
X[not_object_columns] = pd.DataFrame(np_X,columns = not_object_columns)
for columns in X.columns:
    print(columns,end = " : ")
    print(X[columns].value_counts().max())
Drop_index = []
for columns in X.columns:
    if X[columns].value_counts().max() > 2800:
        Drop_index.append(columns)
X = X.drop(Drop_index, axis=1)
not_object_columns =  X.select_dtypes(exclude=["object"]).columns
object_columns = X.select_dtypes(include=["object"]).columns
print(X.shape)
print(Drop_index)
Check_null()
X.describe()

X[X['GarageYrBlt'] == 2207].index
X.loc[X[X['GarageYrBlt'] == 2207].index, 'GarageYrBlt'] = 2007
from scipy.stats import skew

skew_features = X[not_object_columns].apply(lambda x: skew(x)).sort_values(ascending=False)
pd.DataFrame({'skew':skew_features})
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax

skew_index = skew_features[skew_features > 0.5].index

for column in skew_index:
    X[column]= boxcox1p(X[column], boxcox_normmax(X[column]+1))

skew_features2 = X[not_object_columns].apply(lambda x: skew(x)).sort_values(ascending=False)
pd.DataFrame({'skew':skew_features2})
X.update(X['MSSubClass'].astype('int'))
def rank_label_encoding(df, columns):
    Sorce = pd.concat([df.loc[:len(y),columns],y],axis = 1).groupby([columns]).mean()
#    Rank_dict = Sorce.rank()["SalePrice"]  
    Rank_dict = Sorce["SalePrice"].to_dict()
    return df[columns].replace(Rank_dict)

final_X = pd.DataFrame()

for columns in object_columns:
    final_X = pd.concat([final_X, rank_label_encoding(X,columns)], axis=1)

for columns in not_object_columns:
    final_X = pd.concat([final_X, X[columns]], axis=1)
import statsmodels.api as sm

ols = sm.OLS(endog = y, exog = final_X.iloc[:len(y),:])
fit = ols.fit()
df_outlier = fit.outlier_test()
df_outlier[df_outlier["bonf(p)"]<0.025]
outlier_index = df_outlier[df_outlier["bonf(p)"]<0.001].index
print(outlier_index)

final_X = final_X.drop(final_X.index[outlier_index])
y = y.drop(y.index[outlier_index])


train_X = final_X.iloc[:len(y)-100,:]
test_X = final_X.iloc[len(y)-100:len(y),:]
pred_X = final_X.iloc[len(y):,:]

train_y = y[:len(y)-100]
test_y = y[len(y)-100:len(y)]
print(train_X.shape)
print(test_X.shape)
from xgboost import XGBRegressor
xgb_model = XGBRegressor(learning_rate = 0.01, n_estimators = 3300,
                        objective = "reg:linear",
                                     max_depth= 3, min_child_weight=2,
                                     gamma = 0, subsample=0.6,
                                     colsample_bytree=0.7,
                                     scale_pos_weight=1,seed=0, 
                                     reg_alpha= 0, reg_lambda= 1)
xgb_model.fit(train_X, train_y)

from xgboost import XGBRegressor
from sklearn.grid_search import GridSearchCV
def XGBRegressor_cv(x,y):
    cv_params = {'learning_rate': [0.005]}

    other_params = dict(learning_rate = 0.01, n_estimators = 3300,
                        objective = "reg:linear",
                                     max_depth= 3, min_child_weight=2,
                                     gamma = 0, subsample=0.6,
                                     colsample_bytree=0.7,
                                     scale_pos_weight=1,seed=0, 
                                     reg_alpha= 0, reg_lambda= 1)

    model = XGBRegressor(**other_params)
    optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring="neg_mean_squared_log_error", cv=5, verbose=1, n_jobs=4)
    optimized_GBM.fit(x, y)
    evalute_result = optimized_GBM.grid_scores_
    print('每輪迭代運行結果:{0}'.format(evalute_result))
    print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
    print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))
    return model
    

XGBRegressor_cv(train_X,train_y)
from lightgbm import LGBMRegressor

lgbm_model = LGBMRegressor(learning_rate = 0.01, n_estimators = 2900,
                        objective='regression',
                                     max_depth= 3,min_child_weight=0,
                                     gamma = 0, 
                                     subsample=0.6, colsample_bytree=0.6, 
                                     scale_pos_weight=1,seed=0, 
                                     reg_alpha= 0.1, reg_lambda= 0)
lgbm_model.fit(train_X, train_y)

from lightgbm import LGBMRegressor
from sklearn.grid_search import GridSearchCV
def LGBMRegressor_cv(x,y):
    cv_params = {'learning_rate': [0.01]}
    other_params = dict(learning_rate = 0.01, n_estimators = 2900,
                        objective='regression',
                                     max_depth= 3,min_child_weight=0,
                                     gamma = 0, 
                                     subsample=0.6, colsample_bytree=0.6, 
                                     scale_pos_weight=1,seed=0, 
                                     reg_alpha= 0.1, reg_lambda= 0)

    model = LGBMRegressor(**other_params)
    optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring="neg_mean_squared_log_error", cv=5, verbose=1, n_jobs=4)
    optimized_GBM.fit(x, y)
    evalute_result = optimized_GBM.grid_scores_
    print('每輪迭代運行結果:{0}'.format(evalute_result))
    print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
    print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))
    return model
    

LGBMRegressor_cv(train_X,train_y)
from sklearn.svm import SVR

SVR_model = SVR(C = 10, epsilon = 0.1,gamma = 1e-06)

SVR_model.fit(train_X, train_y) 

from sklearn.svm import SVR
from sklearn.grid_search import GridSearchCV
def SVR_cv(x,y):
    cv_params = {"gamma":[10**(-7), 10**(-6)], "epsilon":[0.1]}
    other_params = dict(C = 10, epsilon = 0.005,gamma = 2e-06)

    model = SVR(**other_params)
    optimized_SVR = GridSearchCV(estimator=model, param_grid=cv_params, scoring="neg_mean_squared_log_error", cv=5, verbose=1, n_jobs=4)
    optimized_SVR.fit(x, y)
    evalute_result = optimized_SVR.grid_scores_
    print('每輪迭代運行結果:{0}'.format(evalute_result))
    print('参数的最佳取值：{0}'.format(optimized_SVR.best_params_))
    print('最佳模型得分:{0}'.format(optimized_SVR.best_score_))
    return model
    

SVR_cv(train_X,train_y)

from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import  KFold
kfolds = KFold(n_splits=5)
alphas = [0.0001, 0.0002, 0.0003]

l1ratio = [0.5, 0.6, 0.7, 0.8, 0.7]

elastic_model = ElasticNetCV(max_iter=1e7, alphas= alphas, 
                                        cv=kfolds, l1_ratio= l1ratio)
elastic_model.fit(train_X, train_y)

print(elastic_model.alpha_)
print(elastic_model.l1_ratio_)
from sklearn.model_selection import cross_val_score
def cv_rmse(model, X, y):
    return np.sqrt(-cross_val_score(model, X, y,
                                           scoring = 'neg_mean_squared_error',
                                           cv=kfolds))
cv_error = {"xgb":cv_rmse(xgb_model, train_X, train_y),
           "lgbm":cv_rmse(lgbm_model, train_X, train_y),
           "elastic":cv_rmse(elastic_model, train_X, train_y),
           "SVR":cv_rmse(SVR_model, train_X, train_y)}
cv_error
veri_data = {"xgb":xgb_model.predict(test_X),
            "lgbm":lgbm_model.predict(test_X),
            "elastic": elastic_model.predict(test_X),
            "SVR":SVR_model.predict(test_X)}
veri_error = dict()
for model,v in veri_data.items():
    veri_error[model] = np.power((v - test_y),2).mean()
print(veri_error)
xgb_preds = xgb_model.predict(pred_X)
lgbm_preds = lgbm_model.predict(pred_X)
elastic_preds =  elastic_model.predict(pred_X)
SVR_preds = SVR_model.predict(pred_X)
ans = xgb_preds*0.4 + elastic_preds*0.4 + lgbm_preds*0.2 + SVR_preds*0
submission = pd.read_csv("../input/sample_submission.csv")
submission["SalePrice"] = np.expm1(ans)
submission.to_csv("AIAhw_submission.csv", index=False)