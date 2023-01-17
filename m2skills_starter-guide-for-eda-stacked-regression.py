# importing libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
# loading Data
train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")
train.head()
train.info()
# plotting SalePrice to check for outliers
fig=plt.subplots(figsize=(15,7))
sns.regplot(np.sort(train.SalePrice), np.arange(len(train)), fit_reg = False)
plt.show()
from scipy.stats import norm
fig,ax=plt.subplots(1,2,figsize=(22,7))
sns.set(font_scale=1.25)
#### first figure
sns.distplot(train.SalePrice, ax = ax[0], fit = norm)
#### second figure
sns.distplot(np.log1p(train.SalePrice), ax = ax[1], fit = norm)
plt.show()

# changing saleprice and replacing it with log values
# train.SalePrice = np.log1p(train.SalePrice)
# filling the columns with continous data with 0
cols=["MasVnrArea", "BsmtUnfSF", "TotalBsmtSF", "GarageCars", "BsmtFinSF2", "BsmtFinSF1", "GarageArea"]
for col in cols:
    train[col].fillna(0, inplace=True)
    test[col].fillna(0, inplace=True)
# filling all the columns with categorical values with "None"
cols1 = ["PoolQC" , "MiscFeature", "Alley", "Fence", "FireplaceQu", "GarageQual", "GarageCond", "GarageFinish", "GarageYrBlt", "GarageType", "BsmtExposure", "BsmtCond", "BsmtQual", "BsmtFinType2", "BsmtFinType1", "MasVnrType"]
for col in cols1:
    train[col].fillna("None", inplace=True)
    test[col].fillna("None", inplace=True)
# fill in with mode
cols2 = ["MSZoning", "BsmtFullBath", "BsmtHalfBath", "Utilities", "Functional", "Electrical", "KitchenQual", "SaleType","Exterior1st", "Exterior2nd"]
for col in cols2:
    train[col].fillna(train[col].mode()[0], inplace=True)
    test[col].fillna(test[col].mode()[0], inplace=True)
# filling Lot frontage missing values with mean
train.LotFrontage.fillna(train.LotFrontage.mean(), inplace = True)
test.LotFrontage.fillna(test.LotFrontage.mean(), inplace = True)
# filling NA's with the mean of the column:
# train = train.fillna(train.mean())
# test  = test.fillna(test.mean())
# finding features with skewness greater than 0.75
from scipy.stats import skew
#log transform skewed numeric features:
numeric_feats = train.dtypes[train.dtypes != "object"].index

skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index
print(skewed_feats)
# visualizing some features before and after taking logs

from scipy.stats import norm
fig,ax=plt.subplots(3,2,figsize=(30,20))
sns.set(font_scale=2)
#### first figure
ax[0, 0].set_title('GrLivArea distribution before taking log')
sns.distplot(train.GrLivArea, ax = ax[0, 0], fit = norm)
#### second figure
ax[0, 1].set_title('GrLivArea distribution after taking log')
sns.distplot(np.log1p(train.GrLivArea), ax = ax[0, 1], fit = norm)

#### third figure
ax[1, 0].set_title('LotArea distribution before taking log')
sns.distplot(train.LotArea, ax = ax[1, 0], fit = norm)
#### fourth figure
ax[1, 1].set_title('LotArea distribution after taking log')
sns.distplot(np.log1p(train.LotArea), ax = ax[1, 1], fit = norm)

#### fifth figure
ax[2, 0].set_title('1stFlrSF distribution before taking log')
sns.distplot(train["1stFlrSF"], ax = ax[2, 0], fit = norm)
#### sixth figure
ax[2, 1].set_title('1stFlrSF distribution after taking log')
sns.distplot(np.log1p(train["1stFlrSF"]), ax = ax[2, 1], fit = norm)
plt.subplots_adjust(hspace=0.3,wspace=0.1)

plt.show()
train["totalSFArea"] = train["TotalBsmtSF"] + train["2ndFlrSF"] + train["1stFlrSF"]
test["totalSFArea"] = test["TotalBsmtSF"] + test["2ndFlrSF"] + test["1stFlrSF"]
#correlation matrix for 15 variables with largest correlation
corrmat = train.corr()
f, ax = plt.subplots(figsize=(12, 12))
k = 15 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train[cols].values.T)

# Generate a mask for the upper triangle
mask = np.zeros_like(cm, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True


hm = sns.heatmap(cm, mask=mask, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
f, ax = plt.subplots(figsize=(12, 7))
sns.countplot(y = train.OverallQual)
plt.title("Distribution plot for Overall Quality of the House")
plt.show()
fig,ax=plt.subplots(1,2,figsize=(15,5))
sns.set(font_scale=1.25)
#### first figure
ax[0].set_title('GrLivArea distribution before taking log')
sns.distplot(train.GarageArea, ax = ax[0], fit = norm)
#### second figure
ax[1].set_title('GrLivArea distribution after taking log')
sns.countplot(y=train.GarageCars, ax = ax[1])
plt.show()
plt.figure(figsize = (12, 6))
sns.countplot(x = 'Neighborhood', data = train)
xt = plt.xticks(rotation=90)
plt.show()
fig,ax=plt.subplots(2,2,figsize=(20,20))
#### first figure
sns.regplot(y = train.SalePrice, x = train["1stFlrSF"], ax = ax[0, 0])
#### second figure
sns.regplot(y = train.SalePrice, x = train["2ndFlrSF"], ax = ax[0, 1])
#### third figure
sns.regplot(y = train.SalePrice, x = train.totalSFArea, ax = ax[1, 0])
#### fourth figure
sns.regplot(y = train.SalePrice, x = train.TotalBsmtSF, ax = ax[1, 1])

plt.show()
plt.figure(figsize = (12, 6))
sns.boxplot(x = 'Neighborhood', y = 'SalePrice',  data = train)
xt = plt.xticks(rotation=45)
plt.show()
sns.boxplot(x = 'OverallQual', y = 'SalePrice',  data = train)
plt.show()
fig,ax=plt.subplots(2,2,figsize=(30,20))
#### first figure
ax[0, 0].set_title('Garage Quality and SalePrice')
sns.boxplot(x = 'GarageQual', y = 'SalePrice',  data = train, ax = ax[0, 0])
#### second figure
ax[0, 1].set_title('GrLivArea distribution after taking log')
sns.boxplot(x = 'KitchenQual', y = 'SalePrice',  data = train, ax = ax[0, 1])

#### third figure
ax[1, 0].set_title('LotArea distribution before taking log')
sns.boxplot(x = 'BsmtQual', y = 'SalePrice',  data = train, ax = ax[1, 0])
#### fourth figure
ax[1, 1].set_title('LotArea distribution after taking log')
sns.boxplot(x = 'ExterQual', y = 'SalePrice',  data = train, ax = ax[1, 1])
plt.show()
fig, ax = plt.subplots(2, 1, figsize = (10, 8))
sns.boxplot(x = 'SaleType', y = 'SalePrice', data = train, ax = ax[0])
sns.boxplot(x = 'SaleCondition', y = 'SalePrice', data = train, ax = ax[1])
plt.tight_layout()
plt.show()
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x="YearBuilt", y="SalePrice", data=train)
fig.axis(ymin=0, ymax=800000);
plt.xticks(rotation=90);
plt.show()
fig, ax = plt.subplots(2, 1, figsize = (10, 7))
sns.boxplot(x = 'HouseStyle', y = 'SalePrice', data = train, ax = ax[0])
sns.boxplot(x = 'RoofStyle', y = 'SalePrice', data = train, ax = ax[1])
plt.tight_layout()
plt.show()
## why to normalize skewed features
train[skewed_feats] = np.log1p(train[skewed_feats])
skewed_feats = list(skewed_feats)
skewed_feats.remove("SalePrice")
test[skewed_feats[:len(skewed_feats) - 1]] = np.log1p(test[skewed_feats[:len(skewed_feats) - 1]])
train = pd.get_dummies(train);
test =  pd.get_dummies(test);
# seperating target varaible from the dataset
target = train.SalePrice
train = train.drop(["SalePrice"], axis = 1)
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.linear_model import ElasticNet, SGDRegressor, BayesianRidge
from sklearn.kernel_ridge import KernelRidge
import warnings
warnings.filterwarnings('ignore')
from xgboost import XGBRegressor

def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, train, target, scoring="neg_mean_squared_error", cv = 5))
    return(rmse)
# testing out all the models and printing out the results
models = [Ridge(alpha = 10),
          Lasso(alpha=0.005,max_iter=10000),
          RandomForestRegressor(),
          GradientBoostingRegressor(),
          SVR(),
          LinearSVR(),
          ElasticNet(alpha=0.001,max_iter=10000),
          BayesianRidge(),
          KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5),
          ExtraTreesRegressor(),
          XGBRegressor()
         ]
names = ["Ridge", "Lasso", "RF", "GBR", "SVR", "LinSVR", "Ela","Bay","Ker","Extra", "XGB"]
for name, model in zip(names, models):
    score = rmse_cv(model)
    print("{}: {:.6f}, {:.4f}".format(name,score.mean(),score.std()))
model_ridge = Ridge()
alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]
cv_ridge = [rmse_cv(Ridge(alpha = alpha)).mean() 
            for alpha in alphas]
# plotting cv scores for models with different alpha, this shows how important it is to tune our model before submission
cv_ridge = pd.Series(cv_ridge, index = alphas)
fig, ax = plt.subplots(figsize = (9, 5))
cv_ridge.plot(title = "Validation - Just Do It")
plt.xlabel("alpha")
plt.ylabel("rmse")
plt.show()
model_ridge = Ridge(normalize=True, alpha = 10)
model_ridge.fit(train,target)
plt.rcParams['figure.figsize'] = (8.0, 8.0)
preds = pd.DataFrame({"preds":model_ridge.predict(train), "true":target})
preds["residuals"] = preds["true"] - preds["preds"]
preds.plot(x = "preds", y = "residuals",kind = "scatter")
plt.show()
plt.scatter(preds.preds, preds.true, c = "red", marker = "s", label = "Validation data")
plt.title("Ridge Regression")
plt.xlabel("Predicted values")
plt.ylabel("Real values")
plt.legend(loc = "upper left")
plt.plot([10.5, 13.5], [10.5, 13.5], c = "red")
plt.show()
model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(train, target)
rmse_cv(model_lasso).mean()
coef = pd.Series(model_lasso.coef_, index = train.columns)
print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
imp_coef = pd.concat([coef.sort_values().head(10), coef.sort_values().tail(15)])
plt.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind = "barh")
plt.title("Coefficients in the Lasso Model")
plt.show()
# residual plot
plt.rcParams['figure.figsize'] = (6.0, 6.0)
preds = pd.DataFrame({"preds":model_lasso.predict(train), "true":target})
preds["residuals"] = preds["true"] - preds["preds"]
preds.plot(x = "preds", y = "residuals",kind = "scatter")
plt.show()
class grid():
    def __init__(self,model):
        self.model = model
    
    def grid_get(self,X,y,param_grid):
        grid_search = GridSearchCV(self.model,param_grid,cv=5, scoring="neg_mean_squared_error")
        grid_search.fit(X,y)
        print(grid_search.best_params_, np.sqrt(-grid_search.best_score_))
        grid_search.cv_results_['mean_test_score'] = np.sqrt(-grid_search.cv_results_['mean_test_score'])
        print(pd.DataFrame(grid_search.cv_results_)[['params','mean_test_score','std_test_score']])
grid(Lasso()).grid_get(train,target,{'alpha': [1, 0.1, 0.001, 0.0004,0.0005,0.0007,0.0009],'max_iter':[10000]})
lasso = Lasso(alpha=0.0005,max_iter=10000)
ridge = Ridge(alpha=10)
svr = SVR(gamma= 0.0004,kernel='rbf',C=13,epsilon=0.009)
ker = KernelRidge(alpha=0.2 ,kernel='polynomial',degree=3 , coef0=0.8)
ela = ElasticNet(alpha=0.005,l1_ratio=0.08,max_iter=10000)
bay = BayesianRidge()
import pandas as pd
import numpy as np
from scipy.stats import skew
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, Lasso
from math import sqrt
TARGET = 'SalePrice'
NFOLDS = 6
SEED = 0
NROWS = None
SUBMISSION_FILE = '../input/sample_submission.csv'
## Load the data ##
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

ntrain = train.shape[0]
ntest = test.shape[0]
## Preprocessing ##

y_train = np.log(train[TARGET]+1)

train.drop([TARGET], axis=1, inplace=True)

all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],
                      test.loc[:,'MSSubClass':'SaleCondition']))

#log transform skewed numeric features:
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index

all_data[skewed_feats] = np.log1p(all_data[skewed_feats])

all_data = pd.get_dummies(all_data)

#filling NA's with the mean of the column:
all_data = all_data.fillna(all_data.mean())
#creating matrices for sklearn:
x_train = np.array(all_data[:train.shape[0]])
x_test = np.array(all_data[train.shape[0]:])

kf = KFold(NFOLDS, shuffle=True, random_state=SEED)
# Wrapper class for all the sklearn models
class SklearnWrapper(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)
class XgbWrapper(object):
    def __init__(self, seed=0, params=None):
        self.param = params
        self.param['seed'] = seed
        self.nrounds = params.pop('nrounds', 250)

    def train(self, x_train, y_train):
        dtrain = xgb.DMatrix(x_train, label=y_train)
        self.gbdt = xgb.train(self.param, dtrain, self.nrounds)

    def predict(self, x):
        return self.gbdt.predict(xgb.DMatrix(x))

    
# function to get out of fold indices for different models
def get_oof(clf):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, (train_index, test_index) in enumerate(kf.split(x_train)):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)

et_params = {
    'n_jobs': 16,
    'n_estimators': 200,
    'max_features': 0.5,
    'max_depth': 12,
    'min_samples_leaf': 2,
}

rf_params = {
    'n_jobs': 16,
    'n_estimators': 200,
    'max_features': 0.15,
    'max_depth': 12,
    'min_samples_leaf': 2,
}

xgb_params = {
    'seed': 0,
    'colsample_bytree': 0.7,
    'silent': 1,
    'subsample': 0.7,
    'learning_rate': 0.15,
    'objective': 'reg:linear',
    'max_depth': 6,
    'num_parallel_tree': 1,
    'min_child_weight': 1,
    'eval_metric': 'rmse',
    'nrounds': 500
}



rd_params={
    'alpha': 10
}


ls_params={
    'alpha': 0.010
}
# initialising models
# xg = XgbWrapper(seed=SEED, params=xgb_params)
et = SklearnWrapper(clf=ExtraTreesRegressor, seed=SEED, params=et_params)
gbr= SklearnWrapper(clf=GradientBoostingRegressor, params = {'learning_rate' : 0.1})
rf = SklearnWrapper(clf=RandomForestRegressor, seed=SEED, params=rf_params)
rd = SklearnWrapper(clf=Ridge, seed=SEED, params=rd_params)
ls = SklearnWrapper(clf=Lasso, seed=SEED, params=ls_params)
# getting out of fold indices and evaluating models
gbr_oof_train, gbr_oof_test = get_oof(gbr)
et_oof_train, et_oof_test = get_oof(et)
rf_oof_train, rf_oof_test = get_oof(rf)
rd_oof_train, rd_oof_test = get_oof(rd)
ls_oof_train, ls_oof_test = get_oof(ls)

print("GB-CV: {}".format(sqrt(mean_squared_error(y_train, gbr_oof_train))))
print("ET-CV: {}".format(sqrt(mean_squared_error(y_train, et_oof_train))))
print("RF-CV: {}".format(sqrt(mean_squared_error(y_train, rf_oof_train))))
print("RD-CV: {}".format(sqrt(mean_squared_error(y_train, rd_oof_train))))
print("LS-CV: {}".format(sqrt(mean_squared_error(y_train, ls_oof_train))))
x_train = np.concatenate((gbr_oof_train, et_oof_train, rf_oof_train, rd_oof_train, ls_oof_train), axis=1)
x_test = np.concatenate((gbr_oof_test, et_oof_test, rf_oof_test, rd_oof_test, ls_oof_test), axis=1)

print("{},{}".format(x_train.shape, x_test.shape))

dtrain = xgb.DMatrix(x_train, label=y_train)
dtest = xgb.DMatrix(x_test)

xgb_params = {
    'seed': 0,
    'colsample_bytree': 0.8,
    'silent': 1,
    'subsample': 0.6,
    'learning_rate': 0.015,
    'objective': 'reg:linear',
    'max_depth': 1,
    'num_parallel_tree': 1,
    'min_child_weight': 1,
    'eval_metric': 'rmse',
}

res = xgb.cv(xgb_params, dtrain, num_boost_round=2000, nfold=4, seed=SEED, stratified=False,
             early_stopping_rounds=25, verbose_eval=20, show_stdv=True)

best_nrounds = res.shape[0] - 1
cv_mean = res.iloc[-1, 0]
cv_std = res.iloc[-1, 1]

print('Ensemble-CV: {0}+{1}'.format(cv_mean, cv_std))

gbdt = xgb.train(xgb_params, dtrain, best_nrounds)

submission = pd.read_csv(SUBMISSION_FILE)
submission.iloc[:, 1] = gbdt.predict(dtest)