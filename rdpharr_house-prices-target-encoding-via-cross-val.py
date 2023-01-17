%load_ext autoreload
%autoreload 2
%matplotlib inline

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore",category=DeprecationWarning)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import re
import seaborn as sns
train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
display(train_df.tail()) 
display(test_df.tail())
#list all columns
train_df.columns
#just the numeric columns
train_df.select_dtypes(include=[np.number]).columns
#missing data 
total = train_df.isnull().sum().sort_values(ascending=False)
percent = (train_df.isnull().sum()/train_df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data[missing_data.Total>0]
#outlier check
plt.figure(figsize=(15,5))
ax = sns.scatterplot(x=train_df.GrLivArea, y=train_df.SalePrice, 
                     hue=train_df.Neighborhood, legend=False).set_title("SalePrice vs. SqFt by Neighborhood")
#remove the 2 wierdos
train_df = train_df[train_df.GrLivArea<4500]
plt.figure(figsize=(15,5))
ax = sns.scatterplot(x=train_df.GrLivArea, y=train_df.SalePrice, 
                     hue=train_df.Neighborhood, legend=False).set_title("SalePrice vs. SqFt by Neighborhood")
# date sold to category
train_df['sale_date'] = train_df.MoSold.astype('str') + '_' + train_df.YrSold.astype('str')
test_df['sale_date'] = test_df.MoSold.astype('str') + '_' + test_df.YrSold.astype('str')

#combine baths
train_df['baths'] = train_df.FullBath + train_df.HalfBath*0.5
test_df['baths'] = test_df.FullBath + test_df.HalfBath*0.5
train_df['bsmt_baths'] = train_df.BsmtFullBath + train_df.BsmtHalfBath*0.5
test_df['bsmt_baths'] = test_df.BsmtFullBath + test_df.BsmtHalfBath*0.5

#drop columns
drop_me = ['MoSold','YrSold','Id','FullBath','HalfBath','BsmtFullBath','BsmtHalfBath']
train_df.drop(columns=drop_me, inplace=True)
test_df.drop(columns=drop_me, inplace=True)

#category columns from numeric
new_cats = ['MSSubClass','OverallQual','OverallCond', 'YearBuilt', 'YearRemodAdd',
           'GarageYrBlt', 'GarageCars','EnclosedPorch', '3SsnPorch', 'ScreenPorch',
           'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd','Fireplaces','baths', 'bsmt_baths']
train_df[new_cats] = train_df[new_cats].astype('object')
test_df[new_cats] = test_df[new_cats].astype('object')
train_df.select_dtypes(include=[np.number]).columns
from sklearn.model_selection import train_test_split

X = train_df.drop(columns='SalePrice')
Y = np.log(train_df['SalePrice'])  #convert Y to log(y)
X_train, X_valid, y_train, y_valid = train_test_split(X, Y, test_size=0.2)
X_test = test_df.copy()
from sklearn import model_selection
def cv_mean_enc(col):
    #X_train mean encodings calculated with Kfold to ensure no data leakage
    kf = model_selection.KFold(5, shuffle=False)
    X_train['target']=y_train
    X_train[col+'_target_enc'] = np.nan
    for tr_ind, val_ind in kf.split(X_train):
        X_tr, X_val = X_train.iloc[tr_ind], X_train.iloc[val_ind]
        X_train.loc[X_train.index[val_ind], col+'_target_enc'] = X_val[col].map(X_tr.groupby(col)['target'].mean())
    X_train[col+'_target_enc'].fillna(y_train.mean(), inplace=True)
    
    #X_valid from calculations on X_train
    encodings = X_train.groupby(col)[col+'_target_enc'].mean()
    X_valid[col+'_target_enc'] = X_valid[col].map(encodings)
    X_valid[col+'_target_enc'].fillna(y_train.mean(), inplace=True)
    
    #X_test get calculated from entire data set
    X_valid['target']=y_valid
    all_data = X_train.append(X_valid)
    encodings = all_data.groupby(col)['target'].mean()
    X_test[col+'_target_enc'] = X_test[col].map(encodings)
    X_test[col+'_target_enc'].fillna(all_data['target'].mean(), inplace=True)
    
    #cleanup
    X_train.drop(columns=['target',col],inplace=True)
    X_valid.drop(columns=['target',col],inplace=True)
    X_test.drop(columns=[col],inplace=True)
cols = X_train.select_dtypes(include='object').columns
for c in cols: cv_mean_enc(c)
total = X_train.isnull().sum().sort_values(ascending=False)
percent = (X_train.isnull().sum()/X_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data[missing_data.Total>0]
X_train.fillna(0, inplace=True)
X_valid.fillna(0, inplace=True)
X_test.fillna(0, inplace=True)
dataset = (X_train,y_train,X_valid,y_valid,X_test)
pickle.dump(dataset,open("dataset.pk", 'wb'))
X_train,y_train,X_valid,y_valid,X_test = pickle.load(open("dataset.pk", 'rb'))
import xgboost as xgb

from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
#this is where we save our predictions for the ensemble later
valid_preds = pd.DataFrame()
test_preds = pd.DataFrame()
params = {'max_depth':3,
        'n_estimators':100, 
        'learning_rate':0.1,
        'min_child_weight' : 1,
        'gamma' : 0,
        'subsample' : 1,
        'colsample_bytree':1,
        'reg_alpha':0}
def tune_param(target_param, param_range, params):
    best_score = 100
    best_iteration = 0
    new_value = params[target_param]
    
    for a in param_range:
        params[target_param]=a
        gbm = xgb.XGBRegressor(**params,random_state=13)
        model_xgb = gbm.fit(X_train, y_train,
            eval_metric="rmse",
            eval_set=[(X_train, y_train), (X_valid, y_valid)],
            verbose=False,
            early_stopping_rounds = 3)
        if model_xgb.best_score < best_score:
            new_value = a
            best_score = model_xgb.best_score
            best_iteration = model_xgb.best_iteration
    
    #return best results as parameter dict
    print(f'After tuning {target_param} to {new_value}, model score is {best_score:.3f} after {best_iteration + 1} iterations.')
    params[target_param]=new_value
    return params
params = tune_param('max_depth',range(3,15), params)
params = tune_param('min_child_weight',range(1,11), params)
params = tune_param('gamma',[0,.1,.2,.3,.4,.5], params)
params = tune_param('subsample',[.6,.7,.8,.9,1], params)
params = tune_param('colsample_bytree',[.6,.7,.8,.9,1], params)
params = tune_param('reg_alpha',[0,.1,.5,1,5,10,50,100,500,1000,5000], params)
params = tune_param('learning_rate',[.1,.2,.3,.4,.5,.6,.7,.8], params)
params
gbm = xgb.XGBRegressor(**params,random_state=13)
model_xgb = gbm.fit(X_train, y_train,
    eval_metric="rmse",
    eval_set=[(X_train, y_train), (X_valid, y_valid)],
    verbose=False,
    early_stopping_rounds = 3)
valid_preds['xgb'] = model_xgb.predict(X_valid)
test_preds['xgb'] = model_xgb.predict(X_test)
from sklearn.preprocessing import StandardScaler
s = StandardScaler()
X_train_s = s.fit_transform(X_train)
X_valid_s = s.fit_transform(X_valid)
X_test_s = s.fit_transform(X_test)
def knn_hyperopt(n):
    model_knn = KNeighborsRegressor(n_neighbors=n)
    model_knn.fit(X_train_s,y_train)
    preds = model_knn.predict(X_valid_s)
    rmse = np.sqrt(mean_squared_error(preds,y_valid))
    print(f"for {n} neighbors, rmse is {rmse:.5f}")
    return rmse
n_neigh = [2,3,5,7,9,11]
results = [] 
for n in n_neigh:
    results.append(knn_hyperopt(n))
opt_neigh = n_neigh[np.argmin(results)]
model_knn = KNeighborsRegressor(n_neighbors=opt_neigh)
model_knn.fit(X_train_s,y_train)
valid_preds['knn'] = model_knn.predict(X_valid_s)
test_preds['knn'] = model_knn.predict(X_test_s)
def lasso_hyperopt(alpha):
    reg = Lasso(alpha=alpha)
    reg.fit(X_train,y_train)
    preds = reg.predict(X_valid)
    rmse = np.sqrt(mean_squared_error(preds,y_valid))
    print(f"for alpha:{alpha}, rmse is {rmse:.5f}")
    return rmse
alpha_lasso = [1e-8, 1e-5, 1e-4, 1e-3,1e-2, 0.5, 0.1, 1, 5, 10]
results = []
for a in alpha_lasso:
    results.append(lasso_hyperopt(a))
opt_alpha = alpha_lasso[np.argmin(results)]
reg = Lasso(alpha=opt_alpha)
reg.fit(X_train,y_train)
valid_preds['lasso'] = reg.predict(X_valid)
test_preds['lasso'] = reg.predict(X_test)
def svr_hyperopt(c):
    model_svr = SVR(C=c, gamma='scale')
    model_svr.fit(X_train_s,y_train)
    preds = model_svr.predict(X_valid_s)
    rmse = np.sqrt(mean_squared_error(preds,y_valid))
    print(f"for C:{c}, rmse is {rmse:.5f}")
    return rmse
c_svr = [1e-4, 1e-3,1e-2, 0.5, 0.1, 1, 5, 10, 50, 1e2, 5e2, 1e3, 5e3, 1e4]
#c_svr = [74 + i/10 for i in range(0,20,2)]
results = []
for a in c_svr:
    results.append(svr_hyperopt(a))
opt_c = c_svr[np.argmin(results)]
model_svr = SVR(C=opt_c, gamma='scale')
model_svr.fit(X_train_s,y_train)
valid_preds['lasso'] = model_svr.predict(X_valid)
test_preds['lasso'] = model_svr.predict(X_test)
def stack_hyperopt(alpha):
    reg = Lasso(alpha=alpha)
    reg.fit(valid_preds,y_valid)
    preds = reg.predict(valid_preds)
    rmse = np.sqrt(mean_squared_error(preds,y_valid))
    print(f"for alpha:{alpha}, rmse is {rmse:.5f}")
    return rmse
alpha_lasso = [1e-8, 1e-5, 1e-4, 1e-3,1e-2, 0.5, 0.1, 1, 5, 10]
results = []
for a in alpha_lasso:
    results.append(stack_hyperopt(a))
opt_alpha = alpha_lasso[np.argmin(results)]
reg = Lasso(alpha=opt_alpha)
reg.fit(valid_preds,y_valid)
preds = reg.predict(test_preds)
sub_df = pd.DataFrame()
sub_df['Id']=pd.read_csv("../input/test.csv")['Id']
sub_df['SalePrice']=np.exp(preds)
sub_df.to_csv('submission.csv', index=False)
