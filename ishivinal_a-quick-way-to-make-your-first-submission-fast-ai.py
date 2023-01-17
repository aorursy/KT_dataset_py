 !pip install git+https://github.com/fastai/fastai@2e1ccb58121dc648751e2109fc0fbf6925aa8887 2>/dev/null 1>/dev/null
# !apt update && apt install -y libsm6 libxext6
from fastai.structured import rf_feat_importance
from fastai.structured import train_cats,proc_df
#RandomForest
import math 
import pandas as pd
import numpy as np 
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
#Xgboost
import xgboost as xgb
#CatBoost 
from catboost import CatBoostRegressor
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
Id = test['Id']
test_copy = test.copy()
test_copy["SalePrice"] = np.nan
train_set_data = [train,test_copy]
train_set_data = pd.concat(train_set_data)
len(train_set_data) == len(train)+len(test)
train_cats(train_set_data)
df, y, nas = proc_df(train_set_data, 'SalePrice',max_n_cat=10)
test_df = df[1460:2919]
df = df[0:1460]
y=y[0:1460]
m = RandomForestRegressor(n_jobs=-1,verbose=0)
m.fit(df, y)
fi = rf_feat_importance(m, df)
len(df.columns)
def plot_fi(fi):
    return fi.plot('cols', 'imp', 'barh', figsize=(12,7), legend=False)
plot_fi(fi[fi.imp>0.005])
df = df[fi[fi.imp>0.0005].cols]
def split_vals(a,n): return a[:n].copy(), a[n:].copy()

n_valid = 400  
n_trn = len(df)-n_valid
X_train, X_valid = split_vals(df, n_trn)
y_train, y_valid = split_vals(y, n_trn)

X_train.shape, y_train.shape, X_valid.shape
def rmse(x,y): return math.sqrt(((np.log(x)-np.log(y))**2).mean())

def print_score(m):
    res = [rmse(m.predict(X_train), y_train), rmse(m.predict(X_valid), y_valid),
                m.score(X_train, y_train), m.score(X_valid, y_valid)]
    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)
    print(res)
rf_param_grid = {
                 'max_depth' : [4, 6, 8,12],
                 'n_estimators': [5,10,20,60,100],
                 'max_features': ['sqrt', 'auto', 'log2'],
                 'min_samples_split': [2, 3, 10,20],
                 'min_samples_leaf': [1, 3, 10,18,25],
                 'bootstrap': [True, False],
                 }
m = RandomForestRegressor()
m_r = RandomizedSearchCV(param_distributions=rf_param_grid, 
                                    estimator = m,  
                                    verbose = 0, n_iter = 50, cv = 4)
m_r.fit(X_train, y_train)
print_score(m_r)
xgb_classifier = xgb.XGBRegressor()

gbm_param_grid = {
    'n_estimators': range(1,100),
    'max_depth': range(1, 15),
    'learning_rate': [.1,.13, .16, .19,.3,.6],
    'colsample_bytree': [.6, .7, .8, .9, 1]
}
xgb_random = RandomizedSearchCV(param_distributions=gbm_param_grid, 
                                    estimator = xgb_classifier, 
                                    verbose = 0, n_iter = 50, cv = 4)
xgb_random.fit(X_train,y_train)
print_score(xgb_random)
m_c = CatBoostRegressor(iterations=2000,learning_rate=0.1,depth=3,loss_function='RMSE',l2_leaf_reg=4,border_count=15,verbose=False)
m_c.fit(X_train,y_train)
print_score(m_c)
test_df = test_df[list(X_train.columns)]
y_pred = (m_c.predict(test_df) + m_r.predict(test_df)+xgb_random.predict(test_df))/3
submission = pd.DataFrame({"Id": Id,"SalePrice": y_pred})
submission.to_csv('submission.csv', index=False)