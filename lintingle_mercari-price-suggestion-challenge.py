#-----------------載入套件-------------------------
import pandas as pd
import numpy as np
import scipy

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer

from sklearn.linear_model import Ridge
import xgboost as xgb
import lightgbm as lgb

from sklearn.model_selection import KFold

from sklearn.metrics import mean_squared_error

from scipy.optimize import minimize

import matplotlib.pyplot as plt

import gc

#-------------------資料準備-----------------------
print("Reading in Data")
df_train = pd.read_csv('../input/train.tsv', sep='\t')
df_test = pd.read_csv('../input/test_stg2.tsv', sep='\t')

nrow_train=df_train.shape[0]
nrow_test=df_test.shape[0]
Y_train=np.log1p(df_train["price"])

print('Size of train set',nrow_train)
print('Size of test set',nrow_test)

df=pd.concat([df_train,df_test],axis=0,sort=True)
df_test=pd.DataFrame(df_test["test_id"])


#----------------處理缺失值--------------------
NUM_BRAND=2500
df["category_name"]=df["category_name"].fillna("Other").astype("category")

df["brand_name"]=df["brand_name"].fillna("Unknown")
pop_brands=df["brand_name"].value_counts().index[1:NUM_BRAND+1]
df.loc[~df["brand_name"].isin(pop_brands),"brand_name"]="Other"
df["brand_name"]=df["brand_name"].astype("category")

df["item_description"]=df["item_description"].fillna("None")

df["item_condition_id"]=df["item_condition_id"].astype("category")
df_train.head()
del df_train
gc.collect
#--------------------encode------------------------
NAME_MIN_DF=10
MAX_FEAT_DESCP=50000
print("Encodings")

print("Condition Encoders")
vect_condition=LabelBinarizer(sparse_output=True)
X_Condition=vect_condition.fit_transform(df["item_condition_id"])

print("Shipping Encoders")
vect_shipping=LabelBinarizer(sparse_output=True)
X_Shipping=vect_shipping.fit_transform(df["shipping"])

print("Name Encoders")
count_name=CountVectorizer(min_df=NAME_MIN_DF)
X_name=count_name.fit_transform(df["name"])

print("Category Encoders")
count_category=CountVectorizer()
X_category=count_category.fit_transform(df["category_name"])

print("Brand Encoders")
count_brand=CountVectorizer()
X_brand=count_brand.fit_transform(df["brand_name"])


print("Descp Encoders")
count_descp=TfidfVectorizer(max_features=MAX_FEAT_DESCP,
                            ngram_range=(1,3),
                            stop_words="english")
X_descp=count_descp.fit_transform(df["item_description"])


del df
gc.collect

X=scipy.sparse.hstack([X_Shipping,X_Condition,X_brand,
                       X_category,X_descp,X_name]).tocsr()

del X_descp
del X_brand
del X_category
del X_name
del X_Shipping
del X_Condition
gc.collect

X_train=X[:nrow_train]
X_test=X[nrow_train:]
dtest=xgb.DMatrix(X_test)
del X
gc.collect
#--------------------Cross-Validation------------------------
print("Cross-Validation")
xgb_pred_val_index = np.zeros(X_train.shape[0])
ridge_pred_val_index = np.zeros(X_train.shape[0])
lgb_pred_val_index = np.zeros(X_train.shape[0])
xgb_pred_all_sum=[]
ridge_pred_all_sum=[]
lgb_pred_all_sum=[]
xgb_cv_RMSLE_sum=0
ridge_cv_RMSLE_sum=0
lgb_cv_RMSLE_sum=0


folds=3
kf = KFold(n_splits=folds, random_state=1001)
for i, (train_index, val_index) in enumerate(kf.split(X_train, Y_train)):
    x_train, x_val = X_train[train_index], X_train[val_index]
    y_train, y_val = Y_train[train_index], Y_train[val_index]

    dtrain=xgb.DMatrix(x_train,y_train)
    dval=xgb.DMatrix(x_val)
    deval=xgb.DMatrix(x_val,y_val)
    
    
    params = {
    'booster': 'gblinear',
    'objective': 'reg:linear', 
    'gamma': 0,                
    'max_depth': 10,           
    'lambda': 0,                   
    'subsample': 0.85,             
    'colsample_bytree': 0.9,      
    'min_child_weight': 17,
    'silent': 1,                  
    'eta': 0.4,                 
    'seed': 1001,
    'nthread': 4,                 
    'eval_metric':'rmse'
    }
    
    plst = params.items()
    evallist = [(deval, 'eval'), (dtrain, 'train')]
    num_round=300
    
    model=xgb.train(plst,dtrain,num_round,evallist, verbose_eval=100,early_stopping_rounds=100)
    xgb_pred_val=model.predict(dval)
    xgb_RMSLE=np.sqrt(mean_squared_error(xgb_pred_val,y_val))
    print('\n Fold %02d XGBoost RMSLE: %.6f' % ((i + 1), xgb_RMSLE))
    xgb_pred_all=model.predict(dtest)
    
    del dtrain
    del dval
    del deval
    gc.collect()
    
    params = {
        'boosting': 'gbdt',
        'max_depth': 7,
        'min_data_in_leaf': 80,
        'num_leaves': 30,
        'learning_rate': 0.4,
        'objective': 'regression',
        'metric': 'rmse',
        'nthread': 4,
        'bagging_freq': 1,
        'subsample': 0.9,
        'colsample_bytree': 0.7,
        'min_child_weight': 17,
        'is_unbalance': False,
        'verbose': -1,
        'seed': 1001,
        'max_bin':511,
        'num_threads':4
    }
    
    dtrain = lgb.Dataset(x_train, label=y_train)
    deval = lgb.Dataset(x_val, label=y_val)
    watchlist = [dtrain, deval]
    watchlist_names = ['train', 'val']

    model = lgb.train(params,
    train_set=dtrain,
    num_boost_round=3000,
    valid_sets=watchlist,
    valid_names=watchlist_names,
    early_stopping_rounds=100,
    verbose_eval=300)
    lgb_pred_val = model.predict(x_val)
    lgb_RMSLE = np.sqrt(mean_squared_error(lgb_pred_val,y_val))
    print(' Fold %02d LightGBM RMSLE: %.6f' % ((i + 1), lgb_RMSLE))
    lgb_pred_all = model.predict(X_test)
    
    del dtrain
    del deval
    gc.collect()
    
    
    
    model=Ridge(solver='sag',alpha=4.75)
    model.fit(x_train,y_train)
    ridge_pred_val=model.predict(x_val)
    ridge_RMSLE=np.sqrt(mean_squared_error(ridge_pred_val,y_val))
    print('\n Fold %02d Ridge RMSLE: %.6f' % ((i + 1), ridge_RMSLE))
    ridge_pred_all=model.predict(X_test)
    
    del x_train
    del y_train
    del x_val
    del y_val
    gc.collect()
    
    xgb_pred_val_index[val_index] = xgb_pred_val
    ridge_pred_val_index[val_index] = ridge_pred_val
    lgb_pred_val_index[val_index] = lgb_pred_val
    
    if i > 0:
        xgb_pred_all_sum = xgb_pred_all_sum + xgb_pred_all
        ridge_pred_all_sum = ridge_pred_all_sum + ridge_pred_all
        lgb_pred_all_sum = lgb_pred_all_sum + lgb_pred_all
    else:
        xgb_pred_all_sum = xgb_pred_all
        ridge_pred_all_sum = ridge_pred_all
        lgb_pred_all_sum = lgb_pred_all
    
    xgb_cv_RMSLE_sum = xgb_cv_RMSLE_sum + xgb_RMSLE
    ridge_cv_RMSLE_sum = ridge_cv_RMSLE_sum + ridge_RMSLE
    lgb_cv_RMSLE_sum = lgb_cv_RMSLE_sum + lgb_RMSLE

xgb_cv_avg_score=xgb_cv_RMSLE_sum/folds
ridge_cv_avg_score=ridge_cv_RMSLE_sum/folds
lgb_cv_avg_score=lgb_cv_RMSLE_sum/folds

xgb_val_real_RMSLE=np.sqrt(mean_squared_error(xgb_pred_val_index,Y_train))
ridge_val_real_RMSLE=np.sqrt(mean_squared_error(ridge_pred_val_index,Y_train))
lgb_val_real_RMSLE=np.sqrt(mean_squared_error(lgb_pred_val_index,Y_train))

print('\n Average XGBoost RMSLE(cv):\t%.6f' % xgb_cv_avg_score)
print(' Out-of-fold XGBoost RMSLE:\t%.6f' % xgb_val_real_RMSLE)
print('\n Average LightGBM RMSLE(cv):\t%.6f' % lgb_cv_avg_score)
print(' Out-of-fold LightGBM RMSLE:\t%.6f' % lgb_val_real_RMSLE)
print('\n Average Ridge RMSLE(cv):\t%.6f' % ridge_cv_avg_score)
print(' Out-of-fold Ridge RMSLE:\t%.6f' % ridge_val_real_RMSLE)

xgb_pred_all_avg=xgb_pred_all_sum/folds
ridge_pred_all_avg=ridge_pred_all_sum/folds
lgb_pred_all_avg=lgb_pred_all_sum/folds

#------------blend-------------------
def rmse_min_func(weights):
    final_prediction=0
    for weight,prediction in zip(weights,blend_train):
        final_prediction+=weight*prediction
    return np.sqrt(mean_squared_error(Y_train,final_prediction))

blend_train = []
blend_test = []

blend_train.append(xgb_pred_val_index) 
blend_train.append(lgb_pred_val_index)
blend_train.append(ridge_pred_val_index)
blend_train=np.array(blend_train)

blend_test.append(xgb_pred_all_avg)
blend_test.append(lgb_pred_all_avg)
blend_test.append(ridge_pred_all_avg)
blend_test=np.array(blend_test)

print('\n Finding Blending Weights ...')

res_list=[]
weight_list=[]

for k in range(20):
    starting_value=np.random.uniform(-1,1,len(blend_train))
    bounds=[(-1,1)]*len(blend_train)
    
    res=minimize(
        rmse_min_func,
        starting_value,
        method='L-BFGS-B',
        bounds=bounds,
        options={'disp':False,
        'maxiter':100000})
    
    res_list.append(res['fun'])
    weight_list.append(res['x'])
    print('{iter}\tScore: {score}\tWeights: {weights}'.format(
        iter=(k+1),
        score=round(res['fun'],6),
        weights='\t'.join([str(round(item,10)) for item in res['x']])))

bestSC=np.min(res_list)
bestweight=weight_list[np.argmin(res_list)]

print('\n Ensemble Score:{best_score}'.format(best_score=bestSC))
print('\n Best Weights:{weight}'.format(weight=bestweight))


test_price=np.zeros(len(blend_test[0]))
train_price =np.zeros(len(blend_train[0]))

print('\n Your final model:')
for k in range(len(blend_test)):
    print('%.6f * model-%d'%(bestweight[k],(k+1)))
    test_price+=blend_test[k]*bestweight[k]
    train_price+= blend_train[k] * bestweight[k]

df_test["price"]=np.expm1(test_price)
print("Generatig File")
df_test[["test_id","price"]].to_csv("submission.csv",index=False)
#------------------模型檢視---------------------------
print('\n Making scatter plots of actual vs. predicted prices ...')
x_true = np.expm1(Y_train)
x_pred = np.expm1(xgb_pred_val_index)
cm = plt.cm.get_cmap('RdYlBu')
# Normalized prediction error clipped so the color-coding covers -75% to 75% range
x_diff = np.clip(100 * ((x_pred - x_true) / x_true), -75, 75)
plt.figure(1, figsize=(12, 10))
plt.title('Actual vs. Predicted Prices - XGBoost')
plt.scatter(x_true, x_pred, c=x_diff, s=10, cmap=cm)
plt.colorbar()
plt.plot([x_true.min() - 50, x_true.max() + 50],
         [x_true.min() - 50, x_true.max() + 50],
         'k--',lw=1)
plt.xlabel('Prices')
plt.ylabel('Predicted Prices')
plt.xlim(-50, 2050)
plt.ylim(-50, 2050)
plt.tight_layout()
plt.show()


x_pred = np.expm1(lgb_pred_val_index)
# Normalized prediction error clipped so the color-coding covers -75% to 75% range
x_diff = np.clip(100 * ((x_pred - x_true) / x_true), -75, 75)
plt.figure(1, figsize=(12, 10))
plt.title('Actual vs. Predicted Prices - LightGBM')
plt.scatter(x_true, x_pred, c=x_diff, s=10, cmap=cm)
plt.colorbar()
plt.plot([x_true.min() - 50, x_true.max() + 50],
         [x_true.min() - 50, x_true.max() + 50],
         'k--',lw=1)
plt.xlabel('Prices')
plt.ylabel('Predicted Prices')
plt.xlim(-50, 2050)
plt.ylim(-50, 2050)
plt.tight_layout()
plt.show()


x_pred = np.expm1(ridge_pred_val_index)
# Normalized prediction error clipped so the color-coding covers -75% to 75% range
x_diff = np.clip(100 * ((x_pred - x_true) / x_true), -75, 75)
plt.figure(1, figsize=(12, 10))
plt.title('Actual vs. Predicted Prices - Ridge Regression')
plt.scatter(x_true, x_pred, c=x_diff, s=10, cmap=cm)
plt.colorbar()
plt.plot([x_true.min() - 50, x_true.max() + 50],
         [x_true.min() - 50, x_true.max() + 50],
         'k--',lw=1)
plt.xlabel('Prices')
plt.ylabel('Predicted Prices')
plt.xlim(-50, 2050)
plt.ylim(-50, 2050)
plt.tight_layout()
plt.show()


x_pred = np.expm1(train_price)
# Normalized prediction error clipped so the color-coding covers -75% to 75% range
x_diff = np.clip(100 * ((x_pred - x_true) / x_true), -75, 75)
plt.figure(4, figsize=(12, 10))
plt.title('Actual vs. Predicted Prices - Blend')
plt.scatter(x_true, x_pred, c=x_diff, s=10, cmap=cm)
plt.colorbar()
plt.plot([x_true.min() - 50, x_true.max() + 50],
         [x_true.min() - 50, x_true.max() + 50],
         'k--',lw=1)
plt.xlabel('Prices')
plt.ylabel('Predicted Prices')
plt.xlim(-50, 2050)
plt.ylim(-50, 2050)
plt.tight_layout()
plt.show()