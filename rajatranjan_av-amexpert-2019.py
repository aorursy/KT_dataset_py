# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train=pd.read_csv('/kaggle/input/amexpert/train.csv')

s=pd.read_csv('/kaggle/input/amexpert/sample_submission_Byiv0dS.csv')

coup_item=pd.read_csv('/kaggle/input/amexpert/coupon_item_mapping.csv')

test=pd.read_csv('/kaggle/input/amexpert/test_QyjYwdj.csv')

comp=pd.read_csv('/kaggle/input/amexpert/campaign_data.csv')

tran=pd.read_csv('/kaggle/input/amexpert/customer_transaction_data.csv')

demo=pd.read_csv('/kaggle/input/amexpert/customer_demographics.csv')

item=pd.read_csv('/kaggle/input/amexpert/item_data.csv')

print(train.shape,test.shape,coup_item.shape,comp.shape,tran.shape,demo.shape,item.shape)
train.head()
print(train.shape)

train.redemption_status.value_counts()
print(comp.shape)

comp.head()
df=train.append(test,ignore_index=True)

df.head()
comp['start_date']=pd.to_datetime(comp['start_date'],format='%d/%m/%y',dayfirst=True)

comp['end_date']=pd.to_datetime(comp['end_date'],format='%d/%m/%y',dayfirst=True)



# comp['start_date_d']=comp['start_date'].dt.day.astype('category')

# comp['start_date_m']=comp['start_date'].dt.month.astype('category')

# comp['start_date_y']=comp['start_date'].dt.year.astype('category')

# comp['start_date_w']=comp['start_date'].dt.week.astype('category')





# comp['end_date_d']=comp['end_date'].dt.day.astype('category')

# comp['end_date_m']=comp['end_date'].dt.month.astype('category')

# comp['end_date_y']=comp['end_date'].dt.year.astype('category')

# comp['end_date_w']=comp['end_date'].dt.week.astype('category')





comp['diff_d']=(comp['end_date']-comp['start_date'])/np.timedelta64(1,'D')

comp['diff_m']=(comp['end_date']-comp['start_date'])/np.timedelta64(1,'M')

comp['diff_w']=(comp['end_date']-comp['start_date'])/np.timedelta64(1,'W')



# comp.drop(['start_date','end_date'],axis=1,inplace=True)
comp.describe(include='all').T
comp.head()
df=df.merge(comp,on='campaign_id',how='left')

df.head()
for j in ['brand', 'brand_type', 'category']:

    print(j,item[j].nunique())


for j in ['brand', 'brand_type', 'category']:

    item[j]=item[j].astype('category')

    

coup_item=coup_item.merge(item,on='item_id',how='left')

coup_item.coupon_id.nunique()
coup_item.head(),coup_item.shape
tran=pd.read_csv('/kaggle/input/amexpert/customer_transaction_data.csv')

tran['date']=pd.to_datetime(tran['date'],format='%Y-%m-%d')

tran['date_d']=tran['date'].dt.day.astype('category')

tran['date_m']=tran['date'].dt.month.astype('category')

tran['date_w']=tran['date'].dt.week.astype('category')



# tran.drop('date',axis=1,inplace=True)

tran.head()
tran[tran['quantity']==20]
tran['discount_bin']=tran['coupon_discount'].apply(lambda x: 0 if x>=0 else 1)

tran['marked_price']=tran['selling_price']-tran['other_discount']-tran['coupon_discount']

tran['disc_percent']=(tran['marked_price']-tran['selling_price'])/tran['selling_price']

tran['price_per_quan']=tran['marked_price']/tran['quantity']

tran['marked_by_sale']=tran['marked_price']/tran['selling_price']

tran.columns
tran=tran.merge(coup_item,on='item_id',how='left')

tran.head()
print(tran.shape)

tran=tran[tran.duplicated()==False]

print(tran.shape,train.shape)

# --drop it
tran.head()
tran=tran.merge(tran.groupby(['customer_id','date']).agg({'coupon_id':'count','item_id':'count','disc_percent':sum}).reset_index()

                .rename(columns={'coupon_id':'coupon_aquired','item_id':'item_bought','disc_percent':'tot_disc'}),on=['customer_id','date'],how='left')
tran[(tran['customer_id']==1052) & (tran['coupon_id']==21)]
tran['coupon_to_item']=tran['item_bought']-tran['coupon_aquired']
tran[(tran['customer_id']==413) & (tran['coupon_id']==577)]
df[(df['customer_id']==413) & (df['coupon_id']==577)]
# tran.groupby(['customer_id','coupon_id']).agg({'date':set}).reset_index()

tran.head()
def func(a,b,c):

    if c!=0:

        c=list(c)

        v=0

        for k in c:

            if a<=k and b>k:

                v+=1

        return v

    else:

        return 0

# cc['within']=cc.apply(lambda x: func(x['start_date'],x['end_date'],x['date']),axis=1)
# Magic features

# tran.groupby(['customer_id','date']).agg({'coupon_id':'count','discount_bin':sum,'quantity':sum,'item_id':'count'}).reset_index()
df.head()
# cc=df.merge(tran.groupby(['customer_id','date']).agg({'coupon_id':'count','discount_bin':sum,'quantity':sum,'item_id':'count'}).reset_index(),on=['customer_id','date'],how='left')

# cc.sample(10)

tran.columns
ddf=df.merge(tran.groupby(['customer_id','coupon_id']).agg({'date':set,'discount_bin':sum,'quantity':sum,'item_id':'count',

                                                            'coupon_aquired':sum,'item_bought':'mean','tot_disc':sum}).reset_index(),on=['customer_id','coupon_id'],how='left')

ddf.sample(10)
ddf['coupon_aquired'].fillna(0)
# def new_df(df):



print(ddf.shape)

ddf['date'].replace(np.nan,0,inplace=True)

ddf['discount_bin'].replace(np.nan,-1,inplace=True)

# ddf['quantity'].replace(np.nan,0,inplace=True)

# ddf['item_id'].replace(np.nan,0,inplace=True)

# df['camp_date_within_count']=ddf.apply(lambda x: func(x['start_date'],x['end_date'],x['date']),axis=1)





# df['bin']=ddf['discount_bin'].apply(lambda x: 1 if x!=-1 else 0)

df['within_date']=ddf['date'].apply(lambda x: len(x) if x !=0 else 0)

# df['C1']=ddf['coupon_aquired'].fillna(0)

# df['C2']=ddf['item_bought'].fillna(0)

# df['C3']=ddf['tot_disc'].fillna(0)









# df['within_date_discount']=ddf['discount_bin'].apply(lambda x: x if x >=0 else 0)

#     df['quantity_date']=ddf['quantity']

# df['item_count']=ddf['item_id']



# -- worked good

    # df['quantity_date']=ddf['quantity']

    # df['item_count']=ddf['item_id']

ddf.head()


# %time

# within_date=[]

# from tqdm import tqdm_notebook as tqdm

# for i in tqdm(range(df.shape[0])):

#     st_dt=df.loc[i,'start_date']

#     en_dt=df.loc[i,'end_date']

#     cust=df.loc[i,'customer_id']

#     coup=df.loc[i,'coupon_id']

# #     temp=tran[(tran['date']>=st_dt) & (tran['date']<en_dt) & (tran['customer_id']==cust) & (tran['coupon_id']==coup)]

# #     temp=temp[temp.duplicated()==False]

#     if tran[(tran['date']>=st_dt) & (tran['date']<en_dt) & (tran['customer_id']==cust) & (tran['coupon_id']==coup)].shape[0]>0:

#         within_date.append(1)

#     else:

#         within_date.append(0)

# #     print(temp.shape,df.loc[i,'redemption_status'])



# df['within_date']=within_date

tran.columns
c=['count','nunique']

n=['mean','max','min','sum','std']

nn=['mean','max','min','sum','std','quantile']

# agg_c={'date_d':c,'date_m':c,'date_w':c,'quantity':n,'selling_price':n,'other_discount':n,'coupon_discount':n,'item_id':c,'brand':c,

#        'category':c,'coupon_id':c,'discount_bin':nn,'marked_price':n,'disc_percent':n,'price_per_quan':n,'brand_type':c,'marked_by_sale':n,

#        'coupon_aquired':nn, 'item_bought':nn, 'tot_disc':n, 'coupon_to_item':nn}





agg_c={'date_d':c,'date_m':c,'date_w':c,'quantity':n,'selling_price':n,'other_discount':n,'coupon_discount':n,'item_id':c,'brand':c,

       'category':c,'coupon_id':c,'discount_bin':nn,'marked_price':n,'disc_percent':n,'price_per_quan':n,'brand_type':c,'marked_by_sale':n,

       'coupon_aquired':nn, 'item_bought':nn, 'tot_disc':n, 'coupon_to_item':nn}

trans=tran.groupby(['customer_id']).agg(agg_c)

trans.head()
trans.columns=['F_' + '_'.join(col).strip() for col in trans.columns.values]

trans.reset_index(inplace=True)

trans.head()
trans.shape
df=df.merge(trans,on=['customer_id'],how='left')





# -------to uncomment



# df.head()
df['campaign_type']=df['campaign_type'].astype('category')
# df['campaign_id']=df['campaign_id'].astype('category')

# df['coupon_id']=df['coupon_id'].astype('category')

# df['customer_id']=df['customer_id'].astype('category')

# df['campaign_type']=df['campaign_type'].astype('category')



# df['within_date_discount'].value_counts()
df.info()
df_train=df[df['redemption_status'].isnull()==False].copy()

df_test=df[df['redemption_status'].isnull()==True].copy()



print(df_train.shape,df_test.shape)
df_train.merge(df_train.drop(['id','redemption_status'],axis=1).groupby('campaign_id').mean().reset_index(),on='campaign_id',how='left')
df_train=df_train.merge(df_train.drop(['id','redemption_status'],axis=1).groupby('coupon_id').mean().reset_index(),on='coupon_id',how='left')

df_test=df_test.merge(df_test.drop(['id','redemption_status'],axis=1).groupby('coupon_id').mean().reset_index(),on='coupon_id',how='left')



# df_train=df_train.merge(df_train.drop(['id','redemption_status'],axis=1).groupby('coupon_id_x').mean().reset_index(),on='coupon_id_x',how='left')

# df_test=df_test.merge(df_test.drop(['id','redemption_status'],axis=1).groupby('coupon_id_x').mean().reset_index(),on='coupon_id_x',how='left')







# df_train=new_df(df_train)

# print(df_train.shape)



# df_train.head()
df_train[df_train.redemption_status==1]
import seaborn as sns

%matplotlib inline

import matplotlib.pyplot as plt

plt.figure(figsize=(15,15))



# sns.heatmap(df_train.corr())
from catboost import CatBoostClassifier,Pool, cv

from lightgbm import LGBMClassifier

from sklearn.model_selection import StratifiedKFold,train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score,confusion_matrix,roc_auc_score
df_train.columns
# X,y=df_train.drop(['id','redemption_status'],axis=1),df_train['redemption_status']

# Xtest=df_test.drop(['id','redemption_status'],axis=1)

# col_to_drop=['id','redemption_status','start_date','end_date','F_quantity_min','F_other_discount_max','F_coupon_discount_max','F_discount_bin_min',

#              'F_disc_percent_min','F_brand_type_nunique','F_marked_by_sale_min','customer_id','campaign_id','coupon_id']



col_to_drop=['id','redemption_status','start_date','end_date']



X,y=df_train.drop(col_to_drop,axis=1),df_train['redemption_status']

Xtest=df_test.drop(col_to_drop,axis=1)



# X,y=df_train.drop(['id','redemption_status','start_date','end_date','customer_id','coupon_id','campaign_id'],axis=1),df_train['redemption_status']

# Xtest=df_test.drop(['id','redemption_status','start_date','end_date','customer_id','coupon_id','campaign_id'],axis=1)



# X=pd.get_dummies(X,drop_first=True)



# from sklearn.ensemble import IsolationForest

# clf = IsolationForest(contamination = 'auto', random_state=1994,behaviour="new",bootstrap=True)

# clf.fit(X)

# df_train['iso_out']=clf.predict(X)

# print(df_train['iso_out'].value_counts())











# print(df_train[df_train['iso_out']==1].shape)

# print(df_train[df_train['iso_out']==-1].shape)
# X['iso_out'].value_counts()

# X=X[X.iso_out==1].copy()

print(X.shape,Xtest.shape)

X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.3,random_state = 1994,stratify=y)
X_train.columns
col=['campaign_id', 'coupon_id', 'customer_id', 'campaign_type','within_date', 'within_date_discount']
# # for j in col:

# X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.3,random_state = 1994,stratify=y)

# print('Dropped->',j)

# X_train.drop(col,inplace=True,axis=1)

# X_val.drop(col,inplace=True,axis=1)

# m=LGBMClassifier(n_estimators=1500,random_state=1994,learning_rate=0.03,reg_alpha=0.2,colsample_bytree=0.5,bagging_fraction=0.9)

# # m=RidgeCV(cv=4)

# m.fit(X_train,y_train,eval_set=[(X_train,y_train),(X_val, y_val.values)],eval_metric='auc', early_stopping_rounds=100,verbose=200)

# p=m.predict_proba(X_val)[:,-1]



# print(roc_auc_score(y_val,p))

# print('---------------------')
from scipy.special import logit

m=LGBMClassifier(n_estimators=1500,random_state=1994,learning_rate=0.03,reg_alpha=0.2,colsample_bytree=0.5,reg_lambda=20)

# m=RidgeCV(cv=4)

m.fit(X_train,y_train,eval_set=[(X_train,y_train),(X_val, y_val.values)],eval_metric='auc', early_stopping_rounds=100,verbose=200)

p=m.predict_proba(X_val)[:,-1]

p1=logit(p)

print(roc_auc_score(y_val,p))

print(roc_auc_score(y_val,p1))
p1
m.feature_importances_
confusion_matrix(y_val,p>0.5)
sorted(zip(m.feature_importances_,X_train),reverse=True)
err=[]

y_pred_tot=[]





feature_importance_df = pd.DataFrame()

gr=X.campaign_id_x.values

from sklearn.model_selection import KFold,StratifiedKFold,GroupKFold

fold=GroupKFold(n_splits=10)

i=1

for train_index, test_index in fold.split(X,y,gr):

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]

    y_train, y_test = y[train_index], y[test_index]

    m=LGBMClassifier(n_estimators=5000,random_state=1994,learning_rate=0.03,reg_alpha=0.2,colsample_bytree=0.5)

    m.fit(X_train,y_train,eval_set=[(X_train,y_train),(X_test, y_test)],eval_metric='auc', early_stopping_rounds=200,verbose=200)

    

    preds=m.predict_proba(X_test,num_iteration=m.best_iteration_)[:,-1]

    

    fold_importance_df = pd.DataFrame()

    fold_importance_df["feature"] = X_train.columns

    fold_importance_df["importance"] = m.feature_importances_

    fold_importance_df["fold"] = i + 1

    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    

    

    print("err: ",roc_auc_score(y_test,preds))

    err.append(roc_auc_score(y_test,preds))

    p = m.predict_proba(Xtest)[:,-1]

    i=i+1

    y_pred_tot.append(p)
np.mean(err,0)

all_features = feature_importance_df[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)

all_features.reset_index(inplace=True)

important_features = list(all_features[0:170]['feature'])

all_features[0:170]
df1 = X[important_features]

corr_matrix = df1.corr().abs()



# Select upper triangle of correlation matrix

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))



# Find index of feature columns with correlation greater than 0.95

high_cor = [column for column in upper.columns if any(upper[column] > 0.98)]

print(len(high_cor))

print(high_cor)
features = [i for i in important_features if i not in high_cor]

print(len(features))

print(features)
X=X[features]

Xtest=Xtest[features]
err=[]

y_pred_tot=[]



# feature_importance_df = pd.DataFrame()



from sklearn.model_selection import KFold,StratifiedKFold

fold=GroupKFold(n_splits=15)

i=1

for train_index, test_index in fold.split(X,y,gr):

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]

    y_train, y_test = y[train_index], y[test_index]

    m=LGBMClassifier(n_estimators=5000,random_state=1994,learning_rate=0.03,reg_alpha=0.2,colsample_bytree=0.5)

    m.fit(X_train,y_train,eval_set=[(X_train,y_train),(X_test, y_test)],eval_metric='auc', early_stopping_rounds=200,verbose=200)

    

    preds=m.predict_proba(X_test,num_iteration=m.best_iteration_)[:,-1]

    

#     fold_importance_df = pd.DataFrame()

#     fold_importance_df["feature"] = X_train.columns

#     fold_importance_df["importance"] = m.feature_importances_

#     fold_importance_df["fold"] = i + 1

#     feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    

    

    print("err: ",roc_auc_score(y_test,preds))

    err.append(roc_auc_score(y_test,preds))

    p = m.predict_proba(Xtest)[:,-1]

    i=i+1

    y_pred_tot.append(p)
np.mean(err,0)
s['redemption_status']=np.mean(y_pred_tot,0)

s.head()

sum(s.redemption_status>0.5)
s.to_csv('AV_amex_lgb_folds_v39.csv',index=False)

s.shape
# print(X.shape,Xtest.shape)



# X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.25,random_state = 1994,stratify=y)

# categorical_features_indices = np.where(X_train.dtypes =='category')[0]

# categorical_features_indices
# m=CatBoostClassifier(n_estimators=2500,random_state=1994,learning_rate=0.03,eval_metric='AUC')

# # m=RidgeCV(cv=4)

# m.fit(X_train,y_train,eval_set=[(X_val, y_val.values)], early_stopping_rounds=300,verbose=200,cat_features=categorical_features_indices)

# p=m.predict_proba(X_val)[:,-1]

# print(roc_auc_score(y_val,p))





# 0:	test: 0.7072835	best: 0.7072835 (0)	total: 94.6ms	remaining: 3m 56s

# 200:	test: 0.9846892	best: 0.9846892 (200)	total: 6.49s	remaining: 1m 14s

# 400:	test: 0.9857940	best: 0.9857940 (400)	total: 13.1s	remaining: 1m 8s

# 600:	test: 0.9860940	best: 0.9860980 (590)	total: 19.6s	remaining: 1m 1s

# 800:	test: 0.9861993	best: 0.9862259 (737)	total: 25.9s	remaining: 54.9s

# 1000:	test: 0.9862786	best: 0.9863095 (882)	total: 32.2s	remaining: 48.2s

# 1200:	test: 0.9863154	best: 0.9863839 (1056)	total: 38.5s	remaining: 41.6s

# 1400:	test: 0.9863995	best: 0.9864448 (1265)	total: 44.8s	remaining: 35.1s

# Stopped by overfitting detector  (300 iterations wait)



# bestTest = 0.9864447541

# bestIteration = 1265



# Shrink model to first 1266 iterations.

# 0.9864447540507506
# errCB=[]

# y_pred_tot_cb=[]

# from sklearn.model_selection import KFold,StratifiedKFold

# fold=StratifiedKFold(n_splits=15,shuffle=True,random_state=1994)

# i=1

# for train_index, test_index in fold.split(X,y):

#     X_train, X_test = X.iloc[train_index], X.iloc[test_index]

#     y_train, y_test = y[train_index], y[test_index]

#     m=CatBoostClassifier(n_estimators=5000,random_state=1994,eval_metric='AUC',learning_rate=0.03)

#     m.fit(X_train,y_train,eval_set=[(X_train,y_train),(X_test, y_test)], early_stopping_rounds=200,verbose=200,cat_features=categorical_features_indices)

#     preds=m.predict_proba(X_test)[:,-1]

#     print("err_cb: ",roc_auc_score(y_test,preds))

#     errCB.append(roc_auc_score(y_test,preds))

#     p = m.predict_proba(Xtest)[:,-1]

#     i=i+1

#     y_pred_tot_cb.append(p)
# np.mean(errCB,0)
# s['redemption_status']=np.mean(y_pred_tot_cb,0)

# s.head()
# sum(s.redemption_status>0.5)
# s.to_csv('AV_amex_cb_folds_v28.csv',index=False)

# s.shape
# s['redemption_status']=np.mean(y_pred_tot_cb,0)*0.25+np.mean(y_pred_tot,0)*0.75

# s.head()
# sum(s.redemption_status>0.5)
# s.to_csv('AV_amex_stack2_folds_v28.csv',index=False)

# # s.shape
# print(X.shape,Xtest.shape)



# X=pd.get_dummies(X,drop_first=True)

# Xtest=pd.get_dummies(Xtest,drop_first=True)



# X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.25,random_state = 1994,stratify=y)

# categorical_features_indices = np.where(X_train.dtypes =='category')[0]

# categorical_features_indices
# from xgboost import XGBClassifier



# errxgb=[]

# y_pred_tot_xgb=[]

# from sklearn.model_selection import KFold,StratifiedKFold

# fold=StratifiedKFold(n_splits=10,shuffle=True,random_state=1994)

# i=1

# for train_index, test_index in fold.split(X,y):

#     X_train, X_test = X.iloc[train_index], X.iloc[test_index]

#     y_train, y_test = y[train_index], y[test_index]

#     m=XGBClassifier(n_estimators=5000,random_state=1994,eval_metric='auc',learning_rate=0.03)

#     m.fit(X_train,y_train,eval_set=[(X_train,y_train),(X_test, y_test)], early_stopping_rounds=200,verbose=200)

#     preds=m.predict_proba(X_test)[:,-1]

#     print("err_xgb: ",roc_auc_score(y_test,preds))

#     errxgb.append(roc_auc_score(y_test,preds))

#     p = m.predict_proba(Xtest)[:,-1]

#     i=i+1

#     y_pred_tot_xgb.append(p)
# np.mean(errxgb,0)
# s['redemption_status']=np.mean(y_pred_tot_xgb,0)

# s.head()
# s.to_csv('AV_amex_xgb_folds_v28.csv',index=False)

# s.shape
# s['redemption_status']=(np.mean(y_pred_tot_cb,0)+np.mean(y_pred_tot,0)+np.mean(y_pred_tot_xgb,0))/3

# s.head()
# s.to_csv('AV_amex_stack3_folds_v28.csv',index=False)
# s['redemption_status']=np.mean(y_pred_tot_xgb,0)*0.5+np.mean(y_pred_tot,0)*0.5

# s.head()
# s.to_csv('AV_amex_stack4_folds_v17.csv',index=False)

# s.shape