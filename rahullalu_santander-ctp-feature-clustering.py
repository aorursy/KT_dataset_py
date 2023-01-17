#IMPORTING REQUIRED LIBRARIES

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import matplotlib.pyplot as plt

import seaborn as sns

import math



from lightgbm.sklearn import LGBMRegressor

from lightgbm.sklearn import LGBMClassifier

import lightgbm as lgb

from xgboost.sklearn import XGBClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import KFold,StratifiedKFold

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.decomposition import TruncatedSVD

from sklearn.cluster import KMeans

from sklearn.decomposition import PCA,KernelPCA,NMF



from sklearn.metrics import roc_auc_score,accuracy_score



from catboost import CatBoostClassifier

from sklearn.linear_model import LogisticRegression



import gc

gc.enable()





import warnings

warnings.filterwarnings("ignore")

%matplotlib inline
#All functions



#FUNCTION FOR PROVIDING FEATURE SUMMARY

def feature_summary(df_fa):

    print('DataFrame shape')

    print('rows:',df_fa.shape[0])

    print('cols:',df_fa.shape[1])

    col_list=['Unique_Count','Max','Min','Mean','Std','Skewness','Median']

    df=pd.DataFrame(index=df_fa.columns,columns=col_list)

    #df['Null']=list([len(df_fa[col][df_fa[col].isnull()]) for i,col in enumerate(df_fa.columns)])

    #df['%_Null']=list([len(df_fa[col][df_fa[col].isnull()])/df_fa.shape[0]*100 for i,col in enumerate(df_fa.columns)])

    df['Unique_Count']=list([len(df_fa[col].unique()) for i,col in enumerate(df_fa.columns)])

    #df['Data_type']=list([df_fa[col].dtype for i,col in enumerate(df_fa.columns)])

    for i,col in enumerate(df_fa.columns):

        if 'float' in str(df_fa[col].dtype) or 'int' in str(df_fa[col].dtype):

            df.at[col,'Max']=str(round(df_fa[col].max(),2))

            df.at[col,'Min']=str(round(df_fa[col].min(),2))

            df.at[col,'Mean']=df_fa[col].mean()

            df.at[col,'Std']=df_fa[col].std()

            df.at[col,'Skewness']=df_fa[col].skew()

            df.at[col,'Median']=df_fa[col].median()

            

        

    return(df.fillna('-'))



#DATASET VIEW

path1="../input/"

data_files=list(os.listdir(path1))

df_files=pd.DataFrame(data_files,columns=['File_Name'])

df_files['Size_in_MB']=df_files.File_Name.apply(lambda x:round(os.stat(path1+x).st_size/(1024*1024),2))

df_files
%%time

#READING AVAILABLE FILES DATASET

#HISTORICAL TRANSACTIONS FILE IS A LARGE ONE 

#SO WE WILL BE READING IT IN PARTS

print('reading train dataset...')

df_train=pd.read_csv(path1+'train.csv')

print('reading test dataset...')

df_test=pd.read_csv(path1+'test.csv')

print('submission file')

df_submission=pd.read_csv(path1+'sample_submission.csv')
#CREATING FINAL X, y and test SETS

X=df_train.drop(['ID_code','target'],axis=1)

y=df_train['target']

test=df_test.drop(['ID_code'],axis=1)
df_combi=pd.concat([X,test],axis=0)

df_fs=feature_summary(df_combi)
n_clus=3

cluster = KMeans(n_clusters=n_clus, random_state=0, n_jobs=-1)

model=cluster.fit(df_fs)

df_fs['labels']=model.labels_
df_fs.groupby('labels').count()
print(X.shape,y.shape,test.shape)
for i in range(n_clus):

    f_list=list(df_fs[df_fs.labels==i].index)

    print('cluster id:',i+1,'\tfeature cluster item count:',len(f_list))
%%time

#CREATING FINAL MODEL WITH STRATIFIED KFOLDS

#FOLD COUNT 10

#TRIED XGBClassifier, LGBMClassifier, CatBoostClassifier

#BEST SCORE ACHIEVED BY CatBoostClassifier



param = {

    'bagging_freq': 5,          

    'bagging_fraction': 0.38,   'boost_from_average':'false',   

    'boost': 'gbdt',             'feature_fraction': 0.04,     'learning_rate': 0.0085,

    'max_depth': -1,             'metric':'auc',                'min_data_in_leaf': 80,     'min_sum_hessian_in_leaf': 10.0,

    'num_leaves': 13,            'num_threads': 8,              'tree_learner': 'serial',   'objective': 'binary',

    'reg_alpha': 0.1302650970728192, 'reg_lambda': 0.3603427518866501,'verbosity': -1

}





#DATAFRAMES FOR STORING PREDICTIONS ON TRAIN DATA AS WELL AS TEST DATA

#CAN BE USED FOR ENSEMBLE 

df_preds=pd.DataFrame()

df_preds_x=pd.DataFrame()







for i in range(n_clus):

    f_list=list(df_fs[df_fs.labels==i].index)

    print('Starting predicting cluster:',i+1)

    

    k=1

    splits=10

    avg_score=0

    

    #CREATING STRATIFIED FOLDS

    skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=200)

    print('\nStarting KFold iterations...')

    X1=X[f_list]

    test1=test[f_list]

    

    for train_index,test_index in skf.split(X1,y):

        df_X=X1.iloc[train_index,:]

        df_y=y.iloc[train_index]

        val_X=X1.iloc[test_index,:]

        val_y=y.iloc[test_index]



        #FITTING MODEL

    



        trn_data = lgb.Dataset(df_X, label=df_y)

        val_data = lgb.Dataset(val_X, label=val_y)

        model= lgb.train(param, trn_data, 1000000, valid_sets = [trn_data, val_data], verbose_eval=1000, early_stopping_rounds = 2000)

        col_name='X_'+str(i+1)+'_'+str(k)

#PREDICTING ON VALIDATION DATA

        

        preds_x=model.predict(val_X,num_iteration=model.best_iteration)

        df_preds_x[col_name]=model.predict(X1,num_iteration=model.best_iteration)

#CALCULATING ACCURACY

        acc=roc_auc_score(val_y,preds_x)

        print('Iteration:',k,'  roc_auc_score:',acc)

        

#         col_name='P_'+str(i+1)+'_'+str(k)

        if k==1:

            score=acc

            preds=model.predict(test1,num_iteration=model.best_iteration)

            df_preds[col_name]=preds#model.predict(test,num_iteration=model.best_iteration)

        else:

            preds1=model.predict(test1,num_iteration=model.best_iteration)

            preds=preds+preds1

            df_preds[col_name]=preds1#model.predict(test,num_iteration=model.best_iteration)

        

        if score<acc:

            score=acc

            

        avg_score=avg_score+acc        

        k=k+1

    

    print('\n Best score:',score,' Avg Score:',avg_score/splits)

%%time

#CREATING SUMBISSION FILE

df_preds_x.to_csv('X_features.csv',index=False)

df_preds.to_csv('test_features.csv',index=False)
# %%time



# X=df_preds_x

# test=df_preds



# model=LogisticRegression()

# k=1

# splits=15

# avg_score=0



# #CREATING STRATIFIED FOLDS

# skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=200)

# print('\nStarting KFold iterations...')

# for train_index,test_index in skf.split(X,y):

#     df_X=X.iloc[train_index,:]

#     df_y=y.iloc[train_index]

#     val_X=X.iloc[test_index,:]

#     val_y=y.iloc[test_index]



# #FITTING MODEL

    

#     model.fit(df_X,df_y)

    

    

# #PREDICTING ON VALIDATION DATA

    

#     preds_x=pd.Series(model.predict_proba(val_X)[:,1])

# #CALCULATING ACCURACY

#     acc=roc_auc_score(val_y,preds_x)

#     print('Iteration:',k,'  roc_auc_score:',acc)

#     if k==1:

#         score=acc

#         preds=pd.Series(model.predict_proba(test)[:,1])

        

#     else:

#         preds1=pd.Series(model.predict_proba(test)[:,1])

#         preds=preds+preds1

        

#         if score<acc:

#             score=acc

            

#     avg_score=avg_score+acc        

#     k=k+1

# print('\n Best score:',score,' Avg Score:',avg_score/splits)

# #TAKING AVERAGE OF PREDICTIONS

# preds=preds/splits
df_preds.head()
%%time

#PREPARING SUBMISSION

df_submission['target']=df_preds.mean(axis=1)

df_submission
#CREATING SUMBISSION FILE

df_submission.to_csv('submission.csv',index=False)
df_submission