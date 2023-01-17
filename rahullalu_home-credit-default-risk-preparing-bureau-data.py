import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import gc
import os
import matplotlib.pyplot as plt
import seaborn as sns
import gc
from lightgbm.sklearn import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score,accuracy_score
from sklearn.model_selection import KFold,StratifiedKFold
from xgboost.sklearn import XGBClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
gc.enable()
%matplotlib inline
# Any results you write to the current directory are saved as output.
#Dataset view
path1= "../input/"
data_files=list(os.listdir(path1))
df_files=pd.DataFrame(data_files,columns=['File_Name'])
df_files['Size_in_MB']=df_files.File_Name.apply(lambda x:round(os.stat(path1+x).st_size/(1024*1024),2))
df_files
#All functions

#FUNCTION FOR PROVIDING FEATURE SUMMARY
def feature_summary(df_fa):
    print('DataFrame shape')
    print('rows:',df_fa.shape[0])
    print('cols:',df_fa.shape[1])
    col_list=['Null','Unique_Count','Data_type','Max/Min','Mean','Std','Skewness','Sample_values']
    df=pd.DataFrame(index=df_fa.columns,columns=col_list)
    df['Null']=list([len(df_fa[col][df_fa[col].isnull()]) for i,col in enumerate(df_fa.columns)])
    #df['%_Null']=list([len(df_fa[col][df_fa[col].isnull()])/df_fa.shape[0]*100 for i,col in enumerate(df_fa.columns)])
    df['Unique_Count']=list([len(df_fa[col].unique()) for i,col in enumerate(df_fa.columns)])
    df['Data_type']=list([df_fa[col].dtype for i,col in enumerate(df_fa.columns)])
    for i,col in enumerate(df_fa.columns):
        if 'float' in str(df_fa[col].dtype) or 'int' in str(df_fa[col].dtype):
            df.at[col,'Max/Min']=str(round(df_fa[col].max(),2))+'/'+str(round(df_fa[col].min(),2))
            df.at[col,'Mean']=df_fa[col].mean()
            df.at[col,'Std']=df_fa[col].std()
            df.at[col,'Skewness']=df_fa[col].skew()
        df.at[col,'Sample_values']=list(df_fa[col].unique())
           
    return(df.fillna('-'))

def drop_corr_col(df_corr):
    upper = df_corr.where(np.triu(np.ones(df_corr.shape),
                          k=1).astype(np.bool))
    # Find index of feature columns with correlation greater than 0.999
    to_drop = [column for column in upper.columns if any(upper[column] > 0.999)]
    return(to_drop)
%%time
#Reading train data
train=pd.read_csv(path1+'application_train.csv',usecols=['SK_ID_CURR','TARGET'])
#Reading bureau data
bur=pd.read_csv(path1+'bureau.csv')
print('bureau set reading complete...')
#Reading bureau balance
bur_bal=pd.read_csv(path1+'bureau_balance.csv')
print('bureau balance set reading complete...')
train.head()
bur.head()
bur_fs=feature_summary(bur)
bur_fs
bur_fs[bur_fs.Data_type=='object']
%%time
for col in ['CREDIT_CURRENCY','CREDIT_TYPE','CREDIT_ACTIVE']:
    bur[col]=bur[col].apply(lambda x: str(x).replace(" ","_")) 

dummy=pd.DataFrame()
for col in ['CREDIT_CURRENCY','CREDIT_TYPE','CREDIT_ACTIVE']:
    dummy=pd.concat([dummy,pd.get_dummies(bur[col],prefix='DUM_'+col)],axis=1)
dummy.head()
bur_f=pd.concat([bur.drop(['CREDIT_CURRENCY','CREDIT_TYPE','CREDIT_ACTIVE'],axis=1),dummy],axis=1)
bur_f.head()
bur_f.shape
bur_f['CALC_PER_CREDIT_MAX_OVERDUE']=bur_f['AMT_CREDIT_MAX_OVERDUE']/bur_f['AMT_CREDIT_SUM']
bur_f['CALC_PER_CREDIT_SUM_DEBT']=bur_f['AMT_CREDIT_SUM_DEBT']/bur_f['AMT_CREDIT_SUM']
bur_f['CALC_PER_CREDIT_SUM_LIMIT']=bur_f['AMT_CREDIT_SUM_LIMIT']/bur_f['AMT_CREDIT_SUM']
bur_f['CALC_PER_CREDIT_SUM_OVERDUE']=bur_f['AMT_CREDIT_SUM_OVERDUE']/bur_f['AMT_CREDIT_SUM']
bur_f['CALC_PER_ANNUITY']=bur_f['AMT_ANNUITY']/bur_f['AMT_CREDIT_SUM']
bur_f['CALC_CREDIT_LIMIT_CROSSED']=bur_f['AMT_CREDIT_SUM_LIMIT']-bur_f['AMT_CREDIT_SUM']
bur_f['CALC_CREDIT_PER_DAY']=bur_f['AMT_CREDIT_SUM']/bur_f['DAYS_CREDIT_ENDDATE'].abs()
bur_f['CALC_CREDIT_CLOSED']=(bur_f['DAYS_ENDDATE_FACT'] < 0).astype(int)
bur_f.shape
del bur,dummy
gc.collect()
bur_bal.head()
bur_bal[bur_bal.SK_ID_BUREAU==5715448]
bur_bal['MONTHS_BALANCE']=bur_bal.MONTHS_BALANCE.abs()
%%time
bur_bal_f=bur_bal.groupby(['SK_ID_BUREAU','STATUS']).aggregate({'STATUS':['count'],'MONTHS_BALANCE':['max','min']})
bur_bal_f.reset_index(inplace=True)
bur_bal_f.columns=['SK_ID_BUREAU','STATUS','STATUS_count','MONTHS_BALANCE_max','MONTHS_BALANCE_min']
bur_bal_f.head()
dummy=pd.get_dummies(bur_bal_f['STATUS'],prefix='DUM_STATUS')
dummy.head()
bur_bal_ff=pd.concat([bur_bal_f.drop(['STATUS'],axis=1),dummy],axis=1)
bur_bal_ff.head()
dummy_col=[x for x in bur_bal_ff.columns if 'DUM_' in x]
%%time
for col in dummy_col:
    bur_bal_ff[col]=bur_bal_ff.apply(lambda x: x.STATUS_count if x[col]==1 else 0,axis=1)
bur_bal_ff.head()
bur_bal_ff.drop('STATUS_count',axis=1,inplace=True)
bur_bal_ff.shape
bur_bal_cols=[x for x in list(bur_bal_ff.columns) if x not in ['SK_ID_BUREAU']]
bur_bal_agg={}
bur_bal_name=['SK_ID_BUREAU']
for col in bur_bal_cols:
    if 'DUM_' in col:
        bur_bal_agg[col]=['sum']
        bur_bal_name.append(col)
    elif '_max' in col:
        bur_bal_agg[col]=['max']
        bur_bal_name.append(col)
    elif '_min' in col:
        bur_bal_agg[col]=['min']
        bur_bal_name.append(col)
    else:
        bur_bal_agg[col]=['sum','mean']
        bur_bal_name.append(col+'_'+'sum')
        bur_bal_name.append(col+'_'+'mean')
%%time
bur_bal_fg=bur_bal_ff.groupby('SK_ID_BUREAU').aggregate(bur_bal_agg)
bur_bal_fg.reset_index(inplace=True)
bur_bal_fg.columns=bur_bal_name
bur_bal_fg.head()
del bur_bal,bur_bal_f,bur_bal_ff
gc.collect()
bur_combi=bur_f.join(bur_bal_fg.set_index('SK_ID_BUREAU'),on='SK_ID_BUREAU',lsuffix='_BU', rsuffix='_BUB')
del bur_bal_fg
gc.collect()
bur_combi_fs=feature_summary(bur_combi)
bur_combi_fs
bur_combi_cols=[x for x in list(bur_combi.columns) if x not in ['SK_ID_CURR','SK_ID_BUREAU']]
bur_combi_agg={}
bur_combi_name=['SK_ID_CURR','SK_ID_BUREAU']
for col in bur_combi_cols:
    if 'DUM_' in col:
        bur_combi_agg[col]=['sum']
        bur_combi_name.append(col+'_'+'sum')
    elif 'AMT_' in col:
        bur_combi_agg[col]=['sum','mean','max','min','var','std']
        bur_combi_name.append(col+'_'+'sum')
        bur_combi_name.append(col+'_'+'mean')
        bur_combi_name.append(col+'_'+'max')
        bur_combi_name.append(col+'_'+'min')
        bur_combi_name.append(col+'_'+'var')
        bur_combi_name.append(col+'_'+'std')
    elif 'CNT_' in col:
        bur_combi_agg[col]=['sum','max','min','count']
        bur_combi_name.append(col+'_'+'sum')
        bur_combi_name.append(col+'_'+'max')
        bur_combi_name.append(col+'_'+'min')
        bur_combi_name.append(col+'_'+'count')
    elif 'DAYS_' in col:
        bur_combi_agg[col]=['sum','max','min']
        bur_combi_name.append(col+'_'+'sum')
        bur_combi_name.append(col+'_'+'max')
        bur_combi_name.append(col+'_'+'min')
    elif 'CALC_' in col:
        bur_combi_agg[col]=['mean']
        bur_combi_name.append(col+'_'+'mean')
    else:
        bur_combi_agg[col]=['sum']
        bur_combi_name.append(col+'_'+'sum')
       

%%time
bur_combi_f=bur_combi.groupby(['SK_ID_CURR','SK_ID_BUREAU']).aggregate(bur_combi_agg)                 
bur_combi_f.reset_index(inplace=True)
bur_combi_f.columns=bur_combi_name
bur_combi_f.head()
bur_combi_cols=list(bur_combi_f.columns)
bur_combi_agg={}
bur_combi_name=['SK_ID_CURR']
for col in bur_combi_cols:
    if 'SK_ID_CURR'==col:
        bur_combi_agg[col]=['count']
        bur_combi_name.append('SK_ID_BUREAU_count')
    elif '_sum'==col:
        bur_combi_agg[col]=['sum']
        bur_combi_name.append(col)
    elif '_mean' in col:
        bur_combi_agg[col]=['mean']
        bur_combi_name.append(col)
    elif '_max' in col:
        bur_combi_agg[col]=['max']
        bur_combi_name.append(col)
    elif '_min' in col:
        bur_combi_agg[col]=['min']
        bur_combi_name.append(col)
    elif '_count' in col:
        bur_combi_agg[col]=['sum']
        bur_combi_name.append(col)
    elif '_var' in col:
        bur_combi_agg[col]=['mean']
        bur_combi_name.append(col)
    elif '_std' in col:
        bur_combi_agg[col]=['mean']
        bur_combi_name.append(col)
    else:
        bur_combi_agg[col]=['sum']
        bur_combi_name.append(col)
%%time
bur_combi_fg=bur_combi_f.groupby(['SK_ID_CURR']).aggregate(bur_combi_agg)                 
bur_combi_fg.reset_index(inplace=True)
bur_combi_fg.columns=bur_combi_name
df_bur_target=train.join(bur_combi_fg.set_index('SK_ID_CURR'),on='SK_ID_CURR',lsuffix='_AP', rsuffix='_BU')
df_bur_target.head()
df_bur_target.shape
train_X,test_X,train_y,test_y=train_test_split(df_bur_target.drop(['SK_ID_CURR','TARGET'],axis=1),df_bur_target['TARGET'],random_state=200)
model =LGBMClassifier(learning_rate=0.05,n_estimators=200,n_jobs=-1,reg_alpha=0.1,min_split_gain=.1,verbose=-1)
model.fit(train_X,train_y)
score2=roc_auc_score(test_y,model.predict_proba(test_X)[:,1])
print(score2)
df_bur=df_bur_target.drop(['SK_ID_CURR','TARGET'],axis=1)
%%time
#FEATURE EXCLUSION
score=0
score1=0
score2=0
drop_list=[]
col_list=list(df_bur.columns)


while True:
    score1=0
    score2=0
    for i,col in enumerate(col_list):
        col_list.remove(col)
        train_X,test_X,train_y,test_y=train_test_split(df_bur[col_list],train['TARGET'],random_state=200)
        model =LGBMClassifier(learning_rate=0.05,n_estimators=200,n_jobs=-1,reg_alpha=0.1,min_split_gain=.1,verbose=-1)
        model.fit(train_X,train_y)
        score2=roc_auc_score(test_y,model.predict_proba(test_X)[:,1])
        col_list.extend([col])
#        dummy_1.at[i,'score']=score2
        if score1<score2:
            score1=score2
            col1=col
#        print('dropped col',col,':',score2)
    if score<score1:
        score=score1
        print('dropped col',col1,':',score)
        drop_list.extend([col1])
        col_list.remove(col1)
    else:
        print('Best score achieved')
        break
print(drop_list)
print('best score:',score)
# select_list=['CALC_PER_CREDIT_SUM_DEBT_mean','AMT_CREDIT_MAX_OVERDUE_mean','DAYS_CREDIT_min','SK_ID_BUREAU_count','AMT_CREDIT_SUM_mean',
#              'DAYS_CREDIT_ENDDATE_max','AMT_CREDIT_SUM_LIMIT_max','DUM_STATUS_1_sum','DAYS_CREDIT_sum','AMT_CREDIT_SUM_DEBT_max']
# col_list=[x for x in list(df_bur_target.columns) if x not in select_list]
# col_list.remove('SK_ID_CURR')
# col_list.remove('TARGET')
# %%time
# score=0
# score1=0
# score2=0

# k=1


# while True:
#     score1=0
#     score2=0
#     temp_list=select_list
#     for i,col in enumerate(col_list):
#         try:
#             if k==0:
#                 train_X,test_X,train_y,test_y=train_test_split(df_bur_target[col],df_bur_target['TARGET'],random_state=200)
#                 model =LGBMClassifier(learning_rate=0.05,n_estimators=200,n_jobs=-1,reg_alpha=0.1,min_split_gain=.1,verbose=-1)
#                 model.fit(np.array(train_X).reshape(-1,1),train_y)
#                 score2=roc_auc_score(test_y,model.predict_proba(np.array(test_X).reshape(-1,1))[:,1])
#             else:
#                 temp_list.extend([col])
#                 train_X,test_X,train_y,test_y=train_test_split(df_bur_target[temp_list],df_bur_target['TARGET'],random_state=200)
#                 model =LGBMClassifier(learning_rate=0.05,n_estimators=200,n_jobs=-1,reg_alpha=0.1,min_split_gain=.1,verbose=-1)
#                 model.fit(train_X,train_y)
#                 score2=roc_auc_score(test_y,model.predict_proba(test_X)[:,1])
#                 temp_list.remove(col)
#         except:
#             print('Exception raised exclude:',col)
#             col_list.remove(col)
            
#         if score1<score2:
#             score1=score2
#             col1=col
# #        print('dropped col',col,':',score2)
#     k=k+1
#     if ((score<score1) & (k<=10)):
#         score=score1
#         print('select col',col1,':',score)
#         select_list.extend([col1])
#         col_list.remove(col1)
#     else:
#         print('Best score achieved')
#         break
    
# print(select_list)
# print('best score:',score)