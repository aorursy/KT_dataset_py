
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
from sklearn.preprocessing import RobustScaler,PolynomialFeatures,MinMaxScaler,Binarizer
from sklearn.decomposition import PCA
import warnings
%matplotlib inline
warnings.filterwarnings('ignore')
gc.enable()
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

def cnt_unique(df):
    return(len(df.unique()))
%%time
#Reading POS CASH balance data
pos_cash=pd.read_csv(path1+'POS_CASH_balance.csv')
print('POS_CASH_balance set reading complete...')
pos_cash_fs=feature_summary(pos_cash)
pos_cash_fs
pos_cash.sort_values(['SK_ID_CURR','SK_ID_PREV']).head(30)
pos_cash['MONTHS_BALANCE']=pos_cash['MONTHS_BALANCE'].abs()
pos_cash['CALC_PERC_REMAINING_INSTAL']=pos_cash['CNT_INSTALMENT_FUTURE']/pos_cash['CNT_INSTALMENT']
pos_cash['CALC_CNT_REMAINING_INSTAL']=pos_cash['CNT_INSTALMENT']-pos_cash['CNT_INSTALMENT_FUTURE']
pos_cash['CALC_DAYS_WITHOUT_TOLERANCE']=pos_cash['SK_DPD']-pos_cash['SK_DPD_DEF']
pos_cash['NAME_CONTRACT_STATUS']=pos_cash['NAME_CONTRACT_STATUS'].apply(lambda x: str(x).replace(" ","_")) 
dummy=pd.get_dummies(pos_cash['NAME_CONTRACT_STATUS'],prefix='DUM_NAME_CONTRACT_STATUS')
dummy.head()
pos_cash_f=pd.concat([pos_cash.drop(['NAME_CONTRACT_STATUS'],axis=1),dummy],axis=1)
pos_cash_f.head()
#DEFINING AGGREGATION RULES AND CREATING LIST OF NEW FEATURES
pos_cash_cols=[x for x in list(pos_cash_f.columns) if x not in ['SK_ID_CURR']]
pos_cash_agg={}
pos_cash_name=['SK_ID_CURR','SK_ID_PREV']
for col in pos_cash_cols:
    if 'SK_ID_PREV'==col:
        pos_cash_agg[col]=['count']
        pos_cash_name.append(col+'_'+'count')
    elif 'MONTHS_BALANCE'==col:
        pos_cash_agg[col]=['max','min','count']
        pos_cash_name.append(col+'_'+'max')
        pos_cash_name.append(col+'_'+'min')
        pos_cash_name.append(col+'_'+'count')
    elif 'DUM_' in col:
        pos_cash_agg[col]=['sum','mean','max','min']
        pos_cash_name.append(col+'_'+'sum')
        pos_cash_name.append(col+'_'+'mean')
        pos_cash_name.append(col+'_'+'max')
        pos_cash_name.append(col+'_'+'min')
    elif 'CNT_' in col:
        pos_cash_agg[col]=['max','min','sum','count']
        pos_cash_name.append(col+'_'+'max')
        pos_cash_name.append(col+'_'+'min')
        pos_cash_name.append(col+'_'+'sum')
        pos_cash_name.append(col+'_'+'count')
    else:
        pos_cash_agg[col]=['sum','mean']
        pos_cash_name.append(col+'_'+'sum')
        pos_cash_name.append(col+'_'+'mean')
pos_cash_f.shape
%%time
#AGGREGATING DATA ON SK_ID_CURR,SK_ID_PREV USING RULES CREATED IN PREVIOUS STEP
pos_cash_ff=pos_cash_f.groupby(['SK_ID_CURR','SK_ID_PREV']).aggregate(pos_cash_agg)
pos_cash_ff.reset_index(inplace=True)
pos_cash_ff.columns=pos_cash_name
#DEFINING RULES FOR SECOND AGGREGATION ON SK_ID_CURR
pos_cash_cols=[x for x in list(pos_cash_ff.columns) if x not in ['SK_ID_CURR','SK_ID_PREV']]
pos_cash_agg={}
pos_cash_name=['SK_ID_CURR']
for col in pos_cash_cols:
    if '_sum'==col:
        pos_cash_agg[col]=['sum']
        pos_cash_name.append(col)
    elif '_mean' in col:
        pos_cash_agg[col]=['mean']
        pos_cash_name.append(col)
    elif '_max' in col:
        pos_cash_agg[col]=['max']
        pos_cash_name.append(col)
    elif '_min' in col:
        pos_cash_agg[col]=['min']
        pos_cash_name.append(col)
    elif '_count' in col:
        pos_cash_agg[col]=['sum']
        pos_cash_name.append(col)
    else:
        pos_cash_agg[col]=['sum']
        pos_cash_name.append(col)
%%time
#AGGREGATING DATA ON SK_ID_CURR,SK_ID_PREV USING RULES CREATED IN PREVIOUS STEP
pos_cash_fg=pos_cash_ff.groupby(['SK_ID_CURR']).aggregate(pos_cash_agg)
pos_cash_fg.reset_index(inplace=True)
pos_cash_fg.columns=pos_cash_name
pos_cash_fg.head()
pos_cash_fg.shape
del pos_cash,pos_cash_f,pos_cash_ff
gc.collect()
train=pd.read_csv(path1+'application_train.csv',usecols=['SK_ID_CURR','TARGET'])
df_final=train.join(pos_cash_fg.set_index('SK_ID_CURR'),on='SK_ID_CURR',lsuffix='_AP', rsuffix='_POS')
df_final.shape
df_pos=df_final.drop(['SK_ID_CURR','TARGET'],axis=1)
%%time
train_X,test_X,train_y,test_y=train_test_split(df_pos,train['TARGET'],random_state=200)
model =LGBMClassifier(learning_rate=0.05,n_estimators=200,n_jobs=-1,reg_alpha=0.1,min_split_gain=.1,verbose=-1)
model.fit(train_X,train_y)
score2=roc_auc_score(test_y,model.predict_proba(test_X)[:,1])
print(score2)
%%time
#FEATURE EXCLUSION
score=0
score1=0
score2=0
drop_list=[]
col_list=list(df_pos.columns)


while True:
    score1=0
    score2=0
    for i,col in enumerate(col_list):
        col_list.remove(col)
        train_X,test_X,train_y,test_y=train_test_split(df_pos[col_list],train['TARGET'],random_state=200)
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
# %%time
# #FORWARD FEATURE SELCTION 
# score=0
# score1=0
# score2=0
# select_list=[]
# col_list=list(df_pos.columns)  
# k=0


# while True:
#     score1=0
#     score2=0
#     temp_list=select_list
#     for i,col in enumerate(col_list):
#         if k==0:
#             train_X,test_X,train_y,test_y=train_test_split(df_pos[col],train['TARGET'],random_state=200)
#             model =LGBMClassifier(learning_rate=0.05,n_estimators=200,n_jobs=-1,reg_alpha=0.1,min_split_gain=.1,verbose=-1)
#             model.fit(np.array(train_X).reshape(-1,1),train_y)
#             score2=roc_auc_score(test_y,model.predict_proba(np.array(test_X).reshape(-1,1))[:,1])
#         else:
#             temp_list.extend([col])
#             train_X,test_X,train_y,test_y=train_test_split(df_pos[temp_list],train['TARGET'],random_state=200)
#             model =LGBMClassifier(learning_rate=0.05,n_estimators=200,n_jobs=-1,reg_alpha=0.1,min_split_gain=.1,verbose=-1)
#             model.fit(train_X,train_y)
#             score2=roc_auc_score(test_y,model.predict_proba(test_X)[:,1])
#             temp_list.remove(col)
#         if score1<=score2:
#             score1=score2
#             col1=col
# #        print('dropped col',col,':',score2)
#     k=k+1
#     if score<=score1:
#         score=score1
#         print('select col',col1,':',score)
#         select_list.extend([col1])
#         col_list.remove(col1)
#     else:
#         print('Best score achieved')
#         break
    
# print(select_list)
# print('best score:',score)