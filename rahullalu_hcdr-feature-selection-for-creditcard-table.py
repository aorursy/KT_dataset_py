
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
#Reading credit card balance data
cc_bal=pd.read_csv(path1+'credit_card_balance.csv')
print('credit_card_balance set reading complete...')
cc_bal_fs=feature_summary(cc_bal)
cc_bal_fs
cc_bal.head()
cc_bal['MONTHS_BALANCE']=cc_bal['MONTHS_BALANCE'].abs()
cc_bal['CALC_PERC_BALANCE']=cc_bal['AMT_BALANCE']/cc_bal['AMT_CREDIT_LIMIT_ACTUAL']
cc_bal['CALC_PERC_DRAWINGS_ATM_CURRENT']=cc_bal['AMT_DRAWINGS_ATM_CURRENT']/cc_bal['AMT_CREDIT_LIMIT_ACTUAL']
cc_bal['CALC_PERC_DRAWINGS_CURRENT']=cc_bal['AMT_DRAWINGS_CURRENT']/cc_bal['AMT_CREDIT_LIMIT_ACTUAL']
cc_bal['CALC_PERC_DRAWINGS_OTHER_CURRENT']=cc_bal['AMT_DRAWINGS_OTHER_CURRENT']/cc_bal['AMT_CREDIT_LIMIT_ACTUAL']
cc_bal['CALC_PERC_DRAWINGS_POS_CURRENT']=cc_bal['AMT_DRAWINGS_POS_CURRENT']/cc_bal['AMT_CREDIT_LIMIT_ACTUAL']
cc_bal['CALC_PERC_INST_MIN_REGULARITY']=cc_bal['AMT_INST_MIN_REGULARITY']/cc_bal['AMT_CREDIT_LIMIT_ACTUAL']
cc_bal['CALC_PERC_PAYMENT_CURRENT']=cc_bal['AMT_PAYMENT_CURRENT']/cc_bal['AMT_CREDIT_LIMIT_ACTUAL']
cc_bal['CALC_PERC_PAYMENT_TOTAL_CURRENT']=cc_bal['AMT_PAYMENT_TOTAL_CURRENT']/cc_bal['AMT_CREDIT_LIMIT_ACTUAL']
cc_bal['CALC_PERC_RECEIVABLE_PRINCIPAL']=cc_bal['AMT_RECEIVABLE_PRINCIPAL']/cc_bal['AMT_CREDIT_LIMIT_ACTUAL']
cc_bal['CALC_PERC_RECIVABLE']=cc_bal['AMT_RECIVABLE']/cc_bal['AMT_CREDIT_LIMIT_ACTUAL']
cc_bal['CALC_DAYS_WITHOUT_TOLERANCE']=cc_bal['SK_DPD']-cc_bal['SK_DPD_DEF']

CNT_DRAWING_LIST=['CNT_DRAWINGS_ATM_CURRENT','CNT_DRAWINGS_CURRENT','CNT_DRAWINGS_OTHER_CURRENT','CNT_DRAWINGS_POS_CURRENT']
cc_bal['CALC_CNT_DRAWINGS_TOTAL']=cc_bal[CNT_DRAWING_LIST].sum(axis=1)
cc_bal['NAME_CONTRACT_STATUS']=cc_bal['NAME_CONTRACT_STATUS'].apply(lambda x: str(x).replace(" ","_")) 
dummy=pd.get_dummies(cc_bal['NAME_CONTRACT_STATUS'],prefix='DUM_NAME_CONTRACT_STATUS')
cc_bal_f=pd.concat([cc_bal.drop(['NAME_CONTRACT_STATUS'],axis=1),dummy],axis=1)
cc_bal_f.head(10)
#DEFINING AGGREGATION RULES AND CREATING LIST OF NEW FEATURES
cc_bal_cols=[x for x in list(cc_bal_f.columns) if x not in ['SK_ID_CURR']]
cc_bal_agg={}
cc_bal_name=['SK_ID_CURR','SK_ID_PREV']
for col in cc_bal_cols:
    if 'SK_ID_PREV'==col:
        cc_bal_agg[col]=['count']
        cc_bal_name.append(col+'_'+'count')
    elif 'MONTHS_BALANCE'==col:
        cc_bal_agg[col]=['max','min','count']
        cc_bal_name.append(col+'_'+'max')
        cc_bal_name.append(col+'_'+'min')
        cc_bal_name.append(col+'_'+'count')
    elif 'AMT_' in col:
        cc_bal_agg[col]=['sum','mean','max','min','var','std']
        cc_bal_name.append(col+'_'+'sum')
        cc_bal_name.append(col+'_'+'mean')
        cc_bal_name.append(col+'_'+'max')
        cc_bal_name.append(col+'_'+'min')
        cc_bal_name.append(col+'_'+'var')
        cc_bal_name.append(col+'_'+'std')
    elif 'CNT_' in col:
        cc_bal_agg[col]=['max','min','sum','count']
        cc_bal_name.append(col+'_'+'max')
        cc_bal_name.append(col+'_'+'min')
        cc_bal_name.append(col+'_'+'sum')
        cc_bal_name.append(col+'_'+'count')
    else:
        cc_bal_agg[col]=['mean']
        cc_bal_name.append(col+'_'+'mean')
%%time
#AGGREGATING DATA ON SK_ID_CURR,SK_ID_PREV USING RULES CREATED IN PREVIOUS STEP
cc_bal_ff=cc_bal_f.groupby(['SK_ID_CURR','SK_ID_PREV']).aggregate(cc_bal_agg)
cc_bal_ff.reset_index(inplace=True)
cc_bal_ff.columns=cc_bal_name
cc_bal_ff.head(20)
#DEFINING RULES FOR SECOND AGGREGATION ON SK_ID_CURR
cc_bal_cols=[x for x in list(cc_bal_ff.columns) if x not in ['SK_ID_CURR','SK_ID_PREV']]
cc_bal_agg={}
cc_bal_name=['SK_ID_CURR']
for col in cc_bal_cols:
    if '_sum'==col:
        cc_bal_agg[col]=['sum']
        cc_bal_name.append(col)
    elif '_var' in col:
        cc_bal_agg[col]=['mean']
        cc_bal_name.append(col)
    elif '_std' in col:
        cc_bal_agg[col]=['mean']
        cc_bal_name.append(col)
    elif '_mean' in col:
        cc_bal_agg[col]=['mean']
        cc_bal_name.append(col)
    elif '_max' in col:
        cc_bal_agg[col]=['max']
        cc_bal_name.append(col)
    elif '_min' in col:
        cc_bal_agg[col]=['min']
        cc_bal_name.append(col)
    elif '_count' in col:
        cc_bal_agg[col]=['sum']
        cc_bal_name.append(col)
    else:
        cc_bal_agg[col]=['sum']
        cc_bal_name.append(col)
%%time
#AGGREGATING DATA ON SK_ID_CURR,SK_ID_PREV USING RULES CREATED IN PREVIOUS STEP
# cc_bal_ff.drop(['SK_ID_PREV'],axis=1,inplace=True)
cc_bal_fg=cc_bal_ff.groupby(['SK_ID_CURR']).aggregate(cc_bal_agg)
cc_bal_fg.reset_index(inplace=True)
cc_bal_fg.columns=cc_bal_name
cc_bal_fg.head()
cc_bal_fg.shape
del cc_bal,cc_bal_f,cc_bal_ff
gc.collect()
train=pd.read_csv(path1+'application_train.csv',usecols=['SK_ID_CURR','TARGET'])
df_final=train.join(cc_bal_fg.set_index('SK_ID_CURR'),on='SK_ID_CURR',lsuffix='_AP', rsuffix='_CCB')
df_ccb=df_final.drop(['SK_ID_CURR','TARGET'],axis=1)
df_ccb.shape
%%time
train_X,test_X,train_y,test_y=train_test_split(df_ccb,train['TARGET'],random_state=200)
model =LGBMClassifier(learning_rate=0.05,n_estimators=200,n_jobs=-1,reg_alpha=0.1,min_split_gain=.1,verbose=-1)
model.fit(train_X,train_y)
score2=roc_auc_score(test_y,model.predict_proba(test_X)[:,1])
print(score2)
%%time
#FORWARD FEATURE SELCTION 
score=0
score1=0
score2=0
select_list=[]
col_list=list(df_ccb.columns)  
k=0


while True:
    score1=0
    score2=0
    temp_list=select_list
    for i,col in enumerate(col_list):
        if k==0:
            train_X,test_X,train_y,test_y=train_test_split(df_ccb[col],train['TARGET'],random_state=200)
            model =LGBMClassifier(learning_rate=0.05,n_estimators=200,n_jobs=-1,reg_alpha=0.1,min_split_gain=.1,verbose=-1)
            model.fit(np.array(train_X).reshape(-1,1),train_y)
            score2=roc_auc_score(test_y,model.predict_proba(np.array(test_X).reshape(-1,1))[:,1])
        else:
            temp_list.extend([col])
            train_X,test_X,train_y,test_y=train_test_split(df_ccb[temp_list],train['TARGET'],random_state=200)
            model =LGBMClassifier(learning_rate=0.05,n_estimators=200,n_jobs=-1,reg_alpha=0.1,min_split_gain=.1,verbose=-1)
            model.fit(train_X,train_y)
            score2=roc_auc_score(test_y,model.predict_proba(test_X)[:,1])
            temp_list.remove(col)
        if score1<=score2:
            score1=score2
            col1=col
#        print('dropped col',col,':',score2)
    k=k+1
    if score<=score1:
        score=score1
        print('select col',col1,':',score)
        select_list.extend([col1])
        col_list.remove(col1)
    else:
        print('Best score achieved')
        break
    
print(select_list)
print('best score:',score)