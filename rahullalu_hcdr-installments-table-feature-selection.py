
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
import warnings
%matplotlib inline
warnings.filterwarnings('ignore')
gc.enable()

#DATASET VIEW
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

#FUNCTION USED FOR GROUPING DATA 
def cnt_unique(df):
    return(len(df.unique()))
%%time
#READING INSTALLMENTS_PAYMENT DATA
inst_pay=pd.read_csv(path1+'installments_payments.csv')
print('installments_payments set reading complete...')
#TAKING ABSOLUTE VALUE FOR DAYS_ FEATURES
for col in inst_pay.columns:
    if 'DAYS_' in col:
        inst_pay[col]=inst_pay[col].abs()
%%time
#MANUAL FEATURE ENGINEERING

inst_pay['CALC_DAYS_LATE_PAYMENT']=inst_pay['DAYS_ENTRY_PAYMENT']-inst_pay['DAYS_INSTALMENT']
inst_pay['CALC_PERC_LESS_PAYMENT']=inst_pay['AMT_PAYMENT']/inst_pay['AMT_INSTALMENT']
inst_pay['CALC_PERC_LESS_PAYMENT'].replace(np.inf,0,inplace=True)
inst_pay['CALC_DIFF_INSTALMENT']=inst_pay['AMT_INSTALMENT']-inst_pay['AMT_PAYMENT']
inst_pay['CALC_PERC_DIFF_INSTALMENT']=np.abs(inst_pay['CALC_DIFF_INSTALMENT'])/inst_pay['AMT_INSTALMENT']
inst_pay['CALC_PERC_DIFF_INSTALMENT'].replace(np.inf,0,inplace=True)
inst_pay['CALC_INSTAL_PAID_LATE'] = (inst_pay['CALC_DAYS_LATE_PAYMENT'] > 0).astype(int)
inst_pay['CALC_OVERPAID']= (inst_pay['CALC_DIFF_INSTALMENT'] < 0).astype(int)
#FEATURE SUMMARY
inst_pay_fs=feature_summary(inst_pay)
inst_pay_fs
#DATA VIEW FOR SINGLE SK_ID_CURR
inst_pay[(inst_pay.SK_ID_CURR==100001)].sort_values('SK_ID_PREV')
#DEFINING AGGREGATION RULES AND CREATING LIST OF NEW FEATURES
inst_pay_cols=[x for x in list(inst_pay.columns) if x not in ['SK_ID_CURR','SK_ID_PREV']]
inst_pay_agg={}
inst_pay_name=['SK_ID_CURR','SK_ID_PREV']
for col in inst_pay_cols:
    if 'NUM_INSTALMENT_VERSION'==col:
        inst_pay_agg[col]=[cnt_unique]#CUSTOM FUNCTION FOR COUNTING UNIQUE INSTALMENT_VERSION
        inst_pay_name.append(col+'_'+'unique')
    elif 'NUM_INSTALMENT_NUMBER'==col:
        inst_pay_agg[col]=['max','count']
        inst_pay_name.append(col+'_'+'max')
        inst_pay_name.append(col+'_'+'count')
    elif 'AMT_' in col:
        inst_pay_agg[col]=['sum','mean','max','min','var','std']
        inst_pay_name.append(col+'_'+'sum')
        inst_pay_name.append(col+'_'+'mean')
        inst_pay_name.append(col+'_'+'max')
        inst_pay_name.append(col+'_'+'min')
        inst_pay_name.append(col+'_'+'var')
        inst_pay_name.append(col+'_'+'std')
    elif 'CALC_DAYS_' in col:
        inst_pay_agg[col]=['sum']
        inst_pay_name.append(col+'_'+'sum')
    elif 'DAYS_' in col:
        inst_pay_agg[col]=['sum','max','min']
        inst_pay_name.append(col+'_'+'sum')
        inst_pay_name.append(col+'_'+'max')
        inst_pay_name.append(col+'_'+'min')
    else:
        inst_pay_agg[col]=['mean']
        inst_pay_name.append(col+'_'+'mean')

%%time
#AGGREGATING DATA ON SK_ID_CURR,SK_ID_PREV USING RULES CREATED IN PREVIOUS STEP
inst_pay_f=inst_pay.groupby(['SK_ID_CURR','SK_ID_PREV']).aggregate(inst_pay_agg)
inst_pay_f.reset_index(inplace=True)
inst_pay_f.columns=inst_pay_name
inst_pay_f.head()
#NUMBER OF MISSED INATALLMENTS
inst_pay_f['CALC_NUM_INSTALMENT_MISSED']=inst_pay_f['NUM_INSTALMENT_NUMBER_max']-inst_pay_f['NUM_INSTALMENT_NUMBER_count']
#DEFINING RULES FOR SECOND AGGREGATION ON SK_ID_CURR
inst_pay_cols=[x for x in list(inst_pay_f.columns) if x not in ['SK_ID_PREV']]
inst_pay_agg={}
inst_pay_name=['SK_ID_CURR']
for col in inst_pay_cols:
    if 'SK_ID_CURR'==col:
        inst_pay_agg[col]=['count']
        inst_pay_name.append('SK_ID_PREV_count')
    elif '_unique' in col:
        inst_pay_agg[col]=['sum']
        inst_pay_name.append(col)
    elif '_mean' in col:
        inst_pay_agg[col]=['mean']
        inst_pay_name.append(col)
    elif '_max' in col:
        inst_pay_agg[col]=['max']
        inst_pay_name.append(col)
    elif '_min' in col:
        inst_pay_agg[col]=['min']
        inst_pay_name.append(col)
    elif '_count' in col:
        inst_pay_agg[col]=['sum']
        inst_pay_name.append(col)
    else:
        inst_pay_agg[col]=['sum']
        inst_pay_name.append(col)
%%time
#AGGREGATING DATA ON SK_ID_CURR
inst_pay_f.drop(['SK_ID_PREV'],axis=1,inplace=True)
inst_pay_fg=inst_pay_f.groupby(['SK_ID_CURR']).aggregate(inst_pay_agg)
inst_pay_fg.reset_index(inplace=True)
inst_pay_fg.columns=inst_pay_name
inst_pay_fg.head(10)
#INSTALMENT_VERSION CHANGE
inst_pay_fg['CALC_CNT_INSTALMENT_VERSION_CHG']=inst_pay_fg['NUM_INSTALMENT_VERSION_unique']-inst_pay_fg['SK_ID_PREV_count']
del inst_pay,inst_pay_f
gc.collect()
# %%time
# inst_pay_fg['inst_mean'] = inst_pay_fg.mean(axis=1)
# print('the_mean calculated...')
# inst_pay_fg['inst_sum'] =inst_pay_fg.sum(axis=1)
# print('the_sum calculated...')
# inst_pay_fg['inst_std'] = inst_pay_fg.std(axis=1)
# print('the_std calculated...')
# inst_pay_fg['inst_kur'] = inst_pay_fg.kurtosis(axis=1)
# print('the_kur calculated...')
#READING APPLICATION TRAIN DATA (ONLY 'SK_ID_CURR','TARGET' FIELDS)
#JOINING WITH INSTALLEMENT DATA
train=pd.read_csv(path1+'application_train.csv',usecols=['SK_ID_CURR','TARGET'])
df_final=train.join(inst_pay_fg.set_index('SK_ID_CURR'),on='SK_ID_CURR',lsuffix='_AP', rsuffix='_INP')
df_final.head()
df_final.shape
%%time
#BASELINE SCORE
train_X,test_X,train_y,test_y=train_test_split(df_final.drop(['SK_ID_CURR','TARGET'],axis=1),df_final['TARGET'],random_state=200)
model=LGBMClassifier(learning_rate=0.05,objective='binary',n_estimators=200,n_jobs=-1,reg_alpha=0.1,min_split_gain=.1,verbose=-1)
model.fit(train_X,train_y)
score2=roc_auc_score(test_y,model.predict_proba(test_X)[:,1])
print('BASELINE SCORE:',score2)
%%time
#FORWARD FEATURE SELCTION 
score=0
score1=0
score2=0
select_list=[]
col_list=[x for x in list(df_final.columns) if x not in ['SK_ID_CURR','TARGET']]  
k=0


while True:
    score1=0
    score2=0
    temp_list=select_list
    for i,col in enumerate(col_list):
        if k==0:
            train_X,test_X,train_y,test_y=train_test_split(df_final[col],df_final['TARGET'],random_state=200)
            model =LGBMClassifier(learning_rate=0.05,n_estimators=200,n_jobs=-1,reg_alpha=0.1,min_split_gain=.1,verbose=-1)
            model.fit(np.array(train_X).reshape(-1,1),train_y)
            score2=roc_auc_score(test_y,model.predict_proba(np.array(test_X).reshape(-1,1))[:,1])
        else:
            temp_list.extend([col])
            train_X,test_X,train_y,test_y=train_test_split(df_final[temp_list],df_final['TARGET'],random_state=200)
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
%%time
#FEATURE EXCLUSION
score=0
score1=0
score2=0
drop_list=[]
col_list=[x for x in list(df_final.columns) if x not in ['SK_ID_CURR','TARGET']]


while True:
    score1=0
    score2=0
    for i,col in enumerate(col_list):
        col_list.remove(col)
        train_X,test_X,train_y,test_y=train_test_split(df_final[col_list],df_final['TARGET'],random_state=200)
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