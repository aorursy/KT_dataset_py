#IMPORTING REQUIRED LIBRARIES
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import seaborn as sns
import math

from lightgbm.sklearn import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold,StratifiedKFold

import gc
gc.enable()


import warnings
warnings.filterwarnings("ignore")
%matplotlib inline


#DATASET VIEW
path1="../input/"
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

#FUNCTION FOR READING DICTIONARY ITEMS AND HANDLING KEYERROR
def get_val(x,col):
    try:
        y=x[col]
    except:
        y=np.nan
    return(y)

#FUNCTION FOR CALCULATING RSME
def rsme(y,pred):
    return(mean_squared_error(y,pred)**0.5)
%%time
#READING TRAINING AND TEST DATASET
print('reading train dataset...')
df_train=pd.read_csv(path1+'train.csv',dtype={'fullVisitorId':str,'date':str,'sessionId':str,'visitId':str,'visitStartTime':str})
print('reading test dataset...')
df_test=pd.read_csv(path1+'test.csv',dtype={'fullVisitorId':str,'date':str,'sessionId':str,'visitId':str,'visitStartTime':str})
print('data reading complete')
#CHECKING TOP FIVE TRAIN OBSERVATIONS OR ROWS
df_train.head()
#FEATURE SUMMARY FOR TRAIN DATASET
feature_summary(df_train)
#CHECKING TOP 5 TEST OBSERVATIONS OR ROWS
df_test.head()
#FEATURE SUMMARY FOR TEST DATASET
feature_summary(df_test)
#COMBINING TRAIN AND TEST DATASET
df_combi=pd.concat([df_train,df_test],ignore_index=True)
#STORING NUMBER OF ROWS IN TRAIN DATASET
#AND DELETING BOTH TRAIN & TEST DATAFRAMES
#NUMBER OF ROWS IN TRAIN DATASET WILL HELP IN SPLITING COMBINED DATASET INTO TRAIN & TEST 
train_len=len(df_train)
del df_train,df_test
gc.collect()
#FEATURE SUMMARY FOR COMBINED DATASET
df_combi_fs=feature_summary(df_combi)
df_combi_fs
%%time
#REPLACING false WITH False AND true WITH True
#AS false AND true HAVE NO MEANING IN PYTHON
#ALSO REPLACING 'not available in demo dataset' WITH 'NaN'
j_fields=['device','geoNetwork','totals','trafficSource']

for col in j_fields:
    df_combi[col].replace('false','False',inplace=True,regex=True)
    df_combi[col].replace('true','True',inplace=True,regex=True)
    df_combi[col].replace('not available in demo dataset','NaN',inplace=True,regex=True)
    df_combi[col].replace('(not set)','NaN',inplace=True,regex=True)
    df_combi[col]=df_combi[col].apply(lambda x: eval(x))
%%time
#EXTRACTING FEATURE NAMES UNDER EACH JSON FEATURES
device_cols=list(df_combi['device'][0].keys())
geoNetwork_cols=list(df_combi['geoNetwork'][0].keys())
totals_cols=list(df_combi['totals'][0].keys())
trafficSource_cols=list(df_combi['trafficSource'][0].keys())

for i in range(1,len(df_combi)):
    device_cols=list(set(device_cols) | set(list(df_combi['device'][i].keys())))
    geoNetwork_cols=list(set(geoNetwork_cols) | set(list(df_combi['geoNetwork'][i].keys())))
    totals_cols=list(set(totals_cols) | set(list(df_combi['totals'][i].keys())))
    trafficSource_cols=list(set(trafficSource_cols) | set(list(df_combi['trafficSource'][i].keys())))
device_cols
geoNetwork_cols
totals_cols
trafficSource_cols
%%time
#CONVERTING DEVICE JSON FEATURE INTO INDIVIDUAL SUB FEATURES UNDER IT
for jlist in device_cols:
    col_name='device_'+jlist
    df_combi[col_name]=df_combi['device'].apply(lambda x: get_val(x,jlist))

#DROPPING device JSON FEATURE
df_combi.drop('device',axis=1,inplace=True)
display(df_combi.head())    
%%time
#CONVERTING geoNetwork JSON FEATURE INTO INDIVIDUAL SUB FEATURES UNDER IT
for jlist in geoNetwork_cols:
    col_name='geoNetwork_'+jlist
    df_combi[col_name]=df_combi['geoNetwork'].apply(lambda x: get_val(x,jlist))
    
#DROPPING geoNetwork JSON FEATURE
df_combi.drop('geoNetwork',axis=1,inplace=True)
display(df_combi.head())
%%time
#CONVERTING trafficSource JSON FEATURE INTO INDIVIDUAL SUB FEATURES UNDER IT
for jlist in trafficSource_cols:
    col_name='trafficSource_'+jlist
    df_combi[col_name]=df_combi['trafficSource'].apply(lambda x: get_val(x,jlist)) 


#DROPPING trafficSource JSON FEATURE
df_combi.drop('trafficSource',axis=1,inplace=True)
display(df_combi.head())
%%time
#CONVERTING totals JSON FEATURE INTO INDIVIDUAL SUB FEATURES UNDER IT
for jlist in totals_cols:
    col_name='totals_'+jlist
    df_combi[col_name]=df_combi['totals'].apply(lambda x: get_val(x,jlist)) 
    
#DROPPING totals JSON FEATURE
df_combi.drop('totals',axis=1,inplace=True)
display(df_combi.head())
#DELETING COMBINED FEATURE SUMMARY DATAFRAME
del df_combi_fs
gc.collect()
#CONVERTING ALL STRING 'NaN' VALUES TO np.nan
df_combi.replace('NaN',np.nan,inplace=True)

#CONVERTING VALUES IN FEATURE trafficSource_adwordsClickInfo AS THEY ARE AGAIN Json FEATURES
#WE HAVE TO HANDLE trafficSource_adwordsClickInfo AS OTHER Json FEATURES
df_combi['trafficSource_adwordsClickInfo']=df_combi.trafficSource_adwordsClickInfo.apply(lambda x: str(x))
#FEATURE SUMMARY FOR COMBINED DATASET
df_combi_fs=feature_summary(df_combi)
df_combi_fs
#IDENFIYING ALL FEATURES WITH SAME VALUES IN ALL OBSERVATIONS
col_drop=list(df_combi_fs.index[df_combi_fs.Unique_Count==1])
print('No of Columns to be removed:',len(col_drop))
print(col_drop)

#DROPPING ALL FEATURES WITH SAME VALUES IN ALL OBSERVATIONS
df_combi.drop(col_drop,axis=1,inplace=True)
gc.collect()
#FEATURE SUMMARY FOR COMBINED DATASET
df_combi_fs=feature_summary(df_combi)
df_combi_fs
#DROPPING trafficSource_campaignCode, AS IT HAS ONLY 2 OBSERVATIONS WITH SOME VALUES
df_combi.drop('trafficSource_campaignCode',axis=1,inplace=True)

#GARBAGE COLLECTION
gc.collect()
%%time
#EXTRACTING FEATURES NAMES FROM Json FEATURE trafficSource_adwordsClickInfo
df_combi['trafficSource_adwordsClickInfo']=df_combi.trafficSource_adwordsClickInfo.apply(lambda x: eval(x))
tsadclick_cols=list(df_combi['trafficSource_adwordsClickInfo'][0].keys())

for i in range(1,len(df_combi)):
    tsadclick_cols=list(set(tsadclick_cols) | set(list(df_combi['trafficSource_adwordsClickInfo'][i].keys())))

print('FEATURES UNDER trafficSource_adwordsClickInfo Json FEATURE\n',tsadclick_cols)

#EXTRACTING FEATURES FROM Json FEATURE trafficSource_adwordsClickInfo
for jlist in tsadclick_cols:
    col_name='tsadclick_'+jlist
    df_combi[col_name]=df_combi['trafficSource_adwordsClickInfo'].apply(lambda x: get_val(x,jlist))
    df_combi[col_name]=df_combi[col_name].apply(lambda x: str(x))
    
#DROPPING Json FEATURE trafficSource_adwordsClickInfo
df_combi.drop('trafficSource_adwordsClickInfo',axis=1,inplace=True)
#DROPPING IRRELEVANT FEATURES tsadclick_targetingCriteria AND tsadclick_criteriaParameters
df_combi.drop(['tsadclick_targetingCriteria','tsadclick_criteriaParameters'],axis=1,inplace=True)

#GARBAGE COLLECTION
gc.collect()
# %%time
# #HANDLING SURROGATES IN TRAIN DATASET
# #THERE ARE NO SUCH ENTRIES IN TEST SET
# #SURROGATES: VALUES CAN'T BE ENCODED/DECODED USING utf-8
# for col in df_combi.columns:
#     if ((df_combi[col].dtype==object) & (col!='fullVisitorId')):
#         df_combi[col]=df_combi[col].apply(lambda x: np.nan if x==np.nan else str(x).encode('utf-8', 'replace').decode('utf-8'))
%%time
#REPLACING 'nan' and '(not provided)' WITH np.nan
df_combi.replace("nan",np.nan,inplace=True)
df_combi.replace("(not provided)",np.nan,inplace=True)
df_combi.replace("(NaN)",np.nan,inplace=True)
df_combi['totals_transactionRevenue'].replace(np.nan,0,inplace=True)

#CONVERTING FEATURES TO CORRECT DATATYPE
df_combi['totals_pageviews']=df_combi['totals_pageviews'].astype('float')
df_combi['totals_transactionRevenue']=df_combi['totals_transactionRevenue'].astype('float')
df_combi['totals_hits']=df_combi['totals_hits'].astype('float')
df_combi['tsadclick_page']=df_combi['tsadclick_page'].astype('float')
#FEATURE SUMMARY FOR COMBINED DATASET
df_combi_fs=feature_summary(df_combi)
df_combi_fs
#SPLITING COMBINED DATASET BACK TO TRAIN AND TEST SETS
train=df_combi[:train_len]
test=df_combi[train_len:]
#FEATURE SUMMARY TRAIN SET
feature_summary(train)
#FEATURE SUMMARY TEST SET
feature_summary(test)
for col in train.columns:
    if ((train[col].dtype==object) & (col!='fullVisitorId')):
        train[col]=train[col].apply(lambda x: np.nan if x==np.nan else str(x).encode('utf-8', 'replace').decode('utf-8'))
#WRITING BACK PREPARED TRAIN AND TEST DATASET
train.to_csv('prepared_train.gz', compression='gzip',index=False)
test.to_csv('prepared_test.gz', compression='gzip',index=False)