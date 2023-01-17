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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

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
#DATASET VIEW
path1="../input/ga-customer-revenue-prediction/"
path2='../input/gstore-prepared-dataset/'
data_files=['../input/gstore-prepared-dataset/prepared_train/prepared_train',
            '../input/gstore-prepared-dataset/prepared_test/prepared_test']
df_files=pd.DataFrame(data_files,columns=['File_Name'])
df_files['Size_in_MB']=df_files.File_Name.apply(lambda x:round(os.stat(x).st_size/(1024*1024),2))
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
df_train=pd.read_csv(data_files[0],dtype={'fullVisitorId':str})
print('reading test dataset...')
df_test=pd.read_csv(data_files[1],dtype={'fullVisitorId':str})
print('data reading complete')
#CHECKING TOP FIVE TRAIN OBSERVATIONS OR ROWS
df_train.head()
#FEATURE SUMMARY FOR TRAIN DATASET
feature_summary(df_train)
#CHECKING TOP 5 TEST OBSERVATIONS OR ROWS
df_test.head()
#FEATURE SUMMARY FOR TEST DATASET
feature_summary(df_test)
#CREATING COPY OF TRAIN DATA SET
train_cpy=df_train.copy()
#ADDING ANOTHER FEATURE revenue_status TO INDICATE PRESENCE/ABSENCE OF REVENUE FOR EACH OBSERVATION
df_train['revenue_status']=df_train.totals_transactionRevenue.apply(lambda x: 0 if x==0 else 1)
#VISUALIZATION FUNCTIONS
def revenue_transaction_visualization(df_t,col,col_name):
    df_con=df_t[[col,'totals_transactionRevenue','revenue_status']].groupby(col).aggregate(
        {'totals_transactionRevenue':['mean'],'revenue_status':['count']}).reset_index()
    df_con.columns=[col,'totals_transactionRevenue_mean','revenue_status_count']
    df=df_con.sort_values(by='totals_transactionRevenue_mean',ascending=False)[:20]
    df1=df_con.sort_values(by='revenue_status_count',ascending=False)[:20]
    display('SORTED BY Mean Revenue')
    display(df.style.format(formatter))


    plt.subplots(figsize=(20,5))
    plt.subplot(1,2,1)
    plt.title('REVENUE MEAN BY '+col_name,color='b',fontsize=12)
    plt.xlabel(col_name,color='b',fontsize=12)
    plt.ylabel('Mean Revenue',color='b',fontsize=12)
    plt.bar(range(len(df)),df.totals_transactionRevenue_mean,color='grey')
    plt.xticks(range(len(df)),df[col],rotation=90,fontsize=12)
    plt.yticks(fontsize=12)


    plt.subplot(1,2,2)
    plt.title('NUMBER OF TRANSACTIONS WITH REVENUE BY '+col_name,color='b',fontsize=12)
    plt.xlabel(col_name,color='b',fontsize=12)
    plt.ylabel('Count of Transactions with Revenue',color='b',fontsize=12)
    plt.bar(range(len(df)),df.revenue_status_count,color='orange')
    plt.yticks(fontsize=12)
    plt.xticks(range(len(df)),df[col],rotation=90,fontsize=12)
    plt.show()

    display('SORTED BY Count of Transactions with Revenue')
    display(df1.style.format(formatter))

    plt.subplots(figsize=(20,5))
    plt.subplot(1,2,1)
    plt.title('REVENUE MEAN BY '+col_name,color='b',fontsize=12)
    plt.xlabel(col_name,color='b',fontsize=12)
    plt.ylabel('Mean Revenue',color='b',fontsize=12)
    plt.bar(range(len(df1)),df1.totals_transactionRevenue_mean,color='grey')
    plt.xticks(range(len(df1)),df1[col],rotation=90,fontsize=12)
    plt.yticks(fontsize=12)


    plt.subplot(1,2,2)
    plt.title('NUMBER OF TRANSACTIONS WITH REVENUE BY '+col_name,color='b',fontsize=12)
    plt.xlabel(col_name,color='b',fontsize=12)
    plt.ylabel('Count of Transactions with Revenue',color='b',fontsize=12)
    plt.bar(range(len(df1)),df1.revenue_status_count,color='orange')
    plt.yticks(fontsize=12)
    plt.xticks(range(len(df1)),df1[col],rotation=90,fontsize=12)
    plt.show()
#UNDERSTANDING NUMBER OF TRANSACTIONS GENERATING REVENUE
pie_labels=['Revenue Generated -'+str(df_train['revenue_status'][df_train.revenue_status==1].count()),'No Revenue Generated-'+
            str(df_train['revenue_status'][df_train.revenue_status==0].count())]
pie_share=[df_train['revenue_status'][df_train.revenue_status==1].count()/df_train['revenue_status'].count(),
           df_train['revenue_status'][df_train.revenue_status==0].count()/df_train['revenue_status'].count()]
figureObject, axesObject = plt.subplots(figsize=(6,6))
pie_colors=('green','orange')
pie_explode=(.30,.15)
axesObject.pie(pie_share,labels=pie_labels,explode=pie_explode,autopct='%.2f%%',colors=pie_colors,startangle=45,shadow=True)
axesObject.axis('equal')
plt.title('Percentage of Transactions Generating Revenue and Not Generating Revenue',color='blue',fontsize=12)
plt.show()
#REVENUE GENERATED BY BROWSERS
df_browser=df_train[['device_browser','totals_transactionRevenue','revenue_status']].groupby(df_train.device_browser).aggregate({'totals_transactionRevenue':['mean'],
                                                                                                              'revenue_status':['count']}).reset_index()
df_browser.columns=['device_browser','totals_transactionRevenue_mean','revenue_status_count']
df=df_browser.sort_values(by='totals_transactionRevenue_mean',ascending=False)[df_browser.totals_transactionRevenue_mean>0]
formatter = {'totals_transactionRevenue_mean':'{:4.2f}'}
display(df.style.format(formatter))

plt.subplots(figsize=(20,5))
plt.subplot(1,2,1)
plt.title('REVENUE MEAN BY BROWSER',color='b',fontsize=12)
plt.xlabel('Browsers',color='b',fontsize=12)
plt.ylabel('Mean Revenue',color='b',fontsize=12)
plt.bar(range(len(df)),df.totals_transactionRevenue_mean,color='grey')
plt.xticks(range(len(df)),df.device_browser,rotation=90,fontsize=12)
plt.yticks(fontsize=12)


plt.subplot(1,2,2)
plt.title('NUMBER OF TRANSACTIONS WITH REVENUE BY BROWSER',color='b',fontsize=12)
plt.xlabel('Browsers',color='b',fontsize=12)
plt.ylabel('Count of Transactions with Revenue',color='b',fontsize=12)
plt.bar(range(len(df)),df.revenue_status_count,color='orange')
plt.xticks(range(len(df)),df.device_browser,rotation=90,fontsize=12)
plt.yticks(fontsize=12)
plt.show()
#REVENUE GENERATED BY OPERATING SYSTEM
df_OS=df_train[['device_operatingSystem','totals_transactionRevenue','revenue_status']].groupby(df_train.device_operatingSystem).aggregate({'totals_transactionRevenue':['mean'],
                                                                                                              'revenue_status':['count']}).reset_index()
df_OS.columns=['device_operatingSystem','totals_transactionRevenue_mean','revenue_status_count']
df=df_OS.sort_values(by='totals_transactionRevenue_mean',ascending=False)[df_OS.totals_transactionRevenue_mean>0]
display(df.style.format(formatter))

plt.subplots(figsize=(20,5))
plt.subplot(1,2,1)
plt.title('REVENUE MEAN BY OPERATING SYSTEM',color='b',fontsize=12)
plt.xlabel('Operating Systems',color='b',fontsize=12)
plt.ylabel('Mean Revenue',color='b',fontsize=12)
plt.bar(range(len(df)),df.totals_transactionRevenue_mean,color='grey')
plt.xticks(range(len(df)),df.device_operatingSystem,rotation=90,fontsize=12)
plt.yticks(fontsize=12)


plt.subplot(1,2,2)
plt.title('NUMBER OF TRANSACTIONS WITH REVENUE BY OPERATING SYSTEM',color='b',fontsize=12)
plt.xlabel('Operating Systems',color='b',fontsize=12)
plt.ylabel('Count of Transactions with Revenue',color='b',fontsize=12)
plt.bar(range(len(df)),df.revenue_status_count,color='orange')
plt.yticks(fontsize=12)
plt.xticks(range(len(df)),df.device_operatingSystem,rotation=90,fontsize=12)
plt.show()
#UNDERSTANDING REVENUE, TRANSACTIONS WITH REVENUE AND TOTAL TRANSACTIONS BY DATE
df=df_train[['date','revenue_status','totals_transactionRevenue']]
df['date']=pd.to_datetime(df['date'],format="%Y%m%d")
df=df.groupby('date').aggregate({'revenue_status':['sum','count'],'totals_transactionRevenue':['sum']}).reset_index()
df.columns=['date','transactions_withRevenue','total_transactions','total_revenue']


#PLOT FOR REVENUE BY DATES
plt.figure(figsize=(40,10))
plt.title("REVENUE BY DATES",fontsize=30,color='b')
plt.xlabel("Dates",fontsize=30,color='b')
plt.ylabel("Total Revenue",fontsize=30,color='b')
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.plot(df['date'],df['total_revenue'],color='orange',linewidth=3)
plt.hlines(xmin='2016-07-20',xmax='2017-08-10',y=np.mean(df['total_revenue']),color='r',linestyle='dashed')
plt.text('2017-01-10',np.mean(df['total_revenue'])+2,str('Mean: '+str(round(np.mean(df['total_revenue'])/(10**10),4))),
         color='r',fontsize=25)
plt.hlines(xmin='2016-07-20',xmax='2017-08-10',y=np.max(df['total_revenue']),color='r',linestyle='dashed')
plt.text('2017-01-10',np.max(df['total_revenue'])+2,str('Max: '+str(round(np.max(df['total_revenue'])/(10**10),4))),
         color='r',fontsize=25)
plt.show()

#PLOT FOR TRANSACTIONS WITH REVENUE BY DATES
plt.figure(figsize=(40,10))
plt.title("TRANSACTIONS WITH REVENUE BY DATES",fontsize=30,color='b')
plt.xlabel("Dates",fontsize=30,color='b')
plt.ylabel("Transactions with Revenue",fontsize=30,color='b')
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.plot(df['date'],df['transactions_withRevenue'],color='g',linewidth=3)
plt.hlines(xmin='2016-07-20',xmax='2017-08-10',y=np.mean(df['transactions_withRevenue']),color='r',linestyle='dashed')
plt.text('2017-01-10',np.mean(df['transactions_withRevenue'])+2,str('Mean: '+str(round(np.mean(df['transactions_withRevenue']),4))),
         color='r',fontsize=25)
plt.hlines(xmin='2016-07-20',xmax='2017-08-10',y=np.max(df['transactions_withRevenue']),color='r',linestyle='dashed')
plt.text('2017-01-10',np.max(df['transactions_withRevenue'])+1,str('Max: '+str(round(np.max(df['transactions_withRevenue']),4))),
         color='r',fontsize=25)
plt.show()


#PLOT FOR TOTAL TRANSACTIONS BY DATES
plt.figure(figsize=(40,10))
plt.title("TOTAL TRANSACTIONS BY DATES",fontsize=30,color='b')
plt.xlabel("Dates",fontsize=30,color='b')
plt.ylabel("Total Transactions",fontsize=30,color='b')
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.plot(df['date'],df['total_transactions'],color='r',linewidth=3)
plt.hlines(xmin='2016-07-20',xmax='2017-08-10',y=np.mean(df['total_transactions']),color='g',linestyle='dashed')
plt.text('2017-01-10',np.mean(df['total_transactions'])+2,str('Mean: '+str(round(np.mean(df['total_transactions']),0))),
         color='g',fontsize=25)
plt.hlines(xmin='2016-07-20',xmax='2017-08-10',y=np.max(df['total_transactions']),color='g',linestyle='dashed')
plt.text('2017-01-10',np.max(df['total_transactions'])+2,str('Max: '+str(round(np.max(df['total_transactions']),0))),
         color='g',fontsize=25)
plt.show()




#REVENUE GENERATED BY OPERATING SYSTEM
df_isM=df_train[['device_isMobile','totals_transactionRevenue','revenue_status']].groupby(df_train.device_isMobile).aggregate({'totals_transactionRevenue':['mean'],
                                                                                                              'revenue_status':['count']}).reset_index()
df_isM.columns=['device_isMobile','totals_transactionRevenue_mean','revenue_status_count']
df=df_isM.sort_values(by='totals_transactionRevenue_mean',ascending=False)[df_isM.totals_transactionRevenue_mean>0]
display(df.style.format(formatter))

plt.subplots(figsize=(20,5))
plt.subplot(1,2,1)
plt.title('REVENUE MEAN BY MOBILE',color='b',fontsize=12)
plt.xlabel('Mobile',color='b',fontsize=12)
plt.ylabel('Mean Revenue',color='b',fontsize=12)
plt.bar(range(len(df)),df.totals_transactionRevenue_mean,color='grey')
plt.xticks(range(len(df)),df.device_isMobile,rotation=90,fontsize=12)
plt.yticks(fontsize=12)


plt.subplot(1,2,2)
plt.title('NUMBER OF TRANSACTIONS WITH REVENUE BY MOBILE',color='b',fontsize=12)
plt.xlabel('Mobile',color='b',fontsize=12)
plt.ylabel('Count of Transactions with Revenue',color='b',fontsize=12)
plt.bar(range(len(df)),df.revenue_status_count,color='orange')
plt.yticks(fontsize=12)
plt.xticks(range(len(df)),df.device_isMobile,rotation=90,fontsize=12)
plt.show()
#REVENUE GENERATED BY COUNTRY
revenue_transaction_visualization(df_train,'geoNetwork_country','COUNTRY')
#REVENUE GENERATED BY CITY
revenue_transaction_visualization(df_train,'geoNetwork_city','CITY')
#REVENUE GENERATED BY NETWORKDOMAIN
revenue_transaction_visualization(df_train,'geoNetwork_networkDomain','NETWORKDOMAIN')
#REVENUE GENERATED BY SUBCONTINENT
revenue_transaction_visualization(df_train,'geoNetwork_subContinent','SUBCONTINENT')
#REVENUE GENERATED BY REGION
revenue_transaction_visualization(df_train,'geoNetwork_region','REGION')
#REVENUE GENERATED BY CONTINENT
revenue_transaction_visualization(df_train,'geoNetwork_continent','CONTINENT')
#REVENUE GENERATED BY METRO
revenue_transaction_visualization(df_train,'geoNetwork_metro','METRO')
#REVENUE GENERATED BY AdCONTENT with Train dataset
revenue_transaction_visualization(df_train,'trafficSource_adContent','AdCONTENT')
#ANALYZING AdContent in train and test dataset
print('Train data: Unique AdContent value count')
display(df_train['trafficSource_adContent'].groupby(df_train.trafficSource_adContent).count())
print('Test data: Unique AdContent value count')
display(df_test['trafficSource_adContent'].groupby(df_test.trafficSource_adContent).count())

#REVENUE GENERATED BY isTrueDirect
df_train['trafficSource_isTrueDirect'].replace({np.nan:'False'},inplace=True)
df_isTD=df_train[['trafficSource_isTrueDirect','totals_transactionRevenue','revenue_status']].groupby(df_train.trafficSource_isTrueDirect).aggregate({'totals_transactionRevenue':['mean'],
                                                                                                              'revenue_status':['count']}).reset_index()
df_isTD.columns=['trafficSource_isTrueDirect','totals_transactionRevenue_mean','revenue_status_count']
df=df_isTD.sort_values(by='totals_transactionRevenue_mean',ascending=False)[df_isTD.totals_transactionRevenue_mean>0]
display(df.style.format(formatter))

plt.subplots(figsize=(20,5))
plt.subplot(1,2,1)
plt.title('REVENUE MEAN BY isTrueDirect',color='b',fontsize=12)
plt.xlabel('isTrueDirect',color='b',fontsize=12)
plt.ylabel('Mean Revenue',color='b',fontsize=12)
plt.bar(range(len(df)),df.totals_transactionRevenue_mean,color='grey')
plt.xticks(range(len(df)),df.trafficSource_isTrueDirect,rotation=90,fontsize=12)
plt.yticks(fontsize=12)


plt.subplot(1,2,2)
plt.title('NUMBER OF TRANSACTIONS WITH REVENUE BY isTrueDirect',color='b',fontsize=12)
plt.xlabel('isTrueDirect',color='b',fontsize=12)
plt.ylabel('Count of Transactions with Revenue',color='b',fontsize=12)
plt.bar(range(len(df)),df.revenue_status_count,color='orange')
plt.yticks(fontsize=12)
plt.xticks(range(len(df)),df.trafficSource_isTrueDirect,rotation=90,fontsize=12)
plt.show()
#REVENUE GENERATED BY MEDIUM
revenue_transaction_visualization(df_train,'trafficSource_medium','MEDIUM')

#REVENUE GENERATED BY MEDIUM
revenue_transaction_visualization(df_train,'trafficSource_referralPath','referralPath')
#AS IDENTIFIED IN EDA COUNTRY Anguilla HAS A VERY HIGH VALUE SINGLE VISIT TRANSACTION
print('Dropping following observation:')
display(train_cpy[train_cpy.geoNetwork_country=='Anguilla'])

#DROPING THIS OUTLIER
train_cpy.drop(train_cpy[train_cpy.geoNetwork_country=='Anguilla'].index,axis=0,inplace=True)

#RESETING INDEX
train_cpy.reset_index(drop=True,inplace=True)

#COMBINING TRAIN AND TEST DATASET
df_combi=pd.concat([train_cpy,df_test],ignore_index=True)

#FEATURE SUMMARY FOR COMBINED DATASET
df_combi_fs=feature_summary(df_combi)
display(df_combi_fs)
#EXTRACTING DAY_OF_WEEK, HOUR, DAY, MONTH FROM DATE 
df_combi['date'] = pd.to_datetime(df_combi['visitStartTime'], unit='s')
df_combi['day_of_week'] = df_combi['date'].dt.dayofweek
df_combi['hour'] = df_combi['date'].dt.hour
df_combi['day'] = df_combi['date'].dt.day
df_combi['month'] = df_combi['date'].dt.month

#ADDING ANOTHER FEATURE revenue_status TO INDICATE PRESENCE/ABSENCE OF REVENUE FOR EACH OBSERVATION
df_combi['revenue_status']=df_combi.totals_transactionRevenue.apply(lambda x: 0 if x==0 else 1)
%%time
#CONVERTING ALL THE STRINGS IN CATEGORICAL FEATURES TO LOWER CASE
for col in df_combi.columns:
    if ((df_combi[col].dtype=='object') & (col!='fullVisitorId')):
        df_combi[col]=df_combi[col].apply(lambda x:str(x).lower())
        
#REPLACING STRING 'nan' WITH np.nan
df_combi.replace('nan',np.nan,inplace=True)
%%time
#CONVERTING CATEGORICAL FEATURES (LESS THAN 10 UNIQUE VALUES) TO DUMMIES
df_combi.drop(['device_isMobile'],axis=1,inplace=True)

cat_col=['channelGrouping','device_deviceCategory','tsadclick_slot','tsadclick_adNetworkType','tsadclick_isVideoAd','trafficSource_medium',
        'geoNetwork_continent']

for col in cat_col:
    df_combi[col]=df_combi[col].apply(lambda x: str(x).replace(" ","_"))
    
dummy=pd.DataFrame()

for col in cat_col:
    if col.find('_')!=-1:
        col_name=col.split('_')[1]
    else:
        col_name=col
    dummy=pd.concat([dummy,pd.get_dummies(df_combi[col],prefix=col_name)],axis=1)
    
print('Newly created dummy cols:',len(dummy.columns))
df_combi=pd.concat([df_combi,dummy],axis=1)

df_combi.drop(cat_col,axis=1,inplace=True)
df_combi.head()
#SOME BASIC DATA CLEANUP
df_combi['totals_newVisits'].fillna(0,inplace=True) 
df_combi['totals_bounces'].fillna(0,inplace=True)
df_combi['tsadclick_page'].fillna(0,inplace=True)
df_combi['trafficSource_isTrueDirect'].replace({np.nan:0,'true':1},inplace=True)
%%time
#GENERATING RANKS FOR CATEGORICAL FEATURES WITH UNIQUE VALUES GREATER THAN 10
#RANKS ARE GENERATED USING REVENUE PERCENTAGE
cols=[x for x in df_combi.columns if x not in ['fullVisitorId','sessionId','geoNetwork_networkDomain','tsadclick_gclId']]

for col in cols:
    if df_combi[col].dtype=='object':
        df_combi[col].fillna('others',inplace=True)
        col_list=['revenue_status','totals_transactionRevenue']
        col_list.append(col)
        print(col_list)
        df=df_combi[col_list].groupby(col).aggregate({col:['count'],'revenue_status':['sum'],'totals_transactionRevenue':['sum']}).reset_index()
        df.columns=[col,col+"_count",'revenue_status_sum','totals_transactionRevenue_sum']
        df['revenue_perc']=df['totals_transactionRevenue_sum']/df[col+"_count"]
        df['rank']=df['revenue_perc'].rank(ascending=1)
        
        replace_dict={}
        final_dict={}
        for k,col_val in enumerate(df[col].values):
            replace_dict[col_val]=df.iloc[k,5]
        final_dict[col]=replace_dict
        df_combi.replace(final_dict,inplace=True)
        del df,replace_dict,final_dict
        gc.collect()
#SPLITING COMBINED DATASET BACK TO TRAIN AND TEST SETS
train=df_combi[:len(train_cpy)]
test=df_combi[len(train_cpy):]
train.head()
test.head()
%%time
#REPLACING DOT WITH SPACE IN FEATURE geoNetwork_networkDomain
#THIS IS DONE TO TREAT IT AS TEXT AND WE WILL USE TfidfVectorizer TO EXTRACT FEATURES
train['geoNetwork_networkDomain'].fillna('unknown.unknown',inplace=True)
test['geoNetwork_networkDomain'].fillna('unknown.unknown',inplace=True)

train['geoNetwork_networkDomain']=train.geoNetwork_networkDomain.apply(lambda x: x.replace('.',' '))
test['geoNetwork_networkDomain']=test.geoNetwork_networkDomain.apply(lambda x: x.replace('.',' '))
%%time
#USING TfidfVectorizer TO EXTRACT FEATURES FROM geoNetwork_networkDomain
Tvect=TfidfVectorizer(ngram_range=(1,2),max_features=20000)
vect=Tvect.fit(train['geoNetwork_networkDomain'])
train_vect=vect.transform(train['geoNetwork_networkDomain'])
test_vect=vect.transform(test['geoNetwork_networkDomain'])

#DIMENSIONALITY REDUCTION ON EXTRACTED FEATURES
svd=TruncatedSVD(n_components=10)

#CREATING DATAFRAMES AFTER FEATURE EXTRACTION AND REDUCTION
vect_cols=['vect'+str(x) for x in range(1,11)]
df_train_vect=pd.DataFrame(svd.fit_transform(train_vect),columns=vect_cols)
df_test_vect=pd.DataFrame(svd.fit_transform(test_vect),columns=vect_cols)
#VIEW OF EXTRACTED AND REDUCED FEATURES
print(train_vect.shape,test_vect.shape)
display(df_train_vect.head())
display(df_test_vect.head())
print('Shape of vector dataframes:',df_train_vect.shape,df_test_vect.shape)
X=train.drop(['sessionId','visitId','date','geoNetwork_networkDomain','tsadclick_gclId'],axis=1)
X_test=test.drop(['sessionId','visitId','date','geoNetwork_networkDomain','tsadclick_gclId'],axis=1)   
#REORDERING INDEX FOR TEST DATASET
#THIS IS REQUIRED BEFORE CONCATENATING DATAFRAMES
X_test.reset_index(drop=True,inplace=True)
X_test.head()
#CONCATENATING WITH TEXT FEATURES 
X=pd.concat([X,df_train_vect],axis=1)
X_test=pd.concat([X_test,df_test_vect],axis=1)
#VIEW OF TRAIN AND TEST DATASET SHAPE
print('Before creating aggregated features')
print('Train shape:',X.shape,' Test shape:',X_test.shape)
%%time
agg_func={}
agg_col=['fullVisitorId']
for col in [x for x in X.columns if x not in ['fullVisitorId']]:
    if col=='totals_transactionRevenue':
        agg_func[col]=['sum']
        agg_col.append(str(col)+'_sum')
    elif col=='revenue_status':
        agg_func[col]=['sum']
        agg_col.append(str(col)+'_sum')
    else:
        agg_func[col]=['sum','max','min','mean','var','std']
        agg_col.append(str(col)+'_sum')
        agg_col.append(str(col)+'_max')
        agg_col.append(str(col)+'_min')
        agg_col.append(str(col)+'_mean')
        agg_col.append(str(col)+'_var')
        agg_col.append(str(col)+'_std')
    
X=X.groupby(X.fullVisitorId).aggregate(agg_func).reset_index()
X.columns=agg_col

X_test=X_test.groupby(X_test.fullVisitorId).aggregate(agg_func).reset_index()
X_test.columns=agg_col
%%time

#CREATING y_dummy FOR USING STRATIFIED KFOLD
y_dummy=X.revenue_status_sum.apply(lambda x: 0 if x==0 else 1)

#TARGET FEATURE CONVERTED TO NATURAL LOG
# y=pd.Series(X['totals_transactionRevenue_sum'])
y=X.totals_transactionRevenue_sum.apply(lambda x: np.log1p(x))

#PEPARING DATA FOR TRAINING LGBM MODEL
X=X.drop(['totals_transactionRevenue_sum','fullVisitorId','revenue_status_sum'],axis=1)

#FINAL DATAFRAME FOR SUBMISSION
col=['fullVisitorId','totals_transactionRevenue_sum']
final=X_test[col] 
final.columns=['fullVisitorId','PredictedLogRevenue']

#FINAL TEST FEATURES USED FOR PREDICTING SUBMISSION
X_test=X_test.drop(['fullVisitorId','totals_transactionRevenue_sum','revenue_status_sum'],axis=1)
print('After creating aggregated features')
print('Train shape:',X.shape,' Test shape:',X_test.shape)
%%time
#LGBMRegressor. THIS REQUIRES FURTHER PARAMETER TUNINIG
model=LGBMRegressor(boosting_type='gbdt',num_leaves=31,max_depth=-1,learning_rate=0.01,n_estimators=1000,max_bin=255,subsample_for_bin=50000,
              objective=None,min_split_gain=0,min_child_weight=3,min_child_samples=10,subsample=1,subsample_freq=1,colsample_bytree=1,
              reg_alpha=0.1,reg_lambda=0,seed=17,silent=False,nthread=-1,n_jobs=-1)


k=1
splits=5
avg_score=0


skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=200)
print('\nStarting KFold iterations...')
for train_index,test_index in skf.split(X,y_dummy):
    df_X=X.iloc[train_index,:]
    df_y=y.iloc[train_index]
    val_X=X.iloc[test_index,:]
    val_y=y.iloc[test_index]

    model.fit(df_X,df_y)

    preds_x=pd.Series(model.predict(val_X))
    acc=rsme(val_y,preds_x)
    print('Iteration:',k,'  rmse:',acc)
    
    if k==1:
        score=acc
        model1=model
        preds=pd.Series(model.predict(X_test))
        
    else:
        preds1=pd.Series(model.predict(X_test))
        preds=preds+preds1
        if score>acc:
            score=acc
            model1=model
    avg_score=avg_score+acc        
    k=k+1
print('\n Best score:',score,' Avg Score:',avg_score/splits)
preds=preds/splits
#PREPARING PREDICTED DATA
final['PredictedLogRevenue']=pd.Series(preds)
#GROUPING PREDICTED DATA ON fullVisitorId
final = final.groupby("fullVisitorId")["PredictedLogRevenue"].sum().reset_index()
final.columns = ["fullVisitorId", "PredictedLogRevenue"]

final
#READING SUMISSION FILE
submission=pd.read_csv(path1+'sample_submission.csv')

#CREATING JOIN BETWEEN PREDICTED DATA WITH SUBMISSION FILE
submission=submission.join(final.set_index('fullVisitorId'),on='fullVisitorId',lsuffix='_sub')
submission.drop('PredictedLogRevenue_sub',axis=1,inplace=True)

#HANDLING NaN IN CASE OF MISSING fullVisitorId
submission.fillna(0,inplace=True)

#SUBMITING FILE
submission.to_csv('LGBM_submission.csv',index=False)