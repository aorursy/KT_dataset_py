# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import gc
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#cust.to_csv('train2.csv')
cust=pd.read_csv('../input/normalize/train2.csv',index_col=0,dtype={'fullVisitorId': 'str'}, nrows=None)
#cust_test.to_csv('test2.csv')
cust_test=pd.read_csv('../input/normalize/test2.csv',index_col=0,dtype={'fullVisitorId': 'str'}, nrows=None)

cust.head()
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

def plot_violin(auto_prices, cols, col_y = 'totals.transactionRevenue'):
    for col in cols:
        fig = plt.figure(figsize=(40,20))
        sns.set_style("whitegrid")
        sns.violinplot(col, col_y, data=auto_prices)
        plt.xlabel(col) # Set text for the x axis
        plt.ylabel(col_y)# Set text for y axis
        #plt.tick_params(labelsize=10)
        fig.show()

#dont visualize without transforming date and visitStartTime otherwise it will show irrelevant graph and takes quite a time. 
irrelevant=['totals.transactionRevenue','fullVisitorId','sessionId','visitId','date','visitStartTime']        
cat_cols=[]

for col in cust:
   if(col not in irrelevant):
    cat_cols.append(col)

plot_violin(cust, cat_cols)    
#1
cust['trafficSource.isTrueDirect'].fillna('none',inplace=True)
cust_test['trafficSource.isTrueDirect'].fillna('none',inplace=True)
cust['trafficSource.isTrueDirect']=[False if (i[1]['trafficSource.isTrueDirect']=='none' and i[1]['trafficSource.source']!='(direct)') else i[1]['trafficSource.isTrueDirect'] for i in cust.iterrows()]
cust_test['trafficSource.isTrueDirect']=[False if (i[1]['trafficSource.isTrueDirect']=='none' and i[1]['trafficSource.source']!='(direct)') else i[1]['trafficSource.isTrueDirect'] for i in cust_test.iterrows()]
cust['trafficSource.isTrueDirect']=[True if (i[1]['trafficSource.isTrueDirect']=='none') else i[1]['trafficSource.isTrueDirect'] for i in cust.iterrows()]
cust_test['trafficSource.isTrueDirect']=[True if (i[1]['trafficSource.isTrueDirect']=='none') else i[1]['trafficSource.isTrueDirect'] for i in cust_test.iterrows()]

#2
cust['date_new']=pd.to_datetime(cust['date'], format='%Y%m%d')
cust['date']=cust['date_new']
cust.drop(['date_new'],axis=1,inplace=True)

cust_test['date_new']=pd.to_datetime(cust_test['date'], format='%Y%m%d')
cust_test['date']=cust_test['date_new']
cust_test.drop(['date_new'],axis=1,inplace=True)

#3
cust['visitStartTime_new']=pd.to_datetime(cust['visitStartTime'], unit='s')
cust['visitStartTime']=cust['visitStartTime_new']
cust.drop(['visitStartTime_new'],axis=1,inplace=True)

cust_test['visitStartTime_new']=pd.to_datetime(cust_test['visitStartTime'], unit='s')
cust_test['visitStartTime']=cust_test['visitStartTime_new']
cust_test.drop(['visitStartTime_new'],axis=1,inplace=True)

#cust_test['trafficSource.isTrueDirect'].fillna(False,inplace=True)

#****************************************************************************************************
#4
cust['day']=cust['date'].dt.day
cust['DOW'] = cust['date'].dt.dayofweek
cust['month']=cust['date'].dt.month
cust['year']=cust['date'].dt.year
cust['time']=cust['visitStartTime'].dt.hour
cust.drop(['visitStartTime'],axis=1,inplace=True)

cust_test['day']=cust_test['date'].dt.day
cust_test['DOW'] = cust_test['date'].dt.dayofweek
cust_test['month']=cust_test['date'].dt.month
cust_test['year']=cust_test['date'].dt.year
cust_test['time']=cust_test['visitStartTime'].dt.hour
cust_test.drop(['visitStartTime'],axis=1,inplace=True)


#5
cust['totals.transactionRevenue'] = pd.to_numeric(cust['totals.transactionRevenue'])
cust['totals.transactionRevenue']=cust['totals.transactionRevenue'].apply(np.log)
cust['totals.transactionRevenue'].fillna(0,inplace=True)


#***************************************************************************************
#6
cust['trafficSource.adContent'].fillna('NA',inplace=True)
cust_test['trafficSource.adContent'].fillna('NA',inplace=True)

cust['trafficSource.adwordsClickInfo.gclId'].fillna('other',inplace=True)
cust_test['trafficSource.adwordsClickInfo.gclId'].fillna('other',inplace=True)

cust['trafficSource.adwordsClickInfo.slot'].fillna('other',inplace=True)
cust_test['trafficSource.adwordsClickInfo.slot'].fillna('other',inplace=True)

cust['trafficSource.adwordsClickInfo.isVideoAd'].fillna('True',inplace=True)
cust_test['trafficSource.adwordsClickInfo.isVideoAd'].fillna('True',inplace=True)

cust['trafficSource.adwordsClickInfo.page'].fillna('oth',inplace=True)
cust_test['trafficSource.adwordsClickInfo.page'].fillna('oth',inplace=True)

cust['trafficSource.adwordsClickInfo.adNetworkType'].fillna('other',inplace=True)
cust_test['trafficSource.adwordsClickInfo.adNetworkType'].fillna('other',inplace=True)

cust['trafficSource.keyword'].fillna('other',inplace=True)
cust_test['trafficSource.keyword'].fillna('other',inplace=True)

cust['trafficSource.referralPath'].fillna('other',inplace=True)
cust_test['trafficSource.referralPath'].fillna('other',inplace=True)

#cust.sort_values(by=['totals.transactionRevenue'],inplace=True)
#cust.drop_duplicates(subset = 'sessionId', keep = 'last', inplace = True)

cust['totals.newVisits'].fillna(0,inplace=True)
cust_test['totals.newVisits'].fillna(0,inplace=True)

cust['totals.bounces'].fillna(0,inplace=True)
cust_test['totals.bounces'].fillna(0,inplace=True)

#7
cust.drop(['trafficSource.campaignCode'],axis=1,inplace=True)

#*************************************************************************************************
#8
print("Removing Unique Value columns: ")
for col in cust:
    if(len(cust[col].unique())==1):
        print(col)
        cust.drop([col],axis=1,inplace=True)
        cust_test.drop([col],axis=1,inplace=True)

#9        
for col in cust:
    cust[col]=cust[col].apply(lambda x:str(x).lower())
    cust[col]=cust[col].apply(lambda x: str(x).replace(" ","_"))
for col in cust_test:    
    cust_test[col]=cust_test[col].apply(lambda x:str(x).lower())
    cust_test[col]=cust_test[col].apply(lambda x: str(x).replace(" ","_"))

        
#cust['totals.pageviews'].fillna('none',inplace=True)
#cust_test['totals.pageviews'].fillna('none',inplace=True)
#cust['totals.pageviews']=[i[1]['totals.hits'] if i[1]['totals.pageviews']=='none' else i[1]['totals.pageviews'] for i in cust.iterrows()]
#cust_test['totals.pageviews']=[i[1]['totals.hits'] if i[1]['totals.pageviews']=='none' else i[1]['totals.pageviews'] for i in cust_test.iterrows()]
#cust['totals.pageviews'] = pd.to_numeric(cust['totals.pageviews'])
#cust_test['totals.pageviews'] = pd.to_numeric(cust_test['totals.pageviews'])

gc.collect()
#**************************************************************************************************
cols=['device.browser','device.operatingSystem','trafficSource.source','trafficSource.adContent','trafficSource.keyword',
      'trafficSource.campaign','trafficSource.adwordsClickInfo.page','geoNetwork.region','geoNetwork.metro', 
      'geoNetwork.country','geoNetwork.city']

cust['totals.transactionRevenue']=cust['totals.transactionRevenue'].astype(float)

for col in cols:
    print("Working on : ")
    print(col)
    grouped=cust[[col,'totals.transactionRevenue']].groupby([col]).sum()
    temp=grouped[grouped['totals.transactionRevenue']==0]
    temp=temp['totals.transactionRevenue']
    temp=np.array(temp.keys())
    cust[col].replace(temp,'other',inplace=True)

    temp=grouped[grouped['totals.transactionRevenue']>0]
    temp=temp['totals.transactionRevenue']
    temp=np.array(temp.keys())
    temp2=pd.DataFrame()
    temp2['join']=[i if i not in temp else 'other' for i in cust_test[col]]
    temp2=set(temp2['join'])
    temp2.remove('other')
    temp3=[]
    for i in temp2:
        temp3.append(i)

    cust_test[col].replace(temp3,'other',inplace=True)

cat_cols = ['DOW','trafficSource.adContent','time','channelGrouping','day', 'year', 'month', 'visitNumber', 
            'device.browser','device.deviceCategory','device.isMobile','device.operatingSystem','geoNetwork.city',
            'geoNetwork.continent','geoNetwork.country','geoNetwork.subContinent','geoNetwork.metro','geoNetwork.region',
           'totals.hits','totals.pageviews','trafficSource.isTrueDirect','trafficSource.medium','trafficSource.source']

plot_violin(cust, cat_cols)
cust.to_csv('train3.csv')
cust_test.to_csv('test3.csv')
#cust=pd.read_csv('../input/simple/train3.csv',index_col=0,dtype={'fullVisitorId': 'str'}, nrows=None)
#cust_test=pd.read_csv('../input/simple/test3.csv',index_col=0,dtype={'fullVisitorId': 'str'}, nrows=None)