# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#For Importing Libs
import json
from pandas.io.json import json_normalize
import random
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.plotly as py
import plotly.graph_objs as go
train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')
# for each in list(train_data):
#     print("Column: ",each,"Dtype:",train_data[each].dtype)
train_data.dtypes
columns = ['device', 'geoNetwork', 'totals', 'trafficSource'] # Columns that have json format

dir_path = "../input/" # you can change to your local 

# p is a fractional number to skiprows and read just a random sample of the our dataset. 
p = 0.4 # *** In this case we will use 50% of data set *** #

#Code to transform the json format columns in table
def json_read(df):
    #joining the [ path + df received]
    data_frame = dir_path + df
    
    #Importing the dataset
    df = pd.read_csv(data_frame, 
                     converters={column: json.loads for column in columns}, # loading the json columns properly
                     dtype={'fullVisitorId': 'str'}, # transforming this column to string
                     skiprows=lambda i: i>0 and random.random() > p)# Number of rows that will be imported randomly
    
    for column in columns: #loop to finally transform the columns in data frame
        #It will normalize and set the json to a table
        column_as_df = json_normalize(df[column]) 
        # here will be set the name using the category and subcategory of json columns
        column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns] 
        # after extracting the values, let drop the original columns
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
        
    # Printing the shape of dataframes that was imported     
    print(f"Loaded {os.path.basename(data_frame)}. Shape: {df.shape}")
    return df # returning the df after importing and transforming
%%time 
# %%time is used to calculate the timing of code chunk execution #
df_train = json_read("train.csv") 
# The same to test dataset
df_test = json_read("test.csv")
df_train.head()
df_train.channelGrouping.value_counts()
df_train.channelGrouping.value_counts().plot(kind="bar")
#dataset date range
print(min(df_train["date"]),"-",max(df_train["date"]))
#notice that
# sessionId = fullVisitorId + "_" + visitId
#vistors start time
timedf = pd.DataFrame()
timedf['visitStartTime'] = pd.to_datetime(df_test['visitStartTime'],unit='s')
timedf['timeHH'] = [str(x).split(' ')[1].split(':')[0] for x in timedf['visitStartTime']] # spliting HH from time stamp
print(timedf.head(2))
timedf['timeHH'].value_counts().sort_values(ascending=False).plot('bar',figsize=[12,8]) # maximum visit is during evening 17 - 18, count holds the value from time x:0:0 to x:59:59 
#top 10 device browsers
print(df_train['device.browser'].value_counts().sort_values(ascending=False)[0:10])
df_train['device.browser'].value_counts().sort_values(ascending=False)[0:10].plot('bar',figsize=[12,8])
df_train["device.deviceCategory"].value_counts().plot("bar",figsize=[10,6])
DcDb = df_train[['visitId','device.deviceCategory','device.browser']].groupby(['device.browser','device.deviceCategory']).count().reset_index().sort_values(by = 'visitId',ascending=0)[0:20]
DcDb.columns = ['device.browser','device.deviceCategory','BrowserUsageCount']
DcDb.head(5)
with sns.axes_style(style='ticks'):
    g = sns.factorplot("device.deviceCategory", "BrowserUsageCount", "device.browser", data=DcDb, kind="bar",size=8,aspect=1.5)
    g.set_axis_labels("device.deviceCategory", "BrowserUsageCount")
#Device category and device mobile type count
df_train[df_train['device.deviceCategory'] == 'desktop']['device.isMobile'].value_counts()
DcDm = df_train[['fullVisitorId','device.deviceCategory','device.isMobile']].groupby(['device.deviceCategory','device.isMobile']).count().reset_index()
DcDm.columns = ['device.deviceCategory','device.isMobile','VisitorCount']
DcDm
ndf = df_train[['visitId','device.operatingSystem','device.browser']].groupby(['device.browser','device.operatingSystem']).count().reset_index().sort_values(by = 'visitId',ascending=0)[0:20]
ndf.columns = ['device.browser','device.operatingSystem','OSUsageCount']
ndf.head()
with sns.axes_style(style='ticks'):
    g = sns.factorplot("device.operatingSystem", "OSUsageCount", "device.browser", data=ndf, kind="bar",size=6,aspect=2)
    g.set_axis_labels("device.operatingSystem", "OSUsageCount")
#top 15 countries
df_train['geoNetwork.country'].value_counts()[0:15].plot('bar',figsize=[12,6])
#top 15 cities
df_train.loc[df_train['geoNetwork.city'] == 'not available in demo dataset']['geoNetwork.city'] = np.nan #replacing "not available in demo dataset" with nan
#top 10 countries
df_train['geoNetwork.city'].value_counts()[0:15].plot('bar',figsize=[12,6])
df_train['geoNetwork.continent'].value_counts()
#top conuntry in every continent
top_country_per_continent = df_train.groupby(['geoNetwork.continent','geoNetwork.country']).size().reset_index().sort_values(0,ascending = False).groupby(['geoNetwork.continent']).head(2)
top_country_per_continent.columns = ['geoNetwork.continent','geoNetwork.country','count']

g = sns.factorplot(x="geoNetwork.country", y ='count', hue='geoNetwork.continent',data=top_country_per_continent,kind="bar", size=6, aspect=1.5)
g.set_xticklabels(rotation=90, ha="right")
plt.show()
#top subcontinent 
dfsubcont = df_train['geoNetwork.subContinent'].value_counts().reset_index()[0:15]
dfsubcont.columns = ['geoNetwork.subContinent','count']
g = sns.factorplot(x="geoNetwork.subContinent", y ='count',data=dfsubcont,kind="bar", size=6, aspect=1.5)
g.set_xticklabels(rotation=90, ha="right")
plt.show()
totaldf = df_train[['geoNetwork.country','totals.hits']].groupby('geoNetwork.country').count().reset_index().sort_values(by='totals.hits',ascending=False).reset_index().drop(['index'],axis = 1)
plt.figure(figsize=(8,8))
p = plt.pie(totaldf['totals.hits'][0:10], labels=totaldf['geoNetwork.country'][0:10],autopct='%1.0f%%',pctdistance=0.6, labeldistance=1.1)
#US vs rest of the world
label = ['US','Others']
value =[totaldf.loc[0]['totals.hits'],totaldf.loc[1:len(totaldf)][['totals.hits']].sum()]
plt.figure(figsize=(8,8))
p1 = plt.pie(value, labels=label,autopct='%1.0f%%',pctdistance=0.6, labeldistance=1.1)
print("US alone has 40 percent of all hits")
#US vs rest of the world
label = ['US + India','Others']
value =[totaldf.loc[0:2][['totals.hits']].sum(),totaldf.loc[2:len(totaldf)][['totals.hits']].sum()]
plt.figure(figsize=(8,8))
p1 = plt.pie(value, labels=label,autopct='%1.1f%%',pctdistance=0.6, labeldistance=1.1)
print("US + India alone has around 50 percent of all hits")
dftss = df_train['trafficSource.source'].value_counts().reset_index()[0:9]
dftss.columns = ['trafficSource.source','count']
plt.figure(figsize=(9,9))
p1 = plt.pie(dftss['count'], labels=dftss['trafficSource.source'],autopct='%1.1f%%',pctdistance=0.6, labeldistance=1.1)
print("Top 6 traffic Source")
df_train[['trafficSource.source','geoNetwork.country']]
df_train.head()
#printing nan counts of the columns
for column in df_train.columns:
    if(df_train[column].isnull().values.any()):
        print(column,"has Nan Count: ",df_train[column].isna().sum())
df_train['totals.bounces'].isna().sum()