#Action

#Import Libraries

import datetime
import math
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))

import json # to convert json in df
from pandas.io.json import json_normalize # to normalize the json file
# peak at the dataset
train_head = pd.read_csv("../input/train.csv",nrows=5)
#show train data 
train_head
#peak at the dataset
test_head = pd.read_csv("../input/test.csv",nrows=5)
#show train data 
test_head

#load train dataset
train = pd.read_csv("../input/train.csv", low_memory=False)
#shape and column names train
print (train.shape)
print (train.columns)
# load test dataset
test = pd.read_csv("../input/test.csv", low_memory=False)
#shape and column names of test
print (test.shape)
print (test.columns)
sampleSubmission = pd.read_csv("../input/sample_submission.csv")
#shape and column names of submission file
print (sampleSubmission.shape)
print (sampleSubmission.columns)

#Train:
# sessionId = fullVisitorId + visitId
print(len(train))
print(train.sessionId.nunique())
print(train.fullVisitorId.nunique())
print(train.visitId.nunique())

# sessionid is not unique for somereason, duplicates do exist.

#test:
print(len(test))
print(test.sessionId.nunique())
print(test.fullVisitorId.nunique())
print(test.visitId.nunique())
#Understand the data types 
train.dtypes
#Action
#borrowed code to parse JSON objects, big query another alternative, need to review this code
# removed sampling and added check, takes time to run

columns = ['device', 'geoNetwork', 'totals', 'trafficSource'] # Columns that have json format

dir_path = "../input/" # you can change to your local 

# p is a fractional number to skiprows and read just a random sample of the our dataset. 
#p = 0.07 # *** In this case we will use 50% of data set *** #

#Code to transform the json format columns in table
def json_read(df):
    #joining the [ path + df received]
    data_frame = dir_path + df
    
    #Importing the dataset
    df = pd.read_csv(data_frame, 
                     converters={column: json.loads for column in columns}, # loading the json columns properly
                     dtype={'fullVisitorId': 'str'}) # transforming this column to string
        
    for column in columns: #loop to finally transform the columns in data frame
        #It will normalize and set the json to a table
        column_as_df = json_normalize(df[column]) 
        # here will be set the name using the category and subcategory of json columns
        column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns] 
        # after extracting the values, let drop the original columns
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
        print("check")
 
    # Printing the shape of dataframes that was imported     
    print(f"Loaded {os.path.basename(data_frame)}. Shape: {df.shape}")
    return df # returning the df after importing and transforming


%%time
#Action

#Loading the train again with new function
df_train = json_read("train.csv")
print ("executed time:",datetime.datetime.now())
#peak the dataset
df_train.head()
#Garbage collection, will error if no train and test
import gc
del [[train,test]]
gc.collect()
 %who
    # what variables exist in the program, to ensure it is deleted
#what datatypes ?
df_train.dtypes

# find the null values 
total=df_train.isnull().sum()
total 
#df_train["channelGrouping"].value_counts()
#Describe the Data
#df_train.describe()
# what numeric variables 
numeric_features = df_train.select_dtypes(include=[np.number])
numeric_features.columns


# what non numeric variables 
numeric_features = df_train.select_dtypes(include=[np.object])
numeric_features.columns
#Action
#let fill the missing values
# my priority is address few fields i like, not to replace all the missing fields


def FillingNaValues(df):    # fillna numeric feature
    df['totals.pageviews'].fillna(1, inplace=True) #filling NA's with 1
    df['totals.newVisits'].fillna(0, inplace=True) #filling NA's with 0
    df['totals.bounces'].fillna(0, inplace=True)   #filling NA's with 0
    df["totals.transactionRevenue"] = df["totals.transactionRevenue"].fillna(0.0) #filling NA with zero
    
    return df #return the transformed dataframe
#Action
# replace missing values using the above fucntion
df_train = FillingNaValues(df_train)
print ("executed time:",datetime.datetime.now())
#Action
#change date formating using function (borrowed,quote source)

from datetime import datetime
def date_process(df):
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d") # seting the column as pandas datetime
    df["_weekday"] = df['date'].dt.weekday #extracting week day
    df["_day"] = df['date'].dt.day # extracting day
    df["_month"] = df['date'].dt.month # extracting day
    df["_year"] = df['date'].dt.year # extracting day
    df['_visitHour'] = (df['visitStartTime'].apply(lambda x: str(datetime.fromtimestamp(x).hour))).astype(int)
    
    return df #returning the df after the transformations
#Action
#Call the function for date formating
df_train = date_process(df_train) #calling the function that we created above

df_train.head(2) #printing the first 2 rows of our dataset
#Action
# find the constant columns and remove them 
constant_columns = []
for col in df_train.columns:
    if len(df_train[col].value_counts()) == 1:
        constant_columns.append(col)

#print column names 
constant_columns        
#Action
#delete the columns which has empty values 
for x in constant_columns:
    df_train.drop(x,axis=1, inplace=True)

df_train.shape
df_train.columns
#Action
# Data type conversion object to Int
df_train['totals.bounces'] = df_train['totals.bounces'].astype(str).astype(int)
df_train['totals.hits'] = df_train['totals.hits'].astype(str).astype(int)
df_train['totals.newVisits'] = df_train['totals.newVisits'].astype(str).astype(int)
df_train['totals.pageviews'] = df_train['totals.pageviews'].astype(str).astype(float)
df_train['totals.transactionRevenue'] = df_train['totals.transactionRevenue'].astype(str).astype(float)
df_train.groupby('channelGrouping')['totals.transactionRevenue'].agg('sum')
# Natural log issue pad by 1 
%matplotlib inline
plt.subplot(211)
df_train["totals.transactionRevenue"].hist(bins =5)
plt.ylabel('transactionRevenu')
plt.title('transactionRevenu histogram')

plt.subplot(212)
LogRevenue = np.log(df_train["totals.transactionRevenue"]+1)
LogRevenue.hist(bins =5)
plt.ylabel('log(transactionRevenu)')
plt.show()
# distribution of numercial values 
%matplotlib inline
import matplotlib.pyplot as plt
attributes = ["totals.bounces", "totals.hits","totals.newVisits",
              "totals.pageviews","visitNumber","_visitHour"]

loc[:,attributes].hist(bins =20,figsize =(20,15))
plt.show()
%matplotlib inline
channel_df = df_train['channelGrouping'].value_counts()
channel_df.index.name = 'channelGrouping'
channel_df.sort_index(inplace=True)
channel_df.plot(kind='bar',rot=20, title= 'Channel Distribution -count not revenue',figsize=(14,5))
plt.show()

# sample visitor
sample_visitor = df_train[df_train['fullVisitorId'] == '7813149961404844386'].sort_values(by='visitNumber')

#sample_visitor[['channelGrouping','date','visitId','visitNumber','totals.hits','totals.pageviews','totals.transactionRevenue']].head(30)
sample_visitor
#Action
df_train['target'] = np.log(df_train["totals.transactionRevenue"]+1)
#verify if the transformation happended
df_train[df_train['fullVisitorId'] == '7813149961404844386'].sort_values(by='visitNumber')['target'].head()
#cross tab on device
pd.crosstab(df_train['device.deviceCategory'], df_train['device.isMobile'], margins=False)
#revenue by mobile device 
g1 = df_train.groupby('device.isMobile')['target'].sum()
g1.plot.bar()
plt.show()

# seems like non mobile device generate more revenue
#revenue by mobile device 
g1 = df_train.groupby('device.deviceCategory')['target'].sum()
g1.plot.bar()
plt.show()

# Again Desktop makes more revenue
#revenue by browser 
g1 = df_train.groupby('device.browser')['target'].sum().sort_values()
df =pd.DataFrame(g1)
df =  df[df['target']>0]
df.plot.barh()
plt.show()

# too many browsers, 
# chrome leads the way
#revenue by Operating System 
g1 = df_train.groupby('device.operatingSystem')['target'].sum().sort_values()
df =pd.DataFrame(g1)
df =  df[df['target']>0]
df.plot.barh()
plt.show()
df.plot(kind='bar', stacked=True)
channel_df.plot(kind='bar',rot=20, title= 'Channel Distribution -count not revenue',figsize=(14,5))


# too many browsers, 
# chrome leads the way
df.plot(kind='bar',rot=20, title= 'revenue by OS',figsize=(14,5))
plt.show()
#revenue by date 
g1 = df_train.groupby('date')['target'].sum().sort_values()
df =pd.DataFrame(g1)
df.plot(figsize=(40,5))
plt.show()
# definitely some seaonality going on
#revenue by year 
g1 = df_train.groupby('_year')['target'].sum().sort_values()
df =pd.DataFrame(g1)
df.plot(kind='bar',rot=20, title= 'revenue by year')
plt.show()
# renevue increasing each year
#revenue by year 
g1 = df_train.groupby(['_month'])['target'].sum()
df =pd.DataFrame(g1)
df.plot()
plt.show()
# Dec month has higher revenue
