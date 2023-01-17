import pandas as pd
import numpy as np
from pandas import Series
from pandas import DataFrame
##import json
import requests
import os
import sys
import datetime
twitterArchive_df = pd.read_csv("../input/twitter-archive-enhanced.csv")
imagePredict_df = pd.read_csv('../input/image-predictions.tsv',sep="\t")
# creating DataFrame using append method with the help of Series

tweetOtherInfo_df = DataFrame()
with open('../input/tweet_json.txt','r') as frd:
    column_names = frd.readline().strip().split(",")
    for line in frd.readlines():
        tweetOtherInfo_df = tweetOtherInfo_df.append(Series({key:value for key,value in zip(column_names,line.strip().split(","))}),ignore_index=True)

# changing column order 

tweetOtherInfo_df = tweetOtherInfo_df[['tweet_id','retweet_count','favorite_count']]
# Lets visualize 60 sample datapoints from each dataset
twitterArchive_df.head(60)
imagePredict_df.tail(60)
tweetOtherInfo_df.sample(60)
twitterArchive_df.count()
# many variables of the dataframe,such as 'in_reply_to_status_id' , 'in_reply_to_user_id' , 'retweeted_status_id' , 'retweeted_status_user_id'
#,'retweeted_status_timestamp' & 'expanded_urls' filled with NaN
len(twitterArchive_df)
twitterArchive_df.shape
# So there are 2356 datapoints/rows/observation present in the dataset
twitterArchive_df.tweet_id.is_unique
twitterArchive_df.tweet_id.value_counts().count()
# Custom function to display only unique column names of a DataFrame

def print_unique_columns(df):
    for column in list(df.columns):
        if df[column].value_counts().count() == len(df):
            print (column)
print_unique_columns(twitterArchive_df)
# 'tweet_id','timestamp' & 'text' columns have 2356 unique values
twitterArchive_df.index
twitterArchive_df.columns
twitterArchive_df.info()
twitterArchive_df.describe()
# Exploring data types of several pandas objects
type(twitterArchive_df.timestamp.iloc[0])
type(twitterArchive_df.retweeted_status_timestamp[twitterArchive_df.retweeted_status_timestamp.notnull()].iloc[0])
twitterArchive_df.name.value_counts()
# 745 rows of 'name' column have None values
##twitterArchive_df.in_reply_to_user_id[~twitterArchive_df.in_reply_to_user_id.isnull()].astype("int64").describe()
for text_ in twitterArchive_df.text[:10]:
    print(text_+"\n")
# Above text is a preview of only first 10 value of 'text' column  
twitterArchive_df.puppo.value_counts()
# 'puppo' column has 2326 rows with None values & 30 rows with puppo values
sum(~twitterArchive_df.puppo.duplicated())
##twitterArchive_df[twitterArchive_df.puppo.duplicated()]
# Above output implies that the 'puppo' column is consist of only two type of values : None & puppo itself.
twitterArchive_non_None = twitterArchive_df[~twitterArchive_df.name.isin(['None'])]
twitterArchive_non_None.shape
# 'name' column has 1611 rows with non null values
sum(twitterArchive_non_None.name.duplicated())
# in 'name' column there are 655 duplicated values
##twitterArchive_non_None[twitterArchive_non_None.name.duplicated()]
##twitterArchive_non_None[twitterArchive_non_None.name.duplicated()].name.value_counts()
twitterArchive_non_None[twitterArchive_non_None.name.duplicated()].name.value_counts().sort_values(ascending=False).head(60)
# in 'name' column there are some inaccurate values such as a , an ,the,quite etc.
sum(twitterArchive_df.expanded_urls.duplicated())
# There are 137 duplicate value presents for 'expanded_urls' column/feature/variable
##twitterArchive_non_NaN = twitterArchive_df[~twitterArchive_df.expanded_urls.isnull()]
sum(twitterArchive_df.expanded_urls.isnull()) # >>> output : 59

# OR , both works as same

twitterArchive_non_NaN = twitterArchive_df[~twitterArchive_df.expanded_urls.isin([np.nan])]
sum(twitterArchive_df.expanded_urls.isin([np.nan])) # >>> output : 59
sum(twitterArchive_non_NaN.expanded_urls.str.startswith("http")) # >>> output : 2297
twitterArchive_df.shape[0] == 2297+59
#So it is proven all non NaN entries of the column 'expanded_urls' are actually urls
imagePredict_df.info()
imagePredict_df.index
imagePredict_df.columns
imagePredict_df.describe()
imagePredict_df.shape
# This DataFrame has 2075 rows/observations
imagePredict_df.tweet_id.is_unique
# Finding column with unique values only :-
print_unique_columns(imagePredict_df)
# 'tweet_id' column has 2075 unique values
imagePredict_df.img_num.value_counts()
# 'img_num' column dont have any missing value.
imagePredict_df.p1.value_counts().sort_values()
# Image Prediction Algotithm predict highest number of dogs as 'golden_retriever' type. Second highest dog type is 'Labrador_retriever'.
sum(imagePredict_df.jpg_url.str.startswith("http"))
imagePredict_df.shape[0] == sum(imagePredict_df.jpg_url.str.startswith("http"))
#So it is proven all entries of the column 'jpg_url' are actually urls
#If I need to consider only dog pictures, then either imagePredict_df.p1_dog is True or imagePredict_df.p2_dog is True or imagePredict_df.p3_dog is True

tmp = imagePredict_df[(imagePredict_df.p1_dog == True) | (imagePredict_df.p2_dog == True) | (imagePredict_df.p3_dog == True)]

tmp.shape
#So tmp DataFrame containing only Dog related tweets
sum((tmp.p1_conf>tmp.p2_conf) & (tmp.p2_conf>tmp.p3_conf))
## So p1_conf > p2_conf > p3_conf is the order of a picture confidence level(probabilty) for the Image Predction Algorithm.
#in tmp2 DataFrame either p1_dog is set to True or p2_dog & p3_dog are set to True where p1_dog is set to False.

tmp2 = tmp[((tmp.p1_dog == True) | ((tmp.p1_dog == False) & (tmp.p2_dog == True) & (tmp.p3_dog == True)))]

sum((tmp2.p1_dog == True))
sum((tmp2.p1_conf < tmp2.p2_conf + tmp2.p3_conf) & (tmp2.p1_dog == False))
##tmp2.shape # >>> 1633
#Above 24 implies, if a picture p1_dog set to False ,then there are 24 occasion ,where pictures p2_conf & p3_conf summation is bigger than p1_conf
# In the above case ,despite of p1_conf is set to False,there are 24 times ,when there is a greater probabilty that that its a dog image.
# Now I want to consider only those rows where either p1_conf is maximum along with other confidefence levels or summation of p2_conf & p3_conf is greater than p1_conf while p1_dog is not True. 
sum(((tmp2.p1_conf < tmp2.p2_conf + tmp2.p3_conf) & (tmp2.p1_dog == False)) | (tmp2.p1_dog == True))

##sum(((tmp2.p1_conf > tmp2.p2_conf + tmp2.p3_conf) | (tmp2.p1_dog == True)) & (tmp2.p1_dog == False)) # >>> 77
tweetOtherInfo_df.info()
tweetOtherInfo_df.shape
# This DataFrame consists of 256 rows & 3 columns
tweetOtherInfo_df.index
tweetOtherInfo_df.columns
type(tweetOtherInfo_df.tweet_id.iloc[0])
type(tweetOtherInfo_df.retweet_count.iloc[0])
type(tweetOtherInfo_df.favorite_count.iloc[0])
# So, all columns are String type
print_unique_columns(tweetOtherInfo_df)
tweetOtherInfo_df.tweet_id.is_unique
# 'tweet_id' column has unique values
tweetOtherInfo_df.retweet_count.value_counts(ascending=False).head()
tweetOtherInfo_df.favorite_count.value_counts(ascending=False).head()
# 'retweet_count' & 'favorite_count' column have some values filled with "Not Exist"
tweetOtherInfo_df.describe()
# making copies of dataframes
twitterArchive_df_clean = twitterArchive_df.copy()
twitterArchive_df_clean.tweet_id = twitterArchive_df_clean.tweet_id.astype("str")
twitterArchive_df_clean.tweet_id.dtype
type(twitterArchive_df_clean.tweet_id.iloc[0])
twitterArchive_df_clean.timestamp = pd.to_datetime(twitterArchive_df_clean.timestamp)
##original_df_clean.retweeted_status_timestamp = pd.to_datetime(original_df_clean.retweeted_status_timestamp.fillna(""))
# OR
twitterArchive_df_clean.retweeted_status_timestamp = pd.to_datetime(twitterArchive_df_clean.retweeted_status_timestamp)
twitterArchive_df_clean.timestamp.dtype
type(twitterArchive_df_clean.timestamp.iloc[0])
twitterArchive_df_clean.retweeted_status_timestamp.dtype
twitterArchive_df_clean.doggo = None
twitterArchive_df_clean.doggo = twitterArchive_df_clean.text.str.extract('\\b(doggo|Doggo)\\b', expand=True)[0]
##twitterArchive_df_clean.text.str.extract('(doggo|Doggo)', expand=True)[0].value_counts()
##sum(~twitterArchive_df_clean.doggo.isin([np.nan]))
twitterArchive_df_clean.puppo = None
twitterArchive_df_clean.puppo = twitterArchive_df_clean.text.str.extract('\\b(puppo|Puppo)\\b', expand=True)[0]
twitterArchive_df_clean.pupper = None
twitterArchive_df_clean.pupper = twitterArchive_df_clean.text.str.extract('\\b(pupper|Pupper)\\b', expand=True)[0]
twitterArchive_df_clean.floofer = None
twitterArchive_df_clean.floofer = twitterArchive_df_clean.text.str.extract('\\b(floofer|Floofer)\\b', expand=True)[0]
twitterArchive_df_clean.doggo.value_counts()
twitterArchive_df_clean.puppo.value_counts()
twitterArchive_df_clean.pupper.value_counts()
twitterArchive_df_clean.floofer.value_counts()
twitterArchive_df_clean.doggo.replace("Doggo","doggo",inplace=True)
##t = {'doggo': True, np.nan: False}
##twitterArchive_df_clean.doggo = twitterArchive_df_clean.doggo.map(t)
twitterArchive_df_clean.floofer[twitterArchive_df_clean.doggo.notnull()] = np.nan
twitterArchive_df_clean.floofer.replace("Floofer","floofer",inplace=True)
##t = {'floofer': True, np.nan: False}
##twitterArchive_df_clean.floofer = twitterArchive_df_clean.floofer.map(t)
twitterArchive_df_clean.pupper[twitterArchive_df_clean.doggo.notnull() | twitterArchive_df_clean.floofer.notnull()] = np.nan
twitterArchive_df_clean.pupper.replace("Pupper","pupper",inplace=True)
##t = {'pupper': True, np.nan: False}
##twitterArchive_df_clean.pupper = twitterArchive_df_clean.pupper.map(t)
twitterArchive_df_clean.puppo[twitterArchive_df_clean.doggo.notnull() | twitterArchive_df_clean.floofer.notnull() | twitterArchive_df_clean.pupper.notnull()] = np.nan
twitterArchive_df_clean.puppo.replace("Puppo","puppo",inplace=True)
##t = {'puppo': True, np.nan: False}
##twitterArchive_df_clean.puppo = twitterArchive_df_clean.puppo.map(t)
##sum(twitterArchive_df_clean.doggo[twitterArchive_df_clean.doggo.notnull()])
twitterArchive_df_clean.doggo.count()
twitterArchive_df_clean.doggo.value_counts()
##sum(twitterArchive_df_clean.floofer[twitterArchive_df_clean.floofer.notnull()])
twitterArchive_df_clean.floofer.count()
twitterArchive_df_clean.floofer.value_counts()
##sum(twitterArchive_df_clean.pupper[twitterArchive_df_clean.pupper.notnull()])
twitterArchive_df_clean.pupper.count()
twitterArchive_df_clean.pupper.value_counts()
##sum(twitterArchive_df_clean.puppo[twitterArchive_df_clean.puppo.notnull()])
twitterArchive_df_clean.puppo.count()
twitterArchive_df_clean.puppo.value_counts()
# making copies of dataframes
imagePredict_df_clean = imagePredict_df.copy()
imagePredict_df_clean.tweet_id = imagePredict_df_clean.tweet_id.astype("str")
imagePredict_df_clean.tweet_id.dtype
type(imagePredict_df_clean.tweet_id.iloc[0])
tmp = imagePredict_df_clean[(imagePredict_df_clean.p1_dog == True) | (imagePredict_df_clean.p2_dog == True) | (imagePredict_df_clean.p3_dog == True)]

tmp2 = tmp[((tmp.p1_dog == True) | ((tmp.p1_dog == False) & (tmp.p2_dog == True) & (tmp.p3_dog == True)))]

imagePredict_df_clean = tmp2[((tmp2.p1_conf < tmp2.p2_conf + tmp2.p3_conf) & (tmp2.p1_dog == False)) | (tmp2.p1_dog == True)]
imagePredict_df_clean
#out of 2075 rows only 1556 meets the requirement
imagePredict_df_clean.p1_dog[imagePredict_df_clean.p1_dog==True].count()
imagePredict_df_clean.p1_dog[imagePredict_df_clean.p1_dog==False].count()
imagePredict_df_clean["type"] = None
# if p1_dog is True , we are copying corresponding 'p1' column values to 'type' column

imagePredict_df_clean.type[imagePredict_df_clean.p1_dog == True] = imagePredict_df_clean[imagePredict_df_clean.p1_dog == True].p1
##sum(imagePredict_df_clean.p1_dog == True)
imagePredict_df_clean.type.fillna("mix",inplace=True)
sum(imagePredict_df_clean.type.isnull())
sum(imagePredict_df_clean.type == "mix")
imagePredict_df_clean.type.value_counts()
imagePredict_df_clean.head()
# making copies of dataframes
tweetOtherInfo_df_clean = tweetOtherInfo_df.copy()
tweetOtherInfo_df_clean = tweetOtherInfo_df_clean[~tweetOtherInfo_df_clean.retweet_count.isin(["Not Exist"])]
tweetOtherInfo_df_clean = tweetOtherInfo_df_clean[~tweetOtherInfo_df_clean.favorite_count.isin(["Not Exist"])]
sum(tweetOtherInfo_df_clean.retweet_count.isin(["Not Exist"]))
sum(tweetOtherInfo_df_clean.favorite_count.isin(["Not Exist"]))
tweetOtherInfo_df_clean.retweet_count = tweetOtherInfo_df_clean.retweet_count.astype("int64")
tweetOtherInfo_df_clean.favorite_count = tweetOtherInfo_df_clean.favorite_count.astype("int64")
tweetOtherInfo_df_clean.retweet_count.dtype
tweetOtherInfo_df_clean.retweet_count.dtype
twitterArchive_df_clean["stage"] = (twitterArchive_df_clean.doggo.fillna("")+twitterArchive_df_clean.floofer.fillna("")+twitterArchive_df_clean.pupper.fillna("")+twitterArchive_df_clean.puppo.fillna(""))
twitterArchive_df_clean["stage"] = twitterArchive_df_clean["stage"].replace("",np.nan)
#Dropping columns
twitterArchive_df_clean.drop("doggo",axis=1,inplace=True)
twitterArchive_df_clean.drop("floofer",axis=1,inplace=True)
twitterArchive_df_clean.drop("pupper",axis=1,inplace=True)
twitterArchive_df_clean.drop("puppo",axis=1,inplace=True)
twitterArchive_df_clean.head(60)
twitterArchive_df_clean["stage"].value_counts()
# how = 'inner' , by default
twitterArchive_df_clean = twitterArchive_df_clean.merge(tweetOtherInfo_df_clean,on="tweet_id")
twitterArchive_df_clean
# 11 tweet_id(s) are removed,as they are not present on both tables. 
twitterArchive_df_clean.head(60)
imagePredict_df_clean.head(30)
twitterArchive_df_clean.info()
twitterArchive_df_clean.describe()
print_unique_columns(twitterArchive_df_clean)
##sum(tmp_df.expanded_urls.str.split("/").str[5] == tmp_df.tweet_id)
twitterArchive_df_clean = twitterArchive_df_clean[twitterArchive_df_clean.retweeted_status_timestamp.isnull()]
sum(twitterArchive_df_clean.retweeted_status_timestamp.notnull())
# Any sentence start with <This is > or <Meet > followed by a word starts with a Capital letter and then stop with a <.(dot)>

twitterArchive_df_clean["name"] = twitterArchive_df_clean.text.str.extract('((This is|Meet) ([A-Z][a-z]*)\\.)', expand=True)[2] # [a-z] or \w 

##twitterArchive_df_clean.text.str.extract('((This is|Meet) ([A-Z][a-z]*)\\.)', expand=True)[2].count() # 1217

##twitterArchive_df_clean.text.str.extract('((?!This)(?!News)(?!Impressive)([A-Z][a-z]*)\\.)', expand=True).head(60) #Except <This> Start with a captal leter & ends with a <.dot>

twitterArchive_df_clean["name"].value_counts(ascending=False)
##twitterArchive_df_clean[twitterArchive_df_clean["name"] == "Snoop"]
# in text of each tweet_id(s) there are som erating those are not valid. We ar going to skip those ratings,such as : 7/11 , 9/11 , 4/20 .
## In which 7/11 , 9/11 used as date by @dog_rates. And 4/20 
''' 
'([0-9]{1,4}/(10|[1-9][0-9]{1,3}))'
'([0-9][0-9]?[.]?\d{0,2}/(10|[1-9][0-9]{1,3}))' better
'((?!7/11)(?!9/11)[0-9][0-9]?[.]?\d{0,2}/(10|[1-9][0-9]{1,3}))' exclude better
'''

twitterArchive_df_clean["rating_numerator"],twitterArchive_df_clean["rating_denominator"] = twitterArchive_df_clean.text.str.extract('((?!7/11)(?!9/11)[0-9][0-9]?[.]?\d{0,2}/(10|[1-9][0-9]{1,3}))', expand=True)[0].str.split("/").str

##twitterArchive_df_clean[twitterArchive_df_clean.text.str.extract('((?!7/11)(?!9/11)[0-9][0-9]?[.]?\d{0,2}/(10|[1-9][0-9]{1,3}))', expand=True)[0].values == "0/10"]

## Side Note:-
#50/50 -> 11/10
#182/10 
#11/15
#1776/10 - > 17.76/10
#666/10 -> 6.66/10
#420/10
#4/20
#0/10
twitterArchive_df_clean.rating_numerator.count()
twitterArchive_df_clean[twitterArchive_df_clean.rating_numerator.isnull()]
##twitterArchive_df_clean[twitterArchive_df_clean.rating_denominator == "20"]
twitterArchive_df_clean[twitterArchive_df_clean.rating_numerator == "11.27"]
twitterArchive_df_clean.rating_denominator = twitterArchive_df_clean.rating_denominator[twitterArchive_df_clean.rating_denominator.notnull()].astype("float64")
twitterArchive_df_clean.rating_numerator = twitterArchive_df_clean.rating_numerator[twitterArchive_df_clean.rating_numerator.notnull()].astype("float64")
twitterArchive_df_clean.rating_denominator.dtype
twitterArchive_df_clean.rating_numerator.dtype
# filtering denominator those have values other than 10.0

deno_non10 = twitterArchive_df_clean.rating_denominator[twitterArchive_df_clean.rating_denominator != 10.0]
# change non 10.0 denominator values to 10.0 (except NaN)

twitterArchive_df_clean.rating_denominator[deno_non10.index] = deno_non10 / deno_non10 * 10
# Scaling down numerator values corresponds of non 10.0 denominator values

twitterArchive_df_clean.rating_numerator[deno_non10.index] = twitterArchive_df_clean.rating_numerator[deno_non10.index] / deno_non10 * 10
twitterArchive_df_clean.rating_denominator.value_counts()
twitterArchive_df_clean.rating_denominator[twitterArchive_df_clean.rating_denominator.isnull()]
# inner join  

twitterArchive_df_clean = pd.merge(twitterArchive_df_clean,imagePredict_df_clean[['tweet_id','jpg_url','type']],on="tweet_id")
imagePredict_df_clean.drop('jpg_url', axis=1,inplace=True)
imagePredict_df_clean.drop('type', axis=1,inplace=True)
twitterArchive_df_clean
# 1499 tweet_id(s) are present on both tables. 
imagePredict_df_clean
all_columns = pd.Series(list(twitterArchive_df_clean) + list(imagePredict_df_clean))
all_columns[all_columns.duplicated()]
#So as like expected after merging done & droping columns , only 'tweet_id' column is left on both tables as a common column 
twitterArchive_df_clean.head(60)
twitterArchive_df_clean.info()
twitterArchive_df_clean.in_reply_to_status_id[twitterArchive_df_clean.in_reply_to_status_id.notnull()].dtype
twitterArchive_df_clean.in_reply_to_user_id[twitterArchive_df_clean.in_reply_to_user_id.notnull()].dtype
#So we can see data type of 'in_reply_to_status_id' , 'in_reply_to_user_id' are float64 not string! 
twitterArchive_df_clean.shape
twitterArchive_df_clean.describe()
print_unique_columns(twitterArchive_df_clean)
twitterArchive_df_clean.expanded_urls.is_unique
twitterArchive_df_clean.jpg_url.is_unique
twitterArchive_df_clean.rating_numerator.value_counts()
twitterArchive_df_clean.rating_denominator.value_counts()
twitterArchive_df_clean.source.value_counts()
#Only 3 type of source value is there , so 'source' column data type could be a category.
twitterArchive_df_clean.in_reply_to_status_id = twitterArchive_df_clean.in_reply_to_status_id[twitterArchive_df_clean.in_reply_to_status_id.notnull()].astype("int64").astype("str")
twitterArchive_df_clean.in_reply_to_user_id = twitterArchive_df_clean.in_reply_to_user_id[twitterArchive_df_clean.in_reply_to_user_id.notnull()].astype("int64").astype("str")
type(twitterArchive_df_clean.in_reply_to_status_id[twitterArchive_df_clean.in_reply_to_status_id.notnull()].iloc[0])
type(twitterArchive_df_clean.in_reply_to_user_id[twitterArchive_df_clean.in_reply_to_user_id.notnull()].iloc[0])
twitterArchive_df_clean.source = twitterArchive_df_clean.source.astype('category')
twitterArchive_df_clean.source.dtype
twitterArchive_df_clean.dtypes
# Run only once

del twitterArchive_df_clean["retweeted_status_id"]
del twitterArchive_df_clean["retweeted_status_user_id"]
del twitterArchive_df_clean["retweeted_status_timestamp"]
twitterArchive_df_clean.shape
# Setting 'tweet_id' column as an index column

##twitterArchive_df_clean.set_index("tweet_id",inplace=True)
twitterArchive_df_clean.info()
twitter_archive_master = twitterArchive_df_clean
twitter_archive_master
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sb
import matplotlib
%matplotlib inline
rcParams['figure.figsize'] = 8,4
sb.set_style('whitegrid')
twitter_archive_master.describe()
twitter_archive_master.corr()
twitter_archive_master.retweet_count.corr(twitter_archive_master.favorite_count,method="pearson")
twitter_archive_master.groupby(["source"])["tweet_id","in_reply_to_status_id","in_reply_to_user_id","name","stage"].count()
twitter_archive_master.groupby(["stage"])['rating_numerator','retweet_count','favorite_count'].min()
twitter_archive_master.groupby(["stage"])['rating_numerator','retweet_count','favorite_count'].max()
twitter_archive_master.groupby(["stage"])['retweet_count','favorite_count'].agg(['count','min','max','sum','mean','median','std'])
twitter_archive_master.groupby(["stage","type"])['retweet_count','favorite_count'].mean()
twitter_archive_master.groupby(["stage","type"])['retweet_count','favorite_count'].mean().max()
twitter_archive_master.groupby(["stage","type"])['retweet_count','favorite_count'].mean().idxmax()
twitter_archive_master.rating_numerator.value_counts().plot(kind="bar",title = "Bar Chart for 'rating_numerator'")
color_theme = ['#9902FD', '#FFA07A', '#B0E0E6','#0981FF']
twitter_archive_master.stage.fillna("N.A.").value_counts().plot(kind="pie",colors=color_theme,title="Pie Chart for Different dog stages")
twitter_archive_master.source.value_counts().plot(kind="barh",title="Bar horizontal chart for 'source'")
twitter_archive_master.retweet_count.plot(kind="hist",title="Histogram for 'retweet_count'",xlim=(0,40000))
twitter_archive_master.favorite_count.plot(kind="hist",title= "Histogram for 'favorite_count'",xticks=[0,15000,30000,60000,100000,120000,140000])
twitter_archive_master.plot(kind="scatter",x="retweet_count",y="favorite_count",c=["darkgrey"],s=50,alpha=0.2,title = "'retweet_count' V/S 'favorite_count'")
twitter_archive_master.boxplot(column="favorite_count",by="stage")
twitter_archive_master.groupby(["stage"])["favorite_count"].median()