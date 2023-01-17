import pandas as pd

import numpy as np

import scipy as sp

import altair as alt

import matplotlib.pyplot as plt

import time

import pandasql as ps



import os

os.chdir("/kaggle/input")

print(os.listdir())

#Lets make our console outputs more nice, by applying some settings.

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"

alt.renderers.enable('notebook')

alt.data_transformers.enable('default', max_rows=None)

%matplotlib inline 
#Support functions for the analysis go here:

#Solution comes from: https://stackoverflow.com/questions/34122395/reading-a-csv-with-a-timestamp-column-with-pandas

def date_parser(string_list):

    return [time.ctime(float(x)) for x in string_list]
#First, lets load the data:

#The header throws off our parser, so just ignore and write manually.

acroDF = pd.read_csv("./reddit-scitech-acronyms/acronyms.csv",skiprows=1,parse_dates=[1],date_parser=date_parser,

                     names=["commID","time","user","subreddit","acronym"])



#Nice date formats!

acroDF.head(10)
acroDF.dropna(inplace=True) #There are some NA acronyms.

acroDF.count() #Now every row is fully defined. 

acroDF.describe() #Some basic information.
q1 = """SELECT t1.time,t1.user,t2.user FROM acroDF AS t1 INNER JOIN acroDF AS t2 ON (t1.time = t2.time) AND (t1.commID != t2.commID) """

timeQueryDF = ps.sqldf(q1, locals())

timeQueryDF.head(10)
subredCount = acroDF.groupby("subreddit",as_index=False).count()

subredCount.drop(["commID","time","user"],axis=1,inplace=True)

subredCount.head(5)



srList = (subredCount.sort_values("acronym", ascending=False))["subreddit"].tolist()

alt.Chart(subredCount).mark_bar().encode(

alt.X('subreddit:N',sort=srList),

alt.Y('acronym:Q'))

#will put things in alphabetical order, by default

subredDF = acroDF.groupby(by="subreddit",as_index=False).count()

#dont need these columns.

subredDF.drop(["commID","time","user"],axis=1,inplace=True)

#subredDF = subredDF.sort_values(by="user",ascending=False)



srStatDF = pd.read_csv("./subredditstats/subredditstats.csv",skiprows=1,parse_dates=[2],

                     date_parser=date_parser,names=["subreddit","subscribers","utc_created"])



srStatDF.sort_values("subreddit",ascending=True,inplace=True)



#when we add columns, they are series. So entries will be matched by indices. Looking at the two columns above,

#one has non-ascending indices, so lets reset them

srStatDF.reset_index(drop=True,inplace=True)



srStatDF["acroCount"] = subredDF["acronym"]

#add the acronym count column.

#derive  an acronym ratio.

def f(x,y): #assume we dont have y=0!

    return (x/y)

#we use the clever unpacking notation (*x), because x is typed as a 2ple. we do this down the columns

srStatDF['acroRatio'] = srStatDF[['acroCount','subscribers']].apply(lambda x: f(*x), axis=1)

#display the full datatable

srStatDF.head(10)
#Lets make our altair histogram for this.

#We need to sort our chart in altair by counts.

#I can't seem to get the chart sorted correctly, so we will have to provide a list of subreddits explicitly,

#as per: https://altair-viz.github.io/user_guide/encoding.html?highlight=alt%20order#ordering-marks

#srStatDF.sort_values("subscribers", ascending=False).head(5)

srList = (srStatDF.sort_values("subscribers", ascending=False))["subreddit"].tolist()



alt.Chart(srStatDF, title="Number of Subscribers per Subreddit").mark_bar().encode(

x=alt.X('subreddit:N',sort=srList),

y=alt.Y('subscribers:Q'))



srList = (srStatDF.sort_values("acroRatio", ascending=False))["subreddit"].tolist()

alt.Chart(srStatDF, title="Ratio of Acronyms to Subscribers, per Subreddit").mark_bar().encode(

x=alt.X('subreddit:N',sort=srList),

y=alt.Y('acroRatio:Q'))

#Support local function: our chained calls are too long.



def getcountdf(aDF,subreddit):

    temp = aDF.groupby("acronym",as_index=False).count()

    return temp.sort_values("subreddit",ascending=False).reset_index(drop=True)



acroFocus = acroDF.copy(deep=True) #leave intact for now.

acroFocus.drop(['commID','time',"user"], axis=1, inplace=True)

acroGroup = acroFocus.groupby("subreddit",as_index=False)



#Dict -> subDF return type.

#DataFrame -> Group -> DataFrame -> Dataframe

geneticsCountDF = getcountdf(acroGroup.get_group("genetics"), "genetics")

btcCountDF = getcountdf(acroGroup.get_group("BitcoinMarkets"), "BitcoinMarkets")

nnCountDF = getcountdf(acroGroup.get_group("neuralnetworks"), "neuralnetworks")



geneticsCountDF.head(5)

#Visualization: Bar Chart with top 75 acronyms for each subreddit.

#Same sorting issues as last time, lets do an explicit encoding.



nameList = (geneticsCountDF.sort_values("subreddit", ascending=False))["acronym"].tolist()

limit = 30



genChart = alt.Chart(geneticsCountDF[0:limit],title="Top Acronyms for Genetics Subreddit").mark_bar().encode(

y=alt.Y("acronym:N",sort=nameList[0:limit]),

x=alt.X("subreddit:Q")).properties(width=300)



nameList = (btcCountDF.sort_values("subreddit", ascending=False))["acronym"].tolist()



btcChart = alt.Chart(btcCountDF[0:limit],title="Top Acronyms for BitcoinMarkets Subreddit").mark_bar().encode(

y=alt.Y("acronym:N",sort=nameList[0:limit]),

x=alt.X("subreddit:Q")).properties(width=300)



nameList = (nnCountDF.sort_values("subreddit", ascending=False))["acronym"].tolist()



nnChart = alt.Chart(nnCountDF[0:limit],title="Top Acronyms for Neural Networks Subreddit").mark_bar().encode(

y=alt.Y("acronym:N",sort=nameList[0:limit]),

x=alt.X("subreddit:Q")).properties(width=300)



#Here we see the power of the grammar of graphics and Altair :)

genChart | btcChart | nnChart