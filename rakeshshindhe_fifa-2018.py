# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df= pd.read_csv("/kaggle/input/world-cup-2018-tweets/FIFA.csv",index_col="Date",parse_dates=["Date"]).sort_index()
dates=df.groupby("Date")
df["Hashtags"]=df["Hashtags"].fillna("Unknown")
myres=list(df["Hashtags"].values)
list2=[]

for data in myres:

    seperate_data1 = data.split(",")

    for i in seperate_data1:

        list2.append(i)
myseries = pd.Series(list2)

myseries.value_counts() # here the hashtags are sorted based on the tweet numbers FRA got more tweets 
final_day_tweets = df.loc["2018-07-15"]

final_day_tweets["Place"].value_counts().index[0] # the country with the most tweets on final day match
final_day_tweets["Tweet"].value_counts().sum() # total number of tweets on final day