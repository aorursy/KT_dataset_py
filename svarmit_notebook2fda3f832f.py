# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%pylab inline 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/911.csv")
# to preview data 

df.head()
df# To see the description of data and columns.

df.info()
# removing nulls

df = df.dropna(subset = ['zip', 'title'])
# As this data is temporal data so checking the range of dates

print (min(df.timeStamp),max(df.timeStamp))



# so this data is for nearly 8 months. Now using very basic plot to visualize the spread of data
#Now using very basic plot to visualize the spread of data

# For this we need to convert timeStamp to date format 

df.timeStamp = df.timeStamp.astype('datetime64')

plt.plot(df.timeStamp)
plt.scatter(df.lng,df.lat)
# details of  at  titles

df.title.describe()
# Counting number of unique zip codes

len(df.zip.unique())
#Replace all null with 0

df.zip.fillna(0,inplace=True)
dfZipCodes_group = df.groupby('zip').size()
dfZipCodes_group.plot(kind = 'bar',figsize=(20,10))
# As most of the values are for null/0 zip code and the trend is very distributed so sorting the zip codes and then plotting again 

dfZipCodes_group.sort(ascending=False)
# Group by zip and type of incident

dfZipType = df.groupby(['zip','title']).size()
# looking at 

dfZipType.head()
dfZipType.sort('zip',ascending=False)
