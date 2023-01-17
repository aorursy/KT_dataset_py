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
# import some packages

import matplotlib.pyplot as pp

import seaborn
%matplotlib inline
# see if I can open the cv file

youtube_data = pd.read_csv('../input/data.csv')
# check to see if the file can be read

youtube_data.head()
# rank and grade are given by Socialblade and not important, so I dropped them

data = youtube_data.drop(['Rank', 'Grade'], axis = 1)
# check if the columns have been dropped 

data.head()
# clean the dataset, because any channel with the first character '9' will show up as the number one most sub channel

cols = data.columns.drop('Channel name')
# some channels had '-- ' or '--', which resulted in the data types to be float rather than int

data[cols] = data[cols].replace('-- ', 0)

data[cols] = data[cols].replace('--', 0)
data[cols] = data[cols].astype(int)
data.dtypes
# check if PewDiePie is the number one subscribed channel and only keep the top 5 channels

df = data.sort_values(by=['Subscribers'],ascending=False).head(5)
# test if df works

df
# on average, how many views do each video gets

df['Average views per upload'] = (df['Video views']/df['Video Uploads'])

# change the column to a int, because earlier it was a float and and the numbers were not displaying correctly

df['Average views per upload'] = df['Average views per upload'].astype(int)
# test if the data type changed

df.dtypes
# bar graph of how many views per upload

pp.bar(np.arange(len(df)),df['Average views per upload'], align='center')

pp.xticks(np.arange(len(df)),df['Channel name'])

pp.ylabel('Average views per upload(By 10 Mil)')

pp.title('How Many Views on Average Do Each Most Subscribed Youtuber Get?')

pp.figure(figsize=(12,3))

df