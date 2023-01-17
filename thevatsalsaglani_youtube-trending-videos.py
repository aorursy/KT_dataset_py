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
CA = pd.read_csv('../input/CAvideos.csv')
CA.head()
US = pd.read_csv('../input/USvideos.csv')

# US.info()
#column name as string..
#name of the Data Frame variable..
def numberOfUniqueValues(col_name, dataFrame):
    print("There are {0} unique values.".format((len(dataFrame[col_name].unique()))))
    
numberOfUniqueValues('views', US)
# len(US['views'].unique())
US.columns
print("VideoID:")
numberOfUniqueValues('video_id', US)
##
print("TrendingDates:")
numberOfUniqueValues('trending_date', US)
##
print("Video Title:")
numberOfUniqueValues('title', US)
##
print("Channels:")
numberOfUniqueValues('channel_title', US)
##
print("Tags:")
numberOfUniqueValues('tags', US)
##


#US['trending_date']
trending_date_unique = []
      
# function to get all the unique elemets of a column in a list.
# listName will be an empty list which must be declared before hand
# dataFrame is the name of the data-frame you need to calculate the unique values.
def uniqueValuesList(listName, columnName, dataFrame):
    for data in dataFrame[columnName]:
        if data not in listName:
            listName.append(data)
    return listName

# lets try the function out
uniqueValuesList(trending_date_unique, 'trending_date', US)
#state mapping
# create an empty dictionary
# some = dict()
# for xx in range(0,len(trending_date_unique)):
#     some[trending_date_unique[xx]] = xx
    
def stateMapping(listName, dictionaryName, columnName, dataFrame):
    for unique_names in range(0,len(listName)):
        dictionaryName[listName[unique_names]] = unique_names
    dataFrame[columnName] = dataFrame[columnName].map(dictionaryName)
trending_Date = dict()
stateMapping(trending_date_unique, trending_Date, 'trending_date', US)
US.head()
unique_video_id = []
uniqueValuesList(unique_video_id, 'video_id', US)
video_mapping = dict()
stateMapping(unique_video_id, video_mapping, 'video_id', US)
unique_video_title = []
uniqueValuesList(unique_video_title, 'title', US)
title_mapping = dict()
stateMapping(unique_video_title, title_mapping, 'title', US)
unique_channels = []
uniqueValuesList(unique_channels, 'channel_title', US)
channels_mapping = dict()
stateMapping(unique_channels, channels_mapping, 'channel_title', US)
unique_tags = []
uniqueValuesList(unique_tags, 'tags', US)
tags_mapping = dict()
stateMapping(unique_tags, tags_mapping, 'tags', US)
US.head()
US.columns
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
cols = ['trending_date', 'title', 'channel_title', 'category_id', 'tags' , 'views', 'likes', 'dislikes', 'comment_count']
cm = np.corrcoef(US[cols].values.T)
sb.set(font_scale=2)
hm=sb.heatmap(cm,
               cbar=True,
               annot=True,
               square=True,
               fmt='.1f',
               annot_kws={'size':9},
               yticklabels=cols,
               xticklabels=cols)
plt.show()
# plotting the channels by likes and comment-counts
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
popular_channels_US = (US.groupby('channel_title')['likes', 'views', 'comment_count', 'dislikes'].agg({'likes': 'sum', 'views': 'sum', 'comment_count': 'sum', 'dislikes': 'sum'}).sort_values(by="views", ascending=False))[:10].reset_index()
popular_channels_US
plt.style.available
plt.style.use('seaborn-whitegrid')
popular_channels_US.plot(kind='bar', x="channel_title", y=['views','likes', 'dislikes'])
allPopularChannelsUS = US.groupby('channel_title')['likes', 'views', 'comment_count', 'dislikes'].agg({'likes': 'sum', 'views': 'sum', 'comment_count': 'sum', 'dislikes': ['max', 'min']})
allPopularChannelsUS
allPopularChannelsUS.columns
allPopularVideosUS = US.groupby('title')['views', 'likes', 'dislikes', 'comment_count'].agg({'views': 'sum', 'likes': 'sum', 'dislikes': ['max', 'min'], 'comment_count': 'sum'})
allPopularVideosUS
# sorting the data of Popular Videos in US by sum of views and man of dislikes
allPopularVideosUS = allPopularVideosUS.sort_values([('views','sum'),('dislikes', 'max')], ascending=False)
# allPopularVideosUS
allPopularVideosUS['views']['sum'][len(allPopularVideosUS)-1]
allPopularVideosUS.columns
for gen in range(0, len(allPopularVideosUS)):
    some_args = {'rat': lambda rating: ((rating['likes']['sum'] - rating['dislikes']['max'])/rating['views']['sum'])*((rating['comment_count']['sum']/rating['views']['sum'])*100)}
    allPopularVideosUS = allPopularVideosUS.assign(**some_args)
# df['var2'] = pd.Series([round(val, 2) for val in df['var2']], index = df.index)
# df['var3'] = pd.Series(["{0:.2f}%".format(val * 100) for val in df['var3']], index = df.index)
allPopularVideosUS['rat'] = pd.Series([round(val, 4) for val in allPopularVideosUS['rat']], index=allPopularVideosUS.index)

allPopularVideosUS
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(allPopularVideosUS.loc[:, allPopularVideosUS.columns != 'rat'], allPopularVideosUS['rat'])


# from sklearn import preprocessing
# from sklearn import utils

# lab_enc = preprocessing.LabelEncoder()
# y_train = lab_enc.fit_transform(y_train)
# y_test = lab_enc.fit_transform(y_test)
from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression

regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)
print("Training Socre:{:.4f}".format(regr.score(X_train, y_train)*100))
print("Test Socre:{:.4f}".format(regr.score(X_test, y_test)*100))

