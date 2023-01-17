# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
df = pd.read_csv("/kaggle/input/Youtube Videos.csv")
df.head()
df.shape
df.info()
# by default pandas is not capable of handling the trending date column as the date-time column

# here columns 1 and 5 are our date-time columns



df = pd.read_csv("/kaggle/input/Youtube Videos.csv", parse_dates = [1,5])

df.head()
# as the trending_date column isn't in the correct format pandas is unable to decode it to date-time type



df.info()
df.columns
# drop can be used to drop both rows and columns

# axis =1: Columns

# axis = 0: rows



df.drop(['tags','thumbnail_link', 'description' ], axis = 1, inplace = True)
df.head()
# which video is the most viewd video in the year 17-18



df.sort_values(by = "views", ascending = False)
# finding the videos that trended for most number of days

# if a video id is available 3 times that video trended for 3 days



df['video_id'].value_counts()
# this NAME? ids seem like the dropped videos



df['video_id'].value_counts()[1:]
# let's look at the title video trending for the most days



df[df['video_id'] == 'rRr1qiJRsXk'].head(1)
# we have one column in object format and other in datetime64 format



df.info()
# using pd.to_datetime

# the year is encoded as %y, day as %d and month as %m



df['trending_date'] = pd.to_datetime(df['trending_date'], format = "%y.%d.%m")

df.head()
# we still have an issue as one column has no timezone whereas the other column has a timezone associated with it



df.info()
df['publish_time']
# instead of splitting the column 

# to extract the date from a date-time column we use a method dt.date

# this goes for month, year, time as well



df['publish_time'].dt.date
# the above date format is of object

# converting it back to the date-time format



df['publish_time'] = pd.to_datetime(df['publish_time'].dt.date)
# now check the formats of both the columns



df.info()
df['trending_date']-df['publish_time']
df['days_took_to_trend'] = df['trending_date']-df['publish_time']
df.head()
# as we can see the format of days_took_to_trend isn't int



df.info()
# using the method dt.days to convert the format into int



df['days_took_to_trend'] = (df['trending_date']-df['publish_time']).dt.days

df['days_took_to_trend']
# videos that took zero days to trend



df[df['days_took_to_trend'] == 0]
# let's find out what are count of video in each bucket

# 1: 

# 2:

# 3:

# 0:

df['days_took_to_trend'].value_counts()
# Let's find out which video obtained the maximum views when it was trending on the same day



df_hits = df[df['days_took_to_trend'] == 0]

df_hits.sort_values(by = 'views', ascending = False)
# here we have complete duplicate rows 

# dropping the complete duplicate rows and keeping the first one



df_hits.drop_duplicates(inplace = True)
df_hits.sort_values(by = 'views', ascending = False)
# which particular channel have most number of same day trending videos?



df_hits['channel_title'].value_counts()[:20]
# which channel is having highest number of views when the video was trending on the same day?



df_hits.groupby("channel_title").agg({"views":"sum"}).sort_values(by = 'views', ascending = False)[:20]
# which channel is having highest number of likes when the video was trending on the same day?



df_hits.groupby("channel_title").agg({"likes":"sum"}).sort_values(by = 'likes', ascending = False)[:20]
# finding the count of videos per category for those videos that trended on the same day



df_hits['category_id'].value_counts()
# let's analyse the complete data



df.sort_values(by = "video_id", ascending = False)
# the original data has multiple entries for the same video

# kepping the final entry of each row



df.drop_duplicates(subset = ['video_id'], keep = 'last', inplace = True)

df
# finding the most viewed video of the year 2017-18



df.sort_values(by = "views", ascending  = False)
# what was the most viewed video for T-Series for the year 17-18



df[df['channel_title']=='T-Series'].sort_values(by = "views", ascending  = False)