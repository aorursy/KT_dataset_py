import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
import os

for i in [i for i in os.listdir(".") if "US" in i]:

    print(i)
df = pd.read_csv('../input/youtube-new/USvideos.csv')

display(df.head())

display(df.info())

display(df.describe())
channels = df.groupby('channel_title').count()['video_id'].reset_index('channel_title')

channels = channels[channels['video_id'] > 10]

channels.sort_values('video_id', ascending=False).tail()
rb = df[df['channel_title'].str.contains('Red Bull')][['video_id','publish_time', 'views']]

rb.head(5)
rb['publish_time'] = pd.to_datetime(rb['publish_time']).dt.date

rb['days_passed'] = pd.datetime.now().date() - rb['publish_time']

rb['days_passed'] = rb['days_passed'].dt.days

rb.head()

rb['views per day'] = rb['views'] / rb['days_passed']

rb.head()