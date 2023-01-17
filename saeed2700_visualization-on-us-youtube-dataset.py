import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import datetime

videos = pd.read_csv('/kaggle/input/youtube-new/USvideos.csv')

videos_json = pd.read_json('/kaggle/input/youtube-new/US_category_id.json')
videos.description = videos.description.fillna('no data')

videos.info()
videos.describe()
videos_json.info()
x = []

y = []

for i in range(len(videos_json['items'])):

    x.append(videos_json['items'][i]['snippet']['title'])

    y.append(videos.category_id[videos.category_id == int(videos_json['items'][i]['id'])].count())

df = pd.DataFrame.from_dict({'name':x,'count':y})

df = df.sort_values(by=['count'], ascending=False)

fig, ax = plt.subplots(figsize=(18,6))

plt.bar(df['name'],df['count'])

plt.title('Category frequency')

plt.xlabel('Category')

plt.ylabel('Frequency')

fig.autofmt_xdate()
data = {'year':['2017', '2018'], 'year_count':[0,0], 'month':[str(201700+x) if x < 13 else str(201800+(x-12)) for x in range(1,25)], 'month_count': [0 for x in range(1,25)]}

for i in videos.trending_date:

    date = datetime.datetime.strftime(datetime.datetime.strptime(i, '%y.%d.%m'), "%Y%m%d")

    if date[0:4] == '2017':

        data['year_count'][0] += 1

        data['month_count'][int(date[4:6])-1] += 1

    else:

        data['year_count'][1] += 1

        data['month_count'][int(date[4:6])+11] += 1

month_df = pd.DataFrame.from_dict({'month':data['month'], 'month_count':data['month_count']})

month_df = month_df.sort_values(by=['month_count'], ascending=False)

fig = plt.figure(figsize=(18,6))



plt.subplot2grid((3,2), (0,0))

plt.bar(data['year'], data['year_count'])

plt.xlabel('trending date')

plt.title('Year frequency')



plt.subplot2grid((3,2), (0,1))

plt.bar(data['month'], data['month_count'])

plt.xticks(rotation=45)

plt.xlabel('trending date')

plt.title('Month frequency without sorting')



plt.subplot2grid((3,2),(2,0))

plt.bar(month_df['month'][:8], month_df['month_count'][:8])

plt.xticks(rotation=45)

plt.ylim(2500, 6200)

plt.xlabel('trending date')

plt.title('Month frequency with sorting')



plt.subplot2grid((3,2),(2,1))

plt.bar(month_df['month'][:3], month_df['month_count'][:3])

plt.ylim(6180, 6200)

plt.xlabel('trending date')

plt.title('Month frequency (top 3)')
channels = {'title':[], 'count':[]}

for i in videos.channel_title.unique():

    channels['title'].append(i)

    channels['count'].append(videos.channel_title[videos.channel_title == i].count())

channels_df = pd.DataFrame.from_dict(channels)

channels_df = channels_df.sort_values(by=['count'], ascending=False)

channels_df = channels_df.reset_index(drop=True)

channels_df.head(20)
fig = plt.figure(figsize=(18,6))



plt.subplot2grid((3,2), (0,0))

plt.bar(channels_df['title'][:20], channels_df['count'][:20])

plt.title('Channels frequency (top 20)')

plt.xticks(rotation=90)

plt.ylim(150, 210)



plt.subplot2grid((3,2), (0,1))

plt.bar(channels_df['title'][:10], channels_df['count'][:10])

plt.title('Channels frequency (top 10)')

plt.xticks(rotation=90)

plt.ylim(180, 210)