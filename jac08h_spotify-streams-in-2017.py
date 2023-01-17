import pandas as pd

import numpy as np



import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline
# color range for graphs

colors = [(0, x, 1) for x in np.arange(0,1,0.1)]
songs = pd.read_csv('../input/data.csv')

songs.head(3)
songs['Date'] = pd.to_datetime(songs['Date'])

# work only with songs from 2017

first_2018_day = pd.datetime(2018, 1, 1)

songs = songs[songs['Date'] < first_2018_day]
songs_global = songs[songs['Region']=='global']

s = songs_global.sort_values(['Date', 'Position'])
artists_by_streams = s.groupby('Artist')['Streams'].sum()

artists_by_streams.sort_values(ascending=False, inplace=True)

top10 = artists_by_streams.iloc[:10]

top10
plt.figure(figsize=(20,10))

plt.title('Most streamed artists on spotify in 2017')

plt.bar(x=top10.index, height=top10, color=colors)
songs_by_streams = s.groupby('Track Name')['Streams'].sum()

songs_by_streams.sort_values(ascending=False, inplace=True)

songs_by_streams[:10]
1470919913 / 4467942169 * 100
s.sort_values('Streams', ascending=False).iloc[0]
start = pd.datetime(2017, 1, 1)

end = pd.datetime(2017, 12, 31)



between = s[(s['Date'] >= start) & (s['Date'] <= end)]

lil_peep = between.groupby(['Artist', 'Date'])['Streams'].count()['Lil Peep']
all_days = pd.Series(index=pd.date_range(start, end))

for day in all_days.index:

    if day not in lil_peep.index:

        lil_peep[day] = 0

lil_peep.sort_index(inplace=True)
plt.bar(lil_peep.index, lil_peep)
lil_peep[lil_peep > 0]