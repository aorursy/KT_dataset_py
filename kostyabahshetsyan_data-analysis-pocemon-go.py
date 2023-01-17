import pandas as pd

import numpy as np



import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



from collections import Counter



sns.set(style="whitegrid")

sns.set_color_codes("pastel")
columns = ['latitude', 'longitude', 'appearedLocalTime', 'appearedHour',

'appearedDay', 'city', 'temperature', 'population_density', 'class']



pcm = pd.read_csv('../input/300k.csv', low_memory=False, usecols=columns)
pcm.head()
pcm.info()
uniq_cls = Counter(pcm['class'])



len(uniq_cls)
cnt = pcm.groupby('city')['city'].size()



f, ax = plt.subplots(figsize=(10, 20))

sns.barplot(x=cnt.values, y=cnt.index, color='b', ax=ax)

plt.setp(ax.patches, linewidth=0)

texts = ax.set(title="Count activity by City")
# extract feature local time

pcm['LocalTime'] = pcm.appearedLocalTime.apply(lambda x: x.split("T")[1])



# grouping data by time (hour)

loc_time = pcm.groupby(['city', pcm.LocalTime.map(lambda x: int(x.split(":")[0]))]).size()

loc_time = loc_time.unstack()

loc_time.fillna(0, inplace=True)



# plot grouping data

f, axes = plt.subplots(len(loc_time.columns), 1, figsize=(10, 40),sharex=True)

for i in range(len(loc_time.columns)):

	sns.barplot(x=loc_time.index, y=loc_time[i], ax=axes[i])

	axes[i].set(ylabel="Count", title="Pocemon in City activity at %2d:00 Local time" %(i))

	plt.setp(axes[i].patches, linewidth=0)

	plt.setp(axes[i].get_xticklabels(), rotation=90, fontsize=9)
import plotly.plotly as py