%matplotlib inline

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import pylab as pl
sentinels = {'speaking_line': ['FALSE'], 'normalized_text': ['']}

cols = ['id', 'episode_id',	'number',  'raw_text',	'timestamp_in_ms',	'speaking_line',	

        'character_id',	'location_id',	'raw_character_text',	'raw_location_text',

        'spoken_words',	'normalized_text',	'word_count']

df = pd.read_csv("../input/simpsons_script_lines.csv",

                    names = cols,

                    error_bad_lines=False,

                    warn_bad_lines=False,

                    low_memory=False,

                    na_values = sentinels)

print(len(df))

df = df.dropna()

print(len(df))
df.describe()
lines_ep_raw = df.groupby(['episode_id']).size().sort_values(ascending=True)

lines_per_episode = lines_ep_raw[1:]

lines_per_episode.describe()
lines_per_char_ep = df['word_count'].groupby([df['character_id'], df['episode_id']])

lines_per_char_ep.describe()
data = [ lines_per_episode ]



bins = np.linspace(0, 350,50)

bins2 = np.linspace(0, 350, 100)

plt.figure(1)

plt.boxplot(data)
# Calculate the statistics from the data to create the "best fit" normal distributions

mean = np.mean(data)

var = np.var(data)

sdev = np.sqrt(var)

pl.hist(data,bins,normed = 'true',color = 'blue')

pl.plot(bins2,pl.normpdf(bins2,mean,sdev), color = 'red')

pl.xlabel('lines per episode')