# import libraries

import pandas as pd

import numpy as np



import matplotlib.pylab as plt

import datetime

from mpl_toolkits.basemap import Basemap

from wordcloud import WordCloud, STOPWORDS

import random
%matplotlib inline

from matplotlib.pylab import rcParams
country = "Philippines"
# load dataset

df = pd.read_csv('../input/attacks_data_UTF8.csv',

                 encoding='latin1', parse_dates=['Date'],

                 infer_datetime_format=True,

                 index_col=1,

                )
df['Victims'] = df['Killed'] + df['Injured']
if country is not None:

    dfc=df.loc[df['Country']==country]

else:

    dfc = df
# Just a quick count of the data



country_rank = df.Country.value_counts().rank(numeric_only=True,ascending=False).loc[country]

country_attacks = df.Country.value_counts()[country]

country_killed = dfc.Killed.sum()

country_injured = dfc.Injured.sum()

print("%s is ranked %.0f with %d attacks resulting to %d deaths and %d injuries" % (country, country_rank, country_attacks, country_killed, country_injured))
dfc.City.value_counts().plot(kind='bar', figsize=(17, 7))

plt.title('Number of attacks by city')
dfc.groupby('City').sum()[['Victims','Killed', 'Injured']].sort_values(by='Victims',ascending=0).plot(kind='bar', figsize=(17, 7), subplots=True)
# Attack with most victims

most_victim = dfc.sort_values(by='Victims',ascending=False).head(1)

# most_victim.index.strftime("%Y-%m-%d")

print("Attack with most victims happened on %s on %s with %d killed, %d injuries with a total of %d victims with the following article: \n'%s' \n" % (most_victim.City.values[0], most_victim.index.strftime("%B %d,%Y")[0], most_victim.Killed, most_victim.Injured, most_victim.Victims, "%s" % most_victim.Description.values[0]))

# Attack with most killed

most_killed = dfc.sort_values(by='Killed',ascending=False).head(1)

print("Attack with the most deaths happened on %s on %s with %d killed, %d injuries with a total of %d victims with the following article: \n'%s' \n" % (most_killed.City.values[0], most_killed.index.strftime("%B %d,%Y")[0], most_killed.Killed, most_killed.Injured, most_killed.Victims, "%s" % most_killed.Description.values[0]))

#Attack with most injuries

most_injuries = dfc.sort_values(by='Injured',ascending=False).head(1)

print("Attack with the most injuries happened on %s on %s with %d killed, %d injuries with a total of %d victims with the following article: \n'%s' \n" % (most_injuries.City.values[0], most_injuries.index.strftime("%B %d,%Y")[0], most_injuries.Killed, most_injuries.Injured, most_injuries.Victims, "%s" % most_injuries.Description.values[0]))
# Over the years

dfc.groupby(dfc.index.year).sum()[['Victims','Killed', 'Injured']].sort_values(by='Victims',ascending=0).plot(kind='bar', figsize=(17, 7), subplots=False)
killedbyday = dfc.groupby([dfc.index.map(lambda x: x.weekday),dfc.index.year], sort=True).agg({'Killed': 'sum'})

rcParams['figure.figsize'] = 20, 10

killedbyday.unstack(level=0).plot(kind='bar', subplots=False)

killedbyday.unstack(level=1).plot(kind='bar', subplots=False)
# Check if there is a difference in attack victims by month

killedbymonth = dfc.groupby([dfc.index.map(lambda x: x.month),dfc.index.year], sort=True).agg({'Killed': 'sum'})

rcParams['figure.figsize'] = 20, 10

killedbymonth.unstack(level=0).plot(kind='bar', subplots=False)

killedbymonth.unstack(level=1).plot(kind='bar', subplots=False)
# Word cloud

text = dfc.Description.str.cat(sep=' ')

stopwords = set(STOPWORDS)
wc = WordCloud(background_color="white",max_words=100, stopwords=stopwords, margin=10,

               random_state=1).generate(text)
def grey_color_func(word, font_size, position, orientation, random_state=None, **kwargs):

    return "hsl(0, 0%%, %d%%)" % random.randint(20, 80)
default_colors = wc.to_array()

rcParams['figure.figsize'] = 10, 10

plt.title("Attack description word cloud")

plt.imshow(wc.recolor(color_func=grey_color_func, random_state=3))

plt.axis("off")

plt.figure()