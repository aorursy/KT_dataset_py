import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle, islice
import datetime as dt
%matplotlib inline
import matplotlib.pylab as pylab
import seaborn as sb
pylab.rcParams['figure.figsize'] = 10, 8
data_file = '../input/Tweets.csv'
parser = lambda x: dt.datetime.strptime(x[:-6], '%Y-%m-%d %H:%M:%S')
tweets = pd.read_csv(data_file, index_col = 'tweet_id',
                     parse_dates=[12], date_parser = parser)
pd.options.display.max_rows = 8
tweets[['airline_sentiment','airline', 'retweet_count', 
        'text', 'tweet_created']]
tweets['dow'] = tweets.tweet_created.dt.dayofweek

g = sb.FacetGrid(tweets, row = 'airline_sentiment', 
                 hue = 'airline', legend_out = True,
                 aspect = 4, size = 2.5)
g.map(sb.distplot, 'dow', hist = False)
g.add_legend()
g.axes.flat[0].set_xlim(0,6)
g.axes.flat[2].set_xlabel('Day of Week')
groups = tweets.groupby([tweets.airline, 
                         tweets.airline_sentiment])

retweet_table = groups.retweet_count.apply(sum)
my_colors = list(islice(cycle(['r', 'b', 'g']), 
                        None, len(retweet_table)))
fig, ax = plt.subplots(3, sharex = True)
groups.count().name.plot(kind = 'bar', color = 
                         my_colors, title = 
                         '# of Tweets', ax = ax[0])

retweet_table.plot(kind = 'bar', color= my_colors, 
                   title = '# of Retweets', ax = ax[1])
(retweet_table/groups.count().name).plot(
    kind = 'bar', color = my_colors, 
    title = 'Retweet Efficiency', ax = ax[2])