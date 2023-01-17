import pandas as pd

tweets = pd.read_csv("../input/data_elonmusk.csv", encoding='latin1')

tweets = tweets.assign(Time=pd.to_datetime(tweets.Time)).drop('row ID', axis='columns')

tweets.head(3)
import seaborn as sns

import matplotlib.pyplot as plt

sns.set_style("dark")



(tweets

     .set_index('Time')

     .groupby(pd.TimeGrouper('1D'))

     .Tweet

     .count()

     .value_counts()

     .sort_values(ascending=False)

).plot.bar(figsize=(14, 7), fontsize=16, color='lightcoral')

plt.gca().set_title('@elonmusk number of tweets per day', fontsize=20)
(tweets.Time

     .dt

     .hour

     .value_counts()

     .sort_index()

).plot.bar(figsize=(14, 7), fontsize=16, color='lightcoral')

plt.gca().set_title('@elonmusk tweets per hour of day', fontsize=20)
d = (tweets

     .set_index('Time')

     .groupby(pd.TimeGrouper('1D'))

     .Tweet

     .count()

     .sort_index()

     .reset_index()

    )

fig = plt.figure(figsize=(14, 7))

ax = plt.gca()

sns.regplot(d.index.values, d.Tweet.values, ax=ax, color='lightcoral')

ax.set_title('@elonmusk tweets per day of year', fontsize=20)
tweets['Retweet from'].notnull().value_counts() / len(tweets)
tweets['Retweet from'].value_counts().head(20).plot.bar(

    figsize=(14, 7), fontsize=16, color='lightcoral'

)

plt.gca().set_title('@elonmusk top retweet sources', fontsize=20)

plt.gca().set_xticklabels(plt.gca().get_xticklabels(), rotation=45, ha='right', fontsize=16)

pass
tweets.Tweet.str.contains('https://').value_counts() / len(tweets)
tweets.Tweet.str.contains('@').value_counts() / len(tweets)
import itertools



c = list(

itertools.chain(

    *tweets.Tweet.map(lambda t: [handle.replace(":", "")[1:] for handle in t.split(" ") 

                            if '@' in handle.replace(":", "")]).tolist())

)



pd.Series(c).value_counts().head(20).plot.bar(

    figsize=(14, 7), fontsize=16, color='lightcoral'

)

plt.gca().set_title('@elonmusk top user tags', fontsize=20)

plt.gca().set_xticklabels(plt.gca().get_xticklabels(), rotation=45, ha='right', fontsize=16)

pass
from nltk import word_tokenize

tokens = tweets.Tweet.map(word_tokenize)



def what_does_elon_think_about(x):

    x_l = x.lower()

    x_t = x.title()

    return tweets.loc[tokens.map(lambda sent: x_l in sent or x_t in sent).values]
what_does_elon_think_about('Trump').Tweet.values.tolist()
what_does_elon_think_about('oil').Tweet.values.tolist()
what_does_elon_think_about('life').Tweet.values.tolist()