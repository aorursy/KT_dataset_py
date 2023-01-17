import numpy as np 

import pandas as pd

import matplotlib

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline 

from wordcloud import WordCloud, STOPWORDS
tweets_df = pd.read_csv("/kaggle/input/trump-tweets/trump_tweets.csv")
print(f"data shape: {tweets_df.shape}")
tweets_df.info()
tweets_df.describe()
tweets_df.head()
def missing_data(data):

    total = data.isnull().sum()

    percent = (data.isnull().sum()/data.isnull().count()*100)

    tt = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    types = []

    for col in data.columns:

        dtype = str(data[col].dtype)

        types.append(dtype)

    tt['Types'] = types

    return(np.transpose(tt))
missing_data(tweets_df)
def unique_values(data):

    total = data.count()

    tt = pd.DataFrame(total)

    tt.columns = ['Total']

    uniques = []

    for col in data.columns:

        unique = data[col].nunique()

        uniques.append(unique)

    tt['Uniques'] = uniques

    return(np.transpose(tt))
unique_values(tweets_df)
def most_frequent_values(data):

    total = data.count()

    tt = pd.DataFrame(total)

    tt.columns = ['Total']

    items = []

    vals = []

    for col in data.columns:

        itm = data[col].value_counts().index[0]

        val = data[col].value_counts().values[0]

        items.append(itm)

        vals.append(val)

    tt['Most frequent item'] = items

    tt['Frequence'] = vals

    tt['Percent from total'] = np.round(vals / total * 100, 3)

    return(np.transpose(tt))
most_frequent_values(tweets_df)
tweets_df['datedt'] = pd.to_datetime(tweets_df['date'])
tweets_df['year'] = tweets_df['datedt'].dt.year

tweets_df['month'] = tweets_df['datedt'].dt.month

tweets_df['day'] = tweets_df['datedt'].dt.day

tweets_df['dayofweek'] = tweets_df['datedt'].dt.dayofweek

tweets_df['hour'] = tweets_df['datedt'].dt.hour

tweets_df['minute'] = tweets_df['datedt'].dt.minute
tweets_df['dated'] = tweets_df.apply(lambda x: x.date[0:10], axis=1)

tweets_df['dated'] = pd.to_datetime(tweets_df['dated'])
def plot_count(feature, title, df, size=1):

    f, ax = plt.subplots(1,1, figsize=(4*size,4))

    total = float(len(df))

    g = sns.countplot(df[feature], order = df[feature].value_counts().index[:31], palette='Set3')

    g.set_title("Number and percentage of {}".format(title))

    if(size > 2):

        plt.xticks(rotation=90, size=8)

    for p in ax.patches:

        height = p.get_height()

        ax.text(p.get_x()+p.get_width()/2.,

                height + 0.2,

                '{:1.2f}%'.format(100*height/total),

                ha="center") 

    plt.show()    
plot_count("dayofweek", "tweets per day of week", tweets_df, size=2)
plot_count("hour", "tweets per hour", tweets_df, size=3)
plot_count("day", "tweets per day of month", tweets_df, size=4)
def plot_count(feature, title, df, size=1):

    f, ax = plt.subplots(1,1, figsize=(4*size,4))

    total = float(len(df))

    g = sns.countplot(df[feature], order = df[feature].value_counts().index[:20], palette='Set3')

    g.set_title("Number and percentage of {}".format(title))

    if(size > 2):

        plt.xticks(rotation=90, size=8)

    for p in ax.patches:

        height = p.get_height()

        ax.text(p.get_x()+p.get_width()/2.,

                height + 3,

                '{:1.2f}%'.format(100*height/total),

                ha="center") 

    plt.show()    
stopwords = set(STOPWORDS)



def show_wordcloud(data, title = None):

    wordcloud = WordCloud(

        background_color='white',

        stopwords=stopwords,

        max_words=50,

        max_font_size=40, 

        scale=5,

        random_state=1

    ).generate(str(data))



    fig = plt.figure(1, figsize=(10,10))

    plt.axis('off')

    if title: 

        fig.suptitle(title, fontsize=20)

        fig.subplots_adjust(top=2.3)



    plt.imshow(wordcloud)

    plt.show()
show_wordcloud(tweets_df['text'], title = 'Prevalent words in tweets')
tweets_df['hashtags'] = tweets_df['hashtags'].replace(np.nan, "['None']", regex=True)

tweets_df['hashtags'] = tweets_df['hashtags'].apply(lambda x: x.replace('\\N',''))

tweets_df['hashtags_count'] = tweets_df['hashtags'].apply(lambda x: len(x.split(',')))
tweets_df['hashtags_individual'] = tweets_df['hashtags'].apply(lambda x: x.split(','))

from itertools import chain

all_hashtags = set(chain.from_iterable(list(tweets_df['hashtags_individual'])))

print(f"There are totally: {len(all_hashtags)}: {all_hashtags}")
for hashtag in all_hashtags:

    _d_df = tweets_df.loc[tweets_df.hashtags==hashtag]

    print(f"Hashtag: {hashtag}, tweets: {_d_df.shape[0]}")
def plot_time_variation(df, x='date', y='retweets', hue=None, size=1, title="", is_log=False):

    f, ax = plt.subplots(1,1, figsize=(4*size,3*size))

    g = sns.lineplot(x=x, y=y, hue=hue, data=df)

    plt.xticks(rotation=90)

    if hue:

        plt.title(f'{y} grouped by {hue} | {title}')

    else:

        plt.title(f'{y} | {title}')

    if(is_log):

        ax.set(yscale="log")

    ax.grid(color='black', linestyle='dotted', linewidth=0.75)

    plt.show() 
plot_time_variation(tweets_df, x='dated', y='retweets', title= 'day averages and variation', size=4, is_log=False)
plot_time_variation(tweets_df, x='dated', y='favorites', title='day averages and variation',size=4, is_log=True)
plot_time_variation(tweets_df, x='dated', y='user_followers', title='day average and variation', size=4, is_log=False)
pd.set_option('display.max_colwidth', 150)

tweets_ordered_df = tweets_df.sort_values(by=["retweets"], ascending=False)

tweets_ordered_df[["text", "date", "hashtags","day", "hour",  "dayofweek", "retweets", "favorites"]].head(5)
pd.set_option('display.max_colwidth', 150)

tweets_ordered_df = tweets_df.sort_values(by=["favorites"], ascending=False)

tweets_ordered_df[["text", "date", "hashtags","day", "hour", "dayofweek", "retweets", "favorites"]].head(5)