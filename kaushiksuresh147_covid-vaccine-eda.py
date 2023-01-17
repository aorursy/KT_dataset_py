



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline 

from wordcloud import WordCloud, STOPWORDS

import missingno as mno



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



tweets_df = pd.read_csv("../input/covidvaccine-tweets/covidvaccine.csv")
tweets_df.info()
tweets_df.describe()
tweets_df.head(10)
#Lets visualize the missing values initially!

mno.matrix(tweets_df)
tweets_df.isna().sum()
# lET'S Look into the unique values 
def unique_values(data):

    total = data.count()

    tt = pd.DataFrame(total)

    tt.columns = ['Total']

    uniques = []

    for col in data.columns:

        unique = data[col].nunique()

        uniques.append(unique)

    tt['Uniques'] = uniques

    tt['Percentage']=tt['Uniques']/tt['Total']

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
# He used the same message but tagging different people for higher reach and support to his claim!

tweets_df[tweets_df['user_name']=='katapult']['text'].iloc[4:7]
def plot_count(feature, title, df, size=1, ordered=True):

    f, ax = plt.subplots(1,1, figsize=(4*size,4))

    total = float(len(df))

    if ordered:

        g = sns.countplot(df[feature], order = df[feature].value_counts().index[:20], palette='Set3')

    else:

        g = sns.countplot(df[feature], palette='Set3')

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
plot_count("user_name", "User name", tweets_df,4)
plot_count("user_location", "User location", tweets_df,4)
plot_count("source", "Source", tweets_df,4)

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
india_df = tweets_df.loc[tweets_df.user_location=="India"]

show_wordcloud(india_df['text'], title = 'Prevalent words in tweets from India')
us_df = tweets_df.loc[tweets_df.user_location=="United States"]

show_wordcloud(us_df['text'], title = 'Prevalent words in tweets from US')
us_df = tweets_df.loc[tweets_df.user_location=="Australia"]

show_wordcloud(us_df['text'], title = 'Prevalent words in tweets from Australia')
def plot_features_distribution(features, title, df, isLog=False):

    plt.figure(figsize=(12,6))

    plt.title(title)

    for feature in features:

        if(isLog):

            sns.distplot(np.log1p(df[feature]),kde=True,hist=False, bins=120, label=feature)

        else:

            sns.distplot(df[feature],kde=True,hist=False, bins=120, label=feature)

    plt.xlabel('')

    plt.legend()

    plt.show()
tweets_df['hashtags'] = tweets_df['hashtags'].replace(np.nan, "['None']", regex=True)

tweets_df['hashtags'] = tweets_df['hashtags'].apply(lambda x: x.replace('\\N',''))

tweets_df['hashtags_count'] = tweets_df['hashtags'].apply(lambda x: len(x.split(',')))

plot_features_distribution(['hashtags_count'], 'Hashtags per tweet (all data)', tweets_df)
tweets_df['hashtags_individual'] = tweets_df['hashtags'].apply(lambda x: x.split(','))

from itertools import chain

all_hashtags = set(chain.from_iterable(list(tweets_df['hashtags_individual'])))

print(f"There are totally: {len(all_hashtags)}")

show_wordcloud(tweets_df['hashtags_individual'], title = 'Prevalent words in hashtags')
### Kindly upvote the kernel & dataset if you find it useful!!!