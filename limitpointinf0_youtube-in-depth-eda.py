import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

print(os.listdir("../input"))
pd.read_csv('../input/USvideos.csv').head()
#read in USvideos data
df = pd.read_csv('../input/USvideos.csv')

#read in the json file for mapping category id to its category title
import json
with open('../input/US_category_id.json') as f:
    data = json.load(f)
cats = pd.DataFrame(data['items'])
cats.head()
cats['category'] = cats['snippet'].map(lambda x: x['title'])
cats = cats.drop(['etag', 'kind', 'snippet'], axis=1)
cats.columns = ['category_id', 'category']
cats.category_id = cats.category_id.map(lambda x: int(x))
#print(cats.info())

#joining the videos dataframe
df = pd.merge(df, cats, how='left', on='category_id')
df = df.drop(['video_id', 
              'category_id', 
              'tags', 
              'thumbnail_link', 
              'comments_disabled', 
              'ratings_disabled', 
              'video_error_or_removed',
              'description'], axis=1)

#preprocessing for datetime strings
from datetime import datetime
df.trending_date = pd.to_datetime(df.trending_date, format='%y.%d.%m')
df.publish_time = pd.to_datetime(df.publish_time, format='%Y-%m-%dT%H:%M:%S.%fZ')

#df.head()
plt.figure(figsize=(10,5))
ax = sns.heatmap(df.isnull(), yticklabels=False)
#grabbing time difference in days
df['trend_diff'] = df.trending_date - np.array(df.publish_time.dt.date)
df.trend_diff = df.trend_diff.dt.days

df.trend_diff.describe()
import scipy.stats as ss

#filter to 99%
p = 99
high = df[df.trend_diff < np.percentile(df.trend_diff, 99)]

#define helper functions
def ecdf(data):
    n=len(data)
    x=np.sort(data)
    y=np.arange(1,n+1)/n
    return x,y

def get_exponential(x_range, mu=0, sigma=1, cdf=False):
    x = x_range
    if cdf:
        y = ss.expon.cdf(x, mu, sigma)
    else:
        y = ss.expon.pdf(x, mu, sigma)
    return x, y

x, y = ecdf(high.trend_diff)
x_exp, y_exp = get_exponential(x, mu=np.mean(high.trend_diff), sigma=np.std(high.trend_diff), cdf=True)
plt.figure(figsize=(10,5))
plt.title('Cumulative Distribution of Days Before Trending to 99th percentile')
plt.plot(x, y, label='ecdf to {}%'.format(p))
plt.plot(x_exp, y_exp, label='exp distribution mu={}, sigma={}'.format("%.2f" % np.mean(high.trend_diff), "%.2f" % np.std(high.trend_diff)))
plt.legend()
plt.show()
plt.figure(figsize=(15,5))
plt.title('Days Before Trending by Category')
ax = sns.boxplot(x="category", y="trend_diff", data=high)
plt.xticks(rotation=80)
plt.ylabel('Days Before Trending')
plt.show()
df['likes_views'] = df['likes']/df['views']
df['dislikes_views'] = df['dislikes']/df['views']

plt.figure(figsize=(15,5))
plt.title('Likes/Views Ratio By Category')
ax = sns.boxplot(x="category", y="likes_views", data=df)
plt.xticks(rotation=80)
plt.ylabel('Ratio Likes/Views')
plt.show()

plt.figure(figsize=(15,5))
plt.title('Dislikes/Views Ratio By Category')
ax = sns.boxplot(x="category", y="dislikes_views", data=df)
plt.xticks(rotation=80)
plt.ylabel('Ratio Dislikes/Views')
plt.show()
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
import string

def plot_wordcloud(text, title=None, max = 1000, size=(10,5), title_size=16):
    """plots wordcloud"""
    wordcloud = WordCloud(max_words=max).generate(text)
    plt.figure(figsize=size)
    plt.title(title, size=title_size)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")

def clean_ColText(df, col, stem=True):
    """Takes dataframe and column name and returns a dataframe with cleaned strings in the form of a list of word tokens. 
    Stemming is an option."""
    table = str.maketrans('', '', string.punctuation)
    df[col] = df[col].map(lambda x: x.translate(table)) #remove punctuation
    df[col] = df[col].map(lambda x: x.lower()) #lowercase
    df[col] = df[col].apply(word_tokenize) #tokenize
    stop_words = set(stopwords.words('english'))
    df[col] = df[col].map(lambda x: [y for y in x if not y in stop_words]) #remove stop words
    df[col] = df[col].map(lambda x: [y for y in x if y not in ["’","’","”","“","‘","—"]]) #remove smart quotes and other non alphanums
    if stem:
        porter = PorterStemmer()
        df[col] = df[col].map(lambda x: [porter.stem(y) for y in x])
        return df
    return df

df = clean_ColText(df, 'title')
txt = ' '.join(sum([x for x in df.title],[]))
plot_wordcloud(txt, title='Youtube Video Titles',size=(15,10))
#grab videos with number of views over 75th percentile
for c in df.category.unique():
    temp = df[df.category == c]
    temp = temp[temp.views > np.percentile(temp.views,75)]
    txt = ' '.join(sum([x for x in temp.title],[]))
    plot_wordcloud(txt, title=c)