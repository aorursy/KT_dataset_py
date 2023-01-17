# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/the-tweets-of-wisdom/tweets.csv')
df.head()
df.describe()
df.info()
df.isna().any()
df[df.tweet_content.isna()]
df[df.author_name.isna()]
df = df.dropna()
df.isna().any()
print(len(df.author_name.unique()))
df.handle.value_counts().head(10)
# let's take top 10 authors

top_ten_authors = df.handle.value_counts().head(10).index
top_ten_authors
plt.figure(figsize=(15,10))

sns.distplot(df.author_name.value_counts())
plt.figure(figsize=(15,10))

sns.distplot(df.likes)
plt.figure(figsize=(15,10))

print(df[df['author_name'] == 'Thomas Sowell'].likes.mean(), df[df['author_name'] == 'Thomas Sowell'].likes.std())

sns.distplot(df[df['author_name'] == 'Thomas Sowell'].likes)
plt.figure(figsize=(15,10))

print(df[df['author_name'] == 'Thomas Sowell'].retweets.mean(), df[df['author_name'] == 'Thomas Sowell'].retweets.std())

sns.distplot(df[df['author_name'] == 'Thomas Sowell'].retweets)
plt.figure(figsize=(10,8))

sns.heatmap(df[['likes', 'retweets']].corr(), annot=True)
plt.figure(figsize=(10,10))

p  = pd.concat([df.groupby('handle').likes.mean(), df.groupby('handle').retweets.mean(), df.groupby('handle').likes.count()], axis=1 )

p.columns = ['Mean Likes per Author', 'Mean Retweets per Author' ,'Number of tweets per authors']

sns.heatmap(p.corr() ,annot=True)
df.head()
plt.figure(figsize=(15,10))

# Sort the dataframe by target

target_0 = df[df['handle'] == 'ThomasSowell']

target_1 = df[df['handle'] == 'TheAncientSage']

target_2 = df[df['handle'] == 'orangebook_']



sns.distplot(target_0[['likes']], hist=False)

sns.distplot(target_1[['likes']], hist=False)

sns.distplot(target_2[['likes']], hist=False)



plt.show()
plt.figure(figsize=(15,10))

# Sort the dataframe by target

target_0 = df[df['handle'] == 'ThomasSowell ']

target_1 = df[df['handle'] == 'TheAncientSage']

target_2 = df[df['handle'] == 'orangebook_']



sns.distplot(target_0[['retweets']], hist=False)

sns.distplot(target_1[['retweets']], hist=False)

sns.distplot(target_2[['retweets']], hist=False)

fig, ax = plt.subplots(figsize=(20,10))

sns.boxplot(x="handle", y="likes", data=df[df['handle'].isin(top_ten_authors)], ax=ax)

plt.show()
fig, ax = plt.subplots(figsize=(20,10))

sns.violinplot(x="handle", y="likes", data=df[df['handle'].isin(top_ten_authors)], ax=ax)

plt.show()
l  = top_ten_authors.to_list()

l.remove('ThomasSowell')

fig, ax = plt.subplots(figsize=(15,10))

sns.boxplot(x="handle", y="retweets", data=df[df['handle'].isin(l)], ax=ax)

plt.show()
l  = top_ten_authors.to_list()

l.remove('ThomasSowell')

fig, ax = plt.subplots(figsize=(15,10))

sns.violinplot(x="handle", y="likes", data=df[df['handle'].isin(l)], ax=ax)

plt.show()
# Ratio of Likes and Retweets

fig, ax = plt.subplots(figsize=(15,10))

sns.distplot(df.likes / (1 + df.retweets))
print('Mean of the ratio distribution', (df.likes / (1 + df.retweets)).mean(), '\nStandard Deviation of the ratio distribution',(df.likes / (1 + df.retweets)).std())
sns.distplot(df[df.handle=='ThomasSowell'].likes)
print('Mean of the likes in ThomasSowell tweets', df[df.handle=='ThomasSowell'].likes.mean(), '\nStandard Deviation of the likes in ThomasSowell tweets', df[df.handle=='ThomasSowell'].likes.std())
fig, ax = plt.subplots(3, 3, figsize=(15,15))

for i in range(3):

    for j in range(3):

        sns.distplot(df[df.handle==top_ten_authors[i*3 + j  + 1]].likes, ax=ax[i][j])
for i in range(3):

    for j in range(3):

        print('Mean of the likes in', top_ten_authors[i*3 + j  + 1], 'tweets', df[df.handle==top_ten_authors[i*3 + j  + 1]].likes.mean(), '\nStandard Deviation of the likes in', top_ten_authors[i*3 + j  + 1], 'tweets', df[df.handle==top_ten_authors[i*3 + j + 1]].likes.std(), '\n')
plt.figure(figsize=(15,10))

dis = df.groupby('handle').filter(lambda group: group.size > 10)

sns.distplot(dis.groupby('handle').likes.mean() - dis.groupby('handle').likes.std())
df.head()
import matplotlib.dates as mdates

from matplotlib.dates import DateFormatter
df.created_at.head()
df.created_at = pd.to_datetime(df.created_at, format='%Y-%m-%d %H:%M:%S')
# Create the plot space upon which to plot the data

fig, ax = plt.subplots(figsize=(30, 15))



# Add the x-axis and the y-axis to the plot

ax.plot(df.created_at,

        df.likes, '-o',

        color='purple')



# Clean up the x axis dates (reviewed in lesson 4)

ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))

ax.xaxis.set_major_formatter(DateFormatter("%m"))



plt.show()
df['creation_month'] = df.created_at.dt.month

df['creation_day'] = df.created_at.dt.day

df['creation_year'] = df.created_at.dt.year

df['creation_hour'] = df.created_at.dt.hour
# Create the plot space upon which to plot the data

fig, ax = plt.subplots(figsize=(15, 10))





# Add the x-axis and the y-axis to the plot

ax.plot(np.sort(df.creation_month.unique()),

        df.groupby('creation_month').likes.sum(), '-o',

        color='purple')



plt.show()
plt.subplots(figsize=(15, 10))

sns.countplot(df.creation_month)
# Create the plot space upon which to plot the data

fig, ax = plt.subplots(figsize=(15, 10))





# Add the x-axis and the y-axis to the plot

ax.plot(np.sort(df.creation_month.unique()),

        df.groupby('creation_month').likes.mean(), '-o',

        color='purple')



plt.show()
# Create the plot space upon which to plot the data

fig, ax = plt.subplots(figsize=(15, 10))





# Add the x-axis and the y-axis to the plot

ax.plot(np.sort(df.creation_year.unique()),

        df.groupby('creation_year').likes.sum(), '-o',

        color='purple')



plt.show()
plt.subplots(figsize=(15, 10))

sns.countplot(df.creation_year)
# Create the plot space upon which to plot the data

fig, ax = plt.subplots(figsize=(15, 10))





# Add the x-axis and the y-axis to the plot

ax.plot(np.sort(df.creation_year.unique()),

        df.groupby('creation_year').likes.mean(), '-o',

        color='purple')



plt.show()
# Create the plot space upon which to plot the data

fig, ax = plt.subplots(figsize=(15, 10))





# Add the x-axis and the y-axis to the plot

ax.plot(np.sort(df.creation_hour.unique()),

        df.groupby('creation_hour').likes.sum(), '-o',

        color='purple')



plt.show()
plt.subplots(figsize=(15, 10))

sns.countplot(df.creation_hour)
# Create the plot space upon which to plot the data

fig, ax = plt.subplots(figsize=(15, 10))





# Add the x-axis and the y-axis to the plot

ax.plot(np.sort(df.creation_hour.unique()),

        df.groupby('creation_hour').likes.mean(), '-o',

        color='purple')



plt.show()
plt.subplots(figsize=(25, 15))

sns.countplot(df.creation_month, hue=df.creation_year)
plt.subplots(figsize=(25, 15))

sns.countplot(df.creation_day, hue=df.creation_year)
plt.subplots(figsize=(25, 15))

sns.countplot(df.creation_hour, hue=df.creation_year)
plt.subplots(figsize=(25, 15))

sns.countplot(df.creation_day, hue=df.creation_month)
sns.heatmap(df[['likes', 'creation_year', 'creation_month', 'creation_hour']].corr(), annot=True)
fig, ax = plt.subplots(3, 3, figsize=(15,15))

for i in range(3):

    for j in range(3):

        ax[i][j].plot(np.sort(df[df['handle'] == top_ten_authors[3 * i + j]].creation_year.unique()), df[df['handle'] == top_ten_authors[3 * i + j]].groupby('creation_year').likes.mean(), '-o', color='purple')
from datetime import datetime
(datetime.today() - df.groupby('handle').created_at.min()).dt.days
sns.heatmap(pd.concat([df.groupby('handle').likes.mean(), (datetime.today() - df.groupby('handle').created_at.min()).dt.days], axis=1).corr(), annot=True)
sns.heatmap(pd.concat([df.groupby('handle').likes.sum(), (datetime.today() - df.groupby('handle').created_at.min()).dt.days], axis=1).corr(), annot=True)
from scipy import stats

from scipy.ndimage.interpolation import shift

## Outliers("Viral tweets") are in yellow and else are in blue

fig, ax = plt.subplots(3, 3, figsize=(30,15))

for i in range(3):

    for j in range(3):

        ax[i][j].scatter(df[df['handle'] == top_ten_authors[3 * i + j]].created_at, df[df['handle'] == top_ten_authors[3 * i + j]].likes, facecolors='blue',alpha=.85, s=30)

        ax[i][j].scatter(df[df['handle'] == top_ten_authors[3 * i + j]][(np.abs(stats.zscore(df[df['handle'] == top_ten_authors[3 * i + j]].likes)) > 3)].created_at, df[df['handle'] == top_ten_authors[3 * i + j]][(np.abs(stats.zscore(df[df['handle'] == top_ten_authors[3 * i + j]].likes)) > 3)].likes, color="yellow")
def plot_avg_likes_between_viral_twts(author, ax):

    t = df[df['handle'] == author].sort_values(by = 'created_at').reset_index()

    idx = t[(np.abs(stats.zscore(t.likes)) > 3)].index

    idx_created = t[(np.abs(stats.zscore(t.likes)) > 3)].created_at

    x = t.created_at

    y = [0 for i in range(t.shape[0])]

    y2 = []

    prev = 0

    for i in idx:

        m = t[prev:i].likes.mean()

        for j in range(prev, i):

            y[j] = m

        y[i] = None

        y2.append(t.iloc[i].likes)

        prev = i + 1

    ax.plot(x, y, linewidth=2)

    ax.scatter(idx_created, y2, color="red")

    ax.set_title(author)
## The red ones are the viral tweets and the blue one is average of likes between two consecutive viral tweets

fig, ax = plt.subplots(3, 3, figsize=(30,20))

for i in range(3):

    for j in range(3):

        plot_avg_likes_between_viral_twts(top_ten_authors[3 * i + j], ax[i][j])
def change_in_likes_after_viral_twts(author, ax):

    t = df[df['handle'] == author].sort_values(by = 'created_at').reset_index()

    idx = t[(np.abs(stats.zscore(t.likes)) > 3)].index

    delta_change = []

    prev = 0

    prev_del = 0

    for i in idx:

        m = t[prev:i].likes.mean() - prev_del

        if(np.isnan(m) == False):

            delta_change.append(m)

            prev_del =  t[prev:i].likes.mean()

        prev = i + 1

    sns.distplot(np.array(delta_change) - shift(delta_change, 1, cval=0), ax=ax).set_title(author)
## The red ones are the viral tweets and the blue one is average of likes between two consecutive viral tweets

fig, ax = plt.subplots(3, 3, figsize=(30,20))

for i in range(3):

    for j in range(3):

        change_in_likes_after_viral_twts(top_ten_authors[3 * i + j], ax[i][j])
## Duration between two consecutive tweets distribution

fig, ax = plt.subplots(3, 3, figsize=(30,20))

for i in range(3):

    for j in range(3):

        sns.distplot((df[df['handle'] == top_ten_authors[3 * i + j]].sort_values(by = 'created_at').created_at.diff() / np.timedelta64(1, 'h')).dropna(), ax=ax[i][j])
## Correlation between duration between two tweets and difference of likes both of them have. 

fig, ax = plt.subplots(3, 3, figsize=(30,20))

for i in range(3):

    for j in range(3):

        temp = df[df['handle'] == top_ten_authors[3 * i + j]].sort_values(by = 'created_at')[['created_at', 'likes']].diff().dropna()

        sns.heatmap(pd.concat([temp.created_at.dt.seconds, temp.likes], axis=1).corr(), ax=ax[i][j], annot=True).set_title(top_ten_authors[3 * i + j])
df['word_count'] = df.tweet_content.str.len()
fig, ax = plt.subplots(figsize=(15, 10))

sns.distplot(df.word_count, ax=ax)
# Create the plot space upon which to plot the data

fig, ax = plt.subplots(figsize=(15, 10))





# Add the x-axis and the y-axis to the plot

ax.plot(np.sort(df.creation_year.unique()),

        df.groupby('creation_year').word_count.mean(), '-o',

        color='purple')



plt.show()
## The red ones are the viral tweets and the blue one is average of likes between two consecutive viral tweets

fig, ax = plt.subplots(3, 3, figsize=(30,20))

for i in range(3):

    for j in range(3):

        ax[i][j].plot(np.sort(df[df.handle == top_ten_authors[3*i + j]].creation_year.unique()), df[df.handle == top_ten_authors[3*i + j]].groupby('creation_year').word_count.mean(), '-o', color='purple')

        ax[i][j].set_title(top_ten_authors[3*i + j])
fig, ax = plt.subplots(figsize=(15, 10))

sns.heatmap(df[['word_count', 'likes']].corr(), ax=ax, annot=True)
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
def plot_wordcloud(text, ax, mask=None, max_words=400, max_font_size=120, figure_size=(24.0,16.0), title = None, title_size=40, image_color=False):

    """

    Function Credit: https://www.kaggle.com/aashita/word-clouds-of-various-shapes

    """

    stopwords = set(STOPWORDS)

    more_stopwords = {'one', 'br', 'Po', 'th', 'sayi', 'fo', 'Unknown'}

    stopwords = stopwords.union(more_stopwords)



    wordcloud = WordCloud(background_color='white',

                    stopwords = stopwords,

                    max_words = max_words,

                    max_font_size = max_font_size, 

                    random_state = 42,

                    mask = mask)

    wordcloud.generate(text)

    

    if image_color:

        image_colors = ImageColorGenerator(mask);

        ax.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear");

        ax.title(title, fontdict={'size': title_size,  

                                  'verticalalignment': 'bottom'})

    else:

        ax.imshow(wordcloud);

        ax.set_title(title, fontdict={'size': title_size, 'color': 'green', 

                                  'verticalalignment': 'bottom'})

    ax.axis('off');

        

def plot_the_author(name, ax=plt):

    author_tweets = df[df.handle == name].tweet_content

    plot_wordcloud('\n'.join(author_tweets), max_words=600, max_font_size=120,  title = name + ' tweets', title_size=20, figure_size=(10,12), ax=ax)
fig, ax = plt.subplots(figsize=(15, 10))

plot_wordcloud('\n'.join(df.tweet_content), ax=ax)
fig, ax = plt.subplots(figsize=(15, 10))

viral_tweets_all = df[(np.abs(stats.zscore(df.likes)) > 3)].tweet_content

plot_wordcloud('\n'.join(viral_tweets_all), ax=ax)
## Correlation between duration between two tweets and difference of likes both of them have. 

fig, ax = plt.subplots(3, 3, figsize=(25,20))

for i in range(3):

    for j in range(3):

        plot_the_author(top_ten_authors[3 * i + j], ax=ax[i][j])
def plot_viral_twts_cloud(name, ax):

    author_tweets = df[df.handle == name].tweet_content

    plot_wordcloud('\n'.join(author_tweets), max_words=600, max_font_size=120,  title = name + ' tweets', title_size=20, figure_size=(10,12), ax=ax[0])

    t = df[df['handle'] == name]

    viral_tweets_content = t[(np.abs(stats.zscore(t.likes)) > 3)].tweet_content

    t = df[(df['handle'] == name) & ~((df.creation_year ==2019) & (df.creation_month==9))] # Trying to remove recent tweets

    least_liked_tweets = t.sort_values(by = 'likes').head(10).tweet_content

    plot_wordcloud('\n'.join(viral_tweets_content), max_words=600, max_font_size=120,  title = name + ' viral tweets', title_size=20, figure_size=(10,12), ax=ax[1])

    plot_wordcloud('\n'.join(least_liked_tweets), max_words=600, max_font_size=120,  title = name + ' least liked tweets', title_size=20, figure_size=(10,12), ax=ax[2])
fig, ax = plt.subplots(5, 3, figsize=(30,20))

for i in range(5):

    plot_viral_twts_cloud(top_ten_authors[i], ax[i])
import nltk

from tqdm import tqdm

tqdm.pandas()
df['tweet_tokens'] = df['tweet_content'].progress_apply(nltk.word_tokenize)
en_stopwords = set(nltk.corpus.stopwords.words('english'))

df['tweet_tokens'] = df['tweet_tokens'].progress_apply(lambda x: [item for item in x if item not in en_stopwords])
#function to filter for ADJ/NN bigrams

def rightTypes(ngram):

    if '-pron-' in ngram or 't' in ngram:

        return False

    for word in ngram:

        if word in en_stopwords or word.isspace():

            return False

    acceptable_types = ('JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS')

    second_type = ('NN', 'NNS', 'NNP', 'NNPS')

    tags = nltk.pos_tag(ngram)

    if tags[0][1] in acceptable_types and tags[1][1] in second_type:

        return True

    else:

        return False

#filter bigrams

#filtered_bi = bigramFreqTable[bigramFreqTable.bigram.map(lambda x: rightTypes(x))]

#function to filter for trigrams

def rightTypesTri(ngram):

    if '-pron-' in ngram or 't' in ngram:

        return False

    for word in ngram:

        if word in en_stopwords or word.isspace():

            return False

    first_type = ('JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS')

    third_type = ('JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS')

    tags = nltk.pos_tag(ngram)

    if tags[0][1] in first_type and tags[2][1] in third_type:

        return True

    else:

        return False

#filter trigrams

#filtered_tri = trigramFreqTable[trigramFreqTable.trigram.map(lambda x: rightTypesTri(x))]
def get_bigram_trigrams(tokens, title):

    bigrams = nltk.collocations.BigramAssocMeasures()

    trigrams = nltk.collocations.TrigramAssocMeasures()

    bigramFinder = nltk.collocations.BigramCollocationFinder.from_words(tokens)

    trigramFinder = nltk.collocations.TrigramCollocationFinder.from_words(tokens)

    #bigrams

    bigram_freq = bigramFinder.ngram_fd.items()

    bigramFreqTable = pd.DataFrame(list(bigram_freq), columns=['bigram','freq']).sort_values(by='freq', ascending=False)

    #trigrams

    trigram_freq = trigramFinder.ngram_fd.items()

    trigramFreqTable = pd.DataFrame(list(trigram_freq), columns=['trigram','freq']).sort_values(by='freq', ascending=False)

    filtered_bi = bigramFreqTable[bigramFreqTable.bigram.map(lambda x: rightTypes(x))]

    filtered_tri = trigramFreqTable[trigramFreqTable.trigram.map(lambda x: rightTypesTri(x))]

    print(title)

    print(filtered_bi.head(20))

    print(filtered_tri.head(20))

    return bigramFinder, trigramFinder, bigrams, trigrams
%time generic_bigrams_finders, generic_trigrams_finders, generic_bigrams, generic_trigrams = get_bigram_trigrams(np.concatenate(df.tweet_tokens.to_list()), title="Extracting generic Bi/Tri-grams")
top_five_author_bigram_finder = [None for i in range(5)]

top_five_author_trigram_finder = [None for i in range(5)]

top_five_author_bigrams = [None for i in range(5)]

top_five_author_trigrams = [None for i in range(5)]

for i in range(5):

     top_five_author_bigram_finder[i], top_five_author_trigram_finder[i], top_five_author_bigrams[i], top_five_author_trigrams[i] = get_bigram_trigrams(np.concatenate(df[df.handle == top_ten_authors[i]].tweet_tokens.to_list()), title=top_ten_authors[i])
top_five_author_bigram_finder_v = [None for i in range(5)]

top_five_author_trigram_finder_v = [None for i in range(5)]

top_five_author_bigrams_v = [None for i in range(5)]

top_five_author_trigrams_v = [None for i in range(5)]

for i in range(5):

    t = df[df['handle'] == top_ten_authors[i]]

    top_five_author_bigram_finder_v[i], top_five_author_trigram_finder_v[i], top_five_author_bigrams_v[i], top_five_author_trigrams_v[i] = get_bigram_trigrams(np.concatenate(t[(np.abs(stats.zscore(t.likes)) > 3)].tweet_tokens.to_list()), title=top_ten_authors[i])
def get_pointwise_mi_scores(bigramFinder, trigramFinder, bigrams, trigrams, title):

    #filter for only those with more than 20 occurences

    bigramFinder.apply_freq_filter(20)

    trigramFinder.apply_freq_filter(20)

    bigramPMITable = pd.DataFrame(list(bigramFinder.score_ngrams(bigrams.pmi)), columns=['bigram','PMI']).sort_values(by='PMI', ascending=False)

    trigramPMITable = pd.DataFrame(list(trigramFinder.score_ngrams(trigrams.pmi)), columns=['trigram','PMI']).sort_values(by='PMI', ascending=False)

    print('Exploring Point wise Mututal Information in bigrams and trigrams of ' + title)

    print('-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------')

    print('-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------')

    print(bigramPMITable.head(10))

    print('-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------')

    print(trigramPMITable.head(10))

    print('-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------')

    return bigramPMITable, trigramPMITable
_, __ = get_pointwise_mi_scores(generic_bigrams_finders, generic_trigrams_finders, generic_bigrams, generic_trigrams, "all the tweets")
for i in range(5):

    _, __ = get_pointwise_mi_scores(top_five_author_bigram_finder[i], top_five_author_trigram_finder[i], top_five_author_bigrams[i], top_five_author_trigrams[i], top_ten_authors[i] + "\'s tweets")
def get_t_scores( bigramFinder, trigramFinder, bigrams, trigrams, title):

    bigramTtable = pd.DataFrame(list(bigramFinder.score_ngrams(bigrams.student_t)), columns=['bigram','t']).sort_values(by='t', ascending=False)

    trigramTtable = pd.DataFrame(list(trigramFinder.score_ngrams(trigrams.student_t)), columns=['trigram','t']).sort_values(by='t', ascending=False)

    #filters

    filteredT_bi = bigramTtable[bigramTtable.bigram.map(lambda x: rightTypes(x))]

    filteredT_tri = trigramTtable[trigramTtable.trigram.map(lambda x: rightTypesTri(x))]

    print('Exploring t scores between the words in bigrams and trigrams of ' + title)

    print('-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------')

    print('-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------')

    print(filteredT_bi.head(10))

    print('-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------')

    print(filteredT_tri.head(10))

    print('-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------')

    return filteredT_bi, filteredT_tri
_, __ = get_t_scores(generic_bigrams_finders, generic_trigrams_finders, generic_bigrams, generic_trigrams, "all the tweets")
for i in range(5):

    _, __ = get_t_scores(top_five_author_bigram_finder[i], top_five_author_trigram_finder[i], top_five_author_bigrams[i], top_five_author_trigrams[i], top_ten_authors[i] + "\'s tweets")
from textblob import TextBlob
def  get_sentiments(text):

    s = TextBlob(text).sentiment

    return  s.polarity ,s.subjectivity
df['polarity'], df['subjectivity'] = zip(*df['tweet_content'].progress_apply(get_sentiments))
df.head()
fig, ax = plt.subplots(figsize=(15, 10))

sns.distplot(df.polarity, ax=ax)
fig, ax = plt.subplots(figsize=(15, 10))

sns.distplot(df.subjectivity, ax=ax)
sns.heatmap(df[['subjectivity', 'polarity', 'likes']].corr(), annot=True)
for author in top_ten_authors:

    print("Mean Polarity of "+ author + " is " + str(df[df['handle'] == author].polarity.mean()) + ' and standard deviation is ' + str(df[df['handle'] == author].polarity.std()))
for author in top_ten_authors:

    print("Mean subjectivity of "+ author +  " is " + str(df[df['handle'] == author].subjectivity.mean())  + ' and standard deviation is ' + str(df[df['handle'] == author].subjectivity.std()))