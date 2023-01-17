import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import nltk

from collections import Counter

from nltk import ngrams

from nltk.tokenize import word_tokenize

from wordcloud import WordCloud

from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
df_train=pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')

df_test=pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')
df_train.head()
df_test.tail()
df_data = df_train.append(df_test, sort = False)

df_data.head()
df_data.tail()
def clean_text(tweets):

    

    # Replacing @handle with the word USER

    tweets_handle = tweets.str.replace(r'@[\S]+', 'user')

    

    # Replacing the Hast tag with the word hASH

    tweets_hash = tweets_handle.str.replace(r'#(\S+)','hash')

    

    # Removing the all the Retweets

    tweets_r = tweets_hash.str.replace(r'\brt\b',' ')

    

    # Replacing the URL or Web Address

    tweets_url = tweets_r.str.replace(r'((www\.[\S]+)|(http?://[\S]+))','URL')

    

    # Replacing Two or more dots with one

    tweets_dot = tweets_url.str.replace(r'\.{2,}', ' ')

    

    # Removing all the special Characters

    tweets_special = tweets_dot.str.replace(r'[^\w\d\s]',' ')

    

    # Removing all the non ASCII characters

    tweets_ascii = tweets_special.str.replace(r'[^\x00-\x7F]+',' ')

    

    # Removing the leading and trailing Whitespaces

    tweets_space = tweets_ascii.str.replace(r'^\s+|\s+?$','')

    

    # Replacing multiple Spaces with Single Space

    Dataframe = tweets_space.str.replace(r'\s+',' ')

    

    return Dataframe
df_data['text'] = clean_text(df_data['text'])

df_data['text'] = df_data['text'].apply(str)
df_data.head()
plt.rcParams['figure.figsize'] = [8, 10]

sns_count = sns.countplot(df_train['sentiment'], data = df_train, order = df_train['sentiment'].value_counts().index)
# Top 20 unigrams for the positive sentiment in the text in whole dataset



df_data_pos = " ".join(df_data.loc[df_data.sentiment == 'positive', 'text'])

token_text_pos = word_tokenize(df_data_pos)

unigrams_pos = ngrams(token_text_pos, 1)

frequency_pos = Counter(unigrams_pos)



df_pos = pd.DataFrame(frequency_pos.most_common(20))





# Barplot that shows the top 20 unigrams

plt.rcParams['figure.figsize'] = [13,9]

sns.set(font_scale = 1.5, style = 'whitegrid')



sns_pos_1 = sns.barplot(x = df_pos[1], y = df_pos[0], color = 'lightsteelblue')



# Setting axes labels

sns_pos_1.set(xlabel = 'Occurrence', ylabel = 'Unigrams', title = 'Top 20 Unigrams for the Positive Sentiment');
# Top 20 Bigrams for the positive sentiment in the text in whole dataset

bigrams_pos = ngrams(token_text_pos, 2)

frequency_pos = Counter(bigrams_pos)

df_pos = pd.DataFrame(frequency_pos.most_common(20))



# Barplot that shows the top 20 Bigrams

sns_pos_2 = sns.barplot(x = df_pos[1], y = df_pos[0], color = 'mediumpurple')



# Setting axes labels

sns_pos_2.set(xlabel = 'Occurrence', ylabel = 'Bigrams', title = 'Top 20 Bigrams for the Positive Sentiment');
# Top 20 Trigrams for the positive sentiment in the text in whole dataset

trigrams_pos = ngrams(token_text_pos, 3)

frequency_pos = Counter(trigrams_pos)

df_pos = pd.DataFrame(frequency_pos.most_common(20))



# Barplot that shows the top 20 Trigrams

sns_pos_3 = sns.barplot(x = df_pos[1], y = df_pos[0], color = 'darkcyan')



# Setting axes labels

sns_pos_3.set(xlabel = 'Occurrence', ylabel = 'Trigrams', title = 'Top 20 Trigrams for the Positive Sentiment');
# Top 20 Unigrams for the negative sentiment in text for the whole dataset

df_data_neg = " ".join(df_data.loc[df_data.sentiment == 'negative', 'text'])

token_text_neg = word_tokenize(df_data_neg)

unigrams_neg = ngrams(token_text_neg, 1)

frequency_neg = Counter(unigrams_neg)



df_neg = pd.DataFrame(frequency_neg.most_common(20))



# Barplot that shows the top 20 unigrams

sns_neg_1 = sns.barplot(x = df_neg[1], y = df_neg[0], color = 'lightsteelblue')



# Setting axes labels

sns_neg_1.set(xlabel = 'Occurrence', ylabel = 'Unigrams', title = 'Top 20 Unigrams for the Negative Sentiment');
# Top 20 Bigrams for the negative sentiment in text for the whole dataset

Bigrams_neg = ngrams(token_text_neg, 2)

frequency_neg = Counter(Bigrams_neg)



df_neg = pd.DataFrame(frequency_neg.most_common(20))



# Barplot that shows the top 20 Bigrams

sns_neg_2 = sns.barplot(x = df_neg[1], y = df_neg[0], color = 'mediumpurple')



# Setting axes labels

sns_neg_2.set(xlabel = 'Occurrence', ylabel = 'Bigrams', title = 'Top 20 Bigrams for the Negative Sentiment');
# Top 20 Trigrams for the negative sentiment in text for the whole dataset

Trigrams_neg = ngrams(token_text_neg, 3)

frequency_neg = Counter(Trigrams_neg)



df_neg = pd.DataFrame(frequency_neg.most_common(20))



# Barplot that shows the top 20 Trigrams

sns_neg_3 = sns.barplot(x = df_neg[1], y = df_neg[0], color = 'darkcyan')



# Setting axes labels

sns_neg_3.set(xlabel = 'Occurrence', ylabel = 'Trigrams', title = 'Top 20 Trigrams for the Negative Sentiment');
# Top 20 unigrams for the neutral sentiment in the text in whole dataset



df_data_neu = " ".join(df_data.loc[df_data.sentiment == 'neutral', 'text'])

token_text_neu = word_tokenize(df_data_neu)

unigrams_neu = ngrams(token_text_neu, 1)

frequency_neu = Counter(unigrams_neu)



df_neu = pd.DataFrame(frequency_neu.most_common(20))



# Barplot that shows the top 20 unigrams

sns_neu_1 = sns.barplot(x = df_neu[1], y = df_neu[0], color = 'lightsteelblue')



# Setting axes labels

sns_neu_1.set(xlabel = 'Occurrence', ylabel = 'Unigrams', title = 'Top 20 Unigrams for the Neutral Sentiment');
# Top 20 Bigrams for the neutral sentiment in the text in whole dataset

Bigrams_neu = ngrams(token_text_neu, 2)

frequency_neu = Counter(Bigrams_neu)



df_neu = pd.DataFrame(frequency_neu.most_common(20))



# Barplot that shows the top 20 Bigrams

sns_neu_2 = sns.barplot(x = df_neu[1], y = df_neu[0], color = 'mediumpurple')



# Setting axes labels

sns_neu_2.set(xlabel = 'Occurrence', ylabel = 'Bigrams', title = 'Top 20 Bigrams for the Neutral Sentiment');
# Top 20 Trigrams for the neutral sentiment in the text in whole dataset

Trigrams_neu = ngrams(token_text_neu, 3)

frequency_neu = Counter(Trigrams_neu)



df_neu = pd.DataFrame(frequency_neu.most_common(20))



# Barplot that shows the top 20 Trigrams

sns_neu_3 = sns.barplot(x = df_neu[1], y = df_neu[0], color = 'darkcyan')



# Setting axes labels

sns_neu_3.set(xlabel = 'Occurrence', ylabel = 'Trigrams', title = 'Top 20 Trigrams for the Neutral Sentiment');
# Removing the stop words before plotting



df_data['text'] = df_data['text'].apply(lambda x: " ".join(word for word in x.split() if word not in stop_words))

joined = " ".join(df_data['text'])

tokenize = word_tokenize(joined)

frequency = Counter(tokenize)

df = pd.DataFrame(frequency.most_common(30))



plt.rcParams['figure.figsize'] = [12, 15]

sns.set(font_scale = 1.3, style = 'whitegrid')



# plotting

word_count = sns.barplot(x = df[1], y = df[0], color = 'orange')

word_count.set_title("Word Count Plot")

word_count.set_ylabel("Words")

word_count.set_xlabel("Count");
# Word cloud of the text with the positive sentiment

df_pos = df_data.loc[df_data.sentiment == 'positive', 'text']

k = (' '.join(df_pos))



wordcloud = WordCloud(width = 1000, height = 500, background_color = 'white').generate(k)

plt.figure(figsize=(15, 10))

plt.imshow(wordcloud)

plt.axis('off');
# Word cloud of the text with the negative sentiment

df_neg = df_data.loc[df_data.sentiment == 'negative', 'text']

k = (' '.join(df_neg))



wordcloud = WordCloud(width = 1000, height = 500, background_color = 'white').generate(k)

plt.figure(figsize=(15, 10))

plt.imshow(wordcloud)

plt.axis('off');
# Word cloud of the text with the Neutral sentiment

df_neu = df_data.loc[df_data.sentiment == 'neutral', 'text']

k = (' '.join(df_neu))



wordcloud = WordCloud(width = 1000, height = 500, background_color = 'white').generate(k)

plt.figure(figsize=(15, 10))

plt.imshow(wordcloud)

plt.axis('off');