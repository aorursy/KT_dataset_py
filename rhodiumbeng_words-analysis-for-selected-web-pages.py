import pandas as pd

import numpy as np

import nltk

from nltk.corpus import stopwords

import matplotlib.pyplot as plt

import seaborn as sns

from wordcloud import WordCloud

import re

from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder

from operator import itemgetter
def prepare_stop_words():

    stopwordsList = []

    # Load default stop words and add a few more specific to the text.

    stop_words_list = stopwords.words('english')

    stop_words_list.append('ounce')

    stop_words_list.append('oz')

    stop_words_list.append('ml')

    stop_words_list.append('ii')

    return stop_words_list
def process_text(file):

    input_file = file

    FILEHEADER = 0

    with open(input_file, 'r') as f:

        if FILEHEADER:

            next(f)

        raw_text = f.read()

    # Lowercase and tokenize

    raw_text = raw_text.lower()

    tokens = nltk.word_tokenize(raw_text)

    text = nltk.Text(tokens)

    # Remove unwanted characters, numbers and stop words

    text_content = [''.join(re.split("[ .,;:!?'`|~@#$%^_*()&{}\n\t\-']", word)) for word in text]

    text_content = [word for word in text_content if not re.search(r'\d', word)]

    text_content = [word for word in text_content if word not in stop_words]

    text_content = [word for word in text_content if len(word) !=0]

    return text_content
stop_words = prepare_stop_words()
url1_unigrams = process_text('../input/text-scrapped-from-amazon-search-pages/url1-combined.txt')

url2_unigrams = process_text('../input/text-scrapped-from-amazon-search-pages/url2-combined.txt')

url3_unigrams = process_text('../input/text-scrapped-from-amazon-search-pages/url3-combined.txt')
print(' Number of words in Url1 is', len(url1_unigrams), '\n',

      'Number of words in Url2 is', len(url2_unigrams), '\n',

      'Number of words in Url2 is', len(url3_unigrams))
def create_bigram_scored_list(text_content):

    # setup and score the bigrams using raw frequency

    finder = BigramCollocationFinder.from_words(text_content)

    bigram_measures = BigramAssocMeasures()

    scored = finder.score_ngrams(bigram_measures.raw_freq)

    scored_list = sorted(scored, key = itemgetter(1), reverse=True)

    # Create dictionary of bigrams and weightage

    bigram_dict = {}

    list_length = len(scored_list)

    for i in range(list_length):

        bigram_dict['_'.join(scored_list[i][0])] = int(scored_list[i][1]*len(text_content))

    return bigram_dict
url1_bigrams = create_bigram_scored_list(url1_unigrams)

url2_bigrams = create_bigram_scored_list(url2_unigrams)

url3_bigrams = create_bigram_scored_list(url3_unigrams)
# Set word cloud params and instantiate the word cloud.

wc_max_words = 200
wordCloud = WordCloud(max_words=wc_max_words)

wordCloud.generate_from_frequencies(url1_bigrams)

plt.figure(figsize=[20,15])

plt.title('Most frequently occurring bigrams for Url 1', fontsize=30)

plt.imshow(wordCloud, interpolation='bilinear')

plt.axis("off")

plt.show()
wordCloud = WordCloud(max_words=wc_max_words, colormap='plasma')

wordCloud.generate_from_frequencies(url2_bigrams)

plt.figure(figsize=[20,15])

plt.title('Most frequently occurring bigrams for Url 2', fontsize=30)

plt.imshow(wordCloud, interpolation='bilinear')

plt.axis("off")

plt.show()
wordCloud = WordCloud(max_words=wc_max_words, colormap='cividis')

wordCloud.generate_from_frequencies(url3_bigrams)

plt.figure(figsize=[20,15])

plt.title('Most frequently occurring bigrams for Url 3', fontsize=30)

plt.imshow(wordCloud, interpolation='bilinear')

plt.axis("off")

plt.show()
def create_top_n_dataframe(bigrams, n=20):

    df = pd.DataFrame(bigrams.items(), columns=['words', 'freq'])

    df_n = df[:n]

    return df_n
df_url1 = create_top_n_dataframe(url1_bigrams)

df_url2 = create_top_n_dataframe(url2_bigrams)

df_url3 = create_top_n_dataframe(url3_bigrams)
sns.set(rc={'figure.figsize':(15,8)})
plt.title('Top 20 Words from Url1', fontsize=30)

sns.barplot(x='freq', y='words', data=df_url1)
plt.title('Top 20 Words from Url2', fontsize=30)

sns.barplot(x='freq', y='words', data=df_url2)
plt.title('Top 20 Words from Url3', fontsize=30)

sns.barplot(x='freq', y='words', data=df_url3)
df_url1
df_url2
df_url3