import matplotlib.pyplot as plt

import nltk

import pandas as pd

import re



from collections import Counter

from nltk.corpus import stopwords

from path import Path

from wordcloud import WordCloud
df_books_en = pd.read_csv('../input/key_english.csv')

df_ab_en = pd.read_csv('../input/key_abbreviations_english.csv')

df_KJV = pd.read_csv('../input/t_kjv.csv')



display(df_books_en.head())

display(df_ab_en.head())

display(df_KJV.head())
def get_verse(df_version, book, c, v):

    def find_name(book_name):

        selection = df_books_en.n==book_name

        if selection.sum() == 0: return 0

        return df_books_en.b[selection].iloc[0]



    def find_abv(book_abv):

        selection = df_ab_en.a==book_abv

        if selection.sum() == 0: return 0

        return df_ab_en.b[selection].iloc[0]

    

    df = df_version

    b = find_name(book) + find_abv(book)

    if b == 0: return 'Book not found'

    return df.t[(df.b==b) & (df.c==c) & (df.v==v)].iloc[0]
Gen_1_3 = get_verse(df_KJV, 'Ge', 1, 3)

Gen_1_3
# This function removes all non-alpha-numeric characters

# except underscore (_). Posessive term 's are remove.

# Eg.: Noah's wife >> Noah wife

def clean_punctuation(x):

    return ' '.join(re.findall("(\w+)(?:\'s)?", x))
df_KJV.t = df_KJV.t.apply(clean_punctuation)

Gen_1_3 = get_verse(df_KJV, 'Ge', 1, 3)

Gen_1_3
STOPWORDS = set(stopwords.words('english'))



words_raw = [word.lower() for v in df_KJV.t for word in v.split()]

words = [word for word in words_raw if word not in STOPWORDS]

words_dist_raw = nltk.FreqDist(words_raw)

words_dist = nltk.FreqDist(words)

display(words_dist_raw)

display(words_dist)
plt.figure(figsize=(20, 5))

words_dist_raw.plot(50)

# plt.savefig('words_dist_raw.png')



plt.figure(figsize=(20, 5))

words_dist.plot(50)

# plt.savefig('words_dist.png')
bigram_dist_raw = nltk.FreqDist(nltk.bigrams(words_raw))

bigram_dist = nltk.FreqDist(nltk.bigrams(words))

display(bigram_dist_raw)

display(bigram_dist)
plt.figure(figsize=(20, 5))

bigram_dist_raw.plot(50)

# plt.savefig('bigram_dist_raw.png')



plt.figure(figsize=(20, 5))

bigram_dist.plot(50)

# plt.savefig('bigram_dist.png')
trigram_dist_raw = nltk.FreqDist(nltk.trigrams(words_raw))

trigram_dist = nltk.FreqDist(nltk.trigrams(words))

display(trigram_dist_raw)

display(trigram_dist)
plt.figure(figsize=(20, 5))

trigram_dist_raw.plot(50)

# plt.savefig('trigram_dist_raw.png')
plt.figure(figsize=(20, 5))

trigram_dist.plot(50)

# plt.savefig('trigram_dist.png')
def plot_cloud(freq_dist, n=200, label=''):

    def get_key(x): 

        if type(x) is str: return x.upper()

        if len(x) == 1: return x.upper()

        else: return ' '.join(x).upper()

    

    freq = { get_key(k):v for (k, v) in freq_dist.most_common(n)}

    

    wordcloud = WordCloud(width=800, height=600, max_font_size=80, 

                          background_color="white", colormap='hsv')

    wordcloud = wordcloud.generate_from_frequencies(freq)



    label = f' of {label}' if len(label) > 0 else label

    wordcloud.to_file(f'Bible Cloud{label}.png');



    plt.figure(figsize=[14,21])

    plt.imshow(wordcloud, interpolation="bilinear")

    plt.axis("off")

    plt.show()
plot_cloud(words_dist, label='Unigrams')
plot_cloud(bigram_dist, label='Bigrams')
plot_cloud(trigram_dist, label='Trigrams')