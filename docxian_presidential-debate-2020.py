import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

import matplotlib.pyplot as plt



from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator



import nltk

from nltk.corpus import stopwords

from nltk import bigrams

from nltk.sentiment.vader import SentimentIntensityAnalyzer



from collections import Counter
df = pd.read_csv('../input/us-election-2020-presidential-debates/us_election_2020_1st_presidential_debate.csv')

df.head()
df.shape
# split by speaker

df_CW = df[df.speaker=='Chris Wallace']

df_JB = df[df.speaker=='Vice President Joe Biden']

df_DT = df[df.speaker=='President Donald J. Trump']

df_CW
df_DT
df_JB
print('Number of segments - Chris Wallace             : ', df_CW.shape[0])

print('Number of segments - President Donald J. Trump : ', df_DT.shape[0])

print('Number of segments - Vice President Joe Biden  : ', df_JB.shape[0])
# convert to strings

text_CW = " ".join(txt for txt in df_CW.text)

text_DT = " ".join(txt for txt in df_DT.text)

text_JB = " ".join(txt for txt in df_JB.text)



# compare total text lengths

print('Length of text - Chris Wallace             : ', len(text_CW))

print('Length of text - President Donald J. Trump : ', len(text_DT))

print('Length of text - Vice President Joe Biden  : ', len(text_JB))
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, max_font_size=50, max_words=500,

                      width = 600, height = 400,

                      background_color="white").generate(text_CW)

plt.figure(figsize=(12,8))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show()
wordcloud = WordCloud(stopwords=stopwords, max_font_size=50, max_words=500,

                      width = 600, height = 400,

                      background_color="white").generate(text_DT)

plt.figure(figsize=(12,8))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show()
wordcloud = WordCloud(stopwords=stopwords, max_font_size=50, max_words=500,

                      width = 600, height = 400,

                      background_color="white").generate(text_JB)

plt.figure(figsize=(12,8))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show()
# standard stopwords

my_stopwords = set(nltk.corpus.stopwords.words('english'))

# additional stopwords

my_stopwords = my_stopwords.union({"'s","'ll","'re","n't","'ve","'m"})
# lower case

text = text_CW.lower()

# tokenize text

words = nltk.word_tokenize(text)

# remove single characters

words = [word for word in words if len(word) > 1]

# remove stopwords

words = [word for word in words if word not in my_stopwords]

# count word frequencies

word_freqs = nltk.FreqDist(words)

# plot word frequencies

plt.rcParams['figure.figsize'] = [12, 6]

plt.title('Word Frequency - Chris Wallace')

word_freqs.plot(50)
my_bigrams = bigrams(words)

counts = Counter(my_bigrams)

counts = dict(counts)

# convert dictionary to data frame

dcounts = pd.DataFrame.from_dict(counts, orient='index', columns=['frequency'])

# select only bigrams occuring at least four times

dcounts = dcounts[dcounts.frequency>=4]

# and sort descending

dcounts = dcounts.sort_values(by='frequency', ascending=False)

plt.rcParams['figure.figsize'] = [12, 12]

plt.barh(list(map(str, dcounts.index)), dcounts.frequency)

plt.title('Most frequent bigrams - Chris Wallace')

plt.grid()

plt.show()
# lower case

text = text_DT.lower()

# tokenize text

words = nltk.word_tokenize(text)

# remove single characters

words = [word for word in words if len(word) > 1]

# remove stopwords

words = [word for word in words if word not in my_stopwords]

# count word frequencies

word_freqs = nltk.FreqDist(words)

# plot word frequencies

plt.rcParams['figure.figsize'] = [12, 6]

plt.title('Word Frequency - President Donald J. Trump')

word_freqs.plot(50)
my_bigrams = bigrams(words)

counts = Counter(my_bigrams)

counts = dict(counts)

# convert dictionary to data frame

dcounts = pd.DataFrame.from_dict(counts, orient='index', columns=['frequency'])

# select only bigrams occuring at least four times

dcounts = dcounts[dcounts.frequency>=4]

# and sort descending

dcounts = dcounts.sort_values(by='frequency', ascending=False)

plt.rcParams['figure.figsize'] = [12, 12]

plt.barh(list(map(str, dcounts.index)), dcounts.frequency)

plt.title('Most frequent bigrams - President Donald J. Trump')

plt.grid()

plt.show()
# lower case

text = text_JB.lower()

# tokenize text

words = nltk.word_tokenize(text)

# remove single characters

words = [word for word in words if len(word) > 1]

# remove stopwords

words = [word for word in words if word not in my_stopwords]

# count word frequencies

word_freqs = nltk.FreqDist(words)

# plot word frequencies

plt.rcParams['figure.figsize'] = [12, 6]

plt.title('Word Frequency - Vice President Joe Biden')

word_freqs.plot(50)
my_bigrams = bigrams(words)

counts = Counter(my_bigrams)

counts = dict(counts)

# convert dictionary to data frame

dcounts = pd.DataFrame.from_dict(counts, orient='index', columns=['frequency'])

# select only bigrams occuring at least four times

dcounts = dcounts[dcounts.frequency>=4]

# and sort descending

dcounts = dcounts.sort_values(by='frequency', ascending=False)

plt.rcParams['figure.figsize'] = [12, 12]

plt.barh(list(map(str, dcounts.index)), dcounts.frequency)

plt.title('Most frequent bigrams - Vice President Joe Biden')

plt.grid()

plt.show()
sia = SentimentIntensityAnalyzer()
sent = sia.polarity_scores(text_CW)

sent_val = sent['compound']

sent.pop('compound')

print('CW: sentiment score = ', sent_val)

print('CW: split = ', sent)
sent = sia.polarity_scores(text_DT)

sent_val = sent['compound']

sent.pop('compound')

print('DT: sentiment score = ', sent_val)

print('DT: split = ', sent)
sent = sia.polarity_scores(text_JB)

sent_val = sent['compound']

sent.pop('compound')

print('JB: sentiment score = ', sent_val)

print('JB: split = ', sent)