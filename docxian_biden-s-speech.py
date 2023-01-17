import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt



from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator



import nltk

from nltk.corpus import stopwords

from nltk import bigrams

from nltk.sentiment.vader import SentimentIntensityAnalyzer



from collections import Counter



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv('../input/joe-biden-2020-dnc-speech/joe_biden_dnc_2020.csv')

df.shape
df
# combine rows and show full text

text = " ".join(xx for xx in df.TEXT)

text
# show wordcloud of speech

stopwords_cloud = set(STOPWORDS)



wordcloud = WordCloud(stopwords=stopwords_cloud, max_font_size=50, max_words=250,

                      width = 600, height = 400,

                      background_color="black").generate(text)

plt.figure(figsize=(12,8))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show()
# standard stopwords

my_stopwords = set(nltk.corpus.stopwords.words('english'))

# additional stopwords

my_stopwords = my_stopwords.union({"'s","'ll","'re","n't","'ve","'m"})
# lower case

text = text.lower()
# tokenize text

words = nltk.word_tokenize(text)
print('Number of tokens: ', len(words))
# remove single characters

words = [word for word in words if len(word) > 1]



# remove stopwords

words = [word for word in words if word not in my_stopwords]
print('Number of tokens after cleaning: ', len(words))
# count word frequencies

word_freqs = nltk.FreqDist(words)
# plot word frequencies

plt.rcParams['figure.figsize'] = [12, 6]

word_freqs.plot(50)
# show counts for top 25

top_words = dict(word_freqs.most_common(50))

top_words
my_bigrams = bigrams(words)

counts = Counter(my_bigrams)

counts = dict(counts)
# convert dictionary to data frame

dcounts = pd.DataFrame.from_dict(counts, orient='index', columns=['frequency'])
# select only bigrams occuring at least twice

dcounts = dcounts[dcounts.frequency>=2]

# and sort descending

dcounts = dcounts.sort_values(by='frequency', ascending=False)
plt.rcParams['figure.figsize'] = [12, 12]

plt.barh(list(map(str, dcounts.index)), dcounts.frequency)

plt.title('Most frequent bigrams')

plt.grid()

plt.show()
sia = SentimentIntensityAnalyzer()
# evaluate sentiment by paragraph (= row)

sent_stats_pos = list()

sent_stats_neg = list()

sent_stats_total = list()

for i in range(18+1):

    txt = df.TEXT[i]

    print(txt)

    sent = sia.polarity_scores(txt)

    print('Sentiment §', i, ': ', sent)

    print('\n')

    sent_stats_pos.append(sent['pos'])

    sent_stats_neg.append(sent['neg'])

    sent_stats_total.append(sent['compound'])
# proportion of text that has positive sentiment

plt.rcParams['figure.figsize'] = [8, 6]

plt.bar(x=range(19), height=sent_stats_pos, color='darkgreen')

plt.grid()

plt.title('Development of positive proportion')

plt.xticks(np.arange(18+1))

plt.show()
# proportion of text that has negative sentiment

plt.rcParams['figure.figsize'] = [8, 6]

plt.bar(x=range(19), height=sent_stats_neg, color='red')

plt.grid()

plt.title('Development of negative proportion')

plt.xticks(np.arange(18+1))

plt.show()
# overall score: +1 extremely positive, -1 extremely negative

plt.rcParams['figure.figsize'] = [8, 6]

plt.bar(x=range(19), height=sent_stats_total, color='blue')

plt.grid()

plt.title('Development of overall sentiment score')

plt.xticks(np.arange(18+1))

plt.show()