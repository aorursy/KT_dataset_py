import numpy as np

import pandas as pd

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



import matplotlib.pyplot as plt

import statistics as stats

import seaborn as sns

from wordcloud import WordCloud

from bs4 import BeautifulSoup

import nltk

from nltk.corpus import stopwords
dataset = pd.read_csv("../input/MrTrumpSpeeches.csv", header=0, delimiter="\~", quoting=3, engine='python')
print(dataset.shape)

print(dataset.columns)

print(dataset.head())
subtitles1 = dataset['subtitles'][0]

wordcloud = WordCloud().generate(subtitles1)



plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()



#print(subtitles1)
subtitles1 = BeautifulSoup(dataset['subtitles'][0], "html5lib")

lower_case = subtitles1.get_text().lower() 

words = lower_case.split()

print('Number of words: ',len(words))

words_meaningful = [w for w in words if not w in stopwords.words("english")]

words_stopwords = [w for w in words if w in stopwords.words("english")]

print('Number of stop words: ', len(words_stopwords))



subtitles1_clean = " ".join(words_meaningful)

wordcloud = WordCloud().generate(subtitles1_clean)

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()



#print(subtitles1_clean)
wordcloud = WordCloud().generate(" ".join(words_stopwords))

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()



#print(words_stopwords)
g = sns.pairplot(dataset[['like_count','dislike_count','view_count']], palette='husl', diag_kind='kde',size=4)

plt.show()
print('median: ',stats.median(dataset['like_count']))

print('mean: ',stats.mean(dataset['like_count']))

print('max: ',max(dataset['like_count']))

print('min: ',min(dataset['like_count']))





plt.hist(dataset['like_count'])

plt.show()