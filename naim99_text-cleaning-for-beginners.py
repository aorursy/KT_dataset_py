import nltk

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
# Run only if this is the first ever time using nltk

# nltk.download()
df = pd.read_csv("../input/tweets-data/tweets.csv")

df
[nltk.sent_tokenize(item) for item in df["text"].values]
tokens = [nltk.word_tokenize(item) for item in df["text"].values]

tokens
import string

import re



regex = re.compile(f'[{re.escape(string.punctuation)}]')



tokens_without_punctuation = [regex.sub(u'', word) for words in tokens for word in words if not regex.sub(u'', word) == u'']

tokens_without_punctuation[:10]
from nltk.corpus import stopwords



stop_words = stopwords.words("english")

stop_words.append("via")

stop_words



words = [token for token in tokens_without_punctuation if token not in stop_words]

words[:15]
import re

    

regex = re.compile('http\S+')



tokens_without_links = [regex.sub(u'', word) for word in words if not regex.sub(u'', word) == u'' and not word.startswith("tc")]

tokens_without_links[:20]
from nltk.stem import PorterStemmer



stemmer = PorterStemmer()



stemmed_words = [stemmer.stem(word) for word in tokens_without_links]

stemmed_words[:10]
from collections import Counter



counter = Counter(stemmed_words)

counter.most_common(20)
def plot_words(words, values):

    indexes = np.arange(len(words))

    plt.xticks(indexes, words, rotation=90)

    plt.bar(indexes, values)
plot_words(counter.keys(), counter.values())
most_common_words = [word for word, _ in counter.most_common(20)]

most_common_values = [count for _, count in counter.most_common(20)]
plot_words(most_common_words, most_common_values)
from nltk.stem import WordNetLemmatizer



lemmatizer = WordNetLemmatizer()



lemmatized_words = [lemmatizer.lemmatize(word) for word in tokens_without_links]

lemmatized_words[:20]
lemmatized_counter = Counter(lemmatized_words)

lemmatized_counter.most_common(20)
plot_words(lemmatized_counter.keys(), lemmatized_counter.values())
most_common_words = [word for word, _ in lemmatized_counter.most_common(20)]

most_common_values = [count for _, count in lemmatized_counter.most_common(20)]
plot_words(most_common_words, most_common_values)
pd.DataFrame({

    "words": lemmatized_words

}).to_csv("words.csv", index=False, encoding="UTF-8")