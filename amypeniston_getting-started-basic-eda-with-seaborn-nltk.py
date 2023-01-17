import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from collections import defaultdict, Counter

from sklearn.feature_extraction.text import CountVectorizer

import nltk

from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)

stopwords = stopwords.words('english')



sns.set(style="white", font_scale=1.2)

plt.rcParams["figure.figsize"] = [10,8]

pd.set_option.display_max_columns = 0

pd.set_option.display_max_rows = 0
train = pd.read_csv("../input/nlp-getting-started/train.csv")

test = pd.read_csv("../input/nlp-getting-started/test.csv")
train.head()
train.shape, test.shape, test.shape[0]/train.shape[0]
train.info()
null_counts = pd.DataFrame({"Num_Null": train.isnull().sum()})

null_counts["Pct_Null"] = null_counts["Num_Null"] / train.count() * 100

null_counts
keywords_vc = pd.DataFrame({"Count": train["keyword"].value_counts()})

sns.barplot(y=keywords_vc[0:30].index, x=keywords_vc[0:30]["Count"], orient='h')

plt.title("Top 30 Keywords")

plt.show()
len(train["keyword"].value_counts())
disaster_keywords = train.loc[train["target"] == 1]["keyword"].value_counts()

nondisaster_keywords = train.loc[train["target"] == 0]["keyword"].value_counts()



fig, ax = plt.subplots(1,2, figsize=(20,8))

sns.barplot(y=disaster_keywords[0:30].index, x=disaster_keywords[0:30], orient='h', ax=ax[0], palette="Reds_d")

sns.barplot(y=nondisaster_keywords[0:30].index, x=nondisaster_keywords[0:30], orient='h', ax=ax[1], palette="Blues_d")

ax[0].set_title("Top 30 Keywords - Disaster Tweets")

ax[0].set_xlabel("Keyword Frequency")

ax[1].set_title("Top 30 Keywords - Non-Disaster Tweets")

ax[1].set_xlabel("Keyword Frequency")

plt.tight_layout()

plt.show()
armageddon_tweets = train[(train["keyword"].fillna("").str.contains("armageddon")) & (train["target"] == 0)]

print("An example tweet:\n", armageddon_tweets.iloc[10, 3])

armageddon_tweets.head()
def keyword_disaster_probabilities(x):

    tweets_w_keyword = np.sum(train["keyword"].fillna("").str.contains(x))

    tweets_w_keyword_disaster = np.sum(train["keyword"].fillna("").str.contains(x) & train["target"] == 1)

    return tweets_w_keyword_disaster / tweets_w_keyword



keywords_vc["Disaster_Probability"] = keywords_vc.index.map(keyword_disaster_probabilities)

keywords_vc.head()
keywords_vc.sort_values(by="Disaster_Probability", ascending=False).head(10)
keywords_vc.sort_values(by="Disaster_Probability").head(10)
locations_vc = train["location"].value_counts()

sns.barplot(y=locations_vc[0:30].index, x=locations_vc[0:30], orient='h')

plt.title("Top 30 Locations")

plt.show()
len(train["location"].value_counts())
disaster_locations = train.loc[train["target"] == 1]["location"].value_counts()

nondisaster_locations = train.loc[train["target"] == 0]["location"].value_counts()



fig, ax = plt.subplots(1,2, figsize=(20,8))

sns.barplot(y=disaster_locations[0:30].index, x=disaster_locations[0:30], orient='h', ax=ax[0], palette="Reds_d")

sns.barplot(y=nondisaster_locations[0:30].index, x=nondisaster_locations[0:30], orient='h', ax=ax[1], palette="Blues_d")

ax[0].set_title("Top 30 Locations - Disaster Tweets")

ax[0].set_xlabel("Keyword Frequency")

ax[1].set_title("Top 30 Locations - Non-Disaster Tweets")

ax[1].set_xlabel("Keyword Frequency")

plt.tight_layout()

plt.show()
train["tweet_length"] = train["text"].apply(len)

sns.distplot(train["tweet_length"])

plt.title("Histogram of Tweet Length")

plt.xlabel("Number of Characters")

plt.ylabel("Density")

plt.show()
min(train["tweet_length"]), max(train["tweet_length"])
g = sns.FacetGrid(train, col="target", height=5)

g = g.map(sns.distplot, "tweet_length")

plt.suptitle("Distribution Tweet Length")

plt.show()
def count_words(x):

    return len(x.split())



train["num_words"] = train["text"].apply(count_words)

sns.distplot(train["num_words"], bins=10)

plt.title("Histogram of Number of Words per Tweet")

plt.xlabel("Number of Words")

plt.ylabel("Density")

plt.show()
g = sns.FacetGrid(train, col="target", height=5)

g = g.map(sns.distplot, "num_words")

plt.suptitle("Distribution Number of Words")

plt.show()
def avg_word_length(x):

    return np.sum([len(w) for w in x.split()]) / len(x.split())



train["avg_word_length"] = train["text"].apply(avg_word_length)

sns.distplot(train["avg_word_length"])

plt.title("Histogram of Average Word Length")

plt.xlabel("Average Word Length")

plt.ylabel("Density")

plt.show()
g = sns.FacetGrid(train, col="target", height=5)

g = g.map(sns.distplot, "avg_word_length")
def create_corpus(target):

    corpus = []



    for w in train.loc[train["target"] == target]["text"].str.split():

        for i in w:

            corpus.append(i)

            

    return corpus



def create_corpus_dict(target):

    corpus = create_corpus(target)

            

    stop_dict = defaultdict(int)

    for word in corpus:

        if word in stopwords:

            stop_dict[word] += 1

    return sorted(stop_dict.items(), key=lambda x:x[1], reverse=True)
corpus_disaster_dict = create_corpus_dict(0)

corpus_non_disaster_dict = create_corpus_dict(1)



disaster_x, disaster_y = zip(*corpus_disaster_dict)

non_disaster_x, non_disaster_y = zip(*corpus_non_disaster_dict)



fig, ax = plt.subplots(1,2, figsize=(20,8))

sns.barplot(y=list(disaster_x)[0:30], x=list(disaster_y)[0:30], orient='h', palette="Reds_d", ax=ax[0])

sns.barplot(y=list(non_disaster_x)[0:30], x=list(non_disaster_y)[0:30], orient='h', palette="Blues_d", ax=ax[1]) 

ax[0].set_title("Top 30 Stop Words - Disaster Tweets")

ax[0].set_xlabel("Stop Word Frequency")

ax[1].set_title("Top 30 Stop Words - Non-Disaster Tweets")

ax[1].set_xlabel("Stop Word Frequency")

plt.tight_layout()

plt.show()
corpus_disaster, corpus_non_disaster = create_corpus(1), create_corpus(0)

counter_disaster, counter_non_disaster = Counter(corpus_disaster), Counter(corpus_non_disaster)

x_disaster, y_disaster, x_non_disaster, y_non_disaster = [], [], [], []



counter = 0

for word, count in counter_disaster.most_common()[0:100]:

    if (word not in stopwords and counter < 15):

        counter += 1

        x_disaster.append(word)

        y_disaster.append(count)



counter = 0

for word, count in counter_non_disaster.most_common()[0:100]:

    if (word not in stopwords and counter < 15):

        counter += 1

        x_non_disaster.append(word)

        y_non_disaster.append(count)



fig, ax = plt.subplots(1,2, figsize=(20,8))

sns.barplot(x=y_disaster, y=x_disaster, orient='h', palette="Reds_d", ax=ax[0])

sns.barplot(x=y_non_disaster, y=x_non_disaster, orient='h', palette="Blues_d", ax=ax[1])

ax[0].set_title("Top 15 Non-Stopwords - Disaster Tweets")

ax[0].set_xlabel("Word Frequency")

ax[1].set_title("Top 15 Non-Stopwords - Non-Disaster Tweets")

ax[1].set_xlabel("Word Frequency")

plt.tight_layout()

plt.show()
def bigrams(target):

    corpus = train[train["target"] == target]["text"]

    count_vec = CountVectorizer(ngram_range=(2, 2)).fit(corpus)

    bag_of_words = count_vec.transform(corpus)

    sum_words = bag_of_words.sum(axis=0) 

    words_freq = [(word, sum_words[0, idx]) for word, idx in count_vec.vocabulary_.items()]

    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)

    return words_freq
bigrams_disaster = bigrams(1)[:15]

bigrams_non_disaster = bigrams(0)[:15]



x_disaster, y_disaster = map(list, zip(*bigrams_disaster))

x_non_disaster, y_non_disaster = map(list, zip(*bigrams_non_disaster))



fig, ax = plt.subplots(1,2, figsize=(20,8))

sns.barplot(x=y_disaster, y=x_disaster, orient='h', palette="Reds_d", ax=ax[0])

sns.barplot(x=y_non_disaster, y=x_non_disaster, orient='h', palette="Blues_d", ax=ax[1])



ax[0].set_title("Top 15 Bigrams - Disaster Tweets")

ax[0].set_xlabel("Word Frequency")

ax[1].set_title("Top 15 Bigrams - Non-Disaster Tweets")

ax[1].set_xlabel("Word Frequency")

plt.tight_layout()

plt.show()
target_vc = train["target"].value_counts(normalize=True)

print("Not Disaster: {:.2%}, Disaster: {:.2%}".format(target_vc[0], target_vc[1]))

sns.barplot(x=target_vc.index, y=target_vc)

plt.title("Histogram of Disaster vs. Non-Disaster")

plt.xlabel("0 = Non-Disaster, 1 = Disaster")

plt.show()