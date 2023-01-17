# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

from pathlib import Path

data = {}

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        data[filename] = Path(dirname) / filename



# Any results you write to the current directory are saved as output.

data
from IPython.display import (

    Markdown as md,

    Latex,

    HTML,

)

from tqdm.auto import tqdm
tweets = pd.read_csv(data["twitter_sentiment_data.csv"])
display(tweets.shape)
value_counts = tweets["sentiment"].value_counts()

value_counts.name = "Raw Number"



value_normd = tweets["sentiment"].value_counts(normalize=True)

value_normd.name = "Percentage"



display(pd.concat([value_counts, value_normd], axis=1))
display(tweets.head())
from copy import deepcopy

eda = deepcopy(tweets)

# tqdm.pandas()
sentiment_num2name = {

    -1: "Anti",

     0: "Neutral",

     1: "Pro",

     2: "News",

}

eda["sentiment"] = eda["sentiment"].apply(lambda num: sentiment_num2name[num])

eda.head()
from matplotlib import pyplot as plt

from matplotlib import style



import seaborn as sns



sns.set(font_scale=1.5)

style.use("seaborn-poster")
fig, axes = plt.subplots(1, 2, figsize=(20, 10), dpi=100)



sns.countplot(eda["sentiment"], ax=axes[0])

labels = list(sentiment_num2name.values())



axes[1].pie(eda["sentiment"].value_counts(),

            labels=labels,

            autopct="%1.0f%%",

            startangle=90,

            explode=tuple([0.1] * len(labels)))



fig.suptitle("Distribution of Tweets", fontsize=20)

plt.show()
import re

import nltk

import itertools
top15 = {}



by_sentiment = eda.groupby("sentiment")

for sentiment, group in tqdm(by_sentiment):

    hashtags = group["message"].apply(lambda tweet: re.findall(r"#(\w+)", tweet))

    hashtags = itertools.chain(*hashtags)

    hashtags = [ht.lower() for ht in hashtags]

    

    frequency = nltk.FreqDist(hashtags)

    

    df_hashtags = pd.DataFrame({

        "hashtags": list(frequency.keys()),

        "counts": list(frequency.values()),

    })

    top15_htags = df_hashtags.nlargest(15, columns=["counts"])

    

    top15[sentiment] = top15_htags.reset_index(drop=True)



display(pd.concat(top15, axis=1).head(n=10))
fig, axes = plt.subplots(2, 2, figsize=(28, 20))

counter = 0



for sentiment, top in top15.items():

    sns.barplot(data=top, y="hashtags", x="counts", palette="Blues_d", ax=axes[counter // 2, counter % 2])

    axes[counter // 2, counter % 2].set_title(f"Most frequent Hashtags by {sentiment} (Visually)", fontsize=25)

    counter += 1

plt.show()
def cleaner(tweet):

    tweet = tweet.lower()

    

    to_del = [

        r"@[\w]*",  # strip account mentions

        r"http(s?):\/\/.*\/\w*",  # strip URLs

        r"#\w*",  # strip hashtags

        r"\d+",  # delete numeric values

        r"U+FFFD",  # remove the "character note present" diamond

    ]

    for key in to_del:

        tweet = re.sub(key, "", tweet)

    

    # strip punctuation and special characters

    tweet = re.sub(r"[,.;':@#?!\&/$]+\ *", " ", tweet)

    # strip excess white-space

    tweet = re.sub(r"\s\s+", " ", tweet)

    

    return tweet.lstrip(" ")
eda["message"] = eda["message"].apply(cleaner)

eda.head()
from nltk.tokenize import word_tokenize

from nltk.stem import WordNetLemmatizer 

from nltk.corpus import stopwords, wordnet  
def lemmatizer(df):

    df["length"] = df["message"].str.len()

    df["tokenized"] = df["message"].apply(word_tokenize)

    df["parts-of-speech"] = df["tokenized"].apply(nltk.tag.pos_tag)

    

    def str2wordnet(tag):

        conversion = {"J": wordnet.ADJ, "V": wordnet.VERB, "N": wordnet.NOUN, "R": wordnet.ADV}

        try:

            return conversion[tag[0].upper()]

        except KeyError:

            return wordnet.NOUN

    

    wnl = WordNetLemmatizer()

    df["parts-of-speech"] = df["parts-of-speech"].apply(

        lambda tokens: [(word, str2wordnet(tag)) for word, tag in tokens]

    )

    df["lemmatized"] = df["parts-of-speech"].apply(

        lambda tokens: [wnl.lemmatize(word, tag) for word, tag in tokens]

    )

    df["lemmatized"] = df["lemmatized"].apply(lambda tokens: " ".join(map(str, tokens)))

    

    return df
eda = lemmatizer(eda)

eda.head()
plt.figure(figsize=(15, 15))

sns.boxplot(x="sentiment", y="length", data=eda, palette=("Blues_d"))

plt.title("Tweet Length Distribution for each Sentiment")

plt.show()
from sklearn.feature_extraction.text import CountVectorizer
frequency = {}



by_sentiment = eda.groupby("sentiment")

for sentiment, group in tqdm(by_sentiment):

    cv = CountVectorizer(stop_words="english")

    words = cv.fit_transform(group["lemmatized"])

    

    n_words = words.sum(axis=0)

    word_freq = [(word, n_words[0, idx]) for word, idx in cv.vocabulary_.items()]

    word_freq = sorted(word_freq, key=lambda x: x[1], reverse=True)

    

    freq = pd.DataFrame(word_freq, columns=["word", "freq"])

    

    frequency[sentiment] = freq.head(n=25)



to_view = pd.concat(frequency, axis=1).head(n=25)

display(to_view)
words = {sentiment: " ".join(frequency[sentiment]["word"].values) for sentiment in sentiment_num2name.values()}



cmaps = {

    "Anti": ("Reds", 110),

    "Pro" : ("Greens", 73),

    "News": ("Blues", 0),

    "Neutral": ("Oranges", 10),

}



from wordcloud import WordCloud



wordclouds = {}

for sentiment, (cmap, rand) in tqdm(cmaps.items()):

    wordclouds[sentiment] = WordCloud(

        width=800, height=500, random_state=rand,

        max_font_size=110, background_color="white",

        colormap=cmap

    ).generate(words[sentiment])

    

fig, axes = plt.subplots(2, 2, figsize=(28, 20))

counter = 0



for sentiment, wordcloud in wordclouds.items():

    axes[counter // 2, counter % 2].imshow(wordcloud)

    axes[counter // 2, counter % 2].set_title(sentiment, fontsize=25)

    counter += 1

    

for ax in fig.axes:

    plt.sca(ax)

    plt.axis("off")



plt.show()
import spacy

spacy_en = spacy.load('en')
def crude_entities(tweet):

    as_words = tweet.apply(spacy_en)

    

    def by_label(words, label):

        filtered = [word.text for word in words.ents if word.label_ == label]

        return filtered

    

    def get_top(label, n=10):

        thing = as_words.apply(lambda x: by_label(x, label))

        flattened = itertools.chain(*thing.values)

        

        counter = Counter(flattened)

        topN = counter.most_common(n)

        

        topN_things = [thing for thing, _ in topN]

        

        return thing

    

    entities = pd.DataFrame()

    entities["people"] = get_top("PERSON", n=10)

    entities["geopolitics"] = get_top("GPE", n=10)

    entities["organizations"] = get_top("ORG")

    

    return entities
from collections import Counter
entities = {}



by_sentiment = eda.groupby("sentiment")



for sentiment, group in tqdm(by_sentiment):

    entities[sentiment] = crude_entities(group["lemmatized"])

    

display(pd.concat(entities, axis=1).head(n=10))
# Preprocessing

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import TfidfTransformer 

from sklearn.model_selection import train_test_split, RandomizedSearchCV



# Building classification models

from sklearn.svm import LinearSVC

from sklearn.pipeline import Pipeline

from sklearn.naive_bayes import MultinomialNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression



# Model evaluation

from sklearn import metrics

from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
X_all = tweets["message"]

y_all = tweets["sentiment"]



X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.25, random_state=1337)



X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.25, random_state=1337)
tfidf = TfidfVectorizer()

tfidf.fit_transform(X_train)
def train(tfidf, model, train_data, train_labels, test_data):

    model.fit(tfidf.transform(train_data), train_labels)

    preds = model.predict(tfidf.transform(test_data))

    

    return preds
def grade(model, preds, test_labels):

    print(metrics.classification_report(test_labels, preds))

    

    cm = confusion_matrix(test_labels, preds)

    cm_normd = cm / cm.sum(axis=1).reshape(-1, 1)

    

    heatmap_kwargs = dict(

        cmap="YlGnBu",

        xticklabels=model.classes_,

        yticklabels=model.classes_,

        vmin=0.,

        vmax=1.,

        annot=True,

        annot_kws={"size": 10},

    )

    

    sns.heatmap(cm_normd, **heatmap_kwargs)

    

    plt.title(f"{model.__class__.__name__} Classification")

    plt.ylabel("Ground-truth labels")

    plt.xlabel("Predicted labels")

    plt.plot()
def train_and_grade(tfidf, model, train_data, train_labels, test_data, test_labels):

    preds = train(tfidf, model, train_data, train_labels, test_data)

    grade(model, preds, test_labels)
rf = RandomForestClassifier(max_depth=5, n_estimators=100)

train_and_grade(tfidf, rf, X_train, y_train, X_valid, y_valid)
nb = MultinomialNB()

train_and_grade(tfidf, nb, X_train, y_train, X_valid, y_valid)
knn = KNeighborsClassifier(n_neighbors=5, metric="minkowski", p=2)

train_and_grade(tfidf, knn, X_train, y_train, X_valid, y_valid)
logreg = LogisticRegression(C=1, class_weight="balanced", max_iter=1000)

train_and_grade(tfidf, logreg, X_train, y_train, X_valid, y_valid)
svm_lsvc = LinearSVC(class_weight="balanced")

train_and_grade(tfidf, svm_lsvc, X_train, y_train, X_valid, y_valid)