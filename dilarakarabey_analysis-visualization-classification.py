import pandas as pd

import numpy as np

import re

import datetime as dt

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.feature_extraction.text import CountVectorizer



%matplotlib inline



warnings.simplefilter(action='ignore', category=Warning)

pd.options.mode.chained_assignment = None
fake = pd.read_csv("../input/fake-and-real-news-dataset/Fake.csv", usecols=["title", "subject", "date"]).copy()

real = pd.read_csv("../input/fake-and-real-news-dataset/True.csv", usecols=["title", "subject", "date"]).copy()
fake["label"] = "fake"

real["label"] = "real"

fake.head()
real.head()
news = pd.concat([fake, real], axis=0).sample(frac=1, random_state=1).reset_index(drop=True)

news.head()
news.label.value_counts(dropna=False)
news.subject.value_counts(dropna=False)
news["subject"] = news["subject"].replace({"politicsNews": "Politics",

                                           "worldnews": "World",

                                           "politics": "Politics",

                                           "News": "All",

                                           "left-news": "Left", 

                                           "Government News": "Government",

                                           "US_News": "US",

                                           "Middle-east": "Middle East"})

news.subject.value_counts()
news.date.value_counts(dropna=False)
news[news.date.str.extract(r"^((?!\w+ \d+, \d+))*", expand=False).notnull()]
news.loc[news.date.str.extract(r"^((?!\w+ \d+, \d+))*", expand=False).notnull(), "date"] = np.nan
news[news.date.isnull()].shape[0]
news.info()
news.date = pd.to_datetime(news.date, errors="coerce")

news_grouped = news[["date", "subject", "label"]].groupby(["date", "label"]).count().reset_index()



fig, ax = plt.subplots(figsize=(16,10))

sns.lineplot(x="date", y="subject", hue="label", data=news_grouped, palette="Set2", ax=ax)

plt.title("News Articles Labelled Fake vs. Real")

plt.xlabel("Time")

plt.ylabel("Count")
news_group_by_subj_and_label = news.groupby(by=["label", "subject"]).count().reset_index()



fig1, ax1 = plt.subplots(figsize=(16, 8))

sns.barplot(x="subject", y="title", hue="label", data=news_group_by_subj_and_label, palette="Set2", saturation=0.5, ax=ax1)

plt.title("Fake vs. Real News Articles by Subjects")

plt.xticks(rotation=45, horizontalalignment='center', fontweight='light', fontsize='x-large')

plt.yticks(horizontalalignment='center', fontweight='light', fontsize='large')

plt.xlabel("Subjects", fontsize="large")

plt.ylabel("Amount", fontsize="large")

plt.legend(fontsize="large")
news_clean = news[news.date > dt.datetime(2016,1,1)]

news_clean.date.value_counts().sort_index()
news_clean = news_clean[news_clean.subject.isin(["All", "Politics", "World"])]

news_clean.subject.value_counts()
news_clean.shape[0]
training_data, testing_data = train_test_split(news_clean, random_state=1) #seed for reproducibility



Y_train = training_data["label"].values

Y_test = testing_data["label"].values



def word_counter(data, column, training_set, testing_set):



    cv = CountVectorizer(binary=False, max_df=0.95)

    cv.fit_transform(training_data[column].values)

    

    train_feature_set = cv.transform(training_data[column].values)

    test_feature_set = cv.transform(testing_data[column].values)

    

    return train_feature_set, test_feature_set, cv



X_train, X_test, feature_transformer = word_counter(news_clean, "title", training_data, testing_data)
classifier = LogisticRegression(solver="newton-cg", C=5, penalty="l2", multi_class="multinomial", max_iter=1000)

model = classifier.fit(X_train, Y_train)
predictions = model.predict(X_test)

accuracy = accuracy_score(Y_test, predictions, normalize=True)

print("Our model has {}% prediction accuracy.".format(round(accuracy, 2) * 100))