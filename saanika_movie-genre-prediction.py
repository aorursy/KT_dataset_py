# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os



print("Reading data")

print(os.listdir("../input"))



##For data preprocessing

from bs4 import BeautifulSoup

import re

from nltk.corpus import stopwords



##For Machine Learning

from sklearn import svm

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score, f1_score

# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/wiki_movie_plots_deduped.csv")

df.tail()

df.info()
df['Genre']=df['Genre'].replace('unknown',np.nan)

df=df.dropna(axis=0, subset=['Genre'])

print(df.tail())
print(len(df))

print(df.shape)

a=df['Genre'].value_counts()[:20]

b=a.keys().tolist()

print(b)

df=df[df.Genre.isin(b)]

df=df.reset_index(drop=True)

sns.set(style="white")

genre_to_count=pd.DataFrame({'Genre':a.index, 'Count':a.values})

plt.figure(figsize=(15,10))

sns.barplot(y="Genre", x="Count", data=genre_to_count,palette="Blues_d")
def plotToWords(raw_plot):

    letters_only = re.sub("[^a-zA-Z]", " ", raw_plot)

    lower_case = letters_only.lower()

    words = lower_case.split()

    stops = set(stopwords.words("english"))

    meaningful_words = [w for w in words if not w in stops]

    return (" ".join(meaningful_words))



def preprocess(dataframe):

    clean_train_reviews = []

    for i in range(0,len(dataframe)):

        clean_train_reviews.append(plotToWords(dataframe.iloc[i]['Plot']))

    dataframe['Plot']=clean_train_reviews

    return dataframe



df=preprocess(df)

print(df["Plot"][:10])
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), max_features=4000)

features = tfidf.fit_transform(df.Plot).toarray()

labels = df.Genre

features.shape
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.naive_bayes import MultinomialNB

X_train, X_test, y_train, y_test = train_test_split(df['Plot'], df['Genre'], random_state = 0)

count_vect = CountVectorizer()

X_train_counts = count_vect.fit_transform(X_train)

tfidf_transformer = TfidfTransformer()

X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

clf = MultinomialNB().fit(X_train_tfidf, y_train)

print(clf.predict(count_vect.transform(["zafer sailor living mother nd coastal village izmir separated girlfriend mehtap whose father also sailor nd friend fahriye try help zafer marry someone family famous talented actress asl surprisingly attends zafer boat tour asli zafer find getting know"])))
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import LinearSVC

from sklearn.model_selection import cross_val_score

models = [

    LinearSVC(),

    MultinomialNB(),

    LogisticRegression(random_state=0),

]

CV = 5

cv_df = pd.DataFrame(index=range(CV * len(models)))

entries = []

for model in models:

  model_name = model.__class__.__name__

  accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)

  for fold_idx, accuracy in enumerate(accuracies):

    entries.append((model_name, fold_idx, accuracy))

cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])

import seaborn as sns

sns.boxplot(x='model_name', y='accuracy', data=cv_df)

sns.stripplot(x='model_name', y='accuracy', data=cv_df, 

              size=8, jitter=True, edgecolor="gray", linewidth=2)

plt.show()
cv_df.groupby('model_name').accuracy.mean()
