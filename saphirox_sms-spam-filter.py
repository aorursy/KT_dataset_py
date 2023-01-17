import numpy as np

import pandas as pd

import plotly as py

import matplotlib.pyplot as plt

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go

import colorlover as cl

import operator

%matplotlib inline

import string

import itertools

import re

import warnings

warnings.filterwarnings('ignore')



from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score, GridSearchCV

from sklearn.naive_bayes import BernoulliNB, MultinomialNB, GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix, classification_report

from sklearn.decomposition import PCA

import xgboost as xgb

import seaborn as sns



import nltk

from nltk.corpus import stopwords



from collections import Counter, OrderedDict



import os

print(os.listdir("../input"))



from IPython.display import HTML



RANDOM_STATE = 43
df = pd.read_csv("../input/spam.csv",encoding='latin-1')

df.head()
df = df.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)

df = df.rename(columns= {"v1": "label", "v2": "text"})

df.label = df.label.astype('category') 

df.text = df.text.astype('str')

df.head()
df.describe()
df.label.value_counts()

_, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20, 5))

df.label.value_counts(sort=True).plot(kind='pie', ax=ax2, autopct='%1.0f%%')

df.label.value_counts(sort=True).plot(kind='bar', color=['blue', 'red'], ax=ax1)

ax1.set_title('Objects counts')

ax1.set_ylabel('Count')

ax1.set_xlabel('Label')

ax2.set_title('Objects counts')
def get_number_checker():

    checker_func = np.vectorize(lambda x: re.search("[0-9]{10}", x) != None or re.search("[0-9]{3}-[0-9]{3}-[0-9]{3}", x) != None)

    return df[checker_func(df.text)]



get_number_checker().label.value_counts(sort=True).plot(kind="bar")

checker_func = np.vectorize(lambda x: re.search("[0-9]{10}", x) != None or re.search("[0-9]{3}-[0-9]{3}-[0-9]{3}", x) != None)

df = df.assign(has_phone_number=checker_func(df.text))
MAX_COMMON_WORDS = 100



def clean_from_stop_words_and_punctuation(x):

    return [word.lower() for word in x.split() if word.lower() not in stopwords.words('english') and  word.lower() not in string.punctuation]



def sort_dict_by_value(t):

    return sorted(t, key=lambda x: x[1],reverse=True)



def get_word_arr(label):

    clean_and_join = lambda x: " ".join(clean_from_stop_words_and_punctuation(x))

    cleaned_arr = df[df.label==label].text.apply(clean_and_join)

    splitted_strings = cleaned_arr.apply(lambda word: word.split(" ")).values

    return list(itertools.chain.from_iterable(splitted_strings))



def get_counter_dict(label):

    return sort_dict_by_value( Counter(get_word_arr(label)).most_common(MAX_COMMON_WORDS))



counter_ham = get_counter_dict('ham')

counter_spam = get_counter_dict('spam')
spam_counter_df = pd.DataFrame.from_dict(counter_spam)

spam_counter_df.T
ham_counter_df = pd.DataFrame.from_dict(counter_ham)

ham_counter_df.T
ham_plot = go.Bar(

    x = ham_counter_df.iloc[:, 0],

    y = ham_counter_df.iloc[:, 1],

    name = "Commom spam words"

)



iplot([ham_plot])
iplot([go.Bar(

    x = spam_counter_df.iloc[:, 0],

    y = spam_counter_df.iloc[:, 1],

    marker = dict(

        color=cl.scales['3']['div']['RdYlBu'][0]

    )

)])
len_df = df.assign(len=df.text.apply(lambda x: len(x)))

len_df.head()
_, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(25, 5))

ax = sns.countplot(data=len_df.sort_values(['len'], ascending=False).sample(200), x='len', hue='label', ax=ax1,dodge=True)

ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")

plt.show()

dtc = GridSearchCV(DecisionTreeClassifier(), param_grid = { 'criterion': ['gini', 'entropy'] }, cv=5)

dtc.fit(np.zeros((df.shape[0],1)), df.label).best_score_
vectorizer = CountVectorizer()



X = vectorizer.fit_transform(df.text)

X.shape
df.label = df.label.map({ 'spam': 1, 'ham': 0 })

y = df.label
pca = PCA(n_components=2).fit(X.toarray())

data2D = pca.transform(X.toarray())

fig, ax = plt.subplots(figsize=(20, 15))

ax.set_title('Vectorize plot')

ax.set_ylabel('PC2')

ax.set_xlabel('PC1')

ax.legend(['HAM', 'SPAM'])

sns.scatterplot(data2D[:,0], data2D[:,1], hue=df.label, ax=ax)
bnb = GridSearchCV(BernoulliNB(),{ 'alpha':range(100),}, cv=StratifiedKFold(n_splits=5), refit=True)

cross_val_score(bnb, X, y, cv=5)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3)

fit = bnb.fit(X_train,y_train)

print(classification_report(y_test, fit.predict(X_test)))
mnb = GridSearchCV(MultinomialNB(),{ 'alpha':range(100),}, cv=StratifiedKFold(n_splits=5), refit=True)

fit = mnb.fit(X_train,y_train)

print(classification_report(y_test, fit.predict(X_test)))
gnb = GaussianNB()

fit = gnb.fit(X_train.toarray(), y_train)

print(classification_report(y_test, fit.predict(X_test.toarray())))