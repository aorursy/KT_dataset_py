# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.feature_extraction.text import CountVectorizer

import matplotlib.pyplot as plt

import seaborn as sns

from nltk.tokenize import TweetTokenizer

from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score, classification_report



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv('/kaggle/input/twitter-sentiment-analysis-hatred-speech/train.csv')

test_data = pd.read_csv('/kaggle/input/twitter-sentiment-analysis-hatred-speech/test.csv')



print(train_data.info())

print(train_data.head())

train_labels_count = train_data['label'].value_counts()

sns.barplot(train_labels_count.index,train_labels_count.values)

plt.title('Number of labels'), plt.ylabel('Number'), plt.xlabel('Label')

plt.show()

train_labels_count.plot(kind='pie', autopct = '%1.1f%%')

plt.title('Distribution of labels in percents')

plt.show()

X = train_data['tweet']

y = train_data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)



def data_cleaning(df):

    tw_tokenizer = TweetTokenizer(strip_handles=True, reduce_len=True)

    df = df.apply(lambda x: tw_tokenizer.tokenize(x))

    stopw = stopwords.words('english')

    df = df.apply(lambda x: [item for item in x if item not in stopw] )

    #Count vectorizer accepts strings not lists

    df = df.apply(lambda x: ' '.join(map(str, x)))

    return df



X_train = data_cleaning(X_train)

X_test = data_cleaning(X_test)





vectorizer = CountVectorizer()

tf_train = vectorizer.fit_transform(X_train)

tf_test = vectorizer.transform(X_test)



print(X_train.shape)

print(y_train.shape)

model = MultinomialNB(alpha=0.01)

model.fit(tf_train, y_train)

predictions = model.predict(tf_test)
print(accuracy_score(y_test, predictions))

print(classification_report(y_test, predictions))
# with StratifiedShuffleSplit



X = train_data['tweet']

y = train_data['label']



shuffle_stratified = StratifiedShuffleSplit(n_splits=1, test_size=0.2)



def data_cleaning(df):

    tw_tokenizer = TweetTokenizer(strip_handles=True, reduce_len=True)

    df = df.apply(lambda x: tw_tokenizer.tokenize(x))

    stopw = stopwords.words('english')

    df = df.apply(lambda x: [item for item in x if item not in stopw] )

    #Count vectorizer accepts strings not lists

    df = df.apply(lambda x: ' '.join(map(str, x)))

    return df



X = data_cleaning(X)

tweets = X.values

labels = y.values



for train_index, test_index in shuffle_stratified.split(tweets, labels):

    tweets_train, tweets_test = tweets[train_index], tweets[test_index]

    labels_train, labels_test = labels[train_index], labels[test_index]



vectorizer = CountVectorizer()

tfs_train = vectorizer.fit_transform(tweets_train)

tfs_test = vectorizer.transform(tweets_test)

print(tfs_train.shape)

print(tfs_test.shape)



model = MultinomialNB(alpha=0.01)

model.fit(tfs_train, labels_train)

predictions2 = model.predict(tfs_test)

print(accuracy_score(labels_test, predictions2))

print(classification_report(labels_test, predictions2))
# Applying over-sampling to balance the categories



from imblearn.over_sampling import SMOTE



X = train_data['tweet']

y = train_data['label']



def data_cleaning(df):

    tw_tokenizer = TweetTokenizer(strip_handles=True, reduce_len=True)

    df = df.apply(lambda x: tw_tokenizer.tokenize(x))

    stopw = stopwords.words('english')

    df = df.apply(lambda x: [item for item in x if item not in stopw] )

    #Count vectorizer accepts strings not lists

    df = df.apply(lambda x: ' '.join(map(str, x)))

    return df



X = data_cleaning(X)

tweets = X.values

labels = y.values



print(tweets.shape)



vectorizer = CountVectorizer()

tweets_numerical = vectorizer.fit_transform(tweets)



sm = SMOTE(random_state=12, ratio = 0.75)

x_res, y_res = sm.fit_sample(tweets_numerical, labels)

print(tweets_numerical.shape)

print(x_res.shape)



x_train_res, x_val_res, y_train_res, y_val_res = train_test_split(x_res, y_res, test_size=0.2)



# model = MultinomialNB(alpha=0.01)

# model.fit(x_train_res, y_train_res)

# predictions3 = model.predict(x_val_res)

# print(accuracy_score(y_val_res, predictions3))

# print(classification_report(y_val_res, predictions3))





from sklearn import svm

from sklearn.metrics import classification_report



classifier_linear = svm.SVC(kernel='linear')

classifier_linear.fit(x_train_res, y_train_res)

pred = classifier_linear.predict(x_val_res)

print(accuracy_score(y_val_res, pred))

print(classification_report(y_val_res, pred))

X = train_data['tweet']

y = train_data['label']



shuffle_stratified = StratifiedShuffleSplit(n_splits=1, test_size=0.2)



def data_cleaning(df):

    tw_tokenizer = TweetTokenizer(strip_handles=True, reduce_len=True)

    df = df.apply(lambda x: tw_tokenizer.tokenize(x))

    stopw = stopwords.words('english')

    df = df.apply(lambda x: [item for item in x if item not in stopw] )

    #Count vectorizer accepts strings not lists

    df = df.apply(lambda x: ' '.join(map(str, x)))

    return df



X = data_cleaning(X)

tweets = X.values

labels = y.values



for train_index, test_index in shuffle_stratified.split(tweets, labels):

    tweets_train, tweets_test = tweets[train_index], tweets[test_index]

    labels_train, labels_test = labels[train_index], labels[test_index]



vectorizer = CountVectorizer()

tfs_train = vectorizer.fit_transform(tweets_train)

tfs_test = vectorizer.transform(tweets_test)

print(tfs_train.shape)

print(tfs_test.shape)



from sklearn import svm

from sklearn.metrics import classification_report



classifier_linear = svm.SVC(kernel='linear')

classifier_linear.fit(tfs_train, labels_train)

pred = classifier_linear.predict(tfs_test)

print(accuracy_score(labels_test, pred))

print(classification_report(labels_test, pred))