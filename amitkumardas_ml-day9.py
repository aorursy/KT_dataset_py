# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import warnings



# Task 1: Download the datasets "sentiment_train" and "sentiment_test" from Kaggle and load it as a Python dataframe object

train_ds = pd.read_csv("../input/sentiment_train.csv")

print("Training dataset: ")

print(train_ds.head())

print(train_ds.shape)



# Task 2. Find the number of positive and negative sentiment documents in the dataset



pos_sentiment_count = len(train_ds[train_ds.label == 1])

neg_sentiment_count = len(train_ds[train_ds.label == 0])



print("Count of positive sentiments: ", pos_sentiment_count)

print("Count of negative sentiments: ", neg_sentiment_count)



# Task 3. Create a count plot using Seaborn library



import matplotlib.pyplot as plt

import seaborn as sn



plt.figure( figsize=(6,5))

ax = sn.countplot(x='label', data=train_ds)



# Task 4. Create Bag-of-Words model of all documents



from sklearn.feature_extraction.text import CountVectorizer



# Initialize the CountVectorizer

count_vectorizer = CountVectorizer()

# Create the dictionary from the corpus

feature_vector = count_vectorizer.fit(train_ds.sentence )

# Get the feature names

features = feature_vector.get_feature_names()

print("Total number of features: ", len(features))

train_ds_features = count_vectorizer.transform(train_ds.sentence)

print(train_ds_features.shape)



# Converting the matrix to a dataframe

train_ds_df = pd.DataFrame(train_ds_features.todense())

# Setting the column names to the features i.e. words

train_ds_df.columns = features



print(train_ds[4:12])

print(train_ds_df.iloc[4:12, 204:212])



print(train_ds_df[['brokeback', 'mountain', 'is', 'such', 'horrible', 'movie']][0:1])



# Task 5. Remove not-so-useful features.



# summing up the occurances of features column wise

features_counts = np.sum (train_ds_features.toarray(), axis = 0)

feature_counts_df = pd.DataFrame (dict(features = features, counts = features_counts))

plt.figure( figsize=(12,5))

plt.hist(feature_counts_df.counts, bins=50);

plt.xlabel('Frequency of words')

plt.ylabel('Density')



feat_appear_once = len(feature_counts_df[feature_counts_df.counts == 1])

print("Number of words present only once across documents", feat_appear_once)

useful_feat_count = len(features) - feat_appear_once



# Initialize the CountVectorizer

count_vectorizer = CountVectorizer(max_features = useful_feat_count)

# Create the dictionary from the corpus

feature_vector = count_vectorizer.fit(train_ds.sentence )

# Get the feature names

features = feature_vector.get_feature_names()

# Transform the document into vectors

train_ds_features = count_vectorizer.transform(train_ds.sentence)

# Count the frequency of the features

features_counts = np.sum(train_ds_features.toarray(), axis = 0 )

feature_counts = pd.DataFrame(dict(features = features, counts = features_counts))



print(feature_counts.sort_values('counts', ascending = False)[0:15])



# Task 6. Create count vectors by removing stop words 

from sklearn.feature_extraction import text



my_stop_words = text.ENGLISH_STOP_WORDS



#Printing first few stop words

print("Few stop words: ", list(my_stop_words)[0:10])



# Setting stop words list

count_vectorizer = CountVectorizer(stop_words = my_stop_words, max_features = useful_feat_count)

feature_vector = count_vectorizer.fit(train_ds.sentence )

train_ds_features = count_vectorizer.transform(train_ds.sentence)

features = feature_vector.get_feature_names()

features_counts = np.sum(train_ds_features.toarray(), axis = 0 )

feature_counts = pd.DataFrame(dict(features = features, counts = features_counts))

print(feature_counts.sort_values('counts', ascending = False)[0:15])



# Task 7. Build a classification model (Naive Bayes) using dataset (split it in 80:20 ratio) to predict whether a test document is positive or negative.

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import BernoulliNB

from sklearn import metrics



train_X, test_X, train_y, test_y = train_test_split(train_ds_features, train_ds.label, test_size = 0.2, random_state = 123)

nb_clf = BernoulliNB()

nb_clf.fit(train_X.toarray(), train_y)

test_ds_predicted = nb_clf.predict(test_X.toarray())



#Task 8. Apply the model on the 20% data held out as test data from dataset. Create the confusion matrix.

cm = metrics.confusion_matrix(test_y, test_ds_predicted)

sn.heatmap(cm, annot=True, fmt='.2f');

print(metrics.classification_report(test_y, test_ds_predicted))