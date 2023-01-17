import numpy as np

import pandas as pd

import os

os.listdir('../input')
train = pd.read_csv('../input/train.tsv', delimiter='\t', encoding='utf-8')

test = pd.read_csv('../input/test.tsv', delimiter='\t', encoding='utf-8')
train.head()
train.describe()
import seaborn as sns

from matplotlib import pyplot as plot
fig = plot.figure(figsize=(15, 5))

sns.countplot(data=train, x='Sentiment')

plot.show()
def get_count():

    s0 = train[train.Sentiment == 0].Sentiment.count()

    s1 = train[train.Sentiment == 1].Sentiment.count()

    s2 = train[train.Sentiment == 2].Sentiment.count()

    s3 = train[train.Sentiment == 3].Sentiment.count()

    s4 = train[train.Sentiment == 4].Sentiment.count()

    return s0, s1, s2, s3, s4



s0, s1, s2, s3, s4 = get_count()

print(s0, s1, s2, s3, s4)
df0 = s2 // s0 - 1

df1 = s2 // s1 - 1

df3 = s2 // s3 - 1

df4 = s2 // s4 - 1

 

train = train.append([train[train.Sentiment == 0]] * df0, ignore_index=True)

train = train.append([train[train.Sentiment == 1]] * df1, ignore_index=True)

train = train.append([train[train.Sentiment == 3]] * df3, ignore_index=True)

train = train.append([train[train.Sentiment == 4]] * df4, ignore_index=True)

train = train.append([train[train.Sentiment == 0][0 : s2 % s0]], ignore_index=True)

train = train.append([train[train.Sentiment == 1][0 : s2 % s1]], ignore_index=True)

train = train.append([train[train.Sentiment == 3][0 : s2 % s3]], ignore_index=True)

train = train.append([train[train.Sentiment == 4][0 : s2 % s4]], ignore_index=True)



s0, s1, s2, s3, s4 = get_count()

print(s0, s1, s2, s3, s4)
fig = plot.figure(figsize=(15, 5))

sns.countplot(data = train, x = 'Sentiment')

plot.show()
import re

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer

from joblib import Parallel, delayed

import string 

import time 



lemma = WordNetLemmatizer() 

stopwords  = stopwords.words('english')

stopwords.extend(['cinema', 'film', 'series', 'movie', 'one', 'like', 'story', 'plot'])



def clean_review(review):

    tokens = review.lower().split()

    filtered_tokens = [lemma.lemmatize(w) for w in tokens if w not in stopwords]

    return " ".join(filtered_tokens)



start_time = time.time()

clean_train_data = train.copy()

clean_train_data['Phrase'] = Parallel(n_jobs=4)(delayed(clean_review)(review) for review in train['Phrase'])

end_time = time.time()

print("Cleaning Training Data Time - Processing Time = ", end_time - start_time)



# remove missing values

print("Cleaned entries: ", clean_train_data.shape[0], " out of ", train.shape[0])
from sklearn.model_selection import train_test_split

target = clean_train_data.Sentiment

train_X_, validation_X_, train_y, validation_y = train_test_split(clean_train_data['Phrase'], target, test_size=0.2, random_state=22)
from sklearn.feature_extraction.text import TfidfVectorizer as tfidf



tfidf_vec = tfidf(min_df=3,  max_features=None, ngram_range=(1, 2), use_idf=1)

train_X = tfidf_vec.fit_transform(train_X_)



print("Succesfully vectorized the data.")
from sklearn.naive_bayes import MultinomialNB

from sklearn import metrics



print("Using Multinomial Naive Bayes : \n")

model = MultinomialNB()

model.fit(train_X, train_y)

validation_X = tfidf_vec.transform(validation_X_)

predicted = model.predict(validation_X)

expected = validation_y

print(metrics.classification_report(expected, predicted))

print("Accuracy Score in Cross-Validation Set : ", metrics.accuracy_score(expected, predicted))
from sklearn.svm import LinearSVC

from sklearn import metrics



print("Using Linear SVM : \n")

model = LinearSVC()

model.fit(train_X, train_y)

validation_X = tfidf_vec.transform(validation_X_)

predicted = model.predict(validation_X)

expected = validation_y

print(metrics.classification_report(expected, predicted))

print("Accuracy Score in Cross-Validation Set : ", metrics.accuracy_score(expected, predicted))
from sklearn.model_selection import GridSearchCV

final_model = LinearSVC()

parameters = [{'C': [1, 10, 100, 1000]}]

final_tuned_model = GridSearchCV(final_model, parameters, cv = 2, n_jobs = 5, verbose=True)



final_tuned_model.fit(train_X, train_y)

validation_X = tfidf_vec.transform(validation_X_)

predicted = final_tuned_model.predict(validation_X)

expected = validation_y

print(metrics.classification_report(expected, predicted))

print("Accuracy Score in Cross-Validation Set : ", metrics.accuracy_score(expected, predicted))



print(final_tuned_model.best_score_)

print(final_tuned_model.best_params_)
clean_test_data = test.copy()

start_time = time.time()

clean_test_data['Phrase'] = Parallel(n_jobs=4)(delayed(clean_review)(review) for review in test['Phrase'])

end_time = time.time()

print("Cleaning Testing Data - Processing time = ", end_time - start_time)



# Removing missing values

print("Clean entries: ", clean_test_data.shape[0], " out of ", test.shape[0])

test_X = tfidf_vec.transform(clean_test_data['Phrase'])



test_pred = final_tuned_model.predict(test_X)
sub_file = pd.read_csv('../input/sampleSubmission.csv',sep=',')

sub_file.Sentiment=test_pred

sub_file.to_csv('final_submission.csv',index=False)