import numpy as np

import pandas as pd



df = pd.read_csv("../input/naive-bayes-imdb/train.csv", index_col=0)

df.head()
df.loc[0, 'review']
from sklearn.feature_extraction.text import CountVectorizer



vectorizer = CountVectorizer(binary=True)

vectorizer.fit(df.review)
len(vectorizer.vocabulary_)
from sklearn.model_selection import train_test_split



df_train, df_test = train_test_split(df, test_size=0.2, shuffle=True)
X_train = vectorizer.transform(df_train.review)

X_test = vectorizer.transform(df_test.review)



X_train.shape, X_test.shape
from sklearn.naive_bayes import BernoulliNB



clf = BernoulliNB().fit(X_train, df_train.label)
np.exp(clf.class_log_prior_)
def show_top10(classifier, vectorizer, categories=('neg', 'pos')):

    feature_names = np.asarray(vectorizer.get_feature_names())

    for i, category in enumerate(categories):

        top10 = np.argsort(classifier.feature_log_prob_[i])[-10:]

        print("%s: %s" % (category, " ".join(feature_names[top10])))



show_top10(clf, vectorizer)
from sklearn.metrics import classification_report



predicts = clf.predict(X_train)

print(classification_report(df_train.label, predicts))
predicts = clf.predict(X_test)

print(classification_report(df_test.label, predicts))
count_vect = CountVectorizer(binary=False).fit(df.review)



X_train_counts = count_vect.transform(df_train.review)

X_test_counts = count_vect.transform(df_test.review)
dict(zip(count_vect.inverse_transform(X_train_counts[0])[0], X_train_counts[0].data))
from sklearn.naive_bayes import MultinomialNB



clf = MultinomialNB().fit(X_train_counts, df_train.label)
predicts = clf.predict(X_train_counts)

print(classification_report(df_train.label, predicts))
X_test_counts = count_vect.transform(df_test.review)

predicts = clf.predict(X_test_counts)

print(classification_report(df_test.label, predicts))
show_top10(clf, count_vect)
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS



count_vect = CountVectorizer(stop_words=ENGLISH_STOP_WORDS, binary=False).fit(df.review)



X_train_counts = count_vect.transform(df_train.review)

X_test_counts = count_vect.transform(df_test.review)
clf = MultinomialNB().fit(X_train_counts, df_train.label)
predicts = clf.predict(X_test_counts)

print(classification_report(df_test.label, predicts))
show_top10(clf, count_vect)
from sklearn.feature_extraction.text import TfidfVectorizer



vectorizer = TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS)

vectorizer = vectorizer.fit(df.review)
X_train_vectors = vectorizer.transform(df_train.review)

X_test_vectors = vectorizer.transform(df_test.review)
X_train_vectors[0].data
vectorizer.inverse_transform(X_train_vectors[0])[0][np.argsort(X_train_vectors[0].data)]
clf = MultinomialNB().fit(X_train_vectors, df_train.label)
show_top10(clf, vectorizer)
predicts = clf.predict(X_train_vectors)

print(classification_report(df_train.label, predicts))
X_test_vectors = vectorizer.transform(df_test.review)

predicts = clf.predict(X_test_vectors)

print(classification_report(df_test.label, predicts))
vectorizer = TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS, ngram_range=(1, 2)).fit(df.review)



X_train_vectors = vectorizer.transform(df_train.review)

X_test_vectors = vectorizer.transform(df_test.review)
vectorizer.inverse_transform(X_train_vectors[0])[0][np.argsort(X_train_vectors[0].data)]
clf = MultinomialNB().fit(X_train_vectors, df_train.label)
predicts = clf.predict(X_train_vectors)

print(classification_report(df_train.label, predicts))
X_test_vectors = vectorizer.transform(df_test.review)

predicts = clf.predict(X_test_vectors)

print(classification_report(df_test.label, predicts))
test = pd.read_csv('../input/naive-bayes-imdb/test.csv', index_col=0)

predicted = clf.predict(vectorizer.transform(test.review))



pd.DataFrame({'Predicted': predicted}).to_csv('solution.csv', index_label='Id')