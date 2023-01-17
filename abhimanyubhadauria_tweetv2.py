import pandas as pd

import numpy as np
train=pd.read_csv("../input/nlp-getting-started/train.csv")

test=pd.read_csv("../input/nlp-getting-started/test.csv")
train.head()
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer()

tfidf.fit(train['text'])
X = tfidf.transform(train['text'])

train['text'][1]
print([X[1, tfidf.vocabulary_['fire']]])
print([X[1, tfidf.vocabulary_['forest']]])
from sklearn.model_selection import train_test_split

X = train.text

y = train.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

print("Train set has total {0} entries with {1:.2f}% negative, {2:.2f}% positive".format(len(X_train),

                                                                             (len(X_train[y_train == 0]) / (len(X_train)*1.))*100,

                                                                            (len(X_train[y_train == 1]) / (len(X_train)*1.))*100))
print("Test set has total {0} entries with {1:.2f}% negative, {2:.2f}% positive".format(len(X_test),

                                                                             (len(X_test[y_test == 0]) / (len(X_test)*1.))*100,

                                                                            (len(X_test[y_test == 1]) / (len(X_test)*1.))*100))
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import Pipeline

from sklearn.metrics import accuracy_score

def accuracy_summary(pipeline, X_train, y_train, X_test, y_test):

    sentiment_fit = pipeline.fit(X_train, y_train)

    y_pred = sentiment_fit.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print("accuracy score: {0:.2f}%".format(accuracy*100))

    return accuracy
cv = CountVectorizer()

rf = RandomForestClassifier(class_weight="balanced")

n_features = np.arange(5000,6001,5000)

def nfeature_accuracy_checker(vectorizer=cv, n_features=n_features, stop_words=None, ngram_range=(1, 1), classifier=rf):

    result = []

    print(classifier)

    print("\n")

    for n in n_features:

        vectorizer.set_params(stop_words=stop_words, max_features=n, ngram_range=ngram_range)

        checker_pipeline = Pipeline([

            ('vectorizer', vectorizer),

            ('classifier', classifier)

        ])

        print("Test result for {} features".format(n))

        nfeature_accuracy = accuracy_summary(checker_pipeline, X_train, y_train, X_test, y_test)

        result.append((n,nfeature_accuracy))

    return result

tfidf = TfidfVectorizer()

print("Result for trigram with stop words (Tfidf)\n")

feature_result_tgt = nfeature_accuracy_checker(vectorizer=tfidf,ngram_range=(1,5))
from sklearn.metrics import classification_report

cv = CountVectorizer(max_features=500,ngram_range=(0,1))

pipeline = Pipeline([

        ('vectorizer', cv),

        ('classifier', rf)

    ])

sentiment_fit = pipeline.fit(X_train, y_train)

y_pred = sentiment_fit.predict(X_test)

print(classification_report(y_test, y_pred, target_names=['negative','positive']))
from sklearn.feature_selection import chi2

import matplotlib.pyplot as plt

%matplotlib inline

tfidf = TfidfVectorizer(max_features=30000,ngram_range=(1, 3))

X_tfidf = tfidf.fit_transform(train.text)

y = train.target

chi2score = chi2(X_tfidf, y)[0]

plt.figure(figsize=(12,8))

scores = list(zip(tfidf.get_feature_names(), chi2score))

chi2 = sorted(scores, key=lambda x:x[1])

topchi2 = list(zip(*chi2[-20:]))

x = range(len(topchi2[1]))

labels = topchi2[0]

plt.barh(x,topchi2[1], align='center', alpha=0.5)

plt.plot(topchi2[1], x, '-o', markersize=5, alpha=0.8)

plt.yticks(x, labels)

plt.xlabel('$\chi^2$')

plt.show();
new=test['text']
y_prednew = sentiment_fit.predict(new)
y_prednew
y_prednew.shape
test.shape
test.head()
my_submission = pd.DataFrame({'id': test.id, 'target': y_prednew})

my_submission.to_csv('submission.csv', index=False)
my_submission.head()