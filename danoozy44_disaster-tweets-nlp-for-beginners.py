# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
train['text'].iloc[2]
from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize
stop_words = set(stopwords.words('english'))
word_tokens = word_tokenize(train['text'].iloc[2])

print(word_tokens)
filtered_tweet = [w for w in word_tokens if not w in stop_words]

print(filtered_tweet)
from sklearn.feature_extraction import text
count_vectorizer = text.CountVectorizer()



train_vectors = count_vectorizer.fit_transform(train['text'])

test_vectors = count_vectorizer.transform(test['text'])



train_vectors
print(train_vectors[0].todense().shape)

print(train_vectors[0].todense())
def remove_stopwords(df):

    

    for i in range(len(df)):

        

        word_tokens = word_tokenize(df['text'].loc[i])

        filtered_set = []

        

        for w in word_tokens:

            if w not in stop_words:

                filtered_set.append(w)

                

        filtered_sentence = ' '.join(filtered_set)

        df['text'].iloc[i] = filtered_sentence
remove_stopwords(train)

remove_stopwords(test)
count_vectorizer = text.CountVectorizer()



train_vectors = count_vectorizer.fit_transform(train['text'])

test_vectors = count_vectorizer.transform(test['text'])



print(train_vectors[0].shape)
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import RidgeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.svm import LinearSVC

from sklearn.linear_model import SGDClassifier

from sklearn.naive_bayes import MultinomialNB
clf = RidgeClassifier()

logreg = LogisticRegression()

sgd = SGDClassifier()

svc = LinearSVC()

mnb = MultinomialNB()
clf_scores = cross_val_score(clf, train_vectors, train['target'], cv=10, scoring='f1')

logreg_scores = cross_val_score(logreg, train_vectors, train['target'], cv=10, scoring='f1')

sgd_scores = cross_val_score(sgd, train_vectors, train['target'], cv=10, scoring='f1')

svc_scores = cross_val_score(svc, train_vectors, train['target'], cv=10, scoring='f1')

mnb_scores = cross_val_score(mnb, train_vectors, train['target'], cv=10, scoring='f1')
print("Ridge Classifier: ", np.mean(clf_scores))

print("Logistic Regression: ", np.mean(logreg_scores))

print("Stochastic Gradient Descent Classifier: ", np.mean(sgd_scores))

print("Support Vector Classifier: ", np.mean(svc_scores))

print("Multinomial Naive Bayes: ", np.mean(mnb_scores))
mnb.fit(train_vectors, train['target'])
preds = mnb.predict(test_vectors)

result = pd.DataFrame({'id':test['id'], 'target':preds})
result.to_csv("submission.csv", index=False)