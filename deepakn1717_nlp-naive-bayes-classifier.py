
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfTransformer


import os
print(os.listdir("../input"))

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
subm = pd.read_csv('../input/sample_submission.csv')
train.head()
test.head()
print(train.shape)
print(test.shape)
train.columns
train.info()
test.info()
features = ['toxic', 'severe_toxic', 'obscene', 'threat',
       'insult', 'identity_hate']
lenght = len(features)

for i in range(lenght):
    print(features[i])
for i in range(lenght):
# Create training and test sets
    X_train, X_test, y_train, y_test = train_test_split(train['comment_text'], train[features[i]], test_size=0.33, random_state=53)
# Initialize a CountVectorizer object: count_vectorizer
    count_vectorizer = CountVectorizer(stop_words='english')
# Transform the training data using only the 'text' column values: count_train 
    count_train = count_vectorizer.fit_transform(X_train)
# Transform the test data using only the 'text' column values: count_test 
    count_test = count_vectorizer.transform(X_test)

    count_main_test = count_vectorizer.transform(test.comment_text)

# Prints the first 10 features of the count_vectorizer
    print(count_vectorizer.get_feature_names()[:10])
# Import TfidfVectorizer
    from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize a TfidfVectorizer object: tfidf_vectorizer
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
# Transform the training data: tfidf_train 
    tfidf_train = tfidf_vectorizer.fit_transform(X_train)
# Transform the test data: tfidf_test 
    tfidf_test = tfidf_vectorizer.transform(X_test)

# Prints the first 10 features
    print(tfidf_vectorizer.get_feature_names()[:10])

# Import the necessary modules
    from sklearn.naive_bayes import MultinomialNB
    from sklearn import metrics

# Instantiate a Multinomial Naive Bayes classifier: nb_classifier
    nb_classifier = MultinomialNB()

# Fit the classifier to the training data
    nb_classifier.fit(count_train, y_train)

# Create the predicted tags: pred
    pred = nb_classifier.predict(count_test)

# Calculate the accuracy score: score
    score = metrics.accuracy_score(y_test, pred)
    print(score)

# Calculate the confusion matrix: cm
    cm = metrics.confusion_matrix(y_test, pred)
    print(cm)
pred_main_test = nb_classifier.predict(count_main_test)

pred_df = pd.DataFrame(pred_main_test,columns=['toxic'])
id = test.id