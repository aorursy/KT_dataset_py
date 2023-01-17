import numpy as np

import pandas as pd

import itertools



from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import PassiveAggressiveClassifier

from sklearn.metrics import accuracy_score, confusion_matrix
fake_news_data = pd.read_csv('../input/textdb3/fake_or_real_news.csv')



print(fake_news_data.shape)

fake_news_data.head()
fake_news_data['label'].value_counts()
labels = fake_news_data.label
# The column "text" is used to X and labels used to y

X_train, X_test, y_train, y_test = train_test_split(fake_news_data['text'], labels, test_size=0.2, random_state=42)
# For more information about TfidVectorizer parameters, check the "references" section

tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)



# Fit and transform train data and transform test data

tfidf_train = tfidf_vectorizer.fit_transform(X_train)

tfidf_test = tfidf_vectorizer.transform(X_test)
# Passive Agressive Classifier

pa_clf = PassiveAggressiveClassifier(max_iter=50)



# Fit the classifier

pa_clf.fit(tfidf_train, y_train)



# Make predictions with test data

preds = pa_clf.predict(tfidf_test)
# Accuracy

score = accuracy_score(y_test, preds)



# Print accuracy

print('Accuracy = {}%'.format(round(score * 100, 2)))
confusion_matrix(y_test, preds)