import numpy as np
import pandas as pd
reviews = pd.read_csv('../input/googleplaystore_user_reviews.csv')
reviews.head()
reviews.dropna(inplace=True)
reviews.head()
from sklearn.model_selection import train_test_split
reviews['Sentiment'].unique()
features = reviews.drop('Sentiment', axis=1)
labels = reviews['Sentiment']
features.head()
labels.head()
feature_train, feature_test, label_train, label_test = train_test_split(features['Translated_Review'], labels, test_size=0.3, random_state=101)
import string
from nltk.corpus import stopwords
mess = 'Sample: Hey! this is a sample message.'
def text_processor(mess):
    nopunc = [c for c in mess if c not in string.punctuation]
    nopunc = ''.join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
model = Pipeline([
    ('bow', CountVectorizer(analyzer=text_processor)),
    ('tfidf', TfidfTransformer()),
    ('classifier', MultinomialNB())
])
model.fit(feature_train, label_train)
predictions = model.predict(feature_test)
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(label_test, predictions))
print('\n')
print(classification_report(label_test, predictions))
