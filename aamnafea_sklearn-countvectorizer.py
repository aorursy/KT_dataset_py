import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
train = pd.read_csv("../input/labeledTrainData.tsv", delimiter="\t", header=0)
train.head()
print(train['review'][0])
def review_to_words( raw_review ):
    review_text = BeautifulSoup(raw_review).get_text() 
    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 
    words = letters_only.lower()
    words= words.split()                             
    stops = set(stopwords.words("english"))                  
    meaningful_words = [w for w in words if not w in stops]
    return( " ".join( meaningful_words )) 
clean_review = []
for i in train.index:
    clean_review.append(review_to_words(train['review'][i]))
len(clean_review[0])
print((clean_review[0]))
vectorizer = CountVectorizer(analyzer='word', tokenizer=None, preprocessor=None, stop_words=None, max_features = 5000)
train_data_features = vectorizer.fit_transform(clean_review)
vocab = vectorizer.get_feature_names()
vocab
train_data_features.shape
train_data_features = train_data_features.toarray()
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators=100)
forest.fit(train_data_features, train['sentiment'])
test = pd.read_csv('../input/testData.tsv', delimiter='\t')
test.head()
clean_test_review = []
for i in train.index:
    clean_test_review.append(review_to_words(test['review'][i]))
test_data_features = vectorizer.transform(clean_test_review)
predictions = forest.predict(test_data_features)
submission = pd.DataFrame(data={'id':test['id'], 'sentiment':predictions})
submission.to_csv('Submission.csv', index=False)