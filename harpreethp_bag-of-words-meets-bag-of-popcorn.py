import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
train = pd.read_csv('../input/labeledTrainData.tsv', sep='\t', quoting=3)
test = pd.read_csv('../input/testData.tsv', sep='\t', quoting=3)
def cleaning(raw_review):
    remove_tags = BeautifulSoup(raw_review).get_text()
    letters = re.sub("[^a-zA-Z]"," ", remove_tags)
    lower_case = letters.lower()
    words = lower_case.split()
    stopword = stopwords.words("english")
    meaningful_words = [w for w in words if not w in stopword]
    return(" ".join(meaningful_words))
total_review = len(train["review"])
clean_train_reviews = []
for i in range(0 , total_review):
    clean_train_reviews.append(cleaning(train["review"][i]))
vectorizer = TfidfVectorizer(max_features = 10000)
train_data_feature = vectorizer.fit_transform(clean_train_reviews)
total_test_review = len(test["review"])
clean_test_reviews = []
for i in range(0 , total_test_review):
    clean_test_reviews.append(cleaning(test["review"][i]))
test_data_feature = vectorizer.fit_transform(clean_test_reviews)
logreg = LogisticRegression()
logreg = logreg.fit(train_data_feature, train["sentiment"])
result = logreg.predict(test_data_feature)
output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )
output.to_csv( "Bag_of_Words_output.csv", index=False, quoting=3 )