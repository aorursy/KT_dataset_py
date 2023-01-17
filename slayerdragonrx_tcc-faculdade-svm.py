from sklearn.feature_extraction.text import CountVectorizer

import re

from bs4 import BeautifulSoup

from nltk.corpus import stopwords

from sklearn.neural_network import MLPClassifier

import numpy as np

import pandas as pd

import os
main_file_path = '../input/projeto-tcc/train.tsv'

test_file_path = '../input/projeto-tcc/test.tsv'



train_data_features = ""

test_data_features = ""



clean_train_reviews = []

clean_test_reviews = []



data = pd.read_csv(main_file_path, header=0, delimiter="\t", quoting=3)

num_reviews = data["Phrase"].size
vectorizer = CountVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words=None, max_features=5000)
def review_to_words(raw_review):

    try:

        review_text = BeautifulSoup(raw_review, features="html.parser").get_text()

        letters_only = re.sub("[^a-zA-Z]", " ", review_text)

        words = letters_only.lower().split()

        stops = set(stopwords.words("english"))

        meaningful_words = [w for w in words if not w in stops]

        return " ".join(meaningful_words)

    except Exception as err:

        print(err)
for i in range(0, num_reviews):

    if (i + 1) % 100 == 0:

        print("Review %d of %d\n" % (i + 1, num_reviews))

    clean_train_reviews.append(review_to_words(data["Phrase"][i]))

train_data_features = vectorizer.fit_transform(clean_train_reviews)

train_data_features = train_data_features.toarray() 
clf = svm.SVC(gamma='auto')

svmResult = clf.fit(train_data_features, data["Sentiment"])
test_data = pd.read_csv(test_file_path, header=0, delimiter="\t", quoting=3)

num_reviews = test_data["Phrase"].size



for i in range(0, num_reviews):

    clean_test_reviews.append(review_to_words(test_data["Phrase"][i]))



test_data_features = vectorizer.transform(clean_test_reviews)

test_data_features = test_data_features.toarray()
result_svm = svmResult.predict(test_data_features)

output_svm = pd.DataFrame(data={"PhraseId": test_data["PhraseId"], "Sentiment": result_svm})

output_svm.to_csv("SVM.csv", index=False, quoting=3)