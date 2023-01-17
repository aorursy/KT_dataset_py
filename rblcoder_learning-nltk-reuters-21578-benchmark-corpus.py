import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))

from nltk.corpus import LazyCorpusLoader, CategorizedPlaintextCorpusReader
!head -n 5 '/kaggle/input/reuters/reuters/reuters/cats.txt'
# https://www.kaggle.com/alvations/testing-1000-files-datasets-from-nltk

reuters = LazyCorpusLoader('reuters', CategorizedPlaintextCorpusReader, 

                           '(training|test).*', cat_file='cats.txt', encoding='ISO-8859-2',

                          nltk_data_subdir='/kaggle/input/reuters/reuters/reuters/')

# https://miguelmalvarez.com/2015/03/20/classifying-reuters-21578-collection-with-python-representing-the-data/

reuters.words()
reuters.categories()
len(reuters.categories())
reuters.fileids("jobs")[0]
reuters.words(reuters.fileids("jobs")[0])
reuters.raw(reuters.fileids("jobs")[0])
reuters.categories(reuters.fileids("jobs")[0])
reuters.fileids()[:5]
train_docs = list(filter(lambda doc: doc.startswith("train"),

                        reuters.fileids()));
train_docs[:5]
len(train_docs)
test_docs = list(filter(lambda doc: doc.startswith("test"),

                        reuters.fileids()));
test_docs[:5]
len(test_docs)
train_documents, train_categories = zip(*[(reuters.raw(i), reuters.categories(i)) for i in reuters.fileids() if i.startswith('training/')])

test_documents, test_categories = zip(*[(reuters.raw(i), reuters.categories(i)) for i in reuters.fileids() if i.startswith('test/')])
from sklearn.feature_extraction.text import TfidfVectorizer



vectorizer = TfidfVectorizer(stop_words = 'english')



vectorised_train_documents = vectorizer.fit_transform(train_documents)

vectorised_test_documents = vectorizer.transform(test_documents)
from sklearn.preprocessing import MultiLabelBinarizer



mlb = MultiLabelBinarizer()

train_labels = mlb.fit_transform(train_categories)

test_labels = mlb.transform(test_categories)

# https://towardsdatascience.com/multi-class-text-classification-with-sklearn-and-nltk-in-python-a-software-engineering-use-case-779d4a28ba5

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier()

clf.fit(vectorised_train_documents, train_labels)
from sklearn.metrics import jaccard_similarity_score

print(jaccard_similarity_score(test_labels, clf.predict(vectorised_test_documents)))