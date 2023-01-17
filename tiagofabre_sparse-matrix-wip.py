import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy.sparse import csr_matrix, coo_matrix
import os
print(os.listdir("../input"))
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
tfidf_transformer = TfidfTransformer()
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.describe()
train.shape
train.info()
coo_train = csr_matrix(train.values)

clf = Pipeline([('tfidf', TfidfTransformer()),
                ('clf', MultinomialNB()),])

target = train.pop('target')
clf.fit(coo_train, target.values)