import pandas as pd

import numpy as np

from sklearn.linear_model import SGDClassifier

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer

from sklearn.pipeline import Pipeline

from sklearn.cross_validation import train_test_split

from sklearn import metrics
train = pd.read_json('../input/train.json')
train.interest_level.value_counts(normalize=True)
pipe = Pipeline([('tfidf', TfidfVectorizer()), ('clf', SGDClassifier())])
pipe.fit(train.description, train.interest_level)
pipe.score(train.description, train.interest_level)
train_numer_df = train.select_dtypes(include=['float64', 'int64'])

train_target = train.interest_level
model = SGDClassifier()

model.fit(train_numer_df, train_target)
model.score(train_numer_df, train_target)
features = []

for i in train.features:

    for j in i:

        if j not in features:

            features.append(j)
feat_array = np.ndarray((len(train),len(features)))
for i in range(len(train)):

    for word in train.features.iloc[i]:

        if word in features:

            feat_array[i,features.index(word)] = 1

            #print features.index(word)

    
target_array = np.array(train_target)
svm_model = SGDClassifier()

svm_model.fit(feat_array, train_target)
svm_model.score(feat_array, train_target)
predictions = svm_model.predict(feat_array)

predictions = pd.Series(predictions)

predictions.value_counts()
train_target.value_counts()