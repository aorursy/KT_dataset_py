# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# imports

import os
import re
import json
import warnings

import pandas as pd
import numpy as np
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from matplotlib import pyplot as plt
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import LinearSVC
from IPython.display import FileLink

# config

warnings.filterwarnings('ignore')
data_path = '../input/'
%matplotlib inline
# os.chdir(data_path)
lbl_train = pd.read_csv(data_path+'labeledTrainData.tsv', sep='\t')
print("Shape: {}".format(lbl_train.shape))
print(lbl_train.columns)
unlbl_train = pd.read_csv(data_path+'unlabeledTrainData.tsv', sep='\t', error_bad_lines=False)
print("Shape: {}".format(unlbl_train.shape))
print(unlbl_train.columns)
test = pd.read_csv(data_path+'testData.tsv', sep='\t')
print(f'Shape: {test.shape}')
print(test.columns)
samplesub = pd.read_csv(data_path+'./sampleSubmission.csv')
samplesub.head(3)
# random positive/negative review

print(f'pos:\n{lbl_train.review[np.random.randint(0, 25000)]}')
print(f'neg:\n{test.review[np.random.randint(0, 25000)]}')
# Average number of words in pos & neg reviews

avg_pos_words = lbl_train[lbl_train.sentiment==1].review.apply(lambda x: len(x.split())).mean()
avg_neg_words = lbl_train[lbl_train.sentiment==0].review.apply(lambda x: len(x.split())).mean()

plt.figure(figsize=(10, 3))
plt.barh(['Positive', 'Negative'], [avg_pos_words, avg_neg_words], height=0.5)
plt.xticks(np.arange(0, 300, 25))
plt.xlabel('Average Number of words')
plt.ylabel('Sentiment')
plt.show()
def clean_review(review):
    # remove line breaks
    review = re.sub(r'<br />', '', review)
    # remove punctuations/tokenize
    tokenizer = RegexpTokenizer(r'\w+')
    review = tokenizer.tokenize(review)
    # apply stemming
    stemmer = PorterStemmer()
    review = ' '.join([stemmer.stem(y) for y in review])
    return review

# clean train, test and unlabeled train
lbl_train.review = lbl_train.review.apply(lambda x: clean_review(x))
test.review = test.review.apply(lambda x: clean_review(x))
unlbl_train = unlbl_train.review.apply(lambda x: clean_review(x))
X_train, X_test, y_train, y_test = train_test_split(lbl_train.review, lbl_train.sentiment,
                                                    test_size=0.2, random_state=13)
X_train.shape, X_test.shape
%%time

lg_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_df=0.9)),
    ('lg', LogisticRegression(n_jobs=-1))
])

lg_pipeline.fit(X_train, y_train)
print(f'Accuracy: {np.mean(lg_pipeline.predict(X_test)==y_test)}')
%%time

rfc_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_df=0.9)),
    ('rfc', RandomForestClassifier(n_estimators=100, n_jobs=-1))
])

rfc_pipeline.fit(X_train, y_train)
print(f"Accuracy: {np.mean(rfc_pipeline.predict(X_test)==y_test)}")
%%time

nb_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_df=0.9)),
    ('nb', BernoulliNB())
])

nb_pipeline.fit(X_train, y_train)
print(f'Accuracy: {np.mean(nb_pipeline.predict(X_test)==y_test)}')
# parameter tuning

lg_params = {
    'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
    'lg__C': [0.1, 1, 10],
}

rfc_params = {
    'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
    'rfc__max_features': ['auto', 'sqrt'],
    'rfc__bootstrap': [True, False],
    'rfc__n_estimators': [200, 400, 600],
    'rfc__min_samples_split': [2, 5, 10],
    'rfc__min_samples_leaf': [1, 2, 4]
}
%%time

lg_grid = GridSearchCV(lg_pipeline, param_grid=lg_params, cv=3, verbose=True, n_jobs=-1)
# lg_grid.fit(X_train, y_train)
# lg_grid.best_params_
# print(f"Logistic gridsearch accuracy: {np.mean(lg_grid.predict(X_test)==y_test)}")
%%time

rfc_grid = RandomizedSearchCV(rfc_pipeline, param_distributions=rfc_params, cv=3, verbose=True, n_jobs=-1)
# rfc_grid.fit(X_train, y_train)
# rfc_grid.best_params_
# print(f"Accuracy: {np.mean(rfc_grid.predict(X_test)==y_test)}")
lg_best_params = {'lg__C': 10, 'tfidf__ngram_range': (1, 2)}
rfc_best_params = {'tfidf__ngram_range': (1, 3),
 'rfc__n_estimators': 400,
 'rfc__min_samples_split': 5,
 'rfc__min_samples_leaf': 4,
 'rfc__max_features': 'sqrt',
 'rfc__bootstrap': False}
lg_pipeline.set_params(**lg_best_params)
rfc_pipeline.set_params(**rfc_best_params)

lg_pipeline.fit(lbl_train.review, lbl_train.sentiment)
rfc_pipeline.fit(lbl_train.review, lbl_train.sentiment)
nb_pipeline.set_params(tfidf__ngram_range=(1,3))
nb_pipeline.fit(lbl_train.review, lbl_train.sentiment)
lg_preds = lg_pipeline.predict(test.review)
rfc_preds = rfc_pipeline.predict(test.review)
nb_preds = nb_pipeline.predict(test.review)
# Pick the test prediction by voting

predictions = pd.DataFrame({'lg': lg_preds, 'rfc': rfc_preds, 'nb': nb_preds}).mode(axis=1).rename(columns={0: 'sentiment'})
submission = pd.DataFrame({'id': test.id.values, 'sentiment': predictions.sentiment.values})
submission.to_csv('submission2.csv', index=False)
# FileLink('./submission2.csv')
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, Flatten, Dense, Dropout, BatchNormalization
from keras.layers import LSTM, Bidirectional, GlobalMaxPool1D
from keras.models import Sequential
from keras.optimizers import Adam
# Let's fix vocab size to 20000
vocab_size = 10000

t = Tokenizer(num_words=vocab_size)
t.fit_on_texts(lbl_train.review)
# create sequences to feed into Neural network model
sequences = t.texts_to_sequences(lbl_train.review)

# As the average length of all reviews is around 250, 
# lets the keep the input dim to 250 and pad the sequences if it is less that 250 words
sequences = pad_sequences(sequences, maxlen=150)
# Now the length of each review is 350, and there are 25000 items
sequences.shape
# Network architecture
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=128, input_length=150, name='embed'))
model.add(Bidirectional(LSTM(32, return_sequences=True, name='lstm')))
model.add(GlobalMaxPool1D(name='gmax1'))
model.add(Dense(20, name='dense1'))
# model.add(Flatten(name='flatten'))
model.add(Dropout(0.05, name='drop1'))
model.add(Dense(1, activation='sigmoid', name='softmax'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(sequences, lbl_train.sentiment.values, validation_split=0.2, epochs=5, batch_size=128, verbose=2)
model.summary()
test_sequences = t.texts_to_sequences(test.review)
test_sequences = pad_sequences(test_sequences, maxlen=150)
len(test_sequences)
test_sequences.shape
preds = model.predict(test_sequences)
preds = (preds>0.5)
preds = [int(p) for p in preds]
submission2 = pd.DataFrame({'id': test.id.values, 'sentiment': preds})
submission2.to_csv('submission2.csv', index=False)
FileLink('./submission2.csv')
