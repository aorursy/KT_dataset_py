from collections import Counter
import zipfile

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csr_matrix

from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, TransformerMixin

import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import InputLayer, Dense, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train_archive = zipfile.ZipFile('/kaggle/input/whats-cooking/train.json.zip', 'r')
train_data = pd.read_json(train_archive.open('train.json'))
print('train shape:', train_data.shape)

test_archive = zipfile.ZipFile('/kaggle/input/whats-cooking/test.json.zip', 'r')
test_data = pd.read_json(test_archive.open('test.json'))
print('test shape:', test_data.shape)

sample_submission_archive = zipfile.ZipFile('/kaggle/input/whats-cooking/sample_submission.csv.zip', 'r')
sample_submission_data = pd.read_csv(sample_submission_archive.open('sample_submission.csv'))
train_data
train_data.info()  # clean
train_data['size'] = train_data['ingredients'].apply(len)
with sns.axes_style('darkgrid'), sns.plotting_context('talk'):
    pd.value_counts(train_data['cuisine']).plot.bar(figsize=(12, 5))
    plt.xticks(rotation=80)
    plt.ylabel('number of recipes')
with sns.axes_style('darkgrid'), sns.plotting_context('talk'):
    train_data.groupby('cuisine')['size'].mean().plot.bar(figsize=(12, 5))
    plt.xticks(rotation=80)
    plt.ylabel('average ingredients count')
    plt.xlabel('')
def to_counters(recipes):
    counters = []
    for recipe in recipes:
        counters.append(Counter(recipe))
    return counters


class WordCounterToVectorTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, vocabulary_size=None, accumulate_outliers=False):
        self.vocabulary_size = vocabulary_size 
        self.bias = int(accumulate_outliers)        
        
        
    def fit(self, X, y=None):
        total_count = Counter()
        for word_counts in X:
            for word, count in word_counts.items():
                total_count[word] += 1
        if self.vocabulary_size is None:
            self.vocabulary_size = len(total_count)
        most_common = total_count.most_common()[:self.vocabulary_size]
        self.most_common_ = most_common
        self.vocabulary_ = {word: index + self.bias for index, (word, count) in enumerate(most_common)}
        return self
    
    
    def transform(self, X, y=None):
        rows = []
        cols = []
        data = []
        for row, word_counts in enumerate(X):
            for word, count in word_counts.items():
                if self.bias or word in self.vocabulary_:
                    rows.append(row)
                    cols.append(self.vocabulary_.get(word, 0))
                    data.append(count)
        return csr_matrix((data, (rows, cols)), shape=(len(X), (self.vocabulary_size + self.bias)))
X = train_data['ingredients'].values
y = train_data['cuisine'].values
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
train_index, valid_index = next(sss.split(X, y))
X_train, X_valid = X[train_index], X[valid_index]
y_train, y_valid = y[train_index], y[valid_index]
X_train.shape, X_valid.shape, y_train.shape, y_valid.shape
vectorizer = WordCounterToVectorTransformer()
X_train_sparce = vectorizer.fit_transform(to_counters(X_train))
X_valid_sparce = vectorizer.transform(to_counters(X_valid))
X_train_sparce[X_train_sparce > 1] = 1
X_valid_sparce[X_valid_sparce > 1] = 1
X_train_vec = X_train_sparce.toarray()
X_valid_vec = X_valid_sparce.toarray()

onehot = OneHotEncoder()
y_train_onehot = onehot.fit_transform(y_train.reshape(-1, 1)).toarray()
y_valid_onehot = onehot.transform(y_valid.reshape(-1, 1)).toarray()
def fit_classifier(classifier):
    classifier.fit(X_train_sparce, y_train)
    y_train_pred = classifier.predict(X_train_sparce)
    y_valid_pred = classifier.predict(X_valid_sparce)
    train_acc = accuracy_score(y_train, y_train_pred)
    valid_acc = accuracy_score(y_valid, y_valid_pred)
    print(f'train accuracy: {train_acc:.5f}\nvalidation accuracy: {valid_acc:.5f}')
mnb = MultinomialNB()
fit_classifier(mnb)
knn = KNeighborsClassifier(n_neighbors=10, n_jobs=-1)
fit_classifier(knn)
rfc = RandomForestClassifier(n_estimators=100, 
                             max_depth=60,
                             min_samples_leaf=5, 
                             random_state=0, 
                             class_weight='balanced_subsample',                            
                             max_features=0.2, 
                             n_jobs=-1)
fit_classifier(rfc)
gbc = GradientBoostingClassifier(n_estimators=30, max_features=0.2, random_state=0)
fit_classifier(gbc)
abc = AdaBoostClassifier(n_estimators=100, random_state=0)
fit_classifier(abc)
vc_all_h = VotingClassifier(estimators=[('mnb', mnb), ('knn', knn), ('rfc', rfc), ('gbc', gbc), ('abc', abc)], n_jobs=-1)
fit_classifier(vc_all_h)
vc_best_h = VotingClassifier(estimators=[('mnb', mnb), ('rfc', rfc), ('gbc', gbc)], n_jobs=-1)
fit_classifier(vc_best_h)
vc_all_s = VotingClassifier(estimators=[('mnb', mnb), ('knn', knn), ('rfc', rfc), ('gbc', gbc), ('abc', abc)], voting='soft', n_jobs=-1)
fit_classifier(vc_all_s)
vc_best_s = VotingClassifier(estimators=[('mnb', mnb), ('rfc', rfc), ('gbc', gbc)], voting='soft', n_jobs=-1)
fit_classifier(vc_best_s)
K.clear_session()

def create_model():
    dnn = Sequential()
    dnn.add(InputLayer(input_shape=[X_train_vec.shape[1]]))
    dnn.add(Dense(4000, activation='elu', kernel_initializer='he_normal'))
    dnn.add(Dense(20, activation='softmax'))
    return dnn

dnn = create_model()
dnn.compile(loss='categorical_crossentropy',
            optimizer=Adam(lr=0.0001),
            metrics=["accuracy"])
early_stopping = EarlyStopping(
    monitor='val_loss',
    min_delta=0,
    patience=5,
    verbose=0,
    mode='min',
    restore_best_weights=True)
callbacks = [early_stopping]

dnn.summary()
%%time
history = dnn.fit(X_train_vec, y_train_onehot, 
                  epochs=50, 
                  batch_size=128,
                  validation_data=(X_valid_vec, y_valid_onehot),
                  callbacks=callbacks)
dnn.evaluate(X_valid_vec, y_valid_onehot)
vectorizer_final = WordCounterToVectorTransformer()
X_vec = vectorizer_final.fit_transform(to_counters(X)).toarray()
X_vec[X_vec > 1] = 1

onehot_final = OneHotEncoder()
y_onehot = onehot_final.fit_transform(y.reshape(-1, 1)).toarray()

K.clear_session()

dnn_final = Sequential()
dnn_final.add(InputLayer(input_shape=[X_vec.shape[1]]))
dnn_final.add(Dense(4000, activation='elu', kernel_initializer='he_normal'))
dnn_final.add(Dense(20, activation='softmax'))
dnn_final.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.0001),
                  metrics=["accuracy"])

dnn_final.summary()
%%time
history_final = dnn_final.fit(X_vec, y_onehot, 
                              epochs=6, 
                              batch_size=128)
dnn_final.save('final.h5')
# dnn_final = load_model('final.h5')
X_test = test_data['ingredients'].values
X_test = vectorizer_final.transform(to_counters(X_test)).toarray()

y_test_pred = dnn_final.predict_classes(X_test)
y_test_pred = onehot_final.categories_[0][y_test_pred]

answers = test_data.copy()
answers = answers.drop('ingredients', axis=1)
answers['cuisine'] = y_test_pred
answers.to_csv('answers.csv', index=False)
answers