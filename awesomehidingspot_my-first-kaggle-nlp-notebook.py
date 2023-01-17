# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
def warn(*args, **kwargs):

    pass

import warnings

warnings.warn = warn #remove spooky warning when working on this spooky notebook
import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

np.set_printoptions(suppress=True)



author_data = pd.read_csv('../input/spooky-author-identification/train.zip')

author_data.head()
print(author_data['author'].value_counts())

author_data.info()
author = author_data.copy()

author['text_length'] = author['text'].apply(lambda text: len(text))

author.head()
print(author['text_length'].describe())

ax = author['text_length'].hist(bins=100);

plt.axis([0, 1000, 0, 5000])

plt.xlabel('text length', fontsize=14)

plt.title('Text length distribution', fontsize=18)
author.groupby('author').mean()
X = author_data.drop(['author'], axis=1)

y = author_data.copy()['author']
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y ,test_size=0.2, random_state=42)
len(X_train), len(X_test)
import nltk

nltk.word_tokenize('We don\'t do that here')
stemmer = nltk.PorterStemmer()



for word in ('computing','computed','compulsive'):

    print(word,'=>',stemmer.stem(word))
from sklearn.feature_extraction import stop_words

stop_words.ENGLISH_STOP_WORDS
text_list = 'i myself believe it was beautiful'.split()

less_than_zero = list(filter(lambda x: not x in stop_words.ENGLISH_STOP_WORDS, text_list))

print(less_than_zero)
import re

from collections import Counter



sentence = 'I love real madrid for real but I dont like real betis'

sentence = re.sub(r'\W+', ' ', sentence, flags=re.M) #remove punctuation from string



c = Counter(sentence.split())

print(c.most_common())
c['love'] += 3  

print(c.most_common())
import string



def derive_new_features(data):

    data = data[['text']].copy()



    #Unique words Count

    data['unique_words_count'] = data.text.apply(lambda x: len(set(str(x).split())))     



    #Punctuation count

    data['punctuation_count'] = data.text.apply(lambda x: len([x for x in x.lower().split() if x in string.punctuation]))



    #Upper case words count

    data['uppercase_words_count'] = data.text.apply(lambda x: sum([x.isupper() for x in x.split()]))



    #Title words count

    data['title_words_count'] = data.text.apply(lambda x: sum([x.istitle() for x in x.split()]))



    return data.drop(['text'], axis=1)
derive_new_features(X_train).head()
from sklearn.base import BaseEstimator, TransformerMixin



class TextToWordCounterTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, remove_punctuation=True, lower_case=True, stemming=True, replace_numbers=True, remove_stopwords=True):

        self.remove_punctuation = remove_punctuation

        self.lower_case = lower_case

        self.stemming = stemming

        self.replace_numbers = replace_numbers

        self.remove_stopwords = remove_stopwords

    def fit(self, X, y=None):

        return self

    def transform(self, X, y=None):

        word_counters = []

        for text in X['text']:

            if self.lower_case:

                text = text.lower()

            if self.replace_numbers:

                text = re.sub(r'\d+(?:\.\d*(?:[eE]\d+))?', 'NUMBER', text) #replace any numerical character with 'NUMBER'

            if self.remove_punctuation:

                text = re.sub(r'\W+', ' ', text, flags=re.M)

            if self.remove_stopwords:

                words = [word for word in text.split() if not word in stop_words.ENGLISH_STOP_WORDS]

                text = ' '.join(words)

            word_list = nltk.word_tokenize(text)

            word_count = Counter(word_list)

            if self.stemming:

                stemmed_word_count = Counter()

                for word in word_list:

                    stemmed_word = stemmer.stem(word)

                    stemmed_word_count[stemmed_word] += 1

                word_count = stemmed_word_count

            word_counters.append(word_count)

        return np.array(word_counters)
X_word_count = TextToWordCounterTransformer().fit_transform(X_train)

X_word_count
from scipy.sparse import csr_matrix



class WordCounterToVectorTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, vocabulary_size=1000):

        self.vocabulary_size = vocabulary_size

    def fit(self, X, y=None):

        total_count = Counter()

        for word_count in X:

            for word, count in word_count.items():

                total_count[word] += min(count, 10) 

        most_common = total_count.most_common()[:self.vocabulary_size]

        self.most_common_ = most_common

        self.vocabulary_ = {word: index + 1 for index, (word, count) in enumerate(most_common)}  #spare an index for excluded words

        return self

    def transform(self, X, y=None):

        data = []

        rows = []

        cols = []

        for row, word_count in enumerate(X):

            for word, count in word_count.items():

                data.append(count)

                rows.append(row)

                cols.append(self.vocabulary_.get(word, 0))

        return csr_matrix((data, (rows, cols)), shape=(len(X), self.vocabulary_size + 1))
vectorizer = WordCounterToVectorTransformer(vocabulary_size=1000)

vectorizer.fit_transform(X_word_count).toarray()
vectorizer.most_common_[:10]
from sklearn.pipeline import Pipeline



preprocess_pipeline = Pipeline([

    ('text_to_word_count', TextToWordCounterTransformer()),

    ('word_count_to_vector', WordCounterToVectorTransformer(vocabulary_size=14000)), 

])
class NewFeaturesAdderTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):

        pass

    def fit(self, X, y=None):

        return self

    def transform(self, X, y=None):

        return np.array(derive_new_features(X))
NewFeaturesAdderTransformer().fit_transform(X_train)
from sklearn.compose import ColumnTransformer



full_pipeline = ColumnTransformer([

    ('feature_adder', NewFeaturesAdderTransformer(), ['text']),

    ('text_pipeline', preprocess_pipeline, ['text']),

])
y_train.head()
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
X_train_transformed = full_pipeline.fit_transform(X_train)

y_train_transformed = label_encoder.fit_transform(y_train.values)
label_encoder.classes_    #submission format: id, EAP,HPL,MWS
from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=42, loss='hinge')
from sklearn.model_selection import cross_val_score



cv_score = cross_val_score(

        sgd_clf, 

        X_train_transformed, y_train_transformed, 

        cv=5,

        verbose=3,

    )

print('mean score :', cv_score.mean())
from sklearn.linear_model import LogisticRegression



log_clf = LogisticRegression(random_state=42)

cv_score = cross_val_score(

        log_clf, 

        X_train_transformed, y_train_transformed, 

        cv=5,

        verbose=3,

    )

print('mean score :', cv_score.mean())
from sklearn.naive_bayes import MultinomialNB



mnb_clf = MultinomialNB()

cv_score = cross_val_score(

        mnb_clf, 

        X_train_transformed, y_train_transformed, 

        cv=5,

        verbose=3,

    )

print('mean score :', cv_score.mean())
def multiclass_logloss(actual, predicted, eps=1e-15):

    """Multi class version of Logarithmic Loss metric.

    :param actual: Array containing the actual target classes

    :param predicted: Matrix with class predictions, one probability per class

    """

    # Convert 'actual' to a binary array if it's not already:

    if len(actual.shape) == 1:

        actual2 = np.zeros((actual.shape[0], predicted.shape[1]))

        for i, val in enumerate(actual):

            actual2[i, val] = 1

        actual = actual2



    clip = np.clip(predicted, eps, 1 - eps)

    rows = actual.shape[0]

    vsota = np.sum(actual * np.log(clip))

    return -1.0 / rows * vsota
from sklearn.model_selection import cross_val_predict



y_scores = cross_val_predict(

        log_clf, 

        X_train_transformed, y_train_transformed, 

        cv=5,

        verbose=3,

        method='predict_proba',

    )
pd.DataFrame(columns=label_encoder.classes_, data=y_scores).head()
multiclass_logloss(y_train_transformed, y_scores)
y_scores = cross_val_predict(

        mnb_clf, 

        X_train_transformed, y_train_transformed, 

        cv=5,

        verbose=3,

        method='predict_proba',

    )



multiclass_logloss(y_train_transformed, y_scores)
import itertools



stopword_params = [True, False]

vocabulary_params = [7000, 8000, 9000, 10000]

data_scores = []



for stopword, vocabulary in list(itertools.product(stopword_params, vocabulary_params)):

   

    full_pipeline.set_params(

            text_pipeline__text_to_word_count__remove_stopwords=stopword,

            text_pipeline__word_count_to_vector__vocabulary_size=vocabulary,

        )

    X_train_processed = full_pipeline.fit_transform(X_train)

    y_scores = cross_val_predict(

            log_clf, 

            X_train_processed, y_train_transformed, 

            cv=3,

            method='predict_proba',

        )

    data_scores += [(stopword, vocabulary, multiclass_logloss(y_train_transformed, y_scores))]
for stopword, vocabulary, logloss in data_scores:

    print('remove_stopwords:', stopword, ',', 'vocabulary_size:',vocabulary)

    print('logloss: ', logloss)
full_pipeline.set_params(

            text_pipeline__text_to_word_count__remove_stopwords=False,

            text_pipeline__word_count_to_vector__vocabulary_size=8_000,

        )

X_train_transformed = full_pipeline.fit_transform(X_train)
%%time

from sklearn.model_selection import GridSearchCV



log_grid_params = {

    'penalty': ['L1', 'l2'],

    'dual': [False],

    'tol':[1e-4, 1e-5],

    'class_weight': ['balanced', None],

    'solver': ['newton-cg', 'lbfgs', 'sag', 'saga'],

    'multi_class': ['ovr', 'auto', 'multinomial'],

}

LogisticRegression #



log_grid_search = GridSearchCV(

    estimator=LogisticRegression(random_state=42),

    param_grid=log_grid_params,

    scoring='neg_log_loss',

    cv=3,

    verbose=2,

)



log_grid_search.fit(X_train_transformed, y_train_transformed)
print(log_grid_search.best_params_)
import joblib



joblib.dump(log_grid_search, 'log_grid_best.pkl')

log_clf_best = joblib.load('log_grid_best.pkl').best_estimator_

print(log_clf_best.get_params())
%%time

from sklearn.model_selection import GridSearchCV



sgd_grid_params = {

    'loss': ['log'],

    'penalty' : ['l2'],

    'eta0':[0.1],

    'alpha': [1e-4, 1e-5],

    'tol': [1e-3, 1e-4],

    'epsilon': [0.3, 0.5, 1],

    'learning_rate': ['adaptive'],

    'class_weight': ['balanced', None],

    'average':[True, False],

}



sgd_grid_search = GridSearchCV(

    estimator=SGDClassifier(random_state=42),

    param_grid=sgd_grid_params,

    scoring='neg_log_loss',

    cv=3,

    verbose=2,

)



sgd_grid_search.fit(X_train_transformed, y_train_transformed)
print(sgd_grid_search.best_params_)
joblib.dump(sgd_grid_search, 'sgd_grid_best.pkl')

sgd_clf_best = joblib.load('sgd_grid_best.pkl').best_estimator_

print(sgd_clf_best.get_params())
%%time

from sklearn.model_selection import GridSearchCV



mnb_grid_params = {

    'alpha': [0, 0.25, 0.5, 0.75, 1],

    'fit_prior': [False, True],

}



GridSearchCV



mnb_grid_search = GridSearchCV(

    estimator=MultinomialNB(),

    param_grid=mnb_grid_params,

    scoring='neg_log_loss',

    cv=3,

    verbose=2,

)



mnb_grid_search.fit(X_train_transformed, y_train_transformed)
print(mnb_grid_search.best_params_)
joblib.dump(mnb_grid_search, 'mnb_grid_best.pkl')

mnb_clf_best = joblib.load('mnb_grid_best.pkl').best_estimator_

print(mnb_clf_best.get_params())
from sklearn.ensemble import VotingClassifier



sgd_clf_best.set_params(loss='log') #need log soft voting classifier

estimators=[('log_clf', log_clf_best), ('sgd_clf',sgd_clf_best), ('mnb_clf',mnb_clf_best)]

vot_clf = VotingClassifier(

    estimators=estimators,

    voting='soft',

)
y_scores = cross_val_predict(

        vot_clf, 

        X_train_transformed, y_train_transformed, 

        cv=5,

        verbose=3,

        method='predict_proba',

    )



multiclass_logloss(y_train_transformed, y_scores)
sgd_clf_best.set_params(loss='hinge')


from sklearn.ensemble import StackingClassifier



estimators=[('sgd_clf',sgd_clf_best), ('log_clf', log_clf_best),  ('mnb_clf',mnb_clf_best)]

stk_clf = StackingClassifier(

    estimators=estimators,

)
y_scores = cross_val_predict(

        stk_clf, 

        X_train_transformed, y_train_transformed, 

        cv=5,

        verbose=3,

        method='predict_proba',

    )



multiclass_logloss(y_train_transformed, y_scores)
final_model = stk_clf

final_model.fit(X_train_transformed, y_train_transformed)
X_test_transformed = full_pipeline.transform(X_test)

y_test_transformed = label_encoder.transform(y_test)



y_scores = final_model.predict_proba(X_test_transformed)

y_scores
multiclass_logloss(y_test_transformed, y_scores)
X_transformed = full_pipeline.transform(X)

y_transformed = label_encoder.transform(y)

final_model.fit(X_transformed, y_transformed)
test_data = pd.read_csv('../input/spooky-author-identification/test.zip')

test_data.head()
test_data_prepared = full_pipeline.transform(test_data)

test_scores = final_model.predict_proba(test_data_prepared)

test_scores
submission_file = pd.DataFrame({

    'id': test_data['id'].values,

    'EAP': test_scores[:,0],

    'HPL': test_scores[:,1],

    'MWS': test_scores[:,2],

}) 

submission_file.head()
submission_file.to_csv('submission.csv', index=False)