import numpy as np

from pandas import DataFrame, Series, read_csv
titles = read_csv("/kaggle/input/clickbait-dataset/clickbait_data.csv")

titles_len = len(titles)

clckbt_ratio = len(titles[titles["clickbait"]==0])/titles_len

print("Database lenght : {} \nClickbait ratio: {}".format(titles_len, clckbt_ratio))

titles.head()
for head in titles['headline'][titles['clickbait']==1][:10]:

    print(head)
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(titles['headline'], titles["clickbait"],

                                                    test_size=.1, random_state=42)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=.1, random_state=42)
!pip install contractions



from nltk.stem.wordnet import WordNetLemmatizer

from contractions import contractions_dict

from nltk import word_tokenize

from nltk.tag import pos_tag

import contractions

import string



_lem = WordNetLemmatizer()

contractions_set = set(contr.lower() for contr in contractions_dict)





def remove_contractions(string):

    ''' Expand and count the contractions in a given string.

        Return string and the contractions number '''

    string = string.lower()

    contr_num = sum(1 for contr in contractions_set if contr in string)

    parsed_string = ' '.join(contractions.fix(word) for word in string.split())

    return parsed_string, contr_num





def lemmatise_sentence(sentence):

    

    # remove contarctions and convert to lower case

    sentence, contr_num = remove_contractions(sentence.lower())

    

    # remove punctuation

    sentence = sentence.translate(str.maketrans('', '', string.punctuation+'’‘'))  

    

    # lemmatize words

    lemm_str = ""

    for word, tag in pos_tag(word_tokenize(sentence.lower())):

        if tag.startswith('NN'):

            word_1 = word 

            pos = 'n'

        elif tag.startswith('VB'):

            word_1 = word

            pos = 'v'

        elif tag.startswith('CD'):

            word_1 = 'NUM'

            pos = 'a'

        else:

            word_1 = word

            pos = 'a'

        lemm_str += ' '+_lem.lemmatize(word_1, pos)

    

    return lemm_str, contr_num
from sklearn.base import BaseEstimator, TransformerMixin



class ParseString(BaseEstimator, TransformerMixin):

    def __init__(self):

        self = True

    def fit(self, X, y=None):

        return self

    def transform(self, X, y=None):

        X_prep, contr_list = [], []

        for string in X:

            lemm_str, contr_num = lemmatise_sentence(string)

            X_prep.append(lemm_str)

            contr_list.append(contr_num)

        return DataFrame({"headline": X_prep, "contr num":contr_list})
X_train_prep = ParseString().fit_transform(X=X_train)
import warnings

from sklearn.feature_extraction.text import CountVectorizer

warnings.filterwarnings("ignore", category=UserWarning, module='bs4')    # shut up bs4 URL warning





def make_vocabulary(X, length, rm_words=False, to_del_words_list=None):

    vectorizer = CountVectorizer(max_features=length)

    vectorizer.fit(X)

    vocab = vectorizer.get_feature_names()

    if rm_words:

        for word in to_del_words_list:

            if word in vocab:

                vocab.remove(word)

    return vocab





def save_vocabulary(words_list, txt_file):

    file = open(txt_file, 'w+')

    for word in words_list:

            file.write(str(key)+'\n')

    file.close()

    

    print('Vocabulary stored in "{}"'.format(txt_file))

    



def load_vocabulary(length, txt_file, to_del=None):

    file = open(txt_file, 'r')

    vocab = np.array([file.readline().rstrip().lower() for line in range(length)])

    file.close()

    print('Dictionary loaded.')

    return vocab

    



to_del = ['trump','donald','christmas','obama','president','america','harry','russian','russia','china',

          'american']
X_train_clckb = X_train_prep.headline.values[y_train==1]

X_train_noclckbt = X_train_prep.headline.values[y_train==0]



full_vocab = make_vocabulary(X_train_prep.headline, length=20)

clckbt_vocab = make_vocabulary(X_train_clckb, length=21, rm_words=True, to_del_words_list=to_del)

no_clckbt_vocab = make_vocabulary(X_train_noclckbt, length=20)
common_words = DataFrame({'No Clickbait': no_clckbt_vocab[:20], 

                          'Clickbait': clckbt_vocab[:20], 

                          'Full': full_vocab[:20]})

common_words.transpose()
from scipy.sparse import coo_matrix, hstack

from sklearn.pipeline import Pipeline

from nltk.corpus import stopwords

from collections import Counter





class PreProcess(BaseEstimator, TransformerMixin):

    def __init__(self, vocabulary):

        self.vocabulary = vocabulary

        self.vectorizer = CountVectorizer(vocabulary=self.vocabulary)

        self.stopwords_set = set(stopwords.words('english'))

    def fit(self, X, y=None):

        return self

    def transform(self, X, y=None):

        # bag of words

        X_bag = self.vectorizer.transform(X.headline)

        # meta data

        meta_arr = []

        for i in range(len(X)):

            d = Counter(X.headline.iloc[i].split())

            num_flag = 1 if list(d)[0]=='NUM' else 0

            n_of_words = sum(d.values())

            contr_r = X['contr num'].iloc[i]/n_of_words

            stop_r = sum(d[key] for key in set(d.keys())&self.stopwords_set) / n_of_words

            meta_arr.append([num_flag, contr_r, stop_r, n_of_words])

        meta_arr = coo_matrix(meta_arr)

        return hstack([X_bag, meta_arr])



    

    

full_pipeline = Pipeline([

    ("parse text", ParseString()),

    ("gen features", PreProcess(vocabulary=clckbt_vocab))

])

X_train_prep = full_pipeline.fit_transform(X_train)

X_train_prep.shape
X_train_mini = X_train_prep.toarray()[:1000]

y_train_mini = y_train[:1000]
from sklearn.model_selection import cross_validate

from sklearn.metrics import accuracy_score, precision_score, recall_score, make_scorer



scorers = {

    'precision_score': make_scorer(precision_score),

    'recall_score': make_scorer(recall_score),

    'accuracy_score': make_scorer(accuracy_score)

}





def print_scores(clf_cv, acc=True, prec=True, rec=True):

    '''Print cross validation results.

       Valid only for CrossValidation.

    '''

    if acc:

        print('accuracy: %.3f' % clf_cv['test_accuracy_score'].mean())

    if prec:

        print('precision: %.3f' % clf_cv['test_precision_score'].mean())

    if rec:

        print('recall: %.3f' % clf_cv['test_recall_score'].mean())

    
from sklearn.naive_bayes import MultinomialNB



mnb_clf = MultinomialNB()



mnb_clf_cv = cross_validate(mnb_clf, X_train_mini[:,:200], y_train_mini, cv=5, scoring=scorers, n_jobs=5)

print_scores(mnb_clf_cv)
mnb_clf_cv = cross_validate(mnb_clf, X_train_prep.toarray()[:,:200], y_train, cv=5, scoring=scorers, n_jobs=5)

print_scores(mnb_clf_cv)
mnb_clf.fit(X_train_prep.toarray()[:,:200], y_train)

probabilities = mnb_clf.predict_proba(X_train_prep.toarray()[:,:200])[:, 1]

probabilities = probabilities.reshape((len(probabilities), 1))
X_train_forest = np.concatenate([probabilities, X_train_prep.toarray()[:, 200:]], axis=1)
from sklearn.ensemble import RandomForestClassifier



forest_clf = RandomForestClassifier(random_state=42)



forest_cv = cross_validate(forest_clf, X_train_forest[:1000], y_train_mini, cv=5, scoring=scorers, n_jobs=-1)

print_scores(forest_cv)
from sklearn.model_selection import RandomizedSearchCV

from scipy.stats import reciprocal, uniform



param_distrib = {"n_estimators": list(range(1,500)), "max_depth": reciprocal(2,100)}



rnd_srch_forest = RandomizedSearchCV(forest_clf, param_distributions=param_distrib,

                                     cv=5, scoring='accuracy', random_state=42,

                                     n_iter=100, verbose=5, n_jobs=-1)



rnd_srch_forest.fit(X_train_forest[:1000], y_train_mini)



print('Best score: %.3f' % rnd_srch_forest.best_score_)

print('Best params:', rnd_srch_forest.best_params_)
forest_clf = rnd_srch_forest.best_estimator_



forest_cv = cross_validate(forest_clf, X_train_forest, y_train, cv=5, scoring=scorers, n_jobs=5)

print_scores(forest_cv)
from sklearn.svm import SVC

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline



svc_clf = Pipeline([

    ('scaler', StandardScaler()),

    ('svm', SVC())

])



svc_cv = cross_validate(svc_clf, X_train_forest[:1000], y_train_mini, cv=5, scoring=scorers, n_jobs=-1)

print_scores(svc_cv)
param_distrib = {'svm__kernel': ['rbf', 'poly'],

                 'svm__C': uniform(1,20),

                 'svm__gamma': reciprocal(.0001, .1),

                }

                 



rnd_srch_svc = RandomizedSearchCV(svc_clf, param_distributions=param_distrib,

                                  cv=5, scoring='accuracy', n_iter=1000, verbose=5, n_jobs=-1)



rnd_srch_svc.fit(X_train_forest[:1000], y_train_mini)



print('Best score: %.3f' % rnd_srch_svc.best_score_)

print('Best params:', rnd_srch_svc.best_params_)
svc_clf = rnd_srch_svc.best_estimator_



svc_clf_cv = cross_validate(svc_clf, X_train_forest, y_train, cv=5, scoring=scorers, n_jobs=5)

print_scores(svc_clf_cv)
forest_clf_v1 = RandomForestClassifier(random_state=42)



forest_cv_v1 = cross_validate(forest_clf_v1, X_train_mini, y_train_mini, cv=5, scoring=scorers, n_jobs=-1)

print_scores(forest_cv_v1)
from sklearn.model_selection import RandomizedSearchCV

from scipy.stats import reciprocal, uniform



param_distrib = {"n_estimators": list(range(1,500)), "max_depth": reciprocal(2,100)}



rnd_srch_forest_v1 = RandomizedSearchCV(forest_clf_v1, param_distributions=param_distrib,

                                     cv=5, scoring='accuracy', random_state=42,

                                     n_iter=100, verbose=5, n_jobs=-1)



rnd_srch_forest_v1.fit(X_train_mini, y_train_mini)



print('Best score: %.3f' % rnd_srch_forest_v1.best_score_)

print('Best params:', rnd_srch_forest_v1.best_params_)
forest_clf_v1 = rnd_srch_forest_v1.best_estimator_



forest_cv_v1 = cross_validate(forest_clf_v1, X_train_prep, y_train, cv=5, scoring=scorers, n_jobs=5)

print_scores(forest_cv_v1)
svc_clf_v1 = Pipeline([

    ('scaler', StandardScaler(with_mean=False)),

    ('svm', SVC())

])



svc_cv_v1 = cross_validate(svc_clf_v1, X_train_mini, y_train_mini, cv=5, scoring=scorers, n_jobs=-1)

print_scores(svc_cv_v1)
param_distrib = {'svm__kernel': ['rbf', 'poly'],

                 'svm__C': uniform(1,20),

                 'svm__gamma': reciprocal(.0001, .1),

                }



rnd_srch_svc_v1 = RandomizedSearchCV(svc_clf_v1, param_distributions=param_distrib,

                                  cv=5, scoring='accuracy', n_iter=100, verbose=5, n_jobs=-1)



rnd_srch_svc_v1.fit(X_train_mini, y_train_mini)



print('Best score: %.3f' % rnd_srch_svc_v1.best_score_)

print('Best params:', rnd_srch_svc_v1.best_params_)
svc_clf_v1 = rnd_srch_svc_v1.best_estimator_



svc_clf_cv_v1 = cross_validate(svc_clf_v1, X_train_prep.toarray(), y_train, cv=5, scoring=scorers, n_jobs=5)

print_scores(svc_clf_cv_v1)
X_val_prep = full_pipeline.transform(X_val)
y_val_pred = mnb_clf.predict(X_val_prep.toarray()[:,:200])



print('Score on the test set: %.3f' % accuracy_score(y_val, y_val_pred))

print('Precision: %.3f' % precision_score(y_val, y_val_pred))

print('Recall: %.3f' % recall_score(y_val, y_val_pred))
y_val_proba = mnb_clf.predict_proba(X_val_prep.toarray()[:,:200])[:,1]

y_val_proba = y_val_proba.reshape((len(y_val), 1))

X_val_forest = np.concatenate([y_val_proba, X_val_prep.toarray()[:, 200:]], axis=1)
forest_clf.fit(X_train_forest, y_train)

y_val_pred = forest_clf.predict(X_val_forest)



print('Score on the test set: %.3f' % accuracy_score(y_val, y_val_pred))

print('Precision: %.3f' % precision_score(y_val, y_val_pred))

print('Recall: %.3f' % recall_score(y_val, y_val_pred))
svc_clf.fit(X_train_forest, y_train)

y_val_pred = svc_clf.predict(X_val_forest)



print('Score on the test set: %.3f' % accuracy_score(y_val, y_val_pred))

print('Precision: %.3f' % precision_score(y_val, y_val_pred))

print('Recall: %.3f' % recall_score(y_val, y_val_pred))
forest_clf_v1.fit(X_train_prep, y_train)

y_val_pred = forest_clf_v1.predict(X_val_prep)



print('Score on the test set: %.3f' % accuracy_score(y_val, y_val_pred))

print('Precision: %.3f' % precision_score(y_val, y_val_pred))

print('Recall: %.3f' % recall_score(y_val, y_val_pred))
svc_clf_v1.fit(X_train_prep, y_train)

y_val_pred = svc_clf_v1.predict(X_val_prep)



print('Score on the test set: %.3f' % accuracy_score(y_val, y_val_pred))

print('Precision: %.3f' % precision_score(y_val, y_val_pred))

print('Recall: %.3f' % recall_score(y_val, y_val_pred))
X_test_prep = full_pipeline.transform(X_test)

y_test_pred = svc_clf_v1.predict(X_test_prep)



print('Score on the test set: %.3f' % accuracy_score(y_test, y_test_pred))

print('Precision: %.3f' % precision_score(y_test, y_test_pred))

print('Recall: %.3f' % recall_score(y_test, y_test_pred))