# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np

import pandas as pd

import sklearn

from sklearn.metrics import confusion_matrix

from sklearn.ensemble import RandomForestClassifier 

from functools import partial

import scipy as sp

from sklearn.decomposition import TruncatedSVD,PCA

import collections

import json

import pandas as pd

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer

from stemming.porter2 import stem

from nltk.tokenize import word_tokenize

import collections

import json

import os

import sklearn

from sklearn.metrics import confusion_matrix

import nltk

from nltk import word_tokenize

from collections import Counter

from functools import partial

from math import sqrt

import cv2

import pandas as pd

import numpy as np

import os

from tqdm import tqdm, tqdm_notebook



from sklearn.metrics import cohen_kappa_score, mean_squared_error

from sklearn.metrics import confusion_matrix as sk_cmatrix



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
def get_cat_val(X, arr, cat):

    cat_list = []

    for ind in arr:

        cat_list.append(X[ind, cat])

    return cat_list



def get_dog_val(X, arr, dog):

    dog_list = []

    for ind in arr:

        dog_list.append(X[ind, dog])

    return dog_list

def weighted_average(item):

    try:

        return np.average(item, weights=np.arange(len(item), 0, -1))

    except ZeroDivisionError:

        return 0

def get_train_meta():

    x_train = pd.read_csv("../input/train/train.csv")

    label_annotations = collections.defaultdict(list)

    label_scores = collections.defaultdict(list)

    for file in os.listdir('../input/train_metadata/'):

        tmp = file.split()[0].split('.json')[0]

        key, val = tmp.split('-') 

        try:

            with open('../input/train_metadata/'+file) as f:

                data = json.load(f)

                if data.get('labelAnnotations'):

                    for element in data['labelAnnotations']:

                        label = element['description']

                        score = element['score']



                        label_annotations[key].append(label)

                        label_scores[key].append(score)

                else: 

                    label_annotations[key].append('N/A')

                    label_scores[key].append(-1)

        except FileNotFoundError:

            print('Oopsie')



    x_train['label_annotation'] = x_train['PetID'].map(label_annotations)

    x_train['label_score'] = x_train['PetID'].map(label_scores)



    x_train['label_annotation'] = [[word_tokenize(word) for word in sentence] for sentence in x_train['label_annotation']]



    flatten = lambda l: [item for sublist in l for item in sublist]

    x_train['label_annotation'] = x_train['label_annotation'].map(flatten)

    x_train['label_annotation'] = [[stem(word) for word in sentence] for sentence in x_train['label_annotation']]

    list_to_string = lambda l: ' '.join(l)

    x_train['label_annotation'] = x_train['label_annotation'].map(list_to_string)



    vectorizer = TfidfVectorizer()

    X = vectorizer.fit_transform(x_train['label_annotation'])



    ind = x_train.index.values

    cat_val = get_cat_val(X, ind, vectorizer.vocabulary_['cat'])

    dog_val = get_dog_val(X, ind, vectorizer.vocabulary_['dog'])

    x_train.loc[:, 'label_cat'] = cat_val

    x_train.loc[:, 'label_dog'] = dog_val

    x_train = x_train.drop(columns=['label_annotation'])

    x_train['label_score'] = x_train['label_score'].map(weighted_average)

    return x_train



def get_test_meta():

    x_test = pd.read_csv("../input/test/test.csv")

    label_annotations = collections.defaultdict(list)

    label_scores = collections.defaultdict(list)

    for file in os.listdir('../input/test_metadata/'):

        tmp = file.split()[0].split('.json')[0]

        key, val = tmp.split('-') 

        try:

            with open('../input/test_metadata/'+file) as f:

                data = json.load(f)

                if data.get('labelAnnotations'):

                    for element in data['labelAnnotations']:

                        label = element['description']

                        score = element['score']



                        label_annotations[key].append(label)

                        label_scores[key].append(score)

                else: 

                    label_annotations[key].append('N/A')

                    label_scores[key].append(-1)

        except FileNotFoundError:

            print('Oopsie')



    x_test['label_annotation'] = x_test['PetID'].map(label_annotations)

    x_test['label_score'] = x_test['PetID'].map(label_scores)



    x_test['label_annotation'] = [[word_tokenize(word) for word in sentence] for sentence in x_test['label_annotation']]



    flatten = lambda l: [item for sublist in l for item in sublist]

    x_test['label_annotation'] = x_test['label_annotation'].map(flatten)

    x_test['label_annotation'] = [[stem(word) for word in sentence] for sentence in x_test['label_annotation']]

    list_to_string = lambda l: ' '.join(l)

    x_test['label_annotation'] = x_test['label_annotation'].map(list_to_string)



    vectorizer = TfidfVectorizer()

    X = vectorizer.fit_transform(x_test['label_annotation'])



    ind = x_test.index.values

    cat_val = get_cat_val(X, ind, vectorizer.vocabulary_['cat'])

    dog_val = get_dog_val(X, ind, vectorizer.vocabulary_['dog'])

    x_test.loc[:, 'label_cat'] = cat_val

    x_test.loc[:, 'label_dog'] = dog_val

    x_test = x_test.drop(columns=['label_annotation'])

    x_test['label_score'] = x_test['label_score'].map(weighted_average)

    return x_test

#train = pd.read_csv("../input/petfinder-adoption-prediction/train/train.csv")

#test = pd.read_csv("../input/petfinder-adoption-prediction/test/test.csv")

train = get_train_meta()

test = get_test_meta()
def extract_text_features(text):

    tp = type(text) is str

    if not tp:

        text = "fsdfl"

    features = []

    header = []



    header.append("PetID")

    bag_of_words = nltk.word_tokenize(text)

    # Total amount of words

    header.append("Amount of Words")

    features.append(len(bag_of_words))



    sent_text = nltk.sent_tokenize(text) 





    # Amount of sentences



    header.append("Amount of Sentences")

    features.append(len(sent_text))



    

    if not tp:

        features = list(np.zeros(len(header)-1))

    return header, features

    #print(tokens)

    

def get_train_text_and_name_features():

    training_df = pd.read_csv('../input/train/train.csv')

    text_data = training_df[["PetID","Description","Name","AdoptionSpeed"]]



    text_features_table = []

    name_features_table = []

    counter = 0 

    for (PID,profile,name,a_s) in text_data.values:



        header,features = extract_text_features(profile)

        text_features_table.append([PID]+features)

    text_features_pd = pd.DataFrame(text_features_table,columns=header)

    return text_features_pd





def get_test_text_and_name_features():

    test_df = pd.read_csv('../input/test/test.csv')

    text_data = test_df[["PetID","Description","Name"]]



    text_features_table = []

    name_features_table = []

    counter = 0 

    for (PID,profile,name) in text_data.values:



        header,features = extract_text_features(profile)

        text_features_table.append([PID]+features)

    text_features_pd = pd.DataFrame(text_features_table,columns=header)

    return text_features_pd
des_train = get_train_text_and_name_features()

train['PetID']=train['PetID'].astype(str)

test['PetID']=test['PetID'].astype(str)
des_test = get_test_text_and_name_features()
AoW_train = pd.DataFrame(des_train['Amount of Words'])

AoW_test = pd.DataFrame(des_test['Amount of Words'])
x_train = train.join(AoW_train)

x_test = test.join(AoW_test)


def extract_features(X):

    X_features = X.drop(["Name","RescuerID","Description","PetID"],axis=1)

    return X_features

x_train = extract_features(x_train)

x_testt = extract_features(x_test)
X_mean = x_train.Fee.mean(axis=0)

X_std = x_train.Fee.std(axis=0)

x_train.Fee = (x_train.Fee-X_mean)/X_std

x_testt.Fee = (x_train.Fee-X_mean)/X_std
y_train = train['AdoptionSpeed'].values
import xgboost as xgb

from sklearn.model_selection import StratifiedKFold

xgb_params = {

    'eval_metric': 'rmse',

    'seed': 1337,

    'silent': 1,

}

def run_xgb(params, X_train, X_test):

    n_splits = 5

    verbose_eval = 1000

    num_rounds = 30000

    early_stop = 500



    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1337)



    oof_train = np.zeros((X_train.shape[0]))

    oof_test = np.zeros((X_test.shape[0], n_splits))



    i = 0



    for train_idx, valid_idx in kf.split(X_train, X_train['AdoptionSpeed'].values):



        X_tr = X_train.iloc[train_idx, :]

        X_val = X_train.iloc[valid_idx, :]



        y_tr = X_tr['AdoptionSpeed'].values

        X_tr = X_tr.drop(['AdoptionSpeed'], axis=1)



        y_val = X_val['AdoptionSpeed'].values

        X_val = X_val.drop(['AdoptionSpeed'], axis=1)



        d_train = xgb.DMatrix(data=X_tr, label=y_tr, feature_names=X_tr.columns)

        d_valid = xgb.DMatrix(data=X_val, label=y_val, feature_names=X_val.columns)



        watchlist = [(d_train, 'train'), (d_valid, 'valid')]

        model = xgb.train(dtrain=d_train, num_boost_round=num_rounds, evals=watchlist,

                         early_stopping_rounds=early_stop, verbose_eval=verbose_eval, params=params)



        valid_pred = model.predict(xgb.DMatrix(X_val, feature_names=X_val.columns), ntree_limit=model.best_ntree_limit)

        test_pred = model.predict(xgb.DMatrix(X_test, feature_names=X_test.columns), ntree_limit=model.best_ntree_limit)



        oof_train[valid_idx] = valid_pred

        oof_test[:, i] = test_pred



        i += 1

    return model, oof_train, oof_test

class OptimizedRounder(object):

    def __init__(self):

        self.coef_ = 0

    

    def _kappa_loss(self, coef, X, y):

        preds = pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels = [0, 1, 2, 3, 4])

        return -cohen_kappa_score(y, preds, weights='quadratic')

    

    def fit(self, X, y):

        loss_partial = partial(self._kappa_loss, X = X, y = y)

        initial_coef = [0.5, 1.5, 2.5, 3.5]

        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')

    

    def predict(self, X, coef):

        preds = pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels = [0, 1, 2, 3, 4])

        return preds

    

    def coefficients(self):

        return self.coef_['x']
def qwk(estimator,X,y, additionals = None):

    N = 5

    # compute matrix W

    W = np.zeros((N,N))     

    for i in range(N):

        for j in range(N):

            W[i,j]=((i-j)**2)/((N-1)**2)

            

    # Compute (confusion) matrix O

    actuals = y     

    if additionals is not None:

        preds = estimator.predict(X,additionals)

    else:

        preds = estimator.predict(X)

    O = confusion_matrix(actuals,preds)

    O = O/O.sum()

    

    # Compute Matrix E

    act_hist=np.zeros([N])

    for act in actuals:

        act_hist[act]+=1

    pred_hist=np.zeros([N])

    for pred in preds:

        pred_hist[pred]+=1    

    E = np.outer(act_hist,pred_hist)

    E = E/E.sum()

    

    #Compute the final score

    num = 0

    den = 0

    for i in range(N):

        for j in range(N):

            num+=W[i,j]*O[i,j]

            den+=W[i,j]*E[i,j]

    k = 1-num/den

    

    

    return k

            
model, oof_train, oof_test = run_xgb(xgb_params, x_train,x_testt)

optR = OptimizedRounder()

optR.fit(oof_train, y_train)

coefficients = optR.coefficients()

kappa = qwk(optR,oof_train,y_train, additionals=coefficients)

print("QWK = ", kappa)
coefficients_ = coefficients.copy()

kappa = qwk(optR,oof_train,y_train, additionals=coefficients_)

print("QWK = ", kappa)
test_predictions = optR.predict(oof_test.mean(axis=1), coefficients_).astype(np.int8)
submission = pd.DataFrame({'PetID': x_test['PetID'].values, 'AdoptionSpeed': test_predictions})

submission.to_csv('submission.csv', index=False)