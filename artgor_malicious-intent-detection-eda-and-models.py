import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

import eli5
from nltk.tokenize import TweetTokenizer
import datetime
import lightgbm as lgb
from scipy import stats
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from wordcloud import WordCloud
from collections import Counter
from nltk.corpus import stopwords
from nltk.util import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
pd.set_option('max_colwidth',400)
import msgpack
from sklearn.metrics import roc_auc_score
info = pd.read_csv('../input/train_info.csv')
info.head()
info.injection.value_counts()
with open('../input/train.msgpack', 'rb') as data_file:
    train = msgpack.unpack(data_file)
with open('../input/test.msgpack', 'rb') as data_file:
    test = msgpack.unpack(data_file)
train = pd.DataFrame(train)
test = pd.DataFrame(test)
train.columns = ['id', 'text']
test.columns = ['id', 'text']
train.head(10).values
train['text'] = train['text'].astype(str)
test['text'] = test['text'].astype(str)
vectorizer = TfidfVectorizer(ngram_range=(1, 4), analyzer='char')
full_text = list(train['text'].values) + list(test['text'].values)
vectorizer.fit(full_text)
train_vectorized = vectorizer.transform(train['text'])
test_vectorized = vectorizer.transform(test['text'])
info.injection.value_counts()
info = pd.merge(train, info, on='id')
y = np.array([1 if i == True else 0 for i in info.injection.values])
logreg = LogisticRegression(C=10.0)
scores = cross_val_score(logreg, train_vectorized, y, scoring='roc_auc', n_jobs=-1, cv=5)
print('Cross-validation mean auc {0:.2f}%, std {1:.2f}.'.format(np.mean(scores) * 100, np.std(scores) * 100))
logreg.fit(train_vectorized, y)
eli5.show_weights(logreg, vec=vectorizer,  targets=[0, 1])
eli5.show_prediction(logreg, doc=train['text'].values[100], vec=vectorizer)
# logreg1 = LogisticRegression(C=1.0)
# logreg1.fit(train_vectorized, y)
# logreg2 = LogisticRegression(C=0.1)
# logreg2.fit(train_vectorized, y)
# sub = pd.read_csv('../input/sample_submission.csv')
# pred = (logreg.predict_proba(test_vectorized) + logreg1.predict_proba(test_vectorized) + logreg2.predict_proba(test_vectorized)) / 3
# sub['injection'] = pred
# sub.head()
# sub.to_csv('sub.csv', index=False)
from sklearn.model_selection import StratifiedKFold, KFold
n_fold = 5
folds = KFold(n_splits=n_fold, shuffle=True, random_state=11)
def train_model(X=train_vectorized, X_test=test_vectorized, y=y, params=None, folds=folds, model_type='lgb', plot_feature_importance=False):

    oof = np.zeros(X.shape[0])
    prediction = np.zeros(X_test.shape[0])
    scores = []
    feature_importance = pd.DataFrame()
    for fold_n, (train_index, valid_index) in enumerate(folds.split(X)):
        print('Fold', fold_n, 'started at', time.ctime())
        X_train, X_valid = X[train_index], X[valid_index]
        y_train, y_valid = y[train_index], y[valid_index]
        
        if model_type == 'lgb':
            model = lgb.LGBMRegressor(**params, n_estimators = 20000, nthread = 4, n_jobs = -1)
            model.fit(X_train, y_train, 
                    eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric='rmse',
                    verbose=1000, early_stopping_rounds=200)
            
            y_pred_valid = model.predict(X_valid)
            y_pred = model.predict(X_test, num_iteration=model.best_iteration_)
        
        oof[valid_index] = y_pred_valid.reshape(-1,)
        scores.append(roc_auc_score(y_valid, y_pred_valid))
        
        prediction += y_pred    

    prediction /= n_fold
    
    print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))

    return oof, prediction
import time
params = {'num_leaves': 54,
         'min_data_in_leaf': 79,
         'objective': 'regression',
         'max_depth': 7,
         'learning_rate': 0.018545526395058548,
         "boosting": "gbdt",
         "feature_fraction": 0.8354507676881442,
         "bagging_freq": 3,
         "bagging_fraction": 0.8126672064208567,
         "bagging_seed": 11,
         "metric": 'rmse',
         "lambda_l1": 0.1,
         "verbosity": -1,
         'min_child_weight': 5.343384366323818,
         'reg_alpha': 1.1302650970728192,
         'reg_lambda': 0.3603427518866501,
         'subsample': 0.8767547959893627,}

oof_lgb, prediction_lgb = train_model(X=train_vectorized, X_test=test_vectorized, y=y, folds=folds,
                                                          params=params, model_type='lgb')
sub = pd.read_csv('../input/sample_submission.csv')
sub['injection'] = prediction_lgb
sub.head()
sub.to_csv('sub.csv', index=False)