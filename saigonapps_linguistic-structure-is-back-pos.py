# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



from sklearn.preprocessing import LabelEncoder

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import StratifiedKFold

from sklearn import metrics

import lightgbm as lgb

from time import time

import spacy

nlp = spacy.load('en')



## ~~ Happy modeling ~~ ##

t0 = time()

print("[*] Loading dataframes")

train = pd.read_csv("../input/train.csv", encoding='utf-8')

test = pd.read_csv("../input/test.csv", encoding='utf-8')



# test the extract function

def extract_pos(txt):

    return " ".join([i.pos_ for i in nlp(txt)])



def run_gdbt(x1,y1,x2,y2,x_test,n_iter=5000,seed=42,max_depth=10,lr=0.02):

    params = {

        'boosting_type': 'gbdt',

        'max_depth': max_depth,

        'learning_rate': lr,

        'num_leaves': 20,

        'verbose': 0, 

        'metric': 'multi_logloss',

        'objective': 'multiclass',

        'num_classes': 3,

        'num_threads': 6,

        'bagging_fraction_seed': seed,

        'feature_fraction_seed': seed,

    }

    n_estimators = n_iter

    d_train = lgb.Dataset(x1, label=y1)

    d_valid = lgb.Dataset(x2, label=y2)

    model = lgb.train(params, d_train, n_estimators, [d_train, d_valid], verbose_eval=200,early_stopping_rounds=120)



    y2_hat = model.predict(x2)

    y_hat = model.predict(x_test)

    return y2_hat, y_hat, model



print("Hello world! -> (POS)", extract_pos(u"hello world!"))

print("[*] Extract POS features")

train['pos'] = train.text.apply(extract_pos)

test['pos'] = test.text.apply(extract_pos)



le = LabelEncoder()

y_train = le.fit_transform(train.author.values)







counter = CountVectorizer(stop_words=None, ngram_range=(1,3), input='content',\

                          encoding='utf-8', decode_error='replace', strip_accents='unicode',\

                          lowercase=True, analyzer='word')



counter.fit(train['pos'].values.tolist() + test['pos'].values.tolist())

train_bow = counter.transform(train['pos'].values.tolist())

test_bow = counter.transform(test['pos'].values.tolist())



train_bow = train_bow.toarray()

test_bow = test_bow.toarray()



print("[*] Train shape", train_bow.shape, y_train.shape)

print("[*] Test shape", test_bow.shape)



## main code ##

n_folds = 5

n_classes = 3

cv_scores = []

y_test_cv = 0

y_train_cv = np.zeros([train.shape[0], n_classes])

kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=2017)

for train_index, val_index in kf.split(train.id,y_train):

    x1, x2 = train_bow[train_index], train_bow[val_index]

    y1, y2 = y_train[train_index], y_train[val_index]

    

    y2_hat, y_test_hat, model = run_gdbt(x1, y1, x2, y2, test_bow,n_iter=1000,max_depth=10)

    y_test_cv = y_test_cv + y_test_hat

    y_train_cv[val_index,:] = y2_hat

    cv_scores.append(metrics.log_loss(y2, y2_hat))



modeling_time = time() - t0

print("Mean cv score : ", np.mean(cv_scores),np.std(cv_scores))

print("Modeling time: %0.3fs" % modeling_time)

print("[*] Extract submission dataframe")

y_test_cv = y_test_cv / n_folds

sub = pd.DataFrame({'id': test.id,\

                    '{}'.format(le.inverse_transform([0])[0]): y_test_cv[:,0], \

                    '{}'.format(le.inverse_transform([1])[0]): y_test_cv[:,1], \

                    '{}'.format(le.inverse_transform([2])[0]): y_test_cv[:,2]})

print("DONE!")

## Voila!, you will see POS would help. Linguistic structure is back.(in ~ 3 minutes)

## Mean cv score :  0.80320359782 0.00983586058765

## Modeling time: 177.148s