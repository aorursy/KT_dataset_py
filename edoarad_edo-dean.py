import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import random

from sklearn.ensemble import GradientBoostingClassifier

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
adult = pd.read_csv("../input/adult.csv")
adult.shape
adult.keys()
adult.income = (adult.income==adult.income.unique()[1]).astype(int) # 1 if income>50k, 0 otherwise

adult['y'] = adult.income
def compute_score(y, y_hat): # to be minimized

    assert len(y)==len(y_hat)

    return -np.mean(y*np.log(y_hat) + (1-y)*np.log(1-y_hat))



def predict_baseline(X_train, y_train, X_test):

    return np.array([y_train.mean()]*len(X_test))



def full_cv(predict_func, X, y, folds=10, shuffle=True, verbose=True):

    y = np.array(y)

    index_shuf = np.array(range(len(X)))    

    if shuffle:

        random.shuffle(index_shuf)

        

    X_folds = np.array_split(index_shuf, folds)

    y_folds = np.array_split(index_shuf, folds)

    y_predict = np.array([])

    for k in range(folds):

        if verbose:

            print('\r{:7.5}%  '.format(k*100.0/folds))

        # We use 'list' to copy, in order to 'pop' later on

        X_train = list(X_folds)

        X_test  = X_train.pop(k)

        X_train = np.concatenate(X_train)

        y_train = list(y_folds)

        y_train.pop(k)

        y_train = np.concatenate(y_train)

        new_y_hat = predict_func(X.ix[X_train], y[y_train], X.ix[X_test])

        y_predict = np.concatenate([y_predict, new_y_hat])

    return compute_score(y[index_shuf], y_predict)



baseline_score = full_cv(predict_baseline, adult, adult.y, verbose=False)

print('baseline_score=',baseline_score) # 0.568739686466 (10 folds)

#baseline_score = compute_score(adult.income, np.array([adult.income.mean()]*len(adult.income))) # 0.5520112931
lst = ['Preschool','1st-4th', '5th-6th','7th-8th','9th','10th','11th','12th','HS-grad','Some-college','Bachelors','Prof-school','Masters','Doctorate','Assoc-acdm','Assoc-voc']

adult['edu_num'] = [lst.index(a) for a in adult.education]
r = np.array(range(0,adult.shape[0]))

random.shuffle(r)

train_size = int(0.7*len(r))

train_set = adult.ix[r[:train_size]]

test_size = len(r)-train_size

test_set = adult.ix[r[train_size:]]
print(train_set[['age']].shape, train_set['y'].shape)
clf = GradientBoostingClassifier(learning_rate=0.1, n_estimators=100, max_depth=2, verbose=True)

feature_list = ['age','hours.per.week', 'edu_num']

clf.fit(train_set[feature_list], train_set['y'])
p = clf.predict_proba(test_set[feature_list])[:,1]

sc = score(test_set['y'],p)

print(sc / baseline_score)
clf.feature_importances_