import os

import time

import numpy as np

import pandas as pd

from seaborn import countplot,lineplot, barplot

import matplotlib.pyplot as plt



from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import KFold

from sklearn.metrics import accuracy_score, confusion_matrix



from bayes_opt import BayesianOptimization

import lightgbm as lgb

import xgboost as xgb

import catboost as cb
tr = pd.read_csv('../input/X_train.csv')

te = pd.read_csv('../input/X_test.csv')

target = pd.read_csv('../input/y_train.csv')

ss = pd.read_csv('../input/sample_submission.csv')
tr.head()
tr.shape, te.shape
countplot(y = 'surface', data = target)

plt.show()
len(tr.measurement_number.value_counts())
tr.shape[0] / 128, te.shape[0] / 128
# https://stackoverflow.com/questions/53033620/how-to-convert-euler-angles-to-quaternions-and-get-the-same-euler-angles-back-fr?rq=1

def quaternion_to_euler(x, y, z, w):

    import math

    t0 = +2.0 * (w * x + y * z)

    t1 = +1.0 - 2.0 * (x * x + y * y)

    X = math.atan2(t0, t1)



    t2 = +2.0 * (w * y - z * x)

    t2 = +1.0 if t2 > +1.0 else t2

    t2 = -1.0 if t2 < -1.0 else t2

    Y = math.asin(t2)



    t3 = +2.0 * (w * z + x * y)

    t4 = +1.0 - 2.0 * (y * y + z * z)

    Z = math.atan2(t3, t4)



    return X, Y, Z



def fe(actual):

    new = pd.DataFrame()

    actual['total_angular_velocity'] = (actual['angular_velocity_X'] ** 2 + actual['angular_velocity_Y'] ** 2 + actual['angular_velocity_Z'] ** 2) ** 0.5

    actual['total_linear_acceleration'] = (actual['linear_acceleration_X'] ** 2 + actual['linear_acceleration_Y'] ** 2 + actual['linear_acceleration_Z'] ** 2) ** 0.5

    

    actual['acc_vs_vel'] = actual['total_linear_acceleration'] / actual['total_angular_velocity']

    

    x, y, z, w = actual['orientation_X'].tolist(), actual['orientation_Y'].tolist(), actual['orientation_Z'].tolist(), actual['orientation_W'].tolist()

    nx, ny, nz = [], [], []

    for i in range(len(x)):

        xx, yy, zz = quaternion_to_euler(x[i], y[i], z[i], w[i])

        nx.append(xx)

        ny.append(yy)

        nz.append(zz)

    

    actual['euler_x'] = nx

    actual['euler_y'] = ny

    actual['euler_z'] = nz

    

    actual['total_angle'] = (actual['euler_x'] ** 2 + actual['euler_y'] ** 2 + actual['euler_z'] ** 2) ** 5

    actual['angle_vs_acc'] = actual['total_angle'] / actual['total_linear_acceleration']

    actual['angle_vs_vel'] = actual['total_angle'] / actual['total_angular_velocity']

    

    def f1(x):

        return np.mean(np.diff(np.abs(np.diff(x))))

    

    def f2(x):

        return np.mean(np.abs(np.diff(x)))

    

    for col in actual.columns:

        if col in ['row_id', 'series_id', 'measurement_number']:

            continue

        new[col + '_mean'] = actual.groupby(['series_id'])[col].mean()

        new[col + '_min'] = actual.groupby(['series_id'])[col].min()

        new[col + '_max'] = actual.groupby(['series_id'])[col].max()

        new[col + '_std'] = actual.groupby(['series_id'])[col].std()

        new[col + '_max_to_min'] = new[col + '_max'] / new[col + '_min']

        

        # Change. 1st order.

        new[col + '_mean_abs_change'] = actual.groupby('series_id')[col].apply(f2)

        

        # Change of Change. 2nd order.

        new[col + '_mean_change_of_abs_change'] = actual.groupby('series_id')[col].apply(f1)

        

        new[col + '_abs_max'] = actual.groupby('series_id')[col].apply(lambda x: np.max(np.abs(x)))

        new[col + '_abs_min'] = actual.groupby('series_id')[col].apply(lambda x: np.min(np.abs(x)))



    return new
%%time

tr = fe(tr)

te = fe(te)

tr.head()
tr.head()
le = LabelEncoder()

target['surface'] = le.fit_transform(target['surface'])
tr.fillna(0, inplace = True)

te.fillna(0, inplace = True)
tr.replace(-np.inf, 0, inplace = True)

tr.replace(np.inf, 0, inplace = True)

te.replace(-np.inf, 0, inplace = True)

te.replace(np.inf, 0, inplace = True)
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import KFold, StratifiedKFold

folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=546789)

sub_preds_rf = np.zeros((te.shape[0], 9))

oof_preds_rf = np.zeros((tr.shape[0]))

score = 0

for i, (train_index, test_index) in enumerate(folds.split(tr, target['surface'])):

    print('-'*20, i, '-'*20)

    

    clf =  RandomForestClassifier(n_estimators = 200, n_jobs = -1)

    clf.fit(tr.iloc[train_index], target['surface'][train_index])

    oof_preds_rf[test_index] = clf.predict(tr.iloc[test_index])

    sub_preds_rf += clf.predict_proba(te) / folds.n_splits

    score += clf.score(tr.iloc[test_index], target['surface'][test_index])

    print('score ', clf.score(tr.iloc[test_index], target['surface'][test_index]))

    importances = clf.feature_importances_

    indices = np.argsort(importances)

    features = tr.columns



    hm = 30

    plt.figure(figsize=(7, 10))

    plt.title('Feature Importances')

    plt.barh(range(len(indices[:hm])), importances[indices][:hm], color='b', align='center')

    plt.yticks(range(len(indices[:hm])), [features[i] for i in indices])

    plt.xlabel('Relative Importance')

    plt.show()



print('Avg Accuracy', score / folds.n_splits)
# https://www.kaggle.com/artgor/where-do-the-robots-drive

import itertools



def plot_confusion_matrix(truth, pred, classes, normalize=False, title=''):

    cm = confusion_matrix(truth, pred)

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    

    plt.figure(figsize=(10, 10))

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

    plt.title('Confusion matrix', size=15)

    plt.colorbar(fraction=0.046, pad=0.04)

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, format(cm[i, j], fmt),

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    plt.grid(False)

    plt.tight_layout()
plot_confusion_matrix(target['surface'], oof_preds_rf, le.classes_)
ss['surface'] = le.inverse_transform(sub_preds_rf.argmax(axis=1))

ss.to_csv('rf.csv', index=False)

ss.head(10)