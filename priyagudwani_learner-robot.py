# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from seaborn import countplot,lineplot, barplot

import matplotlib.pyplot as plt





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
tr = pd.read_csv('../input/X_train.csv')

te = pd.read_csv('../input/X_test.csv')

target = pd.read_csv('../input/y_train.csv')

ss = pd.read_csv('../input/sample_submission.csv')





tr.head()
countplot(y = 'surface', data = target)

plt.show()
plt.figure(figsize=(26, 16))

for i, col in enumerate(tr.columns[3:]):

    plt.subplot(3,4, i + 1)

    plt.plot(tr.loc[tr['series_id'] == 1, col])

    plt.title(col)
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

    

    actual['total_angle'] = (actual['euler_x'] ** 2 + actual['euler_y'] ** 2 + actual['euler_z'] ** 2) ** 0.5

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

        new[col + '_Moving_average_10_mean'] = actual.groupby(['series_id'])[col].rolling(window=10).mean().mean(skipna=True)

        new[col + '_Moving_average_16_mean'] = actual.groupby(['series_id'])[col].rolling(window=16).mean().mean(skipna=True)

        new[col + '_Moving_average_10_std'] = actual.groupby(['series_id'])[col].rolling(window=10).std().mean(skipna=True)

        new[col + '_Moving_average_16_std'] = actual.groupby(['series_id'])[col].rolling(window=16).std().mean(skipna=True)

        

        new[col + '_max_to_min'] = new[col + '_max'] / new[col + '_min']

        

        # Change. 1st order.

        new[col + '_mean_abs_change'] = actual.groupby('series_id')[col].apply(f2)

        

        # Change of Change. 2nd order.

        new[col + '_mean_change_of_abs_change'] = actual.groupby('series_id')[col].apply(f1)

        

        new[col + '_abs_max'] = actual.groupby('series_id')[col].apply(lambda x: np.max(np.abs(x)))

        new[col + '_abs_min'] = actual.groupby('series_id')[col].apply(lambda x: np.min(np.abs(x)))



    return new
tr = fe(tr)

te = fe(te)

tr.head()

tr.fillna(0, inplace = True)

te.fillna(0, inplace = True)

tr.replace(-np.inf, 0, inplace = True)

tr.replace(np.inf, 0, inplace = True)

te.replace(-np.inf, 0, inplace = True)

te.replace(np.inf, 0, inplace = True)
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

target['surface'] = le.fit_transform(target['surface'])

target.head()
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import KFold, StratifiedKFold

folds = StratifiedKFold(n_splits=20, shuffle=True, random_state=20)

pred_e = np.zeros((te.shape[0], 9))

pred_r = np.zeros((tr.shape[0]))

score = 0
for i, (train_index, test_index) in enumerate(folds.split(tr, target['surface'])):

    

    clf =  RandomForestClassifier(n_estimators = 200, n_jobs = -1)

    clf.fit(tr.iloc[train_index], target['surface'][train_index])

 

    pred_r[test_index] = clf.predict(tr.iloc[test_index])

    pred_e += clf.predict_proba(te) / folds.n_splits

    

    score += clf.score(tr.iloc[test_index], target['surface'][test_index])

    print('score ', clf.score(tr.iloc[test_index], target['surface'][test_index]))

   
# https://www.kaggle.com/artgor/where-do-the-robots-drive

import itertools

from sklearn.metrics import confusion_matrix



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
plot_confusion_matrix(target['surface'], pred_r, le.classes_)
ss['surface'] = le.inverse_transform(pred_e.argmax(axis=1))

ss.to_csv('rf.csv', index=False)

ss.head(10)