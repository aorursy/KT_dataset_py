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
import gc

import os

import logging

import datetime

import warnings

import numpy as np

import pandas as pd

import seaborn as sns

import lightgbm as lgb

from tqdm import tqdm_notebook

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import mean_squared_error, confusion_matrix

from sklearn.metrics import roc_auc_score, roc_curve

from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings('ignore')
IS_LOCAL = False

PATH="../input/career-con-2019"
%%time

X_train = pd.read_csv(os.path.join(PATH, 'X_train.csv'))

X_test = pd.read_csv(os.path.join(PATH, 'X_test.csv'))

y_train = pd.read_csv(os.path.join(PATH, 'y_train.csv'))
print("Train X: {}\nTrain y: {}\nTest X: {}".format(X_train.shape, y_train.shape, X_test.shape))
X_train.head()
y_train.head()
X_test.head()
def missing_data(data):

    total = data.isnull().sum()

    percent = (data.isnull().sum()/data.isnull().count()*100)

    tt = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    types = []

    for col in data.columns:

        dtype = str(data[col].dtype)

        types.append(dtype)

    tt['Types'] = types

    return(np.transpose(tt))
missing_data(X_train)
missing_data(X_test)
missing_data(y_train)
X_train.describe()
X_test.describe()
y_train.describe()
f, ax = plt.subplots(1,1, figsize=(16,4))

g = sns.countplot(y_train['surface'])

g.set_title("Number of labels for each class")

plt.show()    
def plot_feature_distribution(df1, df2, label1, label2, features):

    i = 0

    sns.set_style('whitegrid')

    plt.figure()

    fig, ax = plt.subplots(2,5,figsize=(16,8))



    for feature in features:

        i += 1

        plt.subplot(2,5,i)

        sns.kdeplot(df1[feature], bw=0.5,label=label1)

        sns.kdeplot(df2[feature], bw=0.5,label=label2)

        plt.xlabel(feature, fontsize=9)

        locs, labels = plt.xticks()

        plt.tick_params(axis='x', which='major', labelsize=8)

        plt.tick_params(axis='y', which='major', labelsize=8)

    plt.show();
features = X_train.columns.values[3:]

plot_feature_distribution(X_train, X_test, 'train', 'test', features)
def plot_feature_class_distribution(classes,tt, features):

    i = 0

    sns.set_style('whitegrid')

    plt.figure()

    fig, ax = plt.subplots(5,2,figsize=(16,24))



    for feature in features:

        i += 1

        plt.subplot(5,2,i)

        for clas in classes:

            ttc = tt[tt['surface']==clas]

            sns.kdeplot(ttc[feature], bw=0.5,label=clas)

        plt.xlabel(feature, fontsize=9)

        locs, labels = plt.xticks()

        plt.tick_params(axis='x', which='major', labelsize=8)

        plt.tick_params(axis='y', which='major', labelsize=8)

    plt.show();
classes = (y_train['surface'].value_counts()).index

tt = X_train.merge(y_train, on='series_id', how='inner')

plot_feature_class_distribution(classes, tt, features)
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



def perform_feature_engineering(actual):

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

    

    def mean_change_of_abs_change(x):

        return np.mean(np.diff(np.abs(np.diff(x))))



    def mean_abs_change(x):

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

        new[col + '_mean_abs_change'] = actual.groupby('series_id')[col].apply(mean_abs_change)

        

        # Change of Change. 2nd order.

        new[col + '_mean_change_of_abs_change'] = actual.groupby('series_id')[col].apply(mean_change_of_abs_change)

        

        new[col + '_abs_max'] = actual.groupby('series_id')[col].apply(lambda x: np.max(np.abs(x)))

        new[col + '_abs_min'] = actual.groupby('series_id')[col].apply(lambda x: np.min(np.abs(x)))



    return new
%%time

X_train = perform_feature_engineering(X_train)

X_test = perform_feature_engineering(X_test)
X_train.head()
X_test.head()
le = LabelEncoder()

y_train['surface'] = le.fit_transform(y_train['surface'])
X_train.fillna(0, inplace = True)

X_train.replace(-np.inf, 0, inplace = True)

X_train.replace(np.inf, 0, inplace = True)

X_test.fillna(0, inplace = True)

X_test.replace(-np.inf, 0, inplace = True)

X_test.replace(np.inf, 0, inplace = True)
folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
sub_preds_rf = np.zeros((X_test.shape[0], 9))

oof_preds_rf = np.zeros((X_train.shape[0]))

score = 0

for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train['surface'])):

    clf =  RandomForestClassifier(n_estimators = 1000, n_jobs = -1)

    clf.fit(X_train.iloc[trn_idx], y_train['surface'][trn_idx])

    oof_preds_rf[val_idx] = clf.predict(X_train.iloc[val_idx])

    sub_preds_rf += clf.predict_proba(X_test) / folds.n_splits

    score += clf.score(X_train.iloc[val_idx], y_train['surface'][val_idx])

    print('Fold: {} score: {}'.format(fold_,clf.score(X_train.iloc[val_idx], y_train['surface'][val_idx])))

print('Avg Accuracy', score / folds.n_splits)
submission = pd.read_csv(os.path.join(PATH,'sample_submission.csv'))

submission['surface'] = le.inverse_transform(sub_preds_rf.argmax(axis=1))

submission.to_csv('rf.csv', index=False)

submission.head(10)