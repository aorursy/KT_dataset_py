# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier, ExtraTreesClassifier,BaggingClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import StratifiedKFold

from lightgbm import LGBMClassifier

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import warnings

warnings.simplefilter("ignore")

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
X_train = pd.read_csv("../input/X_train.csv")

X_test = pd.read_csv("../input/X_test.csv")

y_train = pd.read_csv("../input/y_train.csv")
X_train.head()
y_train.head()
from scipy.stats import kurtosis

from scipy.stats import skew

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



def generate_features(data):

    new_data = pd.DataFrame()

    data['total_angular_velocity'] = (data['angular_velocity_X'] ** 2 + data['angular_velocity_Y'] ** 2 + data['angular_velocity_Z'] ** 2) ** 0.5

    data['total_linear_acceleration'] = (data['linear_acceleration_X'] ** 2 + data['linear_acceleration_Y'] ** 2 + data['linear_acceleration_Z'] ** 2) ** 0.5

    

    data['acc_vs_vel'] = data['total_linear_acceleration'] / data['total_angular_velocity']

    

    x, y, z, w = data['orientation_X'].tolist(), data['orientation_Y'].tolist(), data['orientation_Z'].tolist(), data['orientation_W'].tolist()

    nx, ny, nz = [], [], []

    for i in range(len(x)):

        xx, yy, zz = quaternion_to_euler(x[i], y[i], z[i], w[i])

        nx.append(xx)

        ny.append(yy)

        nz.append(zz)

    

    data['euler_x'] = nx

    data['euler_y'] = ny

    data['euler_z'] = nz

    

    data['total_angle'] = (data['euler_x'] ** 2 + data['euler_y'] ** 2 + data['euler_z'] ** 2) ** 5

    data['angle_vs_acc'] = data['total_angle'] / data['total_linear_acceleration']

    data['angle_vs_vel'] = data['total_angle'] / data['total_angular_velocity']

    

    def mean_change_of_abs_change(x):

        return np.mean(np.diff(np.abs(np.diff(x))))



    def mean_abs_change(x):

        return np.mean(np.abs(np.diff(x)))

    

    for col in data.columns:

        if col in ['row_id', 'series_id', 'measurement_number']:

            continue

        new_data[col + '_mean'] = data.groupby(['series_id'])[col].mean()

        new_data[col + '_min'] = data.groupby(['series_id'])[col].min()

        new_data[col + '_max'] = data.groupby(['series_id'])[col].max()

        new_data[col + '_std'] = data.groupby(['series_id'])[col].std()

        new_data[col + '_max_to_min'] = new_data[col + '_max'] / new_data[col + '_min']

        new_data[col + '_kurtosis'] = data.groupby('series_id')[col].apply(lambda x: kurtosis(x))

        new_data[col + '_skew'] = data.groupby('series_id')[col].apply(lambda x: skew(x))

        

        # 1st order derivative

        new_data[col + '_mean_abs_change'] = data.groupby('series_id')[col].apply(mean_abs_change)

        

        # 2nd order derivative

        new_data[col + '_mean_change_of_abs_change'] = data.groupby('series_id')[col].apply(mean_change_of_abs_change)

        

        new_data[col + '_abs_max'] = data.groupby('series_id')[col].apply(lambda x: np.max(np.abs(x)))

        new_data[col + '_abs_min'] = data.groupby('series_id')[col].apply(lambda x: np.min(np.abs(x)))



    return new_data
X_train = generate_features(X_train)

X_test = generate_features(X_test)
label_encoder = LabelEncoder()

y_train['surface'] = label_encoder.fit_transform(y_train['surface'])
folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
# https://www.kaggle.com/emanueleamcappella/random-forest-hyperparameters-tuning

classifier = [RandomForestClassifier,AdaBoostClassifier, ExtraTreesClassifier,BaggingClassifier, DecisionTreeClassifier]

classifier_avg = []

for model in classifier:

    print("Model : {}". format(model))

    submission_predictions = np.zeros((X_test.shape[0], 9))

    oof_predictions = np.zeros((X_train.shape[0]))

    score = 0

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train['surface'])):

        clf =  model()

        clf.fit(X_train.iloc[trn_idx], y_train['surface'][trn_idx])

        oof_predictions[val_idx] = clf.predict(X_train.iloc[val_idx])

        submission_predictions += clf.predict_proba(X_test) / folds.n_splits

        score += clf.score(X_train.iloc[val_idx], y_train['surface'][val_idx])

        print('Fold: {} score: {}'.format(fold_,clf.score(X_train.iloc[val_idx], y_train['surface'][val_idx])))

    print('Avg Accuracy', score / folds.n_splits)

    classifier_avg.append(score / folds.n_splits)
temp = pd.DataFrame()

temp["Classifier"] = classifier

temp["Average"] = classifier_avg
plt.figure(figsize = (20,8))

sns.barplot(y = temp["Classifier"], x = temp["Average"], orient='h')
submission_predictions = np.zeros((X_test.shape[0], 9))

oof_predictions = np.zeros((X_train.shape[0]))

score = 0

for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train['surface'])):

    clf =  ExtraTreesClassifier(n_estimators=2000, n_jobs=-1)

    clf.fit(X_train.iloc[trn_idx], y_train['surface'][trn_idx])

    oof_predictions[val_idx] = clf.predict(X_train.iloc[val_idx])

    submission_predictions += clf.predict_proba(X_test) / folds.n_splits

    score += clf.score(X_train.iloc[val_idx], y_train['surface'][val_idx])

    print('Fold: {} score: {}'.format(fold_,clf.score(X_train.iloc[val_idx], y_train['surface'][val_idx])))

print('Avg Accuracy', score / folds.n_splits)
submission = pd.read_csv('../input/sample_submission.csv')

submission['surface'] = label_encoder.inverse_transform(submission_predictions.argmax(axis=1))

submission.to_csv('submission.csv', index=False)

submission.head()