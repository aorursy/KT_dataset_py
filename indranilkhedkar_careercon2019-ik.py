# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.|

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.



import math

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import StratifiedKFold

from sklearn.ensemble import RandomForestClassifier
X_train=pd.read_csv('../input/X_train.csv')

y_train=pd.read_csv('../input/y_train.csv')

X_test=pd.read_csv('../input/X_test.csv')

y_test=pd.read_csv('../input/sample_submission.csv')
X_train.describe()
y_train.describe()
X_test.describe()
y_test.describe()
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
def perform_euler_factors_calculation(df):

    df['total_angular_velocity'] = np.sqrt(np.square(df['angular_velocity_X']) + np.square(df['angular_velocity_Y']) + np.square(df['angular_velocity_Z']))

    df['total_linear_acceleration'] = np.sqrt(np.square(df['linear_acceleration_X']) + np.square(df['linear_acceleration_Y']) + np.square(df['linear_acceleration_Z']))

    df['total_xyz'] = np.sqrt(np.square(df['orientation_X']) + np.square(df['orientation_Y']) +

                              np.square(df['orientation_Z']))

    df['acc_vs_vel'] = df['total_linear_acceleration'] / df['total_angular_velocity']

    

    x, y, z, w = df['orientation_X'].tolist(), df['orientation_Y'].tolist(), df['orientation_Z'].tolist(), df['orientation_W'].tolist()

    nx, ny, nz = [], [], []

    for i in range(len(x)):

        xx, yy, zz = quaternion_to_euler(x[i], y[i], z[i], w[i])

        nx.append(xx)

        ny.append(yy)

        nz.append(zz)

    

    df['euler_x'] = nx

    df['euler_y'] = ny

    df['euler_z'] = nz

    

    df['total_angle'] = np.sqrt(np.square(df['euler_x']) + np.square(df['euler_y']) + np.square(df['euler_z']))

    df['angle_vs_acc'] = df['total_angle'] / df['total_linear_acceleration']

    df['angle_vs_vel'] = df['total_angle'] / df['total_angular_velocity']

    return df
def perform_feature_engineering(df):

    df_out = pd.DataFrame()

    

    for col in df.columns:

        if col in ['row_id', 'series_id', 'measurement_number']:

            continue

        df_out[col + '_mean'] = df.groupby(['series_id'])[col].mean()

        df_out[col + '_min'] = df.groupby(['series_id'])[col].min()

        df_out[col + '_max'] = df.groupby(['series_id'])[col].max()

        df_out[col + '_std'] = df.groupby(['series_id'])[col].std()

        df_out[col + '_mad'] = df.groupby(['series_id'])[col].mad()

        df_out[col + '_med'] = df.groupby(['series_id'])[col].median()

        df_out[col + '_skew'] = df.groupby(['series_id'])[col].skew()

        df_out[col + '_range'] = df_out[col + '_max'] - df_out[col + '_min']

        df_out[col + '_max_to_min'] = df_out[col + '_max'] / df_out[col + '_min']

        df_out[col + '_mean_abs_change'] = df.groupby('series_id')[col].apply(lambda x: np.mean(np.abs(np.diff(x))))

        df_out[col + '_mean_change_of_abs_change'] = df.groupby('series_id')[col].apply(lambda x: np.mean(np.diff(np.abs(np.diff(x)))))

        df_out[col + '_abs_max'] = df.groupby('series_id')[col].apply(lambda x: np.max(np.abs(x)))

        df_out[col + '_abs_min'] = df.groupby('series_id')[col].apply(lambda x: np.min(np.abs(x)))

        df_out[col + '_abs_mean'] = df.groupby('series_id')[col].apply(lambda x: np.mean(np.abs(x)))

        df_out[col + '_abs_std'] = df.groupby('series_id')[col].apply(lambda x: np.std(np.abs(x)))

        df_out[col + '_abs_avg'] = (df_out[col + '_abs_min'] + df_out[col + '_abs_max'])/2

        df_out[col + '_abs_range'] = df_out[col + '_abs_max'] - df_out[col + '_abs_min']



    return df_out
X_train = perform_euler_factors_calculation(X_train)
X_test = perform_euler_factors_calculation(X_test)
X_train.shape, X_test.shape
X_train = perform_feature_engineering(X_train[X_train.columns.values[:13]])
X_test = perform_feature_engineering(X_test[X_test.columns.values[:13]])
print("Train X: {}\nTrain y: {}\nTest X: {}".format(X_train.shape, y_train.shape, X_test.shape))
le = LabelEncoder()

y_train['surface'] = le.fit_transform(y_train['surface'])
scaler = StandardScaler()

X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train))

X_test_scaled = pd.DataFrame(scaler.transform(X_test))
folds = StratifiedKFold(n_splits=49, shuffle=True, random_state=41)
folds
sub_preds_rf = np.zeros((X_test_scaled.shape[0], 9))

oof_preds_rf = np.zeros((X_train_scaled.shape[0]))

score = 0

for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train_scaled, y_train['surface'])):

    clf =  RandomForestClassifier(n_estimators = 500, n_jobs = -1)

    clf.fit(X_train_scaled.iloc[trn_idx], y_train['surface'][trn_idx])

    oof_preds_rf[val_idx] = clf.predict(X_train_scaled.iloc[val_idx])

    sub_preds_rf += clf.predict_proba(X_test_scaled) / folds.n_splits

    score += clf.score(X_train_scaled.iloc[val_idx], y_train['surface'][val_idx])

    print('Fold: {} score: {}'.format(fold_,clf.score(X_train_scaled.iloc[val_idx], y_train['surface'][val_idx])))

print('Avg Accuracy', score / folds.n_splits)
sub_preds_rf
from xgboost import XGBClassifier

sub_preds_xgboost = np.zeros((X_test_scaled.shape[0], 9))

oof_preds_xgboost = np.zeros((X_train_scaled.shape[0]))

score = 0

for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train_scaled, y_train['surface'])):

    xgb_clf =  XGBClassifier(n_jobs = -1)

    xgb_clf.fit(X_train_scaled.iloc[trn_idx], y_train['surface'][trn_idx])

    oof_preds_xgboost[val_idx] = xgb_clf.predict(X_train_scaled.iloc[val_idx])

    sub_preds_xgboost += xgb_clf.predict_proba(X_test_scaled) / folds.n_splits

    score += xgb_clf.score(X_train_scaled.iloc[val_idx], y_train['surface'][val_idx])

    print('Fold: {} score: {}'.format(fold_,xgb_clf.score(X_train_scaled.iloc[val_idx], y_train['surface'][val_idx])))

print('Avg Accuracy', score / folds.n_splits)
sub_preds_xgboost
y_test_pred_final = np.array(sub_preds_rf.argmax(axis=1)*0.5+sub_preds_xgboost.argmax(axis=1)*0.5)
y_test_pred_final.shape
y_test_pred_final = np.array([int(val) for val in y_test_pred_final])
y_test_pred_final
y_test['surface'] = le.inverse_transform(y_test_pred_final)

y_test.to_csv('submission_18.csv', index=False)
y_test