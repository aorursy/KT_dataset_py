import gc

import os

import logging

import datetime

import warnings

import numpy as np

import pandas as pd

import seaborn as sns

import itertools

from lightgbm import LGBMClassifier

from tqdm import tqdm_notebook

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import mean_squared_error, confusion_matrix

from sklearn.metrics import roc_auc_score, roc_curve

from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings('ignore')
IS_LOCAL = False

if(IS_LOCAL):

    PATH="../input/careercon/"

else:

    PATH="../input/"

os.listdir(PATH)
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

total = float(len(y_train))

g = sns.countplot(y_train['surface'], order = y_train['surface'].value_counts().index, palette='Set3')

g.set_title("Number and percentage of labels for each class")

for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:1.2f}%'.format(100*height/total),

            ha="center") 

plt.show()    
f, ax = plt.subplots(1,1, figsize=(18,8))

total = float(len(y_train))

g = sns.countplot(y_train['group_id'], order = y_train['group_id'].value_counts().index, palette='Set3')

g.set_title("Number and percentage of group_id")

for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:1.1f}%'.format(100*height/total),

            ha="center", rotation='90') 

plt.show()    
def plot_feature_distribution(df1, df2, label1, label2, features):

    i = 0

    sns.set_style('whitegrid')

    plt.figure()

    fig, ax = plt.subplots(2,5,figsize=(16,8))



    for feature in features:

        i += 1

        plt.subplot(2,5,i)

        sns.distplot(df1[feature], hist=False, label=label1)

        sns.distplot(df2[feature], hist=False, label=label2)

        plt.xlabel(feature, fontsize=9)

        locs, labels = plt.xticks()

        plt.tick_params(axis='x', which='major', labelsize=8)

        plt.tick_params(axis='y', which='major', labelsize=8)

    plt.show();
features = X_train.columns.values[3:13]

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

            sns.distplot(ttc[feature], hist=False,label=clas)

        plt.xlabel(feature, fontsize=9)

        locs, labels = plt.xticks()

        plt.tick_params(axis='x', which='major', labelsize=8)

        plt.tick_params(axis='y', which='major', labelsize=8)

    plt.show();
classes = (y_train['surface'].value_counts()).index

tt = X_train.merge(y_train, on='series_id', how='inner')

plot_feature_class_distribution(classes, tt, features)
fig, ax = plt.subplots(1,1,figsize=(24,6))

tmp = pd.DataFrame(y_train.groupby(['group_id', 'surface'])['series_id'].count().reset_index())

m = tmp.pivot(index='surface', columns='group_id', values='series_id')

s = sns.heatmap(m, linewidths=.1, linecolor='black', annot=True, cmap="YlGnBu")

s.set_title('Number of surface category per group_id', size=16)

plt.show()
f,ax = plt.subplots(figsize=(6,6))

m = X_train.iloc[:,3:].corr()

sns.heatmap(m, annot=True, linecolor='darkblue', linewidths=.1, cmap="YlGnBu", fmt= '.1f',ax=ax)
f,ax = plt.subplots(figsize=(6,6))

m = X_test.iloc[:,3:].corr()

sns.heatmap(m, annot=True, linecolor='darkblue', linewidths=.1, cmap="YlGnBu", fmt= '.1f',ax=ax)
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

    

    def mean_change_of_abs_change(x):

        return np.mean(np.diff(np.abs(np.diff(x))))



    def mean_abs_change(x):

        return np.mean(np.abs(np.diff(x)))

    

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

        df_out[col + '_mean_abs_change'] = df.groupby('series_id')[col].apply(mean_abs_change)

        df_out[col + '_mean_change_of_abs_change'] = df.groupby('series_id')[col].apply(mean_change_of_abs_change)

        df_out[col + '_abs_max'] = df.groupby('series_id')[col].apply(lambda x: np.max(np.abs(x)))

        df_out[col + '_abs_min'] = df.groupby('series_id')[col].apply(lambda x: np.min(np.abs(x)))

        df_out[col + '_abs_mean'] = df.groupby('series_id')[col].apply(lambda x: np.mean(np.abs(x)))

        df_out[col + '_abs_std'] = df.groupby('series_id')[col].apply(lambda x: np.std(np.abs(x)))

        df_out[col + '_abs_avg'] = (df_out[col + '_abs_min'] + df_out[col + '_abs_max'])/2

        df_out[col + '_abs_range'] = df_out[col + '_abs_max'] - df_out[col + '_abs_min']



    return df_out
%%time

X_train = perform_euler_factors_calculation(X_train)
%%time

X_test = perform_euler_factors_calculation(X_test)
X_train.shape, X_test.shape
features = X_train.columns.values[13:23]

plot_feature_distribution(X_train, X_test, 'train', 'test', features)
classes = (y_train['surface'].value_counts()).index

tt = X_train.merge(y_train, on='series_id', how='inner')

plot_feature_class_distribution(classes, tt, features)
USE_ALL_FEATURES = False

if(USE_ALL_FEATURES):

    features = X_train.columns.values

else:

    features = X_train.columns.values[:13]
%%time

X_train = perform_feature_engineering(X_train[features])
%%time

X_test = perform_feature_engineering(X_test[features])
print("Train X: {}\nTrain y: {}\nTest X: {}".format(X_train.shape, y_train.shape, X_test.shape))
X_train.head()
X_test.head()
%%time

correlations = X_train.corr().abs().unstack().sort_values(kind="quicksort").reset_index()

correlations = correlations[correlations['level_0'] != correlations['level_1']]
correlations.head(10)
correlations.tail(10)
n_top_corr = correlations[correlations[0]==1.0].shape[0]

print("There are {} different features pairs with correlation factor 1.0.".format(n_top_corr))
drop_features = list(correlations.head(n_top_corr)['level_0'].unique())

X_train = X_train.drop(drop_features,axis=1)

X_test = X_test.drop(drop_features,axis=1)
print("Train X: {}\nTrain y: {}\nTest X: {}".format(X_train.shape, y_train.shape, X_test.shape))
corr = X_train.corr()

fig, ax = plt.subplots(1,1,figsize=(16,16))

sns.heatmap(corr,  xticklabels=False, yticklabels=False)

plt.show()
le = LabelEncoder()

y_train['surface'] = le.fit_transform(y_train['surface'])
X_train.fillna(0, inplace = True)

X_train.replace(-np.inf, 0, inplace = True)

X_train.replace(np.inf, 0, inplace = True)

X_test.fillna(0, inplace = True)

X_test.replace(-np.inf, 0, inplace = True)

X_test.replace(np.inf, 0, inplace = True)
scaler = StandardScaler()

X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train))

X_test_scaled = pd.DataFrame(scaler.transform(X_test))

print ("Scaled !")
folds = StratifiedKFold(n_splits=49, shuffle=True, random_state=2018)
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
def plot_confusion_matrix(actual, predicted, classes, title='Confusion Matrix'):

    conf_matrix = confusion_matrix(actual, predicted)

    

    plt.figure(figsize=(8, 8))

    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Greens)

    plt.title(title, size=12)

    plt.colorbar(fraction=0.05, pad=0.05)

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=90)

    plt.yticks(tick_marks, classes)



    thresh = conf_matrix.max() / 2.

    for i, j in itertools.product(range(conf_matrix.shape[0]), range(conf_matrix.shape[1])):

        plt.text(j, i, format(conf_matrix[i, j], 'd'),

        horizontalalignment="center", color="white" if conf_matrix[i, j] > thresh else "black")



    plt.ylabel('Actual label')

    plt.xlabel('Predicted label')

    plt.grid(False)

    plt.tight_layout()
plot_confusion_matrix(y_train['surface'], oof_preds_rf, le.classes_, title='Confusion Matrix')
USE_LGB = True

if(USE_LGB):

    sub_preds_lgb = np.zeros((X_test.shape[0], 9))

    oof_preds_lgb = np.zeros((X_train.shape[0]))

    score = 0

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train['surface'])):

        train_x, train_y = X_train.iloc[trn_idx], y_train['surface'][trn_idx]

        valid_x, valid_y = X_train.iloc[val_idx], y_train['surface'][val_idx]

        clf =  LGBMClassifier(

                      nthread=-1,

                      n_estimators=2000,

                      learning_rate=0.01,

                      boosting_type='gbdt',

                      is_unbalance=True,

                      objective='multiclass',

                      numclass=9,

                      silent=-1,

                      verbose=-1,

                      feval=None)

        clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)], 

                     verbose= 1000, early_stopping_rounds= 200)



        oof_preds_lgb[val_idx] = clf.predict(valid_x)

        sub_preds_lgb += clf.predict_proba(X_test) / folds.n_splits

        score += clf.score(valid_x, valid_y)

        print('Fold: {} score: {}'.format(fold_,clf.score(valid_x, valid_y)))

    print('Avg Accuracy', score / folds.n_splits)
submission = pd.read_csv(os.path.join(PATH,'sample_submission.csv'))

submission['surface'] = le.inverse_transform(sub_preds_rf.argmax(axis=1))

submission.to_csv('submission_rf.csv', index=False)

submission.head(10)
USE_LGB = True

if(USE_LGB):

    submission['surface'] = le.inverse_transform(sub_preds_lgb.argmax(axis=1))

    submission.to_csv('submission_lgb.csv', index=False)

    submission.head(10)