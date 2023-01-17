import numpy as np

import pandas as pd

import os

import gc

import matplotlib.pyplot as plt

# %matplotlib inline

from sklearn.preprocessing import StandardScaler

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

import xgboost as xgb

import lightgbm as lgb

from seaborn import countplot,lineplot, barplot

from sklearn import model_selection

from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

import itertools

import json

import time

from sklearn import linear_model

import eli5

from eli5.sklearn import PermutationImportance

from mlxtend.feature_selection import SequentialFeatureSelector as SFS

from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs

import shap

from sklearn.neighbors import NearestNeighbors

from sklearn.feature_selection import GenericUnivariateSelect, SelectPercentile, SelectKBest, f_classif, mutual_info_classif, RFE

import statsmodels.api as sm

import warnings

warnings.filterwarnings('ignore')

from catboost import CatBoostClassifier

from collections import Counter

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2

from sklearn.metrics import confusion_matrix

pd.set_option('max_columns', None)

import datetime

import seaborn as sns

import lightgbm as lgb

from scipy import stats

from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_val_score, GridSearchCV, RepeatedStratifiedKFold

IS_LOCAL = False

if (IS_LOCAL):

    location = "../input/careercon/"

else:

    location = "../input/"

os.listdir(location)
%%time

X_train = pd.read_csv(os.path.join(location, 'X_train.csv'))

X_test = pd.read_csv(os.path.join(location, 'X_test.csv'))

y_train = pd.read_csv(os.path.join(location, 'y_train.csv'))
print("Xtrain: {}\nXtest: {}\nytrain: {}".format(X_train.shape, X_test.shape, y_train.shape))
print('Size of the Xtrain')

print('Numbers of Measurements: {0}\nNumbers of columns: {1}'.format(X_train.shape[0], X_train.shape[1]))
print('Size of the Xtest')

print('Numbers of Measurements: {0}\nNumbers of columns: {1}'.format(X_test.shape[0], X_test.shape[1]))
print('Size of the Labels')

print('Numbers of Measurements: {0}\nNumbers of columns: {1}'.format(y_train.shape[0], y_train.shape[1]))
def show_head(data):

    return(data.head())
show_head(X_train)
show_head(X_test)
show_head(y_train)
def missing_data(data):

    total = data.isnull().sum()

    percent = (total/data.isnull().count()*100)

    miss_column = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    types = []

    for col in data.columns:

        dtype = str(data[col].dtype)

        types.append(dtype)

    miss_column['Types'] = types

    return(np.transpose(miss_column))            
missing_data(X_train)
missing_data(X_test)
missing_data(y_train)
def describe_data(data):

    return(data.describe())
describe_data(X_train)
describe_data(X_test)
describe_data(y_train)
Surface_count = y_train['surface'].value_counts().reset_index().rename(columns = {'index' : 'Labels'})

Surface_count
countplot(y = 'surface', data = y_train)

plt.show()
trace = go.Pie(labels = y_train['surface'].value_counts().index,

              values = y_train['surface'].value_counts().values,

              domain = {'x':[0.55,1]})



data = [trace]

layout = go.Layout(title = 'PieChat Distribution of Floors')

fig = go.Figure(data = data, layout = layout)

py.iplot(fig)
def plot_columns_distribution(df1, df2, label1, label2, columns):

    i = 0

    sns.set_style('whitegrid')

    plt.figure()

    fig, ax = plt.subplots(2,5,figsize=(16,8))



    for col in columns:

        i += 1

        plt.subplot(2,5,i)

        sns.kdeplot(df1[col], bw=0.5,label=label1)

        sns.kdeplot(df2[col], bw=0.5,label=label2)

        plt.xlabel(col, fontsize=9)

        locs, labels = plt.xticks()

        plt.tick_params(axis='x', which='major', labelsize=8)

        plt.tick_params(axis='y', which='major', labelsize=8)

    plt.show();

    

columns = X_train.columns.values[3:]

plot_columns_distribution(X_train, X_test, 'train', 'test', columns)
def plot_columns_class_distribution(classes,series_group, columns):

    i = 0

    sns.set_style('whitegrid')

    plt.figure()

    fig, ax = plt.subplots(5,2,figsize=(16,24))



    for col in columns:

        i += 1

        plt.subplot(5,2,i)

        for clas in classes:

            series_groups = series_group[series_group['surface']==clas]

            sns.kdeplot(series_groups[col], bw=0.5,label=clas)

        plt.xlabel(col, fontsize=9)

        locs, labels = plt.xticks()

        plt.tick_params(axis='x', which='major', labelsize=8)

        plt.tick_params(axis='y', which='major', labelsize=8)

    plt.show();

    

classes = (y_train['surface'].value_counts()).index

series_group = X_train.merge(y_train, on='series_id', how='inner')

plot_columns_class_distribution(classes, series_group, columns)
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



def data_engineering(actual):

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

xtrain = data_engineering(X_train)

xtest = data_engineering(X_test)
describe_data(xtrain)
print("Xtrain: {}\nXtest: {}".format(X_train.shape, X_test.shape))
print("Xtrain: {}\nXtest: {}".format(xtrain.shape, xtest.shape))
show_head(xtest)
corr_xtrain = xtrain.corr()

corr_xtrain
colormap = plt.cm.RdBu

plt.figure(figsize=(24,18))

plt.title('Pearson Correlation of Features', y=1.05, size=20)

sns.heatmap(X_train.astype(float).corr(),linewidths=0.05,vmax=1.0, 

            square=True, cmap=colormap, linecolor='white', annot=True)
le = LabelEncoder()

y_train['surface'] = le.fit_transform(y_train['surface'])
xtrain.fillna(0, inplace = True)

xtrain.replace(-np.inf, 0, inplace = True)

xtrain.replace(np.inf, 0, inplace = True)

xtest.fillna(0, inplace = True)

xtest.replace(-np.inf, 0, inplace = True)

xtest.replace(np.inf, 0, inplace = True)


folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=11)

def feed_model(train,test,label,folds=folds,averaging='usual',clf=None,clf_type='rfc',params=None,

               plot_feature_importance=False,groups=y_train['group_id']):

    forcast = np.zeros((test.shape[0], 9))

    con_pred = np.zeros((train.shape[0]))

    scores = []

    feature_importance = pd.DataFrame()

    for fold_, (train_index, valid_index) in enumerate(folds.split(train, label, groups)):

        print('Fold', fold_, 'started at', time.ctime())

        x_train, x_valid = train.iloc[train_index], train.iloc[valid_index]

        y_train, y_valid = label.iloc[train_index], label.iloc[valid_index]

        

        if clf_type == 'rfc':

            clf = clf

            clf.fit(x_train, y_train)

            Valid_pred = clf.predict(x_valid).reshape(-1,)

            score = accuracy_score(y_valid, Valid_pred)

            Real_pred = clf.predict_proba(test)

            

        con_pred[valid_index] = clf.predict(x_valid).reshape(-1,)

        scores.append(accuracy_score(y_valid, Valid_pred))

        if averaging == 'usual':

            forcast += Real_pred

        elif averaging == 'rank':

            forcast += pd.Series(Real_pred).rank().values

        

    forcast /= folds.n_splits

    

    print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))

    

    if clf_type == 'lgb':

        feature_importance["importance"] /= folds.n_splits

        if plot_feature_importance:

            cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance",

                                                                                                       ascending=False)[:50].index



            best_features = feature_importance[feature_importance.feature.isin(cols)]



            plt.figure(figsize=(16, 12));

            sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False));

            plt.title('LGB Features (avg over folds)');

        

            return Valid_pred, forcast, feature_importance

        return Valid_pred, forcast, scores

    

    else:

        return Valid_pred, forcast, scores

clf = RandomForestClassifier(n_estimators=500,n_jobs=-1,random_state=0)

con_pred_rfc1, forcast_rfc1, scores_rfc1 = feed_model(train=xtrain, test=xtest, label=y_train['surface'],folds=folds,clf_type='rfc',

                                                   plot_feature_importance=True,clf=clf)
FeatureImportance = clf.feature_importances_

indices = np.argsort(FeatureImportance)

features = xtrain.columns



hm = 30

plt.figure(figsize=(16, 12))

plt.title('RFC Features Avg Over Folds')

plt.barh(range(len(indices[:hm])), FeatureImportance[indices][:hm], color='b', align='center')

plt.yticks(range(len(indices[:hm])), [features[i] for i in indices])

plt.xlabel('Importance')

plt.show()
feat_labels = xtrain.columns
for feature in zip(feat_labels, clf.feature_importances_):

    print(feature)
from sklearn.feature_selection import SelectFromModel

from sklearn.metrics import accuracy_score

from sklearn import datasets

sfm = SelectFromModel(clf, threshold=0.001)

sfm.fit(xtrain, y_train['surface'])

for feature_list_index in sfm.get_support(indices=True):

    print(feat_labels[feature_list_index])

    

xtrain_importance = sfm.transform(xtrain)

xtest_importance = sfm.transform(xtest)
folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=11)

def feed_model_importance(train,test,label,folds=folds,averaging='usual',clf=None,clf_type='rfc',params=None,

               plot_feature_importance=False,groups=y_train['group_id']):

    forcast = np.zeros((test.shape[0], 9))

    con_pred = np.zeros((train.shape[0]))

    scores = []

    feature_importance = pd.DataFrame()

    for fold_n, (train_index, valid_index) in enumerate(folds.split(train, label, groups)):

        print('Fold', fold_n, 'started at', time.ctime())

        x_train, x_valid = train[train_index], train[valid_index]

        y_train, y_valid = label[train_index], label[valid_index]

        

        if clf_type == 'rfc':

            clf = clf

            clf.fit(x_train, y_train)

            Valid_pred = clf.predict(x_valid).reshape(-1,)

            score = accuracy_score(y_valid, Valid_pred)

            Real_pred = clf.predict_proba(test)

            

        con_pred[valid_index] += (Valid_pred).reshape(-1,)

        scores.append(accuracy_score(y_valid, Valid_pred))

        if averaging == 'usual':

            forcast += Real_pred

        elif averaging == 'rank':

            forcast += pd.Series(Real_pred).rank().values

        

    forcast /= folds.n_splits

    

    print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))

    

    if clf_type == 'lgb':

        feature_importance["importance"] /= n_folds

        if plot_feature_importance:

            cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(

                by="importance", ascending=False)[:50].index



            best_features = feature_importance.loc[feature_importance.feature.isin(cols)]



            plt.figure(figsize=(16, 12));

            sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False));

            plt.title('LGB Features (avg over folds)');

        

            return Valid_pred, forcast, feature_importance

        return Valid_pred, forcast, scores

    

    else:

        return Valid_pred, forcast, scores

clf1 = RandomForestClassifier(n_estimators=500,n_jobs=-1,random_state=0)

con_pred_rfc, forcast_rfc, scores_rfc = feed_model_importance(train=xtrain_importance, test=xtest_importance, label=y_train['surface'],folds=folds,clf_type='rfc',

                                                   plot_feature_importance=False,clf=clf1)
from keras import models

from keras import layers

from keras.callbacks import EarlyStopping

from keras.callbacks import ModelCheckpoint

from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import cross_val_score

np.random.seed(0)



folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=11)



def create_network():

    network = models.Sequential()

    network.add(layers.Dense(units=16, activation='relu', input_dim= 171))

    network.add(layers.Dense(units=16, activation='relu'))

    network.add(layers.Dense(units=1, activation='sigmoid'))



    network.compile(loss='binary_crossentropy',

                    optimizer='adam',

                    metrics=['accuracy'])

    

    # Return compiled network

    return network

# Wrap Keras model so it can be used by scikit-learn

es = EarlyStopping(monitor='val_loss',mode='auto',verbose=1,patience=20)

mc = ModelCheckpoint('best_model.h5',monitor='val_acc',mode='max',verbose=1,save_best_only=True)

neural_network = KerasClassifier(build_fn=create_network,epochs=10,batch_size=100,verbose=0)

cross_val_score(neural_network, xtrain, y_train['surface'],fit_params={'callbacks': [es,mc]}, cv=folds)
sub = pd.read_csv(os.path.join(location,'sample_submission.csv'))

sub['surface'] = le.inverse_transform(forcast_rfc.argmax(axis=1))

sub.to_csv('Forcasting.csv', index=False)

sub.head(20)