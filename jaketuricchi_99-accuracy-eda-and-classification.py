# -*- coding: utf-8 -*-

"""

Created on Wed Jun 10 08:59:51 2020
@author: jaket

"""

#j2p

### Exercise pattern classification
import math

import pandas as pd

import numpy as np

import os

import matplotlib.pyplot as plt

import warnings 

import seaborn as sns

import sklearn

from datetime import datetime

import calendar

%matplotlib inline

pd.set_option('display.max_rows', 1000)

import warnings

warnings.filterwarnings('ignore')
train_df = pd.read_csv('../input/exercisepatternpredict/pml-training.csv', error_bad_lines=False, index_col=False).drop('Unnamed: 0', axis=1)

test_df = pd.read_csv('../input/exercisepatternpredict/pml-testing.csv', error_bad_lines=False, index_col=False).drop('Unnamed: 0', axis=1)
train_df = train_df.sample(frac=1).reset_index(drop=True)
print(train_df.columns.values)

print(train_df.isna().sum()) 
sns.heatmap(train_df.isnull(), cbar=False) # Heatmap to visualise NAs
train_df.describe()
train_df['classe']=train_df['classe'].astype('category')
freq_plot1=train_df.filter(items=['user_name', 'classe'])

freq_plot1=freq_plot1.groupby(['user_name'])['classe'].agg(counts='value_counts').reset_index()
sns.barplot(data = freq_plot1, x = 'counts', y = 'user_name', hue = 'classe', ci = None)
            

pairplot1=train_df.filter(items=['num_window', 'roll_belt', 'pitch_belt', 'yaw_belt', 'total_accel_belt', 'classe'])

sns.pairplot(pairplot1, hue='classe',  plot_kws = {'alpha': 0.6, 'edgecolor': 'k'},size = 4)
pairplot2=train_df.filter(items=['num_window', 'gyros_belt_x', 'gyros_belt_y', 'accel_belt_x', 'accel_belt_y',  'magnet_belt_x','magnet_belt_y', 'classe'])

sns.pairplot(pairplot2, hue='classe',  plot_kws = {'alpha': 0.6,  'edgecolor': 'k'},size = 4)
pairplot3=train_df.filter(items=['num_window', 'gyros_belt_z', 'accel_belt_z', 'magnet_belt_z', 'classe'])

sns.pairplot(pairplot3, hue='classe',  plot_kws = {'alpha': 0.6, 'edgecolor': 'k'},size = 4)
pairplot4=train_df.filter(items=['roll_arm', 'pitch_arm', 'yaw_arm', 'total_accel_arm', 'classe'])

sns.pairplot(pairplot4, hue='classe',  plot_kws = {'alpha': 0.6, 'edgecolor': 'k'},size = 4)
pairplot5=train_df.filter(items=['num_window', 'gyros_arm_x', 'gyros_arm_y', 'accel_arm_x', 'accel_arm_y',  'magnet_arm_x','magnet_arm_y', 'classe'])

sns.pairplot(pairplot5, hue='classe',  plot_kws = {'alpha': 0.6,  'edgecolor': 'k'},size = 4)
pairplot6=train_df.filter(items=['num_window', 'gyros_arm_z', 'accel_arm_z', 'magnet_arm_z', 'classe'])

sns.pairplot(pairplot6, hue='classe',  plot_kws = {'alpha': 0.6, 'edgecolor': 'k'},size = 4)
pairplot7=train_df.filter(items=['skewness_roll_belt', 'max_roll_belt', 'max_picth_belt', 

                                 'var_total_accel_belt', 'stdev_roll_belt',

                                 'avg_yaw_belt', 'classe'])

sns.pairplot(pairplot7, hue='classe',  plot_kws = {'alpha': 0.6, 'edgecolor': 'k'},size = 4)
print(train_df.isna().sum()) 

train_df = train_df.loc[:, train_df.isnull().mean() < .8] #remove cols with <80% completeness.

test_df = test_df.loc[:, test_df.isnull().mean() < .8] #remove cols with <80% completeness.
train_df = train_df.drop(['raw_timestamp_part_1', 'raw_timestamp_part_2' ,'cvtd_timestamp', 'new_window','num_window'], axis=1)

test_df = test_df.drop(['raw_timestamp_part_1', 'raw_timestamp_part_2' ,'cvtd_timestamp', 'new_window','num_window', 'problem_id'], axis=1)
def zeros_to_ones(x):

    x = np.where(x==0, 1, x)

    return(x)
def feat_eng (df):

    df['x_axis_feat']=df[df.columns[df.columns.to_series().str.contains('_x')]].apply(zeros_to_ones).apply(np.prod, axis=1)

    df['y_axis_feat']=df[df.columns[df.columns.to_series().str.contains('_y')]].apply(zeros_to_ones).apply(np.prod, axis=1)

    df['z_axis_feat']=df[df.columns[df.columns.to_series().str.contains('_z')]].apply(zeros_to_ones).apply(np.prod, axis=1)

    

    # Lets interact all belt, arm, dumbell and forearm variables

    

    df['belt_feat']=df[df.columns[df.columns.to_series().str.contains('_belt')]].apply(zeros_to_ones).apply(np.prod, axis=1)

    df['arm_feat']=df[df.columns[df.columns.to_series().str.contains('_arm')]].apply(zeros_to_ones).apply(np.prod, axis=1)

    df['forearm_feat']=df[df.columns[df.columns.to_series().str.contains('_forearm')]].apply(zeros_to_ones).apply(np.prod, axis=1)

    

    # Let's interact all magnet, accel and gyros variables

    

    df['accel_feat']=df[df.columns[df.columns.to_series().str.contains('accel_')]].apply(zeros_to_ones).apply(np.prod, axis=1)

    df['magnet_feat']=df[df.columns[df.columns.to_series().str.contains('magnet_')]].apply(zeros_to_ones).apply(np.prod, axis=1)

    df['gyros_feat']=df[df.columns[df.columns.to_series().str.contains('gyros_')]].apply(zeros_to_ones).apply(np.prod, axis=1)

    

    return(df)
train_df=feat_eng(train_df)

test_df=feat_eng(test_df)
def Encode_fn(df):

    users=pd.get_dummies(df['user_name']) #OneHot encode username

    df=pd.concat([df, users], axis=1).reset_index(drop=True) #Join to modelling df

    df=df.drop('user_name', axis=1) #Drop original username var

    return(df)
train_df=Encode_fn(train_df)

test_df=Encode_fn(test_df)
train_df['classe']=train_df['classe'].astype('category') # Ensure the target is cat

train_df['target']=train_df['classe'].cat.codes # Label encoding

train_df['target']=train_df['target'].astype('category') # Ensure the target is cat

train_df=train_df.drop('classe', axis=1)
from sklearn.model_selection import train_test_split
X=train_df.drop('target', axis=1).reset_index(drop=True)

y=train_df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,  QuadraticDiscriminantAnalysis

from sklearn.svm import SVC, LinearSVC, NuSVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import confusion_matrix, accuracy_score, log_loss, precision_score, recall_score, f1_score
classifiers = [

    KNeighborsClassifier(3),

    SVC(kernel="rbf", C=0.025, probability=True),

    DecisionTreeClassifier(),

    RandomForestClassifier(),

    AdaBoostClassifier(),

    GradientBoostingClassifier(),

    GaussianNB(),

    LinearDiscriminantAnalysis(),

    QuadraticDiscriminantAnalysis()]
log_cols=["Classifier", "Accuracy", "Log Loss"]

log = pd.DataFrame(columns=log_cols)
for clf in classifiers:

    clf.fit(X_train, y_train)

    name = clf.__class__.__name__

    

    print("="*30)

    print(name)

    

    print('****Results****')

    train_predictions = clf.predict(X_test)

    acc = accuracy_score(y_test, train_predictions)

    

    # calculate score

    precision = precision_score(y_test, train_predictions, average = 'macro') 

    recall = recall_score(y_test, train_predictions, average = 'macro') 

    f_score = f1_score(y_test, train_predictions, average = 'macro')

    

    

    print("Precision: {:.4%}".format(precision))

    print("Recall: {:.4%}".format(recall))

    print("F-score: {:.4%}".format(recall))

    print("Accuracy: {:.4%}".format(acc))

    

    train_predictions = clf.predict_proba(X_test)

    ll = log_loss(y_test, train_predictions)

    print("Log Loss: {}".format(ll))

    

    log_entry = pd.DataFrame([[name, acc*100, ll]], columns=log_cols)

    log = log.append(log_entry)

    

print("="*30)
sns.barplot(x='Accuracy', y='Classifier', data=log, color="b")
sns.set_color_codes("muted")

sns.barplot(x='Log Loss', y='Classifier', data=log, color="g")
rf = RandomForestClassifier(n_estimators=500, random_state = 42)

rf.fit(X_train, y_train);

feat_importances = pd.Series(rf.feature_importances_, index=X_train.columns)

feat_importances.nlargest(25).plot(kind='barh')
from sklearn.model_selection import RandomizedSearchCV
n_estimators = [int(x) for x in np.linspace(start = 10, stop = 20, num = 10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 1000, num = 10)]

max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [2, 4, 10, 100]
bootstrap = [True, False]
random_grid = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf,

               'bootstrap': bootstrap}

print(random_grid)
rf_random = RandomizedSearchCV(estimator = rf, 

                               param_distributions = random_grid, 

                               n_iter = 100, cv = 3, verbose=2, 

                               random_state=42, n_jobs = -1)

# Fit the random search model

rf_random.fit(X_train, y_train)
print(rf_random.best_params_)
best_params_rf = rf_random.best_estimator_

best_params_rf.fit(X_train,y_train)
y_pred_rf = best_params_rf.predict(X_test)
precision = precision_score(y_test, y_pred_rf, average = 'macro') 

recall = recall_score(y_test, y_pred_rf, average = 'macro') 

f_score = f1_score(y_test, y_pred_rf, average = 'macro')

    

    

print("Precision: {:.4%}".format(precision))

print("Recall: {:.4%}".format(recall))

print("F-score: {:.4%}".format(recall))

final_predictions = best_params_rf.predict(test_df)
print(final_predictions)
! p2j Exercise classification.py
 