# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

for f in os.listdir('../input'):

    print(f.ljust(30) + str(round(os.path.getsize('../input/' + f) / 1000000, 2)) + 'MB')





# Any results you write to the current directory are saved as output.
x_df = pd.read_csv('../input/X_train.csv')

x_test_df = pd.read_csv('../input/X_test.csv')

y_df = pd.read_csv('../input/y_train.csv')





x_df.head()
y_df.head()
x_df.describe()
x_test_df.describe()
y_df.describe()
len(y_df['series_id'].value_counts())
len(x_df['series_id'].value_counts())
len(x_df['measurement_number'].value_counts())
print(y_df.head())



print(x_df.isnull().values.any())

print(x_test_df.isnull().values.any())

print(y_df.isnull().values.any())
import seaborn as sns

import matplotlib.pyplot as plt



sns.set(style='darkgrid')

sns.countplot(y = 'surface',

              data = y_df,

              order = y_df['surface'].value_counts().index)

plt.show()
import math

def phi(x):    

    return math.atan2(x['orientation_W']*x['orientation_X'] + x['orientation_Y']*x['orientation_Z'], 1-2*(x['orientation_X']**2+x['orientation_Y']**2))



def theta(x):

    return math.asin(2*(x['orientation_W']*x['orientation_Y'] - x['orientation_Z']*x['orientation_X']))



def chi(x):

    return math.atan2(2*(x['orientation_W']*x['orientation_Z']+x['orientation_X']*x['orientation_Y']), 1-2*(x['orientation_Y']**2+x['orientation_Z']**2))
x_df['phi'] = x_df.apply(phi, axis=1)

x_df['theta'] = x_df.apply(theta, axis=1)

x_df['chi'] = x_df.apply(chi, axis=1)



x_test_df['phi'] = x_test_df.apply(phi, axis=1)

x_test_df['theta'] = x_test_df.apply(theta, axis=1)

x_test_df['chi'] = x_test_df.apply(chi, axis=1)



x_df.head()
f,ax = plt.subplots(figsize=(8, 8))

sns.heatmap(x_df.iloc[:,3:].corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
def plot_feature_distribution(df1, df2, label1, label2, features,a=3,b=6):

    i = 0

    sns.set_style('whitegrid')

    plt.figure()

    fig, ax = plt.subplots(a,b,figsize=(17,9))



    for feature in features:

        i += 1

        plt.subplot(a,b,i)

        sns.kdeplot(df1[feature], bw=0.5,label=label1)

        sns.kdeplot(df2[feature], bw=0.5,label=label2)

        plt.xlabel(feature, fontsize=9)

        locs, labels = plt.xticks()

        plt.tick_params(axis='x', which='major', labelsize=8)

        plt.tick_params(axis='y', which='major', labelsize=8)

    plt.show();
features = x_df.columns.values[3:]

plot_feature_distribution(x_df, x_test_df, 'train', 'test', features)
def plot_feature_class_distribution(classes,tt, features,a=5,b=3):

    i = 0

    sns.set_style('whitegrid')

    plt.figure()

    fig, ax = plt.subplots(a,b,figsize=(16,24))



    for feature in features:

        

        i += 1

        plt.subplot(a,b,i)

        for clas in classes:

            ttc = tt[tt['surface']==clas]

            sns.kdeplot(ttc[feature], bw=0.5,label=clas)

        plt.xlabel(feature, fontsize=9)

        locs, labels = plt.xticks()

        plt.tick_params(axis='x', which='major', labelsize=8)

        plt.tick_params(axis='y', which='major', labelsize=8)

    plt.show();
plt.clf()

plt.cla()

plt.close()

classes = (y_df['surface'].value_counts()).index

print(x_df.head())

data = x_df.merge(y_df, on='series_id', how='inner')

plot_feature_class_distribution(classes, data, features)

x_df.head()


x_df['chi_new'] = x_df['chi']*np.log10(np.absolute(x_df['chi']))

x_test_df['chi_new'] = x_test_df['chi']*np.log10(np.absolute(x_test_df['chi']))







sns.kdeplot(x_df['chi_new'], bw=0.5,label='train')

sns.kdeplot(x_test_df['chi_new'], bw=0.5,label='test')

plt.xlabel('chi_new', fontsize=9)

locs, labels = plt.xticks()

plt.tick_params(axis='x', which='major', labelsize=8)

plt.tick_params(axis='y', which='major', labelsize=8)

x_df['orientation_X_new'] = x_df['orientation_X']*np.absolute(x_df['orientation_X']) #np.log2(x_df['orientation_X'])

x_test_df['orientation_X_new'] = x_test_df['orientation_X']*np.absolute(x_test_df['orientation_X'])#np.log2(x_test_df['orientation_X'])



sns.kdeplot(x_df['orientation_X_new'], bw=0.5,label='train')

sns.kdeplot(x_test_df['orientation_X_new'], bw=0.5,label='test')

plt.xlabel('orientation_X_new', fontsize=9)

locs, labels = plt.xticks()

plt.tick_params(axis='x', which='major', labelsize=8)

plt.tick_params(axis='y', which='major', labelsize=8)

x_df['orientation_Y_new'] = x_df['orientation_Y']*np.absolute(x_df['orientation_Y'])

x_test_df['orientation_Y_new'] = x_test_df['orientation_Y']*np.absolute(x_test_df['orientation_Y'])



sns.kdeplot(x_df['orientation_Y_new'], bw=0.5,label='train')

sns.kdeplot(x_test_df['orientation_Y_new'], bw=0.5,label='test')

plt.xlabel('orientation_Y_new', fontsize=9)

locs, labels = plt.xticks()

plt.tick_params(axis='x', which='major', labelsize=8)

plt.tick_params(axis='y', which='major', labelsize=8)

def normalize_quaternions (data):

    data['norm_quat'] = (data['orientation_X']**2 + data['orientation_Y']**2 + data['orientation_Z']**2 + data['orientation_W']**2)

    data['mod_quat'] = (data['norm_quat'])**0.5

    data['norm_X'] = data['orientation_X'] / data['mod_quat']

    data['norm_Y'] = data['orientation_Y'] / data['mod_quat']

    data['norm_Z'] = data['orientation_Z'] / data['mod_quat']

    data['norm_W'] = data['orientation_W'] / data['mod_quat']

    

    return data
x_df = normalize_quaternions(x_df)

x_test_df = normalize_quaternions(x_test_df)



x_df.head()
def feat_eng(data):

    

    df = pd.DataFrame()

    data['totl_anglr_vel'] = (data['angular_velocity_X']**2 + data['angular_velocity_Y']**2 + data['angular_velocity_Z']**2)** 0.5

    data['totl_linr_acc'] = (data['linear_acceleration_X']**2 + data['linear_acceleration_Y']**2 + data['linear_acceleration_Z']**2)**0.5

    data['totl_xyz'] = (data['orientation_X']**2 + data['orientation_Y']**2 + data['orientation_Z']**2)**0.5

    data['acc_vs_vel'] = data['totl_linr_acc'] / data['totl_anglr_vel']

    

    def mean_change_of_abs_change(x):

        return np.mean(np.diff(np.abs(np.diff(x))))

    

    for col in data.columns:

        if col in ['row_id','series_id','measurement_number']:

            continue

        df[col + '_mean'] = data.groupby(['series_id'])[col].mean()

        df[col + '_median'] = data.groupby(['series_id'])[col].median()

        df[col + '_max'] = data.groupby(['series_id'])[col].max()

        df[col + '_min'] = data.groupby(['series_id'])[col].min()

        df[col + '_std'] = data.groupby(['series_id'])[col].std()

        df[col + '_range'] = df[col + '_max'] - df[col + '_min']

        df[col + '_maxtoMin'] = df[col + '_max'] / df[col + '_min']

        df[col + '_mean_abs_chg'] = data.groupby(['series_id'])[col].apply(lambda x: np.mean(np.abs(np.diff(x))))

        df[col + '_mean_change_of_abs_change'] = data.groupby('series_id')[col].apply(mean_change_of_abs_change)

        df[col + '_abs_max'] = data.groupby(['series_id'])[col].apply(lambda x: np.max(np.abs(x)))

        df[col + '_abs_min'] = data.groupby(['series_id'])[col].apply(lambda x: np.min(np.abs(x)))

        df[col + '_abs_avg'] = (df[col + '_abs_min'] + df[col + '_abs_max'])/2

    return df
x_df = feat_eng(x_df)

x_test_df = feat_eng(x_test_df)

x_df=x_df.reset_index()

x_test_df=x_test_df.reset_index()



print(x_df.shape)

x_df.head()
correlations = x_df.corr().abs().unstack().sort_values(kind="quicksort").reset_index()

correlations = correlations[correlations['level_0'] != correlations['level_1']]

correlations.tail(20)
print(x_df.shape)

print(x_test_df.shape)

x_df = x_df.drop(['acc_vs_vel_max',\

           'acc_vs_vel_min',\

           'mod_quat_max',\

           'mod_quat_min',\

           'norm_quat_max',\

           'norm_quat_min',\

           'phi_max',\

           'phi_min',\

           'totl_anglr_vel_max',\

           'totl_anglr_vel_min',\

           'totl_linr_acc_max',\

           'totl_linr_acc_min',\

           'totl_xyz_max',\

           'totl_xyz_min'], axis=1)



x_test_df = x_test_df.drop(['acc_vs_vel_max',\

           'acc_vs_vel_min',\

           'mod_quat_max',\

           'mod_quat_min',\

           'norm_quat_max',\

           'norm_quat_min',\

           'phi_max',\

           'phi_min',\

           'totl_anglr_vel_max',\

           'totl_anglr_vel_min',\

           'totl_linr_acc_max',\

           'totl_linr_acc_min',\

           'totl_xyz_max',\

           'totl_xyz_min'], axis=1)

from sklearn import preprocessing

le = preprocessing.LabelEncoder()



y_df['surface'] = le.fit_transform(y_df['surface'])

y_df['surface'].value_counts()
from sklearn.model_selection import KFold, StratifiedKFold



folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=59)

predicted = np.zeros((x_test_df.shape[0],9))

measured= np.zeros((x_df.shape[0]))

score = 0
import gc

from sklearn.model_selection import GridSearchCV

from time import time

from operator import itemgetter

from sklearn.ensemble import RandomForestClassifier









def run_gridsearch(X, y, clf, param_grid, cv=5):

    #Run a grid search for best Decision Tree parameters => X:features, y:target

    #clf:classifier, param_grid:parameters, cv:k-foldCV

    #top_params: [dict]

    grid_search = GridSearchCV(clf, param_grid=param_grid, cv=cv, verbose=2)

    start = time()

    grid_search.fit(X, y)

    print(("\nGridSearchCV took {:.2f} seconds for {:d} candidate"

           "parameter settings.").format(time() - start, len(grid_search.cv_results_)))

    top_params = report(grid_search.cv_results_, 2)

    print(grid_search.best_params_)

    return  top_params



def report(grid_scores, n_top=2):

    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]

    print(grid_scores['mean_train_score'])#top_scores[0]

    #for top_score in top_scores:

    #    print('mean_train_score:' + str(top_score['mean_train_score']) +top_score['params'])

    return top_scores
import warnings

from sklearn.ensemble import GradientBoostingClassifier



warnings.filterwarnings("ignore") # Don't want to see the warnings in the notebook

print("-- Grid Parameter Search via 10-fold CV")

# set of parameters to test

'''

param_grid = {"criterion": ["gini"],

              "n_estimators": [500],

              "min_samples_split": [5, 10],

              "max_depth": [20, 25],

              "min_samples_leaf": [20,  50],

              "max_leaf_nodes": [20, 40]

              }

              '''

param_grid = {"loss": ["deviance"],

              "n_estimators": [500],

              "min_samples_split": [5,  20],

              "max_depth": [10,  50],

              "min_samples_leaf": [10,   50],

              "max_leaf_nodes": [20, 40]

              }

seed=43

clf = GradientBoostingClassifier(random_state=seed)

#ts_gs = run_gridsearch(x_df, y_df['surface'], clf, param_grid, cv=3)
gc.collect()
import sklearn.model_selection as ms

X_train, X_test, y_train, y_test = ms.train_test_split(x_df, y_df['surface'], 

                                                                    test_size=0.25, 

                                                                    random_state=seed,

                                                                   stratify= y_df['surface'])
from sklearn.metrics import accuracy_score



#clf = RandomForestClassifier(random_state=seed,criterion='gini', max_depth=20, max_leaf_nodes=40, 

#                            min_samples_leaf=20, min_samples_split=5, n_estimators=500)

clf=RandomForestClassifier(n_estimators = 500, n_jobs = -1)

#clf = GradientBoostingClassifier(random_state=seed,loss= 'deviance', 

#                                 max_depth= 10, max_leaf_nodes= 20, min_samples_leaf= 50, 

#                                 min_samples_split= 5, n_estimators= 100)





clf.fit(x_df, y_df['surface'])

trn_acc = accuracy_score(y_train, clf.predict(X_train))

tst_acc = accuracy_score(y_test, clf.predict(X_test))



print('training accuracy:', trn_acc)

print('test accuracy    :', tst_acc)
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,

                        n_jobs=1, train_sizes=np.linspace(0.2, 1.0, 50)):

    plt.figure(figsize=(10,6))

    plt.title(title)

    if ylim is not None:

        plt.ylim(*ylim)

    plt.xlabel("Training examples")

    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(

        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)

    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)

    test_scores_std = np.std(test_scores, axis=1)



    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,

                     train_scores_mean + train_scores_std, alpha=0.1,

                     color="r")

    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,

                     test_scores_mean + test_scores_std, alpha=0.1, color="g")

    plt.plot(train_sizes, train_scores_mean, '-', color="r",

             label="Training score")

    plt.plot(train_sizes, test_scores_mean, '-', color="g",

             label="Cross-validation score")



    plt.legend(loc="best")

    plt.grid("on")

    return plt
from sklearn import model_selection

from sklearn.model_selection import learning_curve



#kfold = model_selection.StratifiedKFold(n_splits=3, shuffle=True, random_state=0)

#plot_learning_curve(clf, 'Learning Curve', x_df, y_df['surface'], cv=kfold)

#plt.show()
y_pred = clf.predict(x_test_df)
x_test_df.head()


x_test_df['surface'] = le.inverse_transform(y_pred)

sub_df = x_test_df.filter(['series_id','surface'], axis=1)

x_test_df.head()
sub_df.to_csv('submission_last_5.csv', index=False)

print(sub_df.shape)

sub_df.head()
print('Avg Accuracy RF', score / folds.n_splits)
