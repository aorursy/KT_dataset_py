

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import os

from time import time

from sklearn import preprocessing

from sklearn.metrics import accuracy_score

import seaborn as sns

from scipy import stats

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from matplotlib import rcParams

# %matplotlib inline

le = preprocessing.LabelEncoder()

from numba import jit

import itertools

from seaborn import countplot,lineplot, barplot

from numba import jit

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2

from sklearn import preprocessing

from scipy.stats import randint as sp_randint

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import KFold

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import KFold, StratifiedKFold

from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import LeaveOneGroupOut

from sklearn.model_selection import GroupKFold

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import RandomizedSearchCV

from scipy.stats import kurtosis, skew



import matplotlib.style as style

style.use('ggplot')



import warnings

warnings.filterwarnings('ignore')

import gc

gc.enable()
#

def sq_dist(a, b):

    ''' the squared euclidean distance between two samples '''



    return np.sum((a - b) ** 2, axis=1)





def find_run_edges(data, edge):

    ''' examine links between samples. left/right run edges are those samples which do not have a link on that side. '''



    if edge == 'left':

        border1 = 0

        border2 = -1

    elif edge == 'right':

        border1 = -1

        border2 = 0

    else:

        return False



    edge_list = []

    linked_list = []



    for i in range(len(data)):

        dist_list = sq_dist(data[i, border1, :4], data[:, border2, :4])  # distances to rest of samples

        min_dist = np.min(dist_list)

        closest_i = np.argmin(dist_list)  # this is i's closest neighbor

        if closest_i == i:  # this might happen and it's definitely wrong

            print('Sample', i, 'linked with itself. Next closest sample used instead.')

            closest_i = np.argsort(dist_list)[1]

        dist_list = sq_dist(data[closest_i, border2, :4], data[:, border1, :4])  # now find closest_i's closest neighbor

        rev_dist = np.min(dist_list)

        closest_rev = np.argmin(dist_list)  # here it is

        if closest_rev == closest_i:  # again a check

            print('Sample', i, '(back-)linked with itself. Next closest sample used instead.')

            closest_rev = np.argsort(dist_list)[1]

        if (i != closest_rev):  # we found an edge

            edge_list.append(i)

        else:

            linked_list.append([i, closest_i, min_dist])



    return edge_list, linked_list





def find_runs(data, left_edges, right_edges):

    import time

    ''' go through the list of samples & link the closest neighbors into a single run '''



    data_runs = []



    for start_point in left_edges:

        i = start_point

        run_list = [i]

        while i not in right_edges:

            tmp = np.argmin(sq_dist(data[i, -1, :4], data[:, 0, :4]))

            if tmp == i:  # self-linked sample

                tmp = np.argsort(sq_dist(data[i, -1, :4], data[:, 0, :4]))[1]

            i = tmp

            run_list.append(i)

        data_runs.append(np.array(run_list))



    return data_runs
train_X = pd.read_csv('../input/X_train.csv')

test_X  = pd.read_csv('../input/X_test.csv' )
print(train_X['series_id'].tail()) # train_X series_id stop in 3809

test_X['series_id'] = test_X['series_id'] + 3810 # so test_X's series_id should start from 3810
_total = pd.concat([train_X, test_X], axis=0).reset_index(drop=True)

print('total series' , len(_total['series_id'].unique()))

total = _total.iloc[:,3:].values.reshape(-1,128,10)
all_left_edges, all_left_link = find_run_edges(total, edge='left')

all_right_edges, all_right_link = find_run_edges(total, edge='right')

print('Found', len(all_left_edges), 'left edges and', len(all_right_edges), 'right edges.')



all_runs = find_runs(total, all_left_edges, all_right_edges)



flat_list = [series_id for run in all_runs for series_id in run]

print(len(flat_list), len(np.unique(flat_list)))
lost_samples = np.array([ i for i in range(len(total)) if i not in np.concatenate(all_runs) ])

print(lost_samples)

print(len(lost_samples))



lost_run = np.array(lost_samples[find_runs(total[lost_samples], [0], [5])[0]])

all_runs.append(lost_run)
final = pd.DataFrame(index=_total['series_id'].unique())

for run_id in range(len(all_runs)):

    for run_pos in range(len(all_runs[run_id])):

        series_id = all_runs[run_id][run_pos]

        final.at[ series_id, 'run_id'  ] = run_id

        final.at[ series_id, 'run_pos' ] = run_pos



train_y = pd.read_csv('../input/y_train.csv')

final['surface'] = train_y['surface']
for id, surface in zip(final[final['run_id'].notnull()]['run_id'], final[final['surface'].notnull()]['surface']):

    final.loc[final['run_id']==id, 'surface'] = surface





print(final['run_id'].unique())

print(final[final['surface'].isnull()]['run_id'].unique())
new_target = final[final['surface'].notnull()]

new_train_series = final[final['surface'].notnull()].index

new_test_series = final[final['surface'].isnull()]['run_id'].index

new_train = _total[_total['series_id'].isin(new_train_series)]

new_test = _total[_total['series_id'].isin(new_test_series)]


def feat_eng(data):

    df = pd.DataFrame()

    data['totl_anglr_vel'] = (data['angular_velocity_X'] ** 2 + data['angular_velocity_Y'] ** 2 + data[

        'angular_velocity_Z'] ** 2) ** 0.5

    data['totl_linr_acc'] = (data['linear_acceleration_X'] ** 2 + data['linear_acceleration_Y'] ** 2 + data[

        'linear_acceleration_Z'] ** 2) ** 0.5

    data['acc_vs_vel'] = data['totl_linr_acc'] / data['totl_anglr_vel']



    def mean_change_of_abs_change(x):

        return np.mean(np.diff(np.abs(np.diff(x))))



    for col in data.columns:



        if col in ['row_id', 'series_id', 'measurement_number',

                   'orientation_X', 'orientation_Y', 'orientation_Z',

                   'run_id', 'orientation_W']:

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

        df[col + '_abs_avg'] = (df[col + '_abs_min'] + df[col + '_abs_max']) / 2





        df[col + '_kurtosis'] = data.groupby(['series_id'])[col].apply(lambda x:kurtosis(x))



        df[col + '_skew'] = data.groupby(['series_id'])[col].skew()



    return df
data = feat_eng(new_train).reset_index(drop=True)

test = feat_eng(new_test).reset_index(drop=True)



data = data.fillna(0)

test = test.fillna(0)

data = data.replace(-np.inf,0)

data = data.replace(np.inf,0)

test = test.replace(-np.inf,0)

test = test.replace(np.inf,0)

new_target['surface'] = le.fit_transform(new_target['surface'])
print('modelling')

model = RandomForestClassifier(n_estimators=500, max_depth=10, min_samples_split=5, n_jobs=-1)

model.fit(data.values, new_target['surface'].values)

measured = model.predict(data.values)

predicted = model.predict_proba(test)

score = model.score(data.values, new_target['surface'].values)

print("score: {}".format(model.score(data.values, new_target['surface'].values)))

importances = model.feature_importances_

indices = np.argsort(importances)

features = data.columns


gc.collect()



print('Accuracy RF', score )



result = pd.DataFrame(data={'surface':le.inverse_transform(predicted.argmax(axis=1))},index= new_test['series_id'].unique())

new_target['surface'] = le.inverse_transform(new_target['surface'])



df = pd.concat([result, new_target[['surface']]], axis=0)

sub = pd.read_csv('../input/sample_submission.csv')

sub['surface'] = new_target['surface']
sub.to_csv('my_submission.csv', index=0)
print(sub.head())