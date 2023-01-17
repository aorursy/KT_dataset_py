import os

import pickle

import numpy as np

import pandas as pd

from scipy.sparse import hstack

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import TimeSeriesSplit, cross_val_score, GridSearchCV

from sklearn.metrics import roc_auc_score

from sklearn.linear_model import LogisticRegression

from matplotlib import pyplot as plt

import seaborn as sns

SEED = 17

def prepare_sparse_features(path_to_train, path_to_test, path_to_site_dict,

                           vectorizer_params):

    times = ['time%s' % i for i in range(1, 11)]

    train_df = pd.read_csv('/kaggle/input/train_sessions.csv',

                       index_col='session_id', parse_dates=times)

    test_df = pd.read_csv('/kaggle/input/test_sessions.csv',

                      index_col='session_id', parse_dates=times)



    # Sort the data by time

    train_df = train_df.sort_values(by='time1')

    

    # read site -> id mapping provided by competition organizers 

    with open(r"/kaggle/input/site_dic.pkl", 'rb') as f:

        site2id = pickle.load(f)

    # create an inverse id _> site mapping

    id2site = {v:k for (k, v) in site2id.items()}

    # we treat site with id 0 as "unknown"

    id2site[0] = 'unknown'

    

    # Transform data into format which can be fed into TfidfVectorizer

    # This time we prefer to represent sessions with site names, not site ids. 

    # It's less efficient but thus it'll be more convenient to interpret model weights.

    sites = ['site%s' % i for i in range(1, 11)]

    train_sessions = train_df[sites].fillna(0).astype('int').apply(lambda row: 

                                                     ' '.join([id2site[i] for i in row]), axis=1).tolist()

    test_sessions = test_df[sites].fillna(0).astype('int').apply(lambda row: 

                                                     ' '.join([id2site[i] for i in row]), axis=1).tolist()

    # we'll tell TfidfVectorizer that we'd like to split data by whitespaces only 

    # so that it doesn't split by dots (we wouldn't like to have 'mail.google.com' 

    # to be split into 'mail', 'google' and 'com')

    vectorizer = TfidfVectorizer(**vectorizer_params)

    X_train = vectorizer.fit_transform(train_sessions)

    X_test = vectorizer.transform(test_sessions)

    y_train = train_df['target'].astype('int').values

    

    # we'll need site visit times for further feature engineering

    train_times, test_times = train_df[times], test_df[times]

    

    return X_train, X_test, y_train, vectorizer, train_times, test_times

from time import time

%time

X_train_sites, X_test_sites, y_train, vectorizer, train_times, test_times = prepare_sparse_features(1,2,3,

    vectorizer_params={'ngram_range': (1,4), 

                       'max_features': 48371,

                       'tokenizer': lambda s: s.split()}

)

time_split = TimeSeriesSplit(n_splits=10)

def write_to_submission_file(predicted_labels, out_file,

                             target='target', index_label="session_id"):

    predicted_df = pd.DataFrame(predicted_labels,

                                index = np.arange(1, predicted_labels.shape[0] + 1),

                                columns=[target])

    predicted_df.to_csv(out_file, index_label=index_label)

def train_and_predict(model, X_train, y_train, X_test, 

                      cv=time_split, scoring='roc_auc',

                      top_n_features_to_show=30, submission_file_name='submission.csv'):

    

    

    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, 

                            scoring=scoring, n_jobs=4)

    print('CV scores', cv_scores)

    print('CV mean: {}, CV std: {}'.format(cv_scores.mean(), cv_scores.std()))

    model.fit(X_train, y_train)

       

    test_pred = model.predict_proba(X_test)[:, 1]

    write_to_submission_file(test_pred, submission_file_name) 

    

    return cv_scores

session_start_hour = train_times['time1'].apply(lambda ts: ts.hour).values

def add_time_features(times, X_sparse, add_hour=True):

    hour = times['time1'].apply(lambda ts: ts.hour)

    morning = ((hour >= 7) & (hour <= 11)).astype('int').values.reshape(-1, 1)

    day1 = (((hour >= 12) & (hour <= 13)) | ((hour >= 16) & (hour <= 18))).astype('int').values.reshape(-1, 1)

    day2 = ((hour >= 14) & (hour <= 15)).astype('int').values.reshape(-1, 1)

  

    evening = ((hour >= 19) & (hour <= 23)).astype('int').values.reshape(-1, 1)

    night = ((hour >= 0) & (hour <=6)).astype('int').values.reshape(-1, 1)

    

    objects_to_hstack = [X_sparse, morning, day1, day2, evening, night]

    

    

    if add_hour:

        # we'll do it right and scale hour dividing by 24

        objects_to_hstack.append(hour.values.reshape(-1, 1) / 24)

        

        

    X = hstack(objects_to_hstack)

    return X

X_train_with_times1 = add_time_features(train_times, X_train_sites)

X_test_with_times1 = add_time_features(test_times, X_test_sites)

%time

X_train_with_times2 = add_time_features(train_times, X_train_sites, add_hour=False)

X_test_with_times2 = add_time_features(test_times, X_test_sites, add_hour=False)

train_durations = (train_times.max(axis=1) - train_times.min(axis=1)).astype('timedelta64[ms]').astype(int)

test_durations = (test_times.max(axis=1) - test_times.min(axis=1)).astype('timedelta64[ms]').astype(int)



scaler = StandardScaler()

train_dur_scaled = scaler.fit_transform(train_durations.values.reshape(-1, 1))

test_dur_scaled = scaler.transform(test_durations.values.reshape(-1, 1))

X_train_with_time_correct = hstack([X_train_with_times2, train_dur_scaled])

X_test_with_time_correct = hstack([X_test_with_times2, test_dur_scaled])



def add_day_month(times, X_sparse):

    day_of_week = times['time1'].apply(lambda t: t.weekday()).values.reshape(-1, 1)

    month = times['time1'].apply(lambda t: t.month).values.reshape(-1, 1) 

    asds=[201301,201302,201303,201304,201305,201306,201307,201308,201309,201310,201311,201312,201401,201402,201403,201404]

    year_month1 = times['time1'].apply(lambda t: 100 * t.year + t.month).values.reshape(-1, 1)

    # linear trend: time in a form YYYYMM, we'll divide by 1e5 to scale this feature 

    year_month = times['time1'].apply(lambda t: 100 * t.year + t.month).values.reshape(-1, 1) / 1e5

    monthA=(((year_month1 >= 201302) & (year_month1 <= 201304)) | ((year_month1 >= 201309) & (year_month1 <= 201404))).astype('int').reshape(-1, 1)

  

    tmp_scaled_day = StandardScaler().fit_transform(day_of_week)

    tmp_scaled_month = StandardScaler().fit_transform(month)

    tmp_scaled_month_year_month=StandardScaler().fit_transform(year_month)

    objects_to_hstack = [X_sparse, tmp_scaled_day, tmp_scaled_month, tmp_scaled_month_year_month, monthA]

    

        

    X = hstack(objects_to_hstack)

    return X

X_train_final = add_day_month(train_times, X_train_with_time_correct)

X_test_final = add_day_month(test_times, X_test_with_time_correct)

test_times1=((test_times.max(axis=1) - test_times.min(axis=1))).astype('timedelta64[ms]').astype(int)/test_times.count(axis=1)/100

train_times1=((train_times.max(axis=1) - train_times.min(axis=1))).astype('timedelta64[ms]').astype(int)/train_times.count(axis=1)/100

avg90_train = (train_times1 >=90).astype('int').values.reshape(-1, 1)

avg90_test = (test_times1 >=90).astype('int').values.reshape(-1, 1)



X_train_final2 = hstack([X_train_final, avg90_train])

X_test_final2 = hstack([X_test_final, avg90_test])

c_values = np.logspace(-2, 3, 40)

logit = LogisticRegression(C=1, random_state=SEED, solver='liblinear')

logit_grid_searcher = GridSearchCV(estimator=logit, param_grid={'C': c_values},

                                scoring='roc_auc', n_jobs=8, cv=time_split, verbose=1)

logit_grid_searcher.fit(X_train_final2, y_train); 

logit_grid_searcher.best_score_, logit_grid_searcher.best_params_

final_model = logit_grid_searcher.best_estimator_

cv_scores8 = train_and_predict(model=final_model, X_train=X_train_final2, y_train=y_train, 

                               X_test=X_test_final2, 

                               cv=time_split, submission_file_name='subm8.csv')