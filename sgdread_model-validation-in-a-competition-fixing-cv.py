import os

import pickle

import numpy as np

import pandas as pd

from scipy.sparse import hstack

import eli5

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import TimeSeriesSplit, cross_val_score, GridSearchCV

from sklearn.metrics import roc_auc_score

from sklearn.linear_model import LogisticRegression

from matplotlib import pyplot as plt

import seaborn as sns

from IPython.display import display_html
def prepare_sparse_features(path_to_train, path_to_test, path_to_site_dict,

                           vectorizer_params, after_load_fn=None):

    times = ['time%s' % i for i in range(1, 11)]

    train_df = pd.read_csv(path_to_train, index_col='session_id', parse_dates=times)

    test_df = pd.read_csv(path_to_test, index_col='session_id', parse_dates=times)

    train_df = train_df.sort_values(by='time1')

    

    if after_load_fn is not None:

        train_df = after_load_fn(train_df)

        test_df = after_load_fn(test_df)

    

    with open(path_to_site_dict, 'rb') as f:

        site2id = pickle.load(f)

    id2site = {v:k for (k, v) in site2id.items()}

    id2site[0] = 'unknown'

    

    sites = ['site%s' % i for i in range(1, 11)]

    train_sessions = train_df[sites].fillna(0).astype('int').apply(lambda row: ' '.join([id2site[i] for i in row]), axis=1).tolist()

    test_sessions = test_df[sites].fillna(0).astype('int').apply(lambda row: ' '.join([id2site[i] for i in row]), axis=1).tolist()

    vectorizer = TfidfVectorizer(**vectorizer_params)

    X_train = vectorizer.fit_transform(train_sessions)

    X_test = vectorizer.transform(test_sessions)

    y_train = train_df['target'].astype('int').values

    

    train_times, test_times = train_df[times], test_df[times]

    

    return X_train, X_test, y_train, vectorizer, train_times, test_times



def add_time_features(times, X_sparse, add_hour=True):

    hour = times['time1'].apply(lambda ts: ts.hour)

    morning = ((hour >= 7) & (hour <= 11)).astype('int').values.reshape(-1, 1)

    day = ((hour >= 12) & (hour <= 18)).astype('int').values.reshape(-1, 1)

    evening = ((hour >= 19) & (hour <= 23)).astype('int').values.reshape(-1, 1)

    night = ((hour >= 0) & (hour <=6)).astype('int').values.reshape(-1, 1)

    

    objects_to_hstack = [X_sparse, morning, day, evening, night]

    feature_names = ['morning', 'day', 'evening', 'night']

    

    if add_hour:

        # we'll do it right and scale hour dividing by 24

        objects_to_hstack.append(hour.values.reshape(-1, 1) / 24)

        feature_names.append('hour')

        

    X = hstack(objects_to_hstack)

    return X, feature_names



def add_day_month(times, X_sparse):

    day_of_week = times['time1'].apply(lambda t: t.weekday()).values.reshape(-1, 1)

    month = times['time1'].apply(lambda t: t.month).values.reshape(-1, 1) 

    # linear trend: time in a form YYYYMM, we'll divide by 1e5 to scale this feature 

    year_month = times['time1'].apply(lambda t: 100 * t.year + t.month).values.reshape(-1, 1) / 1e5

    

    objects_to_hstack = [X_sparse, day_of_week, month, year_month]

    feature_names = ['day_of_week', 'month', 'year_month']

        

    X = hstack(objects_to_hstack)

    return X, feature_names



def pre_process():

    X_train_with_times1, new_feat_names = add_time_features(train_times, X_train_sites)

    X_test_with_times1, _ = add_time_features(test_times, X_test_sites)

    X_train_with_times1.shape, X_test_with_times1.shape



    X_train_with_times2, new_feat_names = add_time_features(train_times, X_train_sites, add_hour=False)

    X_test_with_times2, _ = add_time_features(test_times, X_test_sites, add_hour=False)

    

    train_durations = (train_times.max(axis=1) - train_times.min(axis=1)).astype('timedelta64[ms]').astype(int)

    test_durations = (test_times.max(axis=1) - test_times.min(axis=1)).astype('timedelta64[ms]').astype(int)



    scaler = StandardScaler()

    train_dur_scaled = scaler.fit_transform(train_durations.values.reshape(-1, 1))

    test_dur_scaled = scaler.transform(test_durations.values.reshape(-1, 1))

    

    X_train_with_time_correct = hstack([X_train_with_times2, train_dur_scaled])

    X_test_with_time_correct = hstack([X_test_with_times2, test_dur_scaled])

    

    X_train_final, more_feat_names = add_day_month(train_times, X_train_with_time_correct)

    X_test_final, _ = add_day_month(test_times, X_test_with_time_correct)    

    

    feat_names = new_feat_names + ['sess_duration'] + more_feat_names

    

    return X_train_final, X_test_final, feat_names



# A helper function for writing predictions to a file

def write_to_submission_file(predicted_labels, out_file,

                             target='target', index_label="session_id"):

    predicted_df = pd.DataFrame(predicted_labels,

                                index = np.arange(1, predicted_labels.shape[0] + 1),

                                columns=[target])

    predicted_df.to_csv(out_file, index_label=index_label)



    

def train_and_predict(model, X_train, y_train, X_test, cv, site_feature_names, 

                      new_feature_names=None, scoring='roc_auc',

                      top_n_features_to_show=30, submission_file_name='submission.csv'):

    

    

    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, 

                            scoring=scoring, n_jobs=4)

    print('CV scores', cv_scores)

    print('CV mean: {}, CV std: {}'.format(cv_scores.mean(), cv_scores.std()))

    model.fit(X_train, y_train)

    

    if new_feature_names:

        all_feature_names = site_feature_names + new_feature_names 

    else: 

        all_feature_names = site_feature_names

    

    display_html(eli5.show_weights(estimator=model, 

                  feature_names=all_feature_names, top=top_n_features_to_show))

    

    if new_feature_names:

        print('New feature weights:')

    

        print(pd.DataFrame({'feature': new_feature_names, 

                        'coef': model.coef_.flatten()[-len(new_feature_names):]}))

    

    test_pred = model.predict_proba(X_test)[:, 1]

    write_to_submission_file(test_pred, submission_file_name) 

    

    return cv_scores



# A helper function for writing predictions to a file

def write_to_submission_file(predicted_labels, out_file,

                             target='target', index_label="session_id"):

    predicted_df = pd.DataFrame(predicted_labels,

                                index = np.arange(1, predicted_labels.shape[0] + 1),

                                columns=[target])

    predicted_df.to_csv(out_file, index_label=index_label)



    

def train_and_predict(model, X_train, y_train, X_test, cv, site_feature_names, 

                      new_feature_names=None, scoring='roc_auc', show_eli=False,

                      top_n_features_to_show=30, submission_file_name='submission.csv'):

    

    

    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, 

                            scoring=scoring, n_jobs=4)

    print('CV scores', cv_scores)

    print('CV mean: {}, CV std: {}'.format(cv_scores.mean(), cv_scores.std()))

    model.fit(X_train, y_train)

    

    if new_feature_names:

        all_feature_names = site_feature_names + new_feature_names 

    else: 

        all_feature_names = site_feature_names



    if show_eli:

        display_html(eli5.show_weights(estimator=model, 

                      feature_names=all_feature_names, top=top_n_features_to_show))

    

    if new_feature_names:

        print('New feature weights:')

    

        print(pd.DataFrame({'feature': new_feature_names, 

                        'coef': model.coef_.flatten()[-len(new_feature_names):]}))

    

    test_pred = model.predict_proba(X_test)[:, 1]

    write_to_submission_file(test_pred, submission_file_name) 

    

    return cv_scores    
PATH_TO_DATA = '../input/'

SEED = 17
time_split = TimeSeriesSplit(n_splits=10)

logit = LogisticRegression(C=1, random_state=SEED, solver='liblinear')
%%time

X_train_sites, X_test_sites, y_train, vectorizer, train_times, test_times = prepare_sparse_features(

    path_to_train=os.path.join(PATH_TO_DATA, 'train_sessions.csv'),

    path_to_test=os.path.join(PATH_TO_DATA, 'test_sessions.csv'),

    path_to_site_dict=os.path.join(PATH_TO_DATA, 'site_dic.pkl'),

    vectorizer_params={'ngram_range': (1, 5), 

                       'max_features': 50000,

                       'tokenizer': lambda s: s.split()}

)



X_train_final, X_test_final, new_feat_names = pre_process()
cv_scores6 = train_and_predict(model=logit, X_train=X_train_final, y_train=y_train,

                               X_test=X_test_final, cv=time_split,

                               site_feature_names=vectorizer.get_feature_names(),

                               new_feature_names=new_feat_names,

                               submission_file_name='subm6.csv')
c_values = np.logspace(-2, 2, 20)

logit_grid_searcher = GridSearchCV(estimator=logit, param_grid={'C': c_values}, scoring='roc_auc', n_jobs=4, cv=time_split, verbose=1)
%%time

logit_grid_searcher.fit(X_train_final, y_train);

logit_grid_searcher.best_score_, logit_grid_searcher.best_params_
final_model = logit_grid_searcher.best_estimator_
cv_scores7 = train_and_predict(model=final_model, X_train=X_train_final, y_train=y_train, 

                               X_test=X_test_final, 

                               site_feature_names=vectorizer.get_feature_names(),

                               new_feature_names=new_feat_names,

                               cv=time_split, submission_file_name='subm7.csv')
cv_scores7 > cv_scores6
import re

import pickle



import pandas as pd

import numpy as np

from pathlib import Path

import seaborn as sns

import matplotlib.pyplot as plt

from pathlib import Path



sns.set()



PATH = Path('../input/')



times = ['time%s' % i for i in range(1, 11)]



def plot_series(df, field, label, **kwargs):

    df = df.copy()

    df['date'] = pd.DatetimeIndex(df[field]).normalize()

    df = (df['date'].value_counts()/len(df)).sort_index().resample('1D').interpolate()

    df.plot(label=label, **kwargs)

    

df_train = pd.read_csv(PATH/'train_sessions.csv', index_col='session_id', parse_dates=times)

df_test = pd.read_csv(PATH/'test_sessions.csv', index_col='session_id', parse_dates=times)
fig , (ax1,ax2) = plt.subplots(2,1,figsize = (16, 12 )) 

fig.suptitle('Year-Month Distributions', fontsize=16)

sns.countplot((df_train.time1.dt.year * 100 + df_train.time1.dt.month).apply(str), ax=ax1)

ax1.set_title("Train distribution") 

sns.countplot((df_test.time1.dt.year * 100 + df_test.time1.dt.month).apply(str), ax=ax2)

ax2.set_title("Test distribution");
plot_series(df_train, 'time1', 'train-all', figsize=(24, 8))

plot_series(df_train[df_train['target']==1], 'time1', 'train-alice')

plot_series(df_test, 'time1', 'test')

plt.legend();
def fix_incorrect_date_formats(df, columns_to_fix):

    for time in columns_to_fix:

        d = df[time]

        d_fix = d[d.dt.day <= 12]

        d_fix = pd.to_datetime(d_fix.apply(str), format='%Y-%d-%m %H:%M:%S')

        df.loc[d_fix.index.values, time] = d_fix

    return df
df_train_fixed = fix_incorrect_date_formats(df_train, times)

df_test_fixed = fix_incorrect_date_formats(df_test, times)

plot_series(df_train_fixed, 'time1', 'train-all', figsize=(24, 8))

plot_series(df_train_fixed[df_train_fixed['target']==1], 'time1', 'train-alice')

plot_series(df_test_fixed, 'time1', 'test')

plt.legend();
fig , (ax1,ax2) = plt.subplots(1,2,figsize = ( 15 , 6 )) 

fig.suptitle('Year-Month Distributions', fontsize=16)

sns.countplot((df_train_fixed.time1.dt.year * 100 + df_train_fixed.time1.dt.month).apply(str), ax=ax1)

ax1.set_title("Train distribution") 

sns.countplot((df_test_fixed.time1.dt.year * 100 + df_test_fixed.time1.dt.month).apply(str), ax=ax2)

ax2.set_title("Test distribution");
%%time

X_train_sites, X_test_sites, y_train, vectorizer, train_times, test_times = prepare_sparse_features(

    after_load_fn=(lambda df: fix_incorrect_date_formats(df, times)), # Applying fix

    path_to_train=os.path.join(PATH_TO_DATA, 'train_sessions.csv'),

    path_to_test=os.path.join(PATH_TO_DATA, 'test_sessions.csv'),

    path_to_site_dict=os.path.join(PATH_TO_DATA, 'site_dic.pkl'),

    vectorizer_params={'ngram_range': (1, 5), 

                       'max_features': 50000,

                       'tokenizer': lambda s: s.split()}

)



X_train_final, X_test_final, new_feat_names = pre_process()
time_split = TimeSeriesSplit(n_splits=10)

logit = LogisticRegression(C=1, random_state=SEED, solver='liblinear')
cv_scores8 = train_and_predict(model=logit, X_train=X_train_final, y_train=y_train,

                               X_test=X_test_final, cv=time_split,

                               site_feature_names=vectorizer.get_feature_names(),

                               new_feature_names=new_feat_names,

                               submission_file_name='subm8.csv')
c_values = np.logspace(-2, 2, 20)

logit_grid_searcher = GridSearchCV(estimator=logit, param_grid={'C': c_values}, scoring='roc_auc', n_jobs=4, cv=time_split, verbose=1)
%%time

logit_grid_searcher.fit(X_train_final, y_train);

logit_grid_searcher.best_score_, logit_grid_searcher.best_params_
final_model = logit_grid_searcher.best_estimator_
cv_scores9 = train_and_predict(model=final_model, X_train=X_train_final, y_train=y_train, 

                               X_test=X_test_final, 

                               site_feature_names=vectorizer.get_feature_names(),

                               new_feature_names=new_feat_names,

                               cv=time_split, submission_file_name='subm9.csv')
cv_scores9 > cv_scores8