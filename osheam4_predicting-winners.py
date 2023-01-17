import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn import svm

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import classification_report

from sklearn.preprocessing import scale

from sklearn import preprocessing

from sklearn.ensemble import RandomForestClassifier as RF
forms = pd.read_csv('../input/forms.csv')

runners = pd.read_csv('../input/runners.csv')

markets = pd.read_csv('../input/markets.csv')

racecard = pd.merge(forms, runners, on=['horse_id', 'market_id'], suffixes=('', '_y'))

racecard = pd.merge(racecard, markets, left_on='market_id', right_on='id', how='left')

dropCols = [c for c in racecard.columns if c.lower()[:3] == 'tip']

racecard.drop(dropCols, axis=1, inplace=True)

racecard['win'] = np.where(racecard['position'] == 1, 1, 0)
X_train = racecard[racecard['market_id'] < 6000]

X_test = racecard[racecard['market_id'] >= 6000]
trainer_perc_win = X_train.groupby(['trainer_id', 'venue_id'])['win'].apply(lambda x: float(sum(x)) / len(x))

trainer_perc_win = trainer_perc_win.to_frame()

#trainer_perc_win[(trainer_perc_win['win'] > 0.6) & (trainer_perc_win['win'] < 0.9)]

trainer_perc_win.reset_index(col_level=1, inplace=True)





X_train = pd.merge(X_train, trainer_perc_win, on=['trainer_id', 'venue_id'], how='left')

X_train.rename(columns={'win_y':'trainer_venue_winratio'}, inplace=True)

X_train.rename(columns={'win_x':'win'}, inplace=True)



X_train['trainer_venue_winratio'].fillna(0, inplace=True)

X_train['win'].fillna(0, inplace=True)



X_test = pd.merge(X_test, trainer_perc_win, on=['trainer_id', 'venue_id'], how='left')

X_test.rename(columns={'win_y':'trainer_venue_winratio'}, inplace=True)

X_test.rename(columns={'win_x':'win'}, inplace=True)



X_test['trainer_venue_winratio'].fillna(0, inplace=True)

X_test['win'].fillna(0, inplace=True)
def getconditionform(df, type):

    if df['condition_id'] in range(1,3):

        return df['heavy_%s' %type]

    elif df['condition_id'] in range(4,8):

        return df['soft_%s' %type]

    else:

        return df['good_%s' %type]

    

X_train['condition_wins'] = X_train.apply(lambda x: getconditionform(x, 'wins'), axis=1)

X_train['condition_places'] = X_train.apply(lambda x: getconditionform(x, 'places'), axis=1)

X_test['condition_wins'] = X_test.apply(lambda x: getconditionform(x, 'wins'), axis=1)

X_test['condition_places'] = X_test.apply(lambda x: getconditionform(x, 'places'), axis=1)
rider_perc_win = X_train.groupby(['rider_id'])['win'].apply(lambda x: float(sum(x)) / len(x))

rider_perc_win = rider_perc_win.to_frame()

rider_perc_win.reset_index(col_level=1, inplace=True)

rider_perc_win



X_train = pd.merge(X_train, rider_perc_win, on=['rider_id'], how='left')

X_train.rename(columns={'win_y':'rider_winratio'}, inplace=True)

X_train.rename(columns={'win_x':'win'}, inplace=True)



X_train['rider_winratio'].fillna(0, inplace=True)

X_train['win'].fillna(0, inplace=True)



X_test = pd.merge(X_test, rider_perc_win, on=['rider_id'], how='left')

X_test.rename(columns={'win_y':'rider_winratio'}, inplace=True)

X_test.rename(columns={'win_x':'win'}, inplace=True)



X_test['rider_winratio'].fillna(0, inplace=True)

X_test['win'].fillna(0, inplace=True)
cols = ['class_level_id', 'overall_wins', 'overall_places', 'days_since_last_run',

    'trainer_id', 'rider_id', 'handicap_weight', 'form_rating_one', 'form_rating_two',

    'class_stronger_places', 'trainer_venue_winratio']





#clf = svm.SVC(kernel='linear', C=1, probability=True)

clf = GaussianNB()

gnb_model1 = clf.fit(X_train[cols], X_train['win'])

y_true, y_pred = X_test['win'], gnb_model1.predict(X_test[cols])

print(classification_report(y_true, y_pred))
cols = ['class_level_id', 'overall_wins', 'overall_places', 'days_since_last_run',

    'trainer_id', 'rider_id', 'handicap_weight', 'form_rating_one', 'form_rating_two',

    'class_stronger_places', 'trainer_venue_winratio', 'condition_wins', 'condition_places',

        'track_wins', 'track_places', 'track_distance_wins', 'track_distance_places']



#clf = svm.SVC(kernel='linear', C=1, probability=True)

clf = GaussianNB()

gnb_model2 = clf.fit(X_train[cols], X_train['win'])

y_true, y_pred = X_test['win'], gnb_model2.predict(X_test[cols])

print(classification_report(y_true, y_pred))
Xs = scale(X_train[cols])

Xstest = scale(X_test[cols])

gnb_model3 = clf.fit(Xs, X_train['win'])

y_true, y_pred = X_test['win'], gnb_model3.predict(Xstest)

print(classification_report(y_true, y_pred))
normalized_X_train = preprocessing.normalize(X_train[cols])

normalized_X_test = preprocessing.normalize(X_test[cols])

gnb_model4 = clf.fit(normalized_X_train, X_train['win'])

y_true, y_pred = X_test['win'], gnb_model4.predict(normalized_X_test)

print(classification_report(y_true, y_pred))
clf = RF( n_estimators = 100, verbose = 0, n_jobs = -1 )

rf_model1 = clf.fit(X_train[cols], X_train['win'])

y_true, y_pred = X_test['win'], rf_model1.predict(X_test[cols])

print(classification_report(y_true, y_pred))
rf_model2 = clf.fit(Xs, X_train['win'])

y_true, y_pred = X_test['win'], rf_model2.predict(Xstest)

print(classification_report(y_true, y_pred))
cols = ['class_level_id', 'overall_wins', 'overall_places', 'days_since_last_run',

    'trainer_id', 'rider_id', 'handicap_weight', 'form_rating_one', 'form_rating_two',

    'class_stronger_places', 'trainer_venue_winratio', 'condition_wins', 'condition_places',

        'track_wins', 'track_places', 'track_distance_wins', 'track_distance_places', 'rider_winratio']

rf_model1 = clf.fit(X_train[cols], X_train['win'])

y_true, y_pred = X_test['win'], rf_model1.predict(X_test[cols])

print(classification_report(y_true, y_pred))