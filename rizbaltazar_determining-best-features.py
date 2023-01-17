# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import timeit



start_time_total = timeit.default_timer()



SEED = 1
lol_data = pd.read_csv('../input/league-of-legends-diamond-ranked-games-10-min/high_diamond_ranked_10min.csv', index_col='gameId')

lol_data.head()
lol_data.info()
sns.boxplot(x="blueWins", y="blueGoldDiff", data=lol_data)
sns.boxplot(x="blueWins", y="blueExperienceDiff", data=lol_data)
y = lol_data.blueWins

X = lol_data.drop(columns='blueWins')
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split



train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=SEED)

rf_model = RandomForestClassifier(random_state=SEED)

rf_model.fit(train_X, train_y)

pred = rf_model.predict(val_X)

baseline_score = accuracy_score(val_y, pred)

print('Accuracy: %.2f%%' %(baseline_score*100))
r = lol_data.drop('blueWins', axis=1).corr()

plt.figure(figsize=(20, 12))

sns.heatmap(r, annot=True, fmt='.2f', center= 0)
redundant_data = ['redFirstBlood', 'redKills', 'redDeaths', 'redGoldDiff', 'redExperienceDiff', 'redGoldPerMin', 'redCSPerMin', 'blueGoldPerMin', 'blueCSPerMin']

clean_data = lol_data.drop(redundant_data, axis=1)



r = clean_data.drop('blueWins', axis=1).corr()

plt.figure(figsize=(20, 12))

sns.heatmap(r, annot=True, fmt='.2f', center= 0);
y_clean = clean_data.blueWins

X_clean = clean_data.drop(columns='blueWins')

train_X, val_X, train_y, val_y = train_test_split(X_clean, y_clean, random_state=SEED)

rf_model = RandomForestClassifier(random_state=SEED)

rf_model.fit(train_X, train_y)

pred = rf_model.predict(val_X)

score = accuracy_score(val_y, pred)

print('Accuracy: %.2f%% vs baseline accuracy: %.2f%%' %(score*100, baseline_score*100))
import eli5

from eli5.sklearn import PermutationImportance

from sklearn.svm import SVC

from sklearn.feature_selection import SelectFromModel



start_time = timeit.default_timer()

perm = PermutationImportance(rf_model).fit(val_X, val_y)

elapsed = timeit.default_timer() - start_time

print('Elapsed time: %s s' %elapsed)

eli5.show_weights(perm, feature_names=val_X.columns.tolist(), top=None)
loss = clean_data[clean_data.blueWins==0]

won = clean_data[clean_data.blueWins==1]

loss = loss.drop(columns='blueWins')

won = won.drop(columns='blueWins')

loss.head()
won.head()
import shap



def shap_row(df, model, row_to_show=0):

    data_for_prediction = df.iloc[row_to_show]

    data_for_prediction_array = data_for_prediction.values.reshape(1, -1)

    model.predict_proba(data_for_prediction_array)

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(data_for_prediction)

    shap.initjs()

    return shap.force_plot(explainer.expected_value[1], shap_values[1], data_for_prediction)
plot = shap_row(loss, rf_model, 1)

plot
plot = shap_row(won, rf_model, 1)

plot
loss_small_diff = loss[(abs(loss.blueGoldDiff)<500) & (abs(loss.blueExperienceDiff)<500)]

loss_small_diff.head()
plot = shap_row(loss_small_diff, rf_model, 0)

plot
start_time = timeit.default_timer()



explainer = shap.TreeExplainer(rf_model)

shap_values_all = explainer.shap_values(val_X)

shap.summary_plot(shap_values_all[0], val_X)



elapsed = timeit.default_timer() - start_time

print('Elapsed time: %s s' %elapsed)
from sklearn.feature_selection import SelectKBest, f_classif



def k_features(model, train_X, val_X, train_y, val_y, k=1):

    selector = SelectKBest(f_classif, k=k)

    X_new = selector.fit_transform(train_X, train_y)

    selected_features = pd.DataFrame(selector.inverse_transform(X_new), 

                                     index=train_X.index, 

                                     columns=train_X.columns)

    selected_cols = selected_features.columns[selected_features.var() != 0]



    kbest_X = train_X[selected_cols]

    kval_X = val_X[selected_cols]



    model.fit(kbest_X, train_y)

    pred = model.predict(kval_X)

    score = accuracy_score(val_y, pred)

    return score



def plot_results(results):

    plt.plot(list(results.keys()), list(results.values()))

    plt.xlabel('# of features')

    plt.ylabel('Score (accuracy)')

    plt.show()

    

def scores(results):

    key_min = min(results.keys(), key=(lambda k: results[k]))

    key_max = max(results.keys(), key=(lambda k: results[k]))

    

    print('Highest score at %d features of %.2f%%' %(key_max, results[key_max]*100))

    print('Lowest score at %d features of %.2f%%' %(key_min, results[key_min]*100))

    return key_max
start_time = timeit.default_timer()

results_rf = {}



rf_model = RandomForestClassifier(random_state=SEED)



for i in range(1, len(train_X.columns)): # len(train_X.columns)

    score = k_features(rf_model, train_X, val_X, train_y, val_y, i)

#     print('Accuracy: %.2f%% %s' %(score*100, selected_cols.tolist()))

    results_rf[i] = score

    

plot_results(results_rf)

elapsed = timeit.default_timer() - start_time

print('Elapsed time: %.2f s' %elapsed)
k_num = scores(results_rf)

selector = SelectKBest(f_classif, k=k_num)

X_new = selector.fit_transform(train_X, train_y)

selected_features = pd.DataFrame(selector.inverse_transform(X_new), 

                                 index=train_X.index, 

                                 columns=train_X.columns)

selected_cols = selected_features.columns[selected_features.var() != 0]



kbest_X = train_X[selected_cols]

kval_X = val_X[selected_cols]



rf_model.fit(kbest_X, train_y)

pred = rf_model.predict(kval_X)

score = accuracy_score(val_y, pred)

print('\nAccuracy: %.2f%% vs the baseline score %.2f%% \n\nBest Features using RandomForestClassifier: %s' %(score*100, baseline_score*100, selected_cols.tolist()))

not_selected = selected_features.columns[selected_features.var() == 0]

print('\nFeatures not selected in RandomForestClassifier: %s' %not_selected.tolist())
from xgboost import XGBClassifier



xgb_model = XGBClassifier()

xgb_model.fit(train_X, train_y)

pred = xgb_model.predict(val_X)

score = accuracy_score(val_y, pred)

print('Accuracy: %.2f%% vs the baseline score of %.2f%%' %(score*100, baseline_score*100))



perm = PermutationImportance(xgb_model).fit(val_X, val_y)

eli5.show_weights(perm, feature_names=val_X.columns.tolist())
start_time = timeit.default_timer()

results_xgb = {}



for i in range(1, len(train_X.columns)): # len(train_X.columns)

    score = k_features(xgb_model, train_X, val_X, train_y, val_y, i)

#     print('Accuracy: %.2f%% %s' %(score*100, selected_cols.tolist()))

    results_xgb[i] = score



# print(results_xgb)

plot_results(results_xgb)

elapsed = timeit.default_timer() - start_time

print('Elapsed time: %.2f s' %elapsed)
k_num = scores(results_xgb)

selector = SelectKBest(f_classif, k=k_num)

X_new = selector.fit_transform(train_X, train_y)

selected_features = pd.DataFrame(selector.inverse_transform(X_new), 

                                 index=train_X.index, 

                                 columns=train_X.columns)

selected_cols = selected_features.columns[selected_features.var() != 0]



kbest_X = train_X[selected_cols]

kval_X = val_X[selected_cols]



xgb_model.fit(kbest_X, train_y)

pred = xgb_model.predict(kval_X)

score = accuracy_score(val_y, pred)

print('\nAccuracy: %.2f%% vs the baseline score %.2f%% \n\nBest features using XGBoost: %s' %(score*100, baseline_score*100, selected_cols.tolist()))

not_selected = selected_features.columns[selected_features.var() == 0]

print('\nFeatures not selected in XGBClassifier: %s' %not_selected.tolist())
# Helper function

def make_diffs(df):

    df['killDiff'] = df['blueKills'] - df['redKills']

    df['assistDiff'] = df['blueAssists'] - df['redAssists']

    df['avgLevelDiff'] = df['blueAvgLevel'] - df['redAvgLevel']

    df['eliteMonstersDiff'] = df['blueEliteMonsters'] - df['redEliteMonsters']

    df['towersDiff'] = df['blueTowersDestroyed'] - df['redTowersDestroyed']

    df['wardsPlacedDiff'] = df['blueWardsPlaced'] - df['redWardsPlaced']

    df['wardsDestroyedDiff'] = df['blueWardsDestroyed'] - df['redWardsDestroyed']

    df['minionsDiff'] = df['blueTotalMinionsKilled'] - df['redTotalMinionsKilled']

    df['jungleMinionsDiff'] = df['blueTotalJungleMinionsKilled'] - df['redTotalJungleMinionsKilled']

    df.head()

    return df



clean_data = lol_data.copy()

y = clean_data.blueWins

# clean_data.columns

clean_data = clean_data.drop(columns='blueWins', axis=1)

clean_data.info()
clean_data = make_diffs(clean_data)

print(clean_data.columns.tolist())
r = clean_data.corr()

plt.figure(figsize=(20, 12))

sns.heatmap(r, annot=True, fmt='.2f', center= 0)
correlations = ['redFirstBlood', 'redKills', 'redDeaths', 'blueCSPerMin', 'blueGoldPerMin', 'blueTotalGold', 

                'redGoldDiff', 'redExperienceDiff', 'redTotalGold', 'redCSPerMin', 'redGoldPerMin', 

                'blueTotalExperience', 'redTotalExperience']



other_data = ['blueWardsPlaced', 'blueWardsDestroyed', 'blueKills', 'blueDeaths', 'blueAssists', 

              'blueEliteMonsters', 'blueDragons', 'blueHeralds', 'blueTowersDestroyed', 

              'blueAvgLevel', 'blueTotalMinionsKilled', 'blueTotalJungleMinionsKilled', 

              'redWardsPlaced', 'redWardsDestroyed', 'redAssists', 'redEliteMonsters', 

              'redDragons', 'redHeralds', 'redTowersDestroyed', 'redAvgLevel', 

              'redTotalMinionsKilled', 'redTotalJungleMinionsKilled']

team_diff = clean_data.drop(correlations, axis=1)

team_diff.head()
r = team_diff.corr()

plt.figure(figsize=(20, 12))

sns.heatmap(r, annot=True, fmt='.2f', center= 0)
train_X, val_X, train_y, val_y = train_test_split(team_diff, y, random_state=SEED)

diff_model = RandomForestClassifier(random_state=SEED)

diff_model.fit(train_X, train_y)

pred = diff_model.predict(val_X)

score = accuracy_score(val_y, pred)

print('Accuracy: %.2f%% vs the baseline score %.2f%%' %(score*100, baseline_score*100))
perm = PermutationImportance(diff_model).fit(val_X, val_y)

eli5.show_weights(perm, feature_names=val_X.columns.tolist(), top=None)
sel = SelectFromModel(perm, threshold=0.02, prefit=True)

X_trans = sel.transform(val_X)

X_trans
pi_features = team_diff.filter(['blueGoldDiff', 'killDiff', 'blueExperienceDiff', 'jungleMinionsDiff', 

                                'redTotalMinionsKilled', 'redEliteMonsters', 'redTotalJungleMinionsKilled', 

                                'blueDeaths', 'blueDragons'], axis=1)

tpi_X, vpi_X, tpi_y, vpi_y = train_test_split(pi_features, y, random_state=SEED)

pi_model = RandomForestClassifier(random_state=SEED)

pi_model.fit(tpi_X, tpi_y)

pred = pi_model.predict(vpi_X)

score = accuracy_score(vpi_y, pred)

print('Accuracy: %.2f%% vs the baseline score %.2f%%' %(score*100, baseline_score*100))
start_time = timeit.default_timer()



explainer = shap.TreeExplainer(diff_model)

shap_values_all = explainer.shap_values(val_X)

shap.summary_plot(shap_values_all[0], val_X)



elapsed = timeit.default_timer() - start_time

print('Elapsed time: %s s' %elapsed)
start_time = timeit.default_timer()

results_rf = {}



for i in range(1, len(train_X.columns)): # len(train_X.columns)

    score = k_features(diff_model, train_X, val_X, train_y, val_y, i)

#     print('Accuracy: %.2f%% %s' %(score*100, selected_cols.tolist()))

    results_rf[i] = score

    

plot_results(results_rf)

elapsed = timeit.default_timer() - start_time

print('Elapsed time: %.2f s' %elapsed)
k_num = scores(results_rf)

selector = SelectKBest(f_classif, k=k_num)

X_new = selector.fit_transform(train_X, train_y)

selected_features = pd.DataFrame(selector.inverse_transform(X_new), 

                                 index=train_X.index, 

                                 columns=train_X.columns)

selected_cols = selected_features.columns[selected_features.var() != 0]



kbest_X = train_X[selected_cols]

kval_X = val_X[selected_cols]



diff_model.fit(kbest_X, train_y)

pred = diff_model.predict(kval_X)
score = accuracy_score(val_y, pred)

print('Accuracy: %.2f%% vs a baseline score of %.2f%%\n\nBest Features using RandomForestClassifier: %s' %(score*100, baseline_score*100, selected_cols.tolist()))

not_selected = selected_features.columns[selected_features.var() == 0]

print('\nFeatures not selected in RandomForestClassifier: %s' %not_selected.tolist())
end_time_total = timeit.default_timer() - start_time_total

print('Elapsed time: %.2f s' %end_time_total)