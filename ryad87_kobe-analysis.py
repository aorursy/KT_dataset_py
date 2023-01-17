import numpy as np

import pandas as pd

import scipy as sp

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.ensemble import RandomForestClassifier

from sklearn.cross_validation import KFold

#from sklearn.ensemble import RandomForestRegressor

#from sklearn.metrics import confusion_matrix

#import time



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Einlesen der Daten

raw_data = pd.read_csv("../input/data.csv")
# Ausgabe der ersten Zeilen der Daten, um Überblick über Daten zu bekommen

raw_data.head()
# Welche Würfe müssen nicht vorhergesagt werden (gemessene Würfe ohne Missing Values)?

shot_made_flag = raw_data['shot_made_flag']

no_missing_values = raw_data[pd.notnull(shot_made_flag)]



# Ausgabe der ersten Würfe, die nicht vorhergesagt werden müssen

no_missing_values.filter(items=['shot_id', 'shot_made_flag']).head()
# 

opacity = 0.01

plt.figure(figsize=(10,10))



# Plotten der loc_x und loc_y Koordinaten von gemessenen Würfen

plt.subplot(121)

plt.scatter(no_missing_values.loc_x, no_missing_values.loc_y, color='red', alpha=opacity)

plt.title('loc_x and loc_y')



# Plotten der lon und lat Koordinaten von gemessenen Würfen

plt.subplot(122)

plt.scatter(no_missing_values.lon, no_missing_values.lat, color='green', alpha=opacity)

plt.title('lat and lon')



# --> Visuell zu erkennen: Offensichtlich korrelieren lon/lat und loc_c/loc_y



# Treffer/kein Treffer gegen Schussdistanz

plt.figure(figsize=(10,1))

plt.scatter(no_missing_values['shot_distance'], no_missing_values['shot_made_flag'], color='blue', alpha=opacity)
# Berechnen der restlichen Spielzeit anhand von minutes_remaining und seconds_remaining

raw_data['game_seconds_remaining'] = raw_data['minutes_remaining'] * 60 + raw_data['seconds_remaining']

raw_data.filter(items=['shot_id', 'shot_made_flag', 'game_seconds_remaining']).head()
# Ausgabe aller Saisons

print(raw_data['season'].unique())

# --> Wir können offensichtlich den zweiten Teil der Season (bspw. 01 aus "2000-01")

# als ordinale Variable verwenden.



def split_season(season):

    """

    Hilfsfunktion zum bekommen des zweiten Teils der Season (bspw. 01 aus "2000-01").

    :param season: Eingabestring (bspw. 2000-01)

    :return: Zweites Element bei Split auf "-" bspw. 01 aus 2000-01

    """

    return season.split("-")[1]



# Speichern der season_ordinal durch Anwendung der split_season() auf season

raw_data['season_ordinal'] = raw_data['season'].apply(split_season)

raw_data.filter(items=['shot_id', 'shot_made_flag', 'season', 'season_ordinal']).head()
# Wir benötigen viele Spalten für die Berechnung der Vorhersage nicht, diese müssen aus dem Datensatz

# entfernt werden.

# Wir benötigen: action_type, combined_action_type, period, playoffs, shot_made_flag, shot_type,

#                opponent, game_seconds_remaining, season_ordinal

#

# Alle anderen Spalten können entfernt werden.

not_needed = ['shot_id', 'team_id', 'team_name', 'shot_zone_area', 'shot_zone_range',

              'shot_zone_basic', 'matchup', 'lon', 'lat', 'seconds_remaining', 'minutes_remaining',

              'shot_distance', 'loc_x', 'loc_y', 'game_event_id', 'game_id', 'game_date', 'season']



# Entfernen der Spalten in not_needed aus raw_data

for drop in not_needed:

    if drop in raw_data:

        raw_data = raw_data.drop(drop, 1)



# Ausgabe der verbleibenden Daten

raw_data.head()
# Wir müssen kategorische Spalten (bspw. action_type) in ordinale Daten umwandeln

categorical_data = ['action_type', 'combined_shot_type', 'shot_type', 'opponent', 'period', 'season']



# Wir nutzen pd.get_dummies() zur Umwandlung von kategorischen in ordinale Daten und entfernen

# gleichzeitig mit drop() die kategorischen Spalten.

for categorical_column in categorical_data:

    if categorical_column in raw_data:

        raw_data = pd.concat([raw_data, pd.get_dummies(raw_data[categorical_column], prefix=categorical_column)], 1)

        raw_data = raw_data.drop(categorical_column, 1)



raw_data.head()
# measured_data: Daten mit gemessenen Treffern/Fehlversuchen

measured_data = raw_data[pd.notnull(raw_data['shot_made_flag'])]



# prediction_data: Vorauszusagenende Daten exkl. shot_made_flag

prediction_data = raw_data[pd.isnull(raw_data['shot_made_flag'])]

prediction_data = prediction_data.drop('shot_made_flag', 1)



# train_data: Trainingsdaten gemessener Werte exkl. shot_made_flag

# train_data_shot: Trainingsdaten shot_made_flag gemessener Werte

train_data = measured_data.drop('shot_made_flag', 1)

train_data_shot = measured_data['shot_made_flag']
def log_loss(actual, predicted, epsilon=1e-15):

    """

    Berechnet die Verlustfunktion (engl.: Log Loss). Sie ordnet jeder Entscheidung

    in Form einer Punktschätzung, einer Bereichsschätzung oder eines Tests den

    Schaden zu, der durch eine vom wahren Parameter abweichende Entscheidung entsteht.

    :param actual: Wahrer Parameter

    :param predicted: Vorhergesagter Parameter

    :return: Log-Loss

    """

    predicted = sp.maximum(epsilon, predicted)

    predicted = sp.minimum(1-epsilon, predicted)

    log_loss = sum(actual * sp.log(predicted) + sp.subtract(1, actual)

                   * sp.log(sp.subtract(1, predicted)))

    log_loss = log_loss * -1.0/len(actual)

    return log_loss
"""

# Wir wollen das Random Forest Klassifikationsverfahren verwenden



# find the best n_estimators for RandomForestClassifier

#print('Finding best n_estimators for RandomForestClassifier...')

min_score = 100000

best_n = 0

scores_n = []

range_n = np.logspace(0, 2, num=3).astype(int)

for n in range_n:

    #print("the number of trees : {0}".format(n))

    #t1 = time.time()

    

    rfc_score = 0.

    rfc = RandomForestClassifier(n_estimators=n)

    for train_k, test_k in KFold(len(train_data), n_folds=10, shuffle=True):

        rfc.fit(train_data.iloc[train_k], train_data_shot.iloc[train_k])

        #rfc_score += rfc.score(train_data_x.iloc[test_k], train_data_y.iloc[test_k])/10

        prediction = rfc.predict(train_data.iloc[test_k])

        rfc_score += log_loss(train_data_shot.iloc[test_k], prediction) / 10

    scores_n.append(rfc_score)

    if rfc_score < min_score:

        min_score = rfc_score

        best_n = n

        

    #t2 = time.time()

    #print('Done processing {0} trees ({1:.3f}sec)'.format(n, t2-t1))

print("best_n: {}, min_score: {}".format(best_n, min_score))





# find best max_depth for RandomForestClassifier

#print('Finding best max_depth for RandomForestClassifier...')

min_score = 100000

best_m = 0

scores_m = []

range_m = np.logspace(0, 2, num=3).astype(int)

for m in range_m:

    #print("the max depth : {0}".format(m))

    #t1 = time.time()

    

    rfc_score = 0.

    rfc = RandomForestClassifier(max_depth=m, n_estimators=best_n)

    for train_k, test_k in KFold(len(train_data), n_folds=10, shuffle=True):

        rfc.fit(train_data.iloc[train_k], train_data_shot.iloc[train_k])

        #rfc_score += rfc.score(train_data_x.iloc[test_k], train_data_y.iloc[test_k])/10

        prediction = rfc.predict(train_data.iloc[test_k])

        rfc_score += log_loss(train_data_shot.iloc[test_k], prediction) / 10

    scores_m.append(rfc_score)

    if rfc_score < min_score:

        min_score = rfc_score

        best_m = m

    

    #t2 = time.time()

    #print('Done processing {0} trees ({1:.3f}sec)'.format(m, t2-t1))

print("best_m: {}, min_score: {}".format(best_m, min_score))

"""
# Durch Visualisierung der Parameter range_n/range_m und scores_n/scores_m

# können wir prüfen, ob die gewählten Parameter die besten sind.

"""

plt.figure(figsize=(10,5))

plt.subplot(121)

plt.plot(range_n, scores_n)

plt.ylabel('score')

plt.xlabel('number of trees')



plt.subplot(122)

plt.plot(range_m, scores_m)

plt.ylabel('score')

plt.xlabel('max depth')

"""
# berechne RandomForestClassifier Modell und fitte mit Traningsdaten

# RandomForestClassifier

# - n_estimators: Anzahl Bäume

# - max_depth: Anzahl Tiefe der Bäume

n_estimators = 15

max_depth = 10

model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)

model.fit(train_data, train_data_shot)



# berechne Vorhersage für Missing Values aus shot_made_flag

prediction = model.predict_proba(prediction_data)



# lese Beispiel-Abgabe und ersetze Dummy Values (0.5 für shot_made_flag) mit Vorhersagen

sample_submission = pd.read_csv("../input/sample_submission.csv")

sample_submission['shot_made_flag'] = prediction



# lese Roh-Daten erneut, um sämtliche Ursprungsdaten zu haben (inkl. Missing Values)

raw_data2 = pd.read_csv("../input/data.csv")

sub_index = 0

for index, row in pd.DataFrame(raw_data2).iterrows():

    # setze Missing Values für shot_made_flag mit Vorhersagen

    if sub_index < len(sample_submission) and sample_submission.loc[sub_index]['shot_id'] == row['shot_id']:

        raw_data2.loc[index, 'shot_made_flag'] = sample_submission.loc[sub_index]['shot_made_flag']

        sub_index += 1



# speichere Datensatz inkl. Vorhersagen in final_submission.csv

raw_data2.to_csv("final_submission.csv", index=False)



# zeige nun shot_id und shot_made_flag aller Datensätze (inkl. Vorhersagen) an

raw_data2.filter(items=['shot_id', 'shot_made_flag']).head(10)
plt.figure(figsize=(5, 5))



# filtere gemessene Würfe (shot_made_flag == 0 oder 1), um vorhergesagte Werte zu plotten

predicted_data = raw_data2[raw_data2['shot_made_flag'] > 0]

predicted_data = predicted_data[predicted_data['shot_made_flag'] < 1]



plt.scatter(predicted_data['shot_distance'], predicted_data['shot_made_flag'], color='blue', alpha=0.4)

# --> Zu sehen: Die Wahrscheinlichkeit, dass ein Wurf ein Treffer ist, nimmt ab, je größer

#               die Distanz wird. Interessant: Einige Würfe ab 45 ft haben eine hohe Treffer-

#               wahrscheinlichkeit.