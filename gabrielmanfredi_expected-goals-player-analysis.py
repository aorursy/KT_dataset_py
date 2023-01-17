import pandas as pd

import numpy as np

import matplotlib as plt

from matplotlib import pyplot

import scipy as sp

from xgboost import XGBClassifier

import sklearn

from sklearn.metrics import average_precision_score

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report,confusion_matrix

from sklearn.metrics import cohen_kappa_score

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import RandomForestClassifier

from sklearn.neural_network import MLPClassifier
filename = 'events.csv'

events = pd.read_csv('../input/events.csv')

shots = events[(events.event_type==1)]

shots_prediction = shots.iloc[:,-6:]

dummies = pd.get_dummies(shots_prediction, columns=['location', 'bodypart','assist_method', 'situation'])

dummies.columns = ['is_goal', 'fast_break', 'loc_centre_box', 'loc_diff_angle_lr', 'diff_angle_left', 'diff_angle_right', 'left_side_box', 'left_side_6ybox', 'right_side_box', 'right_side_6ybox', 'close_range', 'penalty', 'outside_box', 'long_range', 'more_35y', 'more_40y', 'not_recorded', 'right_foot', 'left_foot', 'header', 'no_assist', 'assist_pass', 'assist_cross', 'assist_header', 'assist_through_ball', 'open_play', 'set_piece', 'corner', 'free_kick']

dummies.head()
X = dummies.iloc[:,1:]

y = dummies.iloc[:,0]

print(X.shape)

print(y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=1)
classifier = XGBClassifier(objective='binary:logistic', max_depth=5, n_estimators=100)

classifier.fit(X_train, y_train)
accuracy = classifier.score(X_test, y_test)

y_pred = classifier.predict_proba(X_test)

predict = classifier.predict(X_test)

y_total = y_train.count()

y_positive = y_train.sum()

auc_roc = roc_auc_score(y_test, y_pred[:, 1])

print('The training set contains {} examples (shots) of which {} are positives (goals).'.format(y_total, y_positive))

print('The accuracy of classifying whether a shot is goal or not is {:.2f} %'.format(accuracy*100))

print('Our classifier obtains an AUC-ROC of {:.4f}.'.format(auc_roc))
auc_pr_baseline = y_positive / y_total

print('The baseline performance for AUC-PR is {:.2f}. This is the AUC-PR that what we would get by random guessing.'.format(auc_pr_baseline))



auc_pr = average_precision_score(y_test, y_pred[:, 1])

print('Our classifier obtains an AUC-PR of {:.4f}.'.format(auc_pr))

cohen_kappa = cohen_kappa_score(y_test,predict)

print('Our classifier obtains a Cohen Kappa of {:.4f}.'.format(cohen_kappa))
print('Confusion Matrix:')

print(confusion_matrix(y_test,predict))

print('Report:')

print(classification_report(y_test,predict))
predictions = X_test.copy()

predictions['true_goals'] = y_test

predictions['expected_goals'] = y_pred[:,1]

predictions['difference'] = predictions['expected_goals'] - predictions['true_goals']

predictions = predictions.iloc[:,28:31]

predictions.head()
logistic_regression = LogisticRegression()

logistic_regression.fit(X_train, y_train)
logistic_regression.score(X_train, y_train)

accuracy = logistic_regression.score(X_test, y_test)

y_pred = logistic_regression.predict_proba(X_test)

accuracy_logreg = logistic_regression.score(X_test, y_test)

y_pred_logreg = logistic_regression.predict_proba(X_test)

predict_logreg = logistic_regression.predict(X_test)

y_total_logreg = y_train.count()

y_positive_logreg = y_train.sum()

auc_roc_logreg = roc_auc_score(y_test, y_pred_logreg[:, 1])

print('The training set contains {} examples (shots) of which {} are positives (goals).'.format(y_total_logreg, y_positive_logreg))

print('The accuracy of classifying whether a shot is goal or not is {:.2f} %'.format(accuracy_logreg*100))

print('Our classifier obtains an AUC-ROC of {:.4f}.'.format(auc_roc_logreg))



auc_pr_baseline = y_positive / y_total

print('The baseline performance for AUC-PR is {:.4f}. This is the AUC-PR that what we would get by random guessing.'.format(auc_pr_baseline))

auc_pr_logreg = average_precision_score(y_test, y_pred_logreg[:, 1])

print('Our classifier obtains an AUC-PR of {:.4f}.'.format(auc_pr_logreg))

cohen_kappa_logreg = cohen_kappa_score(y_test,predict_logreg)

print('Our classifier obtains a Cohen Kappa of {:.4f}.'.format(cohen_kappa_logreg))
print('Confusion Matrix:')

print(confusion_matrix(y_test,predict_logreg))

print('Report:')

print(classification_report(y_test,predict_logreg))
coefficients = pd.Series(logistic_regression.coef_[0], X_train.columns)

print(coefficients)
mlp = MLPClassifier(random_state=0, hidden_layer_sizes=(28,28,28,28), max_iter=2000, activation='relu')

mlp.fit(X_train, y_train)
mlp.score(X_train, y_train)

mlp.score(X_test, y_test)

accuracy = mlp.score(X_test, y_test)

print('The accuracy of classifying whether a shot is goal or not is {:.2f} %.'.format(accuracy*100))

y_pred = mlp.predict_proba(X_test)

predict = mlp.predict(X_test)

y_total = y_train.count()

y_positive = y_train.sum()

print('The training set contains {} examples of which {} are positives.'.format(y_total, y_positive))

auc_roc = roc_auc_score(y_test, y_pred[:,1])

print('Our MLP classifier obtains an AUC-ROC of {:.4f}.'.format(auc_roc))

auc_pr_baseline = y_positive / y_total

print('The baseline performance for AUC-PR is {:.4f}. This is what we would get by random guessing'.format(auc_pr_baseline))

auc_pr = average_precision_score(y_test, y_pred[:,1])

print('Our MLP classifier obtains an AUC-PR of {:.4f}.'.format(auc_pr))

cohen_kappa = cohen_kappa_score(y_test,predict)

print('Our classifier obtains a Cohen Kappa of {:.4f}.'.format(cohen_kappa))

MSE = sklearn.metrics.mean_squared_error(y_test, y_pred[:,1])

print('Our MLP classifier obtains a Mean Squared Error (MSE) of {:.4f}.'.format(MSE))
print('Confusion Matrix:')

print(confusion_matrix(y_test,predict))

print('Report:')

print(classification_report(y_test,predict))
predictions = X_test.copy()

predictions['true_goals'] = y_test

predictions['expected_goals'] = y_pred[:,1]

predictions['difference'] = predictions['expected_goals'] - predictions['true_goals']

predictions = predictions.iloc[:,28:31]
ypred2 = mlp.predict_proba(X_train)

predictions_train = X_train.copy()

predictions_train['true_goals'] = y_train

predictions_train['expected_goals'] = ypred2[:,1]

predictions_train['difference'] = predictions_train['expected_goals'] - predictions_train['true_goals']

predictions_train = predictions_train.iloc[:,28:31]

all_predictions = pd.concat([predictions, predictions_train], axis=0)

events2 = pd.concat([events, all_predictions], axis=1)

shots2 = events2[events2.event_type==1]
xG_players = shots2[['player', 'event_type', 'true_goals', 'expected_goals', 'difference']].groupby('player').sum()

xG_players.columns = ['n_shots', 'goals_scored', 'expected_goals', 'difference']

xG_players[['goals_scored', 'expected_goals']].corr()

xG_players.sort_values(['difference', 'goals_scored'])
xG_players.sort_values(['expected_goals'], ascending=False)
xG_players['xG_per_shot_ratio'] = xG_players['expected_goals'] / xG_players['n_shots']

xG_players[xG_players.n_shots>100].sort_values(['xG_per_shot_ratio', 'goals_scored'])
headers = events2[(events2.event_type==1) & (events2.bodypart==3)]

headers_players = headers[['player', 'event_type', 'true_goals', 'expected_goals', 'difference']].groupby('player').sum()

headers_players.columns = ['n_headers', 'goals_scored', 'expected_goals', 'difference']

headers_players.sort_values(['difference'])

left_foot = events2[(events2.event_type==1) & (events2.bodypart==2)]

left_foot_players = left_foot[['player', 'event_type', 'true_goals', 'expected_goals', 'difference']].groupby('player').sum()

left_foot_players.columns = ['n_left_foot_shots', 'goals_scored', 'expected_goals', 'difference']

left_foot_players.sort_values(['difference'])

left_foot_players.loc['cristiano ronaldo']
right_foot = events2[(events2.event_type==1) & (events2.bodypart==1)]

right_foot_players = right_foot[['player', 'event_type', 'true_goals', 'expected_goals', 'difference']].groupby('player').sum()

right_foot_players.columns = ['n_right_foot_shots', 'goals_scored', 'expected_goals', 'difference']

right_foot_players.sort_values(['difference'])
outside_box = shots2[(shots2.location==15)]

outbox_players = outside_box[['player', 'event_type', 'true_goals', 'expected_goals', 'difference']].groupby('player').sum()

outbox_players.columns = ['n_outside_box_shots', 'goals_scored', 'expected_goals', 'difference']

outbox_players.sort_values(['difference'])

passes_and_throughballs = pd.concat([shots2[shots2.assist_method==1], shots2[shots2.assist_method==4]])

assisting_players = passes_and_throughballs[['player2', 'event_type', 'true_goals', 'expected_goals', 'difference']].groupby('player2').sum()

assisting_players['xGoals_per_pass'] = assisting_players['expected_goals'] / assisting_players['event_type']

assisting_players.columns = ['n_passes', 'goals_scored_from_passes', 'xGoals_from_passes', 'difference', 'xGoals_per_pass']



assisting_players[assisting_players.n_passes > 100].sort_values(['xGoals_per_pass'], ascending=False)



crosses = shots2[shots2.assist_method==2]

crosses_players = shots2[['player2', 'event_type', 'true_goals', 'expected_goals', 'difference']].groupby('player2').sum()

crosses_players.columns = ['n_crosses', 'goals_scored_from_crosses', 'xGoals_from_crosses', 'difference']

crosses_players['xGoals_per_cross'] = crosses_players['xGoals_from_crosses'] / crosses_players['n_crosses']

crosses_players.columns = ['n_crosses', 'goals_scored_from_crosses', 'xGoals_from_crosses', 'difference', 'xGoals_per_cross']

crosses_players[crosses_players.n_crosses > 50].sort_values(['xGoals_per_cross'], ascending=False)



print('Passes and Through-Balls:')

assisting_players.sort_values(['difference'], ascending=False)
print('Crosses:')

crosses_players.sort_values(['difference'], ascending=False)