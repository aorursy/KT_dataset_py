import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import numpy as np 

from sklearn import svm

from sklearn.svm import LinearSVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.cross_validation import train_test_split

from sklearn import metrics

from sklearn.model_selection import cross_val_score

from sklearn import preprocessing

from sklearn.feature_selection import RFE

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

import graphviz

from sklearn import tree

import warnings

warnings.filterwarnings('ignore')

import pickle
df = pd.read_csv('../input/1617-boxscore-edited-wl-ha/2016-17_teamBoxScore.csv')

df.head()
df.tail()
df.shape
df.columns
df.isnull().sum().max()
#correlation matrix



corrmat = df.corr()

f, ax = plt.subplots(figsize=(20,18))

sns.heatmap(corrmat, vmax=.8, square=True)
#team result



k = 9

cols = corrmat.nlargest(k, 'teamRslt')['teamRslt'].index

f, ax = plt.subplots(figsize=(10,6))

cm = np.corrcoef(df[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
#team points



k = 12

cols = corrmat.nlargest(k, 'teamPTS')['teamPTS'].index

f, ax = plt.subplots(figsize=(10,6))

cm = np.corrcoef(df[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
#opponent points



k = 12

cols = corrmat.nlargest(k, 'opptPTS')['opptPTS'].index

f, ax = plt.subplots(figsize=(10,6))

cm = np.corrcoef(df[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
#location; h/a



k = 12

cols = corrmat.nlargest(k, 'teamLoc')['teamLoc'].index

f, ax = plt.subplots(figsize=(10,6))

cm = np.corrcoef(df[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
# team days off



k = 12

cols = corrmat.nlargest(k, 'teamDayOff')['teamDayOff'].index

f, ax = plt.subplots(figsize=(10,6))

cm = np.corrcoef(df[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
# team turnovers



k = 12

cols = corrmat.nlargest(k, 'teamTO')['teamTO'].index

f, ax = plt.subplots(figsize=(10,6))

cm = np.corrcoef(df[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
#scatter plots



cols1 = ['opptPTS', 'teamDrtg', 'teamPF', 'teamTO', 'teamORB', 'teamFGA']

sns.pairplot(df[cols1], size=2.5)

plt.show()
#prepare x and y



feature_cols = ['opptPTS', 'teamDrtg', 'teamPF', 'teamTO', 'teamORB', 'teamFGA']

x = df[feature_cols]

y = df['teamRslt']

x.head()
#train test split, standardize data



x_train, x_test, y_train, y_test = train_test_split(x, y , test_size=0.4, random_state=2)
print(x_train.shape)

print(x_test.shape)

print(y_train.shape)

print(y_test.shape)
#knn 



knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(x_train, y_train)

pred = knn.predict(x_test)

print(metrics.accuracy_score(y_test, pred))

print(knn.predict_proba(x_test))
#linear svm



clf = LinearSVC(random_state=2)

clf.fit(x_train, y_train)

print(clf.coef_)

print(clf.intercept_)

pred = (clf.predict(x_test))

#print(pred)

print(metrics.accuracy_score(y_test, pred))
#random forrest classifier



clf = RandomForestClassifier()

clf.fit(x_train, y_train)
print(clf.feature_importances_)
pred = clf.predict(x_test)

#print(pred)

#print(clf.predict_proba(x_test))

print(metrics.accuracy_score(y_test, pred))
# Gradient Treee Boosting



clfgtb = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(x_train, y_train)

clfgtb.score(x_test, y_test)
df2 = pd.read_csv('../input/1718-boxscore-wl-ha/2017-18_teamBoxScore.csv')

df2.head()
#prepare x and y



new_feature_cols = ['opptPTS', 'teamDrtg', 'teamPF', 'teamTO', 'teamORB', 'teamFGA']

x_new = df2[new_feature_cols]

y_new = df2['teamRslt']

x.head()
# gradient tree boosting



clfgtb = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(x, y)

clfgtb.score(x_new, y_new)
filename = 'nba_pred_modelv1.sav'

pickle.dump(clfgtb, open(filename, 'wb'))
#games

games = ['PHX1 vs SAS2', 'DET1 vs PHI2', 'MIN1 vs BOS2', 'NY1 vs MIA2', 'TOR1 vs MIL2', 'CHI1 vs DAL2', 'UTA1 vs DEN2', 'WSH1 vs MEM2', 'ATL1 vs POR2', 'CHA1 vs LAL2']



g1 = [[101.3, 111.9, 22.3, 15.9, 11.3, 87.1]]

g2 = [[108.4, 106.7, 18.7, 14.4, 10.3, 86.0]]

g3 = [[103.4, 109.8, 18.0, 13.2, 10.3, 84.8]]

g4 = [[102.2, 108.0, 20.8, 15.3, 10.8, 86.0]]

g5 = [[105.9, 105.4, 22.1, 13.9, 9.4, 86.5]]

g6 = [[101.6, 108.7, 19.2, 13.8, 9.4, 88.6]]

g7 = [[107.9, 106.8, 20.1, 14.4, 8.4, 82.4]]

g8 = [[98.9, 106.4, 21.7, 13.7, 9.9, 86.1]]

g9 = [[102.5, 110.9, 19.7, 15.5, 9.5, 84.7]]

g10 = [[106.7, 107.4, 18.1, 13.1, 10.5, 86.4]]
nba_pred_modelv1 = pickle.load(open(filename, 'rb'))
pred1 = nba_pred_modelv1.predict(g1)

prob1 = nba_pred_modelv1.predict_proba(g1)

print(pred1)

print(prob1)
pred2 = nba_pred_modelv1.predict(g2)

prob2 = nba_pred_modelv1.predict_proba(g2)

print(pred2)

print(prob2)
pred3 = nba_pred_modelv1.predict(g3)

prob3 = nba_pred_modelv1.predict_proba(g3)

print(pred3)

print(prob3)
pred4 = nba_pred_modelv1.predict(g4)

prob4 = nba_pred_modelv1.predict_proba(g4)

print(pred4)

print(prob4)
pred5 = nba_pred_modelv1.predict(g5)

prob5 = nba_pred_modelv1.predict_proba(g5)

print(pred5)

print(prob5)
pred6 = nba_pred_modelv1.predict(g6)

prob6 = nba_pred_modelv1.predict_proba(g6)

print(pred6)

print(prob6)
pred7 = nba_pred_modelv1.predict(g7)

prob7 = nba_pred_modelv1.predict_proba(g7)

print(pred7)

print(prob7)
pred8 = nba_pred_modelv1.predict(g8)

prob8 = nba_pred_modelv1.predict_proba(g8)

print(pred8)

print(prob8)
pred9 = nba_pred_modelv1.predict(g9)

prob9 = nba_pred_modelv1.predict_proba(g9)

print(pred9)

print(prob9)
pred10 = nba_pred_modelv1.predict(g10)

prob10 = nba_pred_modelv1.predict_proba(g10)

print(pred10)

print(prob10)
d = {'Game': games, 'Prediction':[pred1, pred2, pred3, pred4, pred5, pred6, pred7, pred8, pred9, pred10], 'Probability (1, 2)': [prob1, prob2, prob3, prob4, prob5, prob6, prob7, prob8, prob9, prob10], 'Actual Result': [2 , 2, 2, 1, 1, 1, 2, 1, 2, 1]}

df3 = pd.DataFrame(data = d)

df3