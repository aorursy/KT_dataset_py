# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/FIFA 2018 Statistics.csv')
data.head()
len(data)
data.isna().sum()
sns.set(rc={'figure.figsize':(8,6)})

sns.distplot(data['Goal Scored'])
data['Goal Scored'].value_counts(normalize=True)
goals_scored = 0

for i in range(0,3):

    goals_scored += data['Goal Scored'].value_counts(normalize=True)[i]

print("Teams scored less than 3 goals in %s%% of the games "%(100*round(goals_scored,4)))
data.head()
attacks = data[['Team', 'Goal Scored', 'Round']]
all_attacks = attacks.groupby('Team').sum()

all_attacks = all_attacks.reset_index()

all_attacks = all_attacks.sort_values('Goal Scored', ascending=False)
g = sns.catplot(x='Team', y='Goal Scored', data=all_attacks, kind="bar", aspect=2)

g.set_xticklabels(rotation=90)
group_attack = attacks[attacks.Round == 'Group Stage']

group_attack = group_attack.groupby('Team').sum()

group_attack = group_attack.reset_index()

group_attack = group_attack.sort_values('Goal Scored', ascending=False)
g = sns.catplot(x='Team', y='Goal Scored', data=group_attack, kind="bar", aspect=2)

g.set_xticklabels(rotation=90)
mean_matchs = data.groupby('Team').mean()

mean_matchs = mean_matchs.reset_index()
g = sns.jointplot(mean_matchs['Goal Scored'], mean_matchs['Attempts'], kind="kde", height=7)
mean_matchs['Precision'] = mean_matchs['Goal Scored'] / mean_matchs['Attempts']
g = sns.catplot(x='Team', y='Precision', data=mean_matchs.sort_values('Precision', ascending=False), kind="bar", aspect=2)

g.set_xticklabels(rotation=90)
mean_matchs.head()
data = pd.read_csv('../input/FIFA 2018 Statistics.csv')
from sklearn.preprocessing import LabelEncoder



lb_labels = LabelEncoder()

data["Man of the Match"] = lb_labels.fit_transform(data['Man of the Match'])

data["Round"] = lb_labels.fit_transform(data['Round'])

data["PSO"] = lb_labels.fit_transform(data['PSO'])
data = data.drop(['Date', '1st Goal', 'Own goals', 'Own goal Time'], axis=1)
data = data.rename(columns = {'Man of the Match':'MoM'})
data.columns = data.columns.str.replace(" ", "_")
data.head()
data.corr()['MoM'].sort_values(ascending=False)
X = data.drop(['MoM'], axis=1)

y = data['MoM']
mom_corr = data.corr()['MoM']
mom_corr = pd.DataFrame({'col':mom_corr.index, 'correlation':mom_corr.values})
no_corr_cols = mom_corr[(mom_corr.correlation < 0.1) & (mom_corr.correlation > -0.1)]

no_corr_cols = list(no_corr_cols.col)
# Droping columns with no correlation

X = X.drop(no_corr_cols, axis=1)
from sklearn.model_selection import train_test_split



indices = data.index.values.tolist()



X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(X, y, indices, test_size=0.1, random_state=42)

# test_size = 0.1 because not enough to have a good precision in the model
len(X_train)
len(X_test)
# Removing teams and getting test teams

X_train = X_train.drop(['Team', 'Opponent'], axis=1)

test_teams = X_test[['Team', 'Opponent']]

X_test = X_test.drop(['Team', 'Opponent'], axis=1)
results_mom = pd.DataFrame({

    "Team": test_teams["Team"],

    "Opponent": test_teams['Opponent'],

    "MoM_true": y_test

    })
from sklearn.neighbors import KNeighborsClassifier



neighbors = range(2,10)

precision_knn = dict()



for i in neighbors:

    clf = KNeighborsClassifier(n_neighbors=i)

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    results_mom['MoM_pred'] = y_pred

    precision = 100*round(clf.score(X_test, y_test),4)

    precision_knn[i] = precision

    print('Neighbors : ', i, '-> Precision : %s' %precision)

    

best_neighbors = max(precision_knn, key=precision_knn.get)
from sklearn.ensemble import RandomForestClassifier



n_estimators = range(2,15)

precision_rdf = dict()



for i in n_estimators:

    clf = RandomForestClassifier(n_estimators=i,random_state=42)

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    results_mom['MoM_pred'] = y_pred

    precision = 100*round(clf.score(X_test, y_test),4)

    precision_rdf[i] = precision

    print('Estimators : ', i, '-> Precision : %s' %precision)



best_estimators = max(precision_rdf, key=precision_rdf.get)
from sklearn.linear_model import LogisticRegression



classifiers = [KNeighborsClassifier(n_neighbors=best_neighbors), LogisticRegression(random_state=42), 

               RandomForestClassifier(n_estimators=best_estimators, random_state=42)]

names = ['KNN', 'Logistic', 'Random Forest']



for name, clf in zip(names, classifiers):

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    results_mom['MoM_pred'] = y_pred

    precision = 100*round(clf.score(X_test, y_test),4)

    print('Model : ', name, '-> Precision : %s' %precision)
clf = LogisticRegression(random_state=42)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

results_mom['MoM_pred'] = y_pred

precision = 100*round(clf.score(X_test, y_test),4)

print('Precision : %s' %precision)
results_mom