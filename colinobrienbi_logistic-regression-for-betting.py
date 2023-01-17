# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sqlite3 



with sqlite3.connect('../input/database.sqlite') as con:

    sql = "SELECT * FROM Match"

    match_data = pd.read_sql_query(sql, con)

match_data.head()
match_data.dtypes
match_data = match_data.loc[match_data['league_id'] == 1729]



columns = ['season',

           'stage',

           'match_api_id',

           'home_team_api_id',

           'away_team_api_id',

           'home_team_goal',

           'away_team_goal']



match_data = match_data[columns]

match_data['goal_difference'] = match_data['home_team_goal'] - match_data['away_team_goal']



match_data.head()
columns=['season', 

         'stage', 

         'match_id', 

         'team', 

         'opponent', 

         'team_goals', 

         'opponent_goals', 

         'goal_difference']



md1 = match_data

md2 = match_data



md1.columns = columns

md2.columns = columns



md2 = md2.rename(columns={'opponent':'team','team':'opponent','team_goals':'opponent_goals','opponent_goals':'team_goals'})



md1['home_away'] = 1

md2['home_away'] = 0



md2['goal_difference'] = md2['goal_difference'] *-1



match_data = md1.append(md2)

match_data.sort_values(['season', 'team', 'stage'], inplace=True)

match_data.head()
match_data['avg_team_goals'] = (match_data.groupby(['season','team'])['team_goals'].shift(1)+

match_data.groupby(['season','team'])['team_goals'].shift(2)+

match_data.groupby(['season','team'])['team_goals'].shift(3))/3



match_data['avg_goal_diff'] = (match_data.groupby(['season','team'])['goal_difference'].shift(1)+

match_data.groupby(['season','team'])['goal_difference'].shift(2)+

match_data.groupby(['season','team'])['goal_difference'].shift(3))/3



match_data['avg_goals_conc'] = (match_data.groupby(['season','team'])['opponent_goals'].shift(1)+

match_data.groupby(['season','team'])['opponent_goals'].shift(2)+

match_data.groupby(['season','team'])['opponent_goals'].shift(3))/3



match_data['avg_goals_conc'] = match_data['avg_goals_conc'] *-1



match_data = match_data.loc[match_data['avg_team_goals'].notnull()]
match_data.head()
match_data = match_data[['match_id',

                         'home_away',

                         'stage',

                         'team_goals',

                         'avg_team_goals',

                         'avg_goal_diff',

                         'avg_goals_conc']]



match_data = match_data.reset_index(drop=True)



match_data['stage'] = pd.qcut(match_data['stage'], 5, labels=[1,2,3,4,5])
md1 = match_data.loc[match_data['home_away'] == 1]

md2 = match_data.loc[match_data['home_away'] == 0]
match_data = md1.merge(md2, on='match_id', how='left')

match_data['total_goals'] = match_data['team_goals_x'] + match_data['team_goals_y']

match_data = match_data.drop(['home_away_x', 'home_away_y', 'stage_y', 'match_id', 'team_goals_x', 'team_goals_y'], axis=1)

match_data.head()
match_data = match_data.rename(columns={'stage_x':'stage'})

match_data['stage'] = pd.qcut(match_data['stage'], 5, labels=[1,2,3,4,5])
X = match_data[['stage', 'avg_team_goals_x', 'avg_goal_diff_x', 'avg_goals_conc_x',

                'avg_team_goals_y', 'avg_goal_diff_y', 'avg_goals_conc_y']]



match_data['total_goals'] = np.where(match_data['total_goals']>2.5, 1, 0)



y = match_data['total_goals']



X = X.as_matrix()

y = y.as_matrix()



from sklearn.preprocessing import OneHotEncoder

onehotencoder = OneHotEncoder(categorical_features = [0])

X = onehotencoder.fit_transform(X).toarray()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)



from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)

X_test = sc_X.transform(X_test)
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state = 0)

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

y_pred_p = classifier.predict_proba(X_test)
final_pred = pd.concat([pd.DataFrame(y_test),pd.DataFrame(y_pred), pd.DataFrame(y_pred_p)], axis=1)

final_pred.columns = ['real', 'pred', 'prob_0', 'prob_1']

final_pred['hit'] = np.where((final_pred['real']==final_pred['pred']), 1, 0)

sum(final_pred.hit)/len(final_pred)
final_pred.head()
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, y_pred)
from sklearn.model_selection import cross_val_score

classifier = LogisticRegression(random_state = 0)

print(cross_val_score(classifier, X, y,cv=5))
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier()

classifier.fit(X_train, y_train).feature_importances_
classifier = RandomForestClassifier()

print(cross_val_score(classifier, X, y,cv=10))
len(X_test)
from sklearn.svm import SVC

classifier = SVC()

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

confusion_matrix(y_test, y_pred)
classifier.score(X,y)
from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors = 5)

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

confusion_matrix(y_test, y_pred)