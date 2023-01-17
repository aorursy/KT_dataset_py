# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.externals import joblib

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("./"))

# Any results you write to the current directory are saved as output.
matches = pd.read_csv('../input/WorldCupMatches.csv')
players = pd.read_csv('../input/WorldCupPlayers.csv')
cups = pd.read_csv('../input/WorldCups.csv')
matches = matches.dropna()
players = players.dropna()
cups = cups.dropna()

cups
#number of WC championship ploting
plt.figure(figsize=(12,6))
sns.countplot(cups['Winner'])
#replace German DR and Germany FR by Germany
#replace Soviet Union by Russia
def replace_name(df):
    if(df['Home Team Name'] in ['German DR', 'Germany FR']):
        df['Home Team Name'] = 'Germany'
    elif(df['Home Team Name'] == 'Soviet Union'):
        df['Home Team Name'] = 'Russia'
    
    if(df['Away Team Name'] in ['German DR', 'Germany FR']):
        df['Away Team Name'] = 'Germany'
    elif(df['Away Team Name'] == 'Soviet Union'):
        df['Away Team Name'] = 'Russia'
    return df
    
matches = matches.apply(replace_name, axis='columns')

matches
#create a dictionary of football team
team_name = {}
index = 0
for idx, row in matches.iterrows():
    name = row['Home Team Name']
    if(name not in team_name.keys()):
        team_name[name] = index
        index += 1
    name = row['Away Team Name']
    if(name not in team_name.keys()):
        team_name[name] = index
        index += 1
        
team_name
#drop unecessary columns
dropped_matches = matches.drop(['Datetime', 'Stadium', 'Referee', 'Assistant 1', 'Assistant 2', 'RoundID','Win conditions',
             'Home Team Initials', 'Away Team Initials', 'Half-time Home Goals', 'Half-time Away Goals',
             'Attendance', 'City', 'MatchID', 'Stage'], 1)

#Make a serie counting the number of time each team became WC champion
championships = cups['Winner'].map(lambda p: 'Germany' if p=='Germany FR' else p).value_counts()
championships
#append 'Home Team Championships' and 'Away Team Championships': Number of times being the champion of WC
dropped_matches['Home Team Championship'] = 0
dropped_matches['Away Team Championship'] = 0

def count_championship(df):
  if(championships.get(df['Home Team Name']) != None):
    df['Home Team Championship'] = championships.get(df['Home Team Name'])
  if(championships.get(df['Away Team Name']) != None):
    df['Away Team Championship'] = championships.get(df['Away Team Name'])
  return df


dropped_matches = dropped_matches.apply(count_championship, axis='columns')

dropped_matches
#find who won: Home win: 1, Away win: 2, Draw: 0
dropped_matches['Winner'] = '-'

def find_winner(df):
    if(int(df['Home Team Goals']) == int(df['Away Team Goals'])):
        df['Winner'] = 0
    elif(int(df['Home Team Goals']) > int(df['Away Team Goals'])):
        df['Winner'] = 1
    else:
        df['Winner'] = 2
    return df

dropped_matches = dropped_matches.apply(find_winner, axis='columns')

dropped_matches
#replace team name by id in team_name dictionary

def replace_team_name_by_id(df):
    df['Home Team Name'] = team_name[df['Home Team Name']]
    df['Away Team Name'] = team_name[df['Away Team Name']]
    #df['Winner'] = team_name[df['Winner']]
    return df

teamid_matches = dropped_matches.apply(replace_team_name_by_id, axis='columns')
teamid_matches
#drop unecessary columns
teamid_matches = teamid_matches.drop(['Year', 'Home Team Goals', 'Away Team Goals'], 1)

teamid_matches
#TRAINING STEPS


#create numpy array
X = teamid_matches.loc[:,['Home Team Name', 'Away Team Name', 'Home Team Championship','Away Team Championship']]
X = np.array(X).astype('float64')

#append data: simply exchange 'home team name' with 'away team name', 'home team championship' with 'away team championship', and replace the result
_X = X.copy()

_X[:,0] = X[:,1]
_X[:,1] = X[:,0]
_X[:,2] = X[:,3]
_X[:,3] = X[:,2]

y = dropped_matches.loc[:,['Winner']]
y = np.array(y).astype('int')
y = np.reshape(y,(1,850))
y = y[0]


_y = y.copy()

for i in range(len(_y)):
  if(_y[i]==1):
    _y[i] = 2
  elif(_y[i] ==2):
    _y[i] = 1
    
#===========
    
X = np.concatenate((X,_X), axis= 0)

y = np.concatenate((y,_y))



#shuffle and split test, train
X,y = shuffle(X,y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

print(y_train)

#train by svm
# param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
#               'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
param_grid = {'C': [1e3],
              'gamma': [0.0001] }
svm_model = SVC(kernel='rbf', class_weight='balanced', probability=True)
svm_model.fit(X, y)
#print("Best estimator found by grid search:")
#print(svm_model.best_estimator_)
print("Predicting on the test set")
#t0 = time()
y_pred = svm_model.predict(X_test)
#print("done in %0.3fs" % (time() - t0))

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred, labels=range(3)))
def prediction(team1, team2):
  id1 = team_name[team1]
  id2 = team_name[team2]
  championship1 = championships.get(team1) if championships.get(team1) != None else 0
  championship2 = championships.get(team2) if championships.get(team2) != None else 0

  x = np.array([id1, id2, championship1, championship2]).astype('float64')
  x = np.reshape(x, (1,-1))
  _y = svm_model.predict_proba(x)[0]

  text = ('Chance for '+team1+' to win '+team2+' is {}\nChance for '+team2+' to win '+team1+' is {}\nChance for '+team1+' and '+team2+' draw is {}').format(_y[1]*100,_y[2]*100,_y[0]*100)
  return _y[0], text
#predict match between France and Uruguay
prob1, text1 = prediction('France', 'Uruguay')
print(text1)
#predict match between Brazil and Belgium
prob2, text2 = prediction('Brazil','Belgium')
print(text2)
#predict matches between England and Sweeden
prob3, text3 = prediction('England','Sweden')
print(text3)