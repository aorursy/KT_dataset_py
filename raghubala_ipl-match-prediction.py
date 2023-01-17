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
#Read the matches file

matches = pd.read_csv("/kaggle/input/ipl/matches.csv")
#View info, head, tail, describe

matches.head(3)
matches.tail(3)
matches.describe()

matches.info()
#Check for thumb rull 1 - Nulls

matches.isnull().sum().sort_values(ascending = False)

#Fix null for column - umpire3

matches.drop(['umpire3'], inplace = True, axis = 1)
#Fix null for column - city

matches[pd.isnull(matches['city'])]

matches['city'] = matches['city'].fillna("DUBAI")

#Fix null for column - player_of_match

matches[pd.isnull(matches['player_of_match'])]

matches['player_of_match'] = matches['player_of_match'].fillna('RAGHU')
#Fix null for column - winner

matches[pd.isnull(matches['winner'])]
matches['winner'] = matches['winner'].fillna('DRAW')
#Fix null for column - umpire1

matches['umpire1'].value_counts() #Mode of the column
matches['umpire1'] = matches['umpire1'].fillna('HDPK Dharmasena')
#Fix null for column - umpire2

matches['umpire2'].value_counts() #Mode of the column
matches['umpire2'] = matches['umpire2'].fillna('SJA Taufel')
matches.info()
# season, city, team1, team2, toss_winner, toss_decision, winner

df = matches.drop(['id', 'date', 'result', 'dl_applied', 'win_by_runs', 'win_by_wickets',

'player_of_match', 'venue',

'umpire1', 'umpire2'], axis = 1)
df.info()

#Label Encode city, toss_decision using Label Encoder class

from sklearn import preprocessing

le = preprocessing.LabelEncoder()
df['city'] = le.fit_transform(df['city'])

df['toss_decision'] = le.fit_transform(df['toss_decision'])
df.head(5)
df.team1.value_counts()

#Label Encode team1, team2, winner, toss_winner, using Mapper

teamMapper = {"Mumbai Indians": 0, "Chennai Super Kings": 1, "Kings XI Punjab":2, "Royal Challengers Bangalore" : 3,

"Kolkata Knight Riders" : 4,"Delhi Daredevils" : 5,"Rajasthan Royals" : 6,"Sunrisers Hyderabad" :

7,"Deccan Chargers" : 8,

"Pune Warriors" : 9,"Gujarat Lions" : 10,"Rising Pune Supergiant" : 11, "Kochi Tuskers Kerala":

12,

"Rising Pune Supergiants": 13, 'DRAW': 14}
for dataset in [df]:

    dataset['team1'] = dataset['team1'].map(teamMapper)
df.team1.value_counts()
for dataset in [df]:

    dataset['team2'] = dataset['team2'].map(teamMapper)

    dataset['toss_winner'] = dataset['toss_winner'].map(teamMapper)

    dataset['winner'] = dataset['winner'].map(teamMapper)
df.head(5)
#X and Y #First Iteration



X = df.drop(['winner','toss_winner','toss_decision'],axis = 1)

y = df['winner']

#X and Y #Second Iteration

X = df.drop(['winner'], axis = 1)

y = df['winner']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#Goal 1 - predict win based on team1, team2, city, season, y = winner

#Model Training - Gaussian NB

from sklearn.naive_bayes import GaussianNB

clf_gnb = GaussianNB()

clf_gnb.fit(X_train, y_train)
y_prediction_gnb = clf_gnb.predict(X_test)
accuracy_score_gnb = clf_gnb.score(X_train, y_train)*100

print (accuracy_score_gnb)



# 32.80898876404494
#Model Training - KNN

from sklearn.neighbors import KNeighborsClassifier

clf_knn = KNeighborsClassifier(n_neighbors=3)

clf_knn.fit(X_train, y_train)
y_prediction_gnb = clf_knn.predict(X_test)

accuracy_score_knn = clf_knn.score(X_train, y_train)*100

print (accuracy_score_knn)



# 62.69662921348315