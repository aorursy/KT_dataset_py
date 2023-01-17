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
#Importing the dataset as a dataframe
matches = pd.read_csv("/kaggle/input/ipl/matches.csv")
matches.head(5)
matches.tail(5)
matches.describe().transpose()
matches.isnull().sum().sort_values(ascending = False)
#Fixing nulls for umpire3 
matches.drop(['umpire3'], inplace = True, axis = 1)
matches.head()
matches.isnull().sum().sort_values(ascending = False)
#Fixing nulls for 'city'
matches[pd.isnull(matches['city'])]
#Replacing Nulls
matches['city'] = matches['city'].fillna('Dubai')
#Fixing nulls for 'player_of_match'
matches[pd.isnull(matches['player_of_match'])]
matches.dropna(subset = ['winner', 'player_of_match'], inplace=True)
matches['umpire1'].value_counts()
matches['umpire1'] = matches['umpire1'].fillna('HDPK Dharmasena')
matches['umpire2'].value_counts()
matches['umpire2'] = matches['umpire2'].fillna('SJA Taufel')
matches.info()
df = matches[['season','team1','team2','toss_winner','toss_decision','venue','winner']]
df.info()
#Label Encode venue, toss_decision using Label Encoder
from sklearn import preprocessing
le = preprocessing.LabelEncoder()

df['venue'] = le.fit_transform(df['venue'])
df['toss_decision'] = le.fit_transform(df['toss_decision'])
df.info()
df.head(5)
df.replace('Rising Pune Supergiant', 'Rising Pune Supergiants', inplace = True)
df['team1'].value_counts()
#LabelEncoding for team1, team2, toss_winner, winner using Mapper

teamMapper = { 'Mumbai Indians':0, 'Kings XI Punjab': 1, 'Chennai Super Kings': 2, 'Royal Challengers Bangalore': 3, 'Kolkata Knight Riders':4, 
              'Delhi Daredevils': 5, 'Rajasthan Royals': 6, 'Sunrisers Hyderabad': 7, 'Deccan Chargers': 8, 'Pune Warriors': 9, 'Gujarat Lions': 10,
             'Rising Pune Supergiants': 11, 'Kochi Tuskers Kerala': 12}
for dataset in [df]:
    dataset['team1'] = dataset['team1'].map(teamMapper)
    dataset['team2'] = dataset['team2'].map(teamMapper)
    dataset['toss_winner'] = dataset['toss_winner'].map(teamMapper)
    dataset['winner'] = dataset['winner'].map(teamMapper)
df.head()
df.info()
#splitting into x and y
x = df.drop(['winner', 'toss_winner', 'toss_decision'], axis = 1) #Features
y = df['winner'] #Labels
#splitting train and test set
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)
#Model Training where (x = season, team1, team2, toss_winner, toss_decision, venue) 

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train) 

#Predicting Y value
y_pred_knn = knn.predict(x_test)

#59.36
print(knn.score(x_train, y_train)*100)
#Model Training where (x = season, team1, team2, venue) 

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train) 

#Predicting Y value
y_pred_knn = knn.predict(x_test)

#58.69
print(knn.score(x_train, y_train)*100)