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
#read matches file
df = pd.read_csv('/kaggle/input/ipl/matches.csv')


#view info head ,tail, describe
df.head(2)


df.tail(2)
df.describe()
df.info()
#check for thumb rull no 1 - Nulls
df.isnull().sum().sort_values(ascending =False)

#fix null for col - umpire3
df.drop(['umpire3'], inplace = True, axis=1)


#fix null for col - city
df[pd.isnull(df["city"])]
df['city'] = df['city'].fillna('Dubai')

#fix null for col - player_of_match
df[pd.isnull(df["player_of_match"])]
df['player_of_match'] = df['player_of_match'].fillna('Tej')
#fix null for col - winner               
df[pd.isnull(df["winner"])]
df['winner'] = df['winner'].fillna('Draw')

#fix null for col - umpire1,umpire2
df['umpire1'].value_counts() #636 #mean , median ,mode
#mean = sales/ age
#median = Monday to friday ,sat, sun ,mon 7 days
df['umpire1'] = df['umpire1'].fillna('HDPK Dharmasena')
df['umpire2'].value_counts()
df['umpire2'] = df['umpire2'].fillna('SJA Taufel')
df.info()
#season ,city,team1, team2, toss_winner , toss decision, winner
matches = df.drop(['id','date','result','dl_applied','player_of_match','venue','umpire1','umpire2'], axis= 1)


matches.info()

#label encode city,toss_decision using label encoder class
from sklearn import preprocessing
le = preprocessing.LabelEncoder()


matches['city'] = le.fit_transform(matches['city'])
matches['toss_decision'] = le.fit_transform(matches['toss_decision'])

matches.head()
matches.team2.value_counts()
#label encoding team2,team2, winner, toss_winner using mapper
teamMapper = {"Mumbai Indians":0,"Chennai Super Kings":1,"Kings XI Punjab":2,"Royal Challengers Bangalore":3
              ,"Kolkata Knight Riders":4,
              "Delhi Daredevils":5,"Rajasthan Royals":6,"Sunrisers Hyderabad":7,"Deccan Chargers":8
              ,"Pune Warriors":9,"Gujarat Lions":10,"Rising Pune Supergiant":11,"Kochi Tuskers Kerala":12
              ,"Rising Pune Supergiants":13,"Draw":14}

for dataset in [matches]:
    dataset['team1']=dataset['team1'].map(teamMapper)
    
        
        
for dataset in [matches]:
    dataset['team2']=dataset['team2'].map(teamMapper)
for dataset in [matches]:
    dataset['winner']=dataset['winner'].map(teamMapper)
    dataset['toss_winner']=dataset['toss_winner'].map(teamMapper)
    
matches.head()
#check info
matches.info()
#x & y
X = matches.drop(["winner"],axis=1)
y = df['winner']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)


# goal 1- predict win base on team 1, team2, city ,season, y=winner

from sklearn.naive_bayes import GaussianNB 


gnb = GaussianNB()

gnb.fit(X_train, y_train) 
y_prediction = gnb.predict(X_test)

acc_gnb= gnb.score(X_test, y_test)*100
print(acc_gnb)
from sklearn.neighbors import KNeighborsClassifier
clf_knn =  KNeighborsClassifier(n_neighbors=3)
clf_knn.fit(X_train, y_train)

y_pred_knn = clf_knn.predict(X_test)
print("Train score",clf_knn.score(X_train, y_train)*100)
