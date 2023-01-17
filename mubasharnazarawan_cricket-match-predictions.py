# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/ContinousDataset.csv")
data.head()
labels = data[['Winner']]

data = data[['Team 1', 'Team 2', 'Ground', 'Host_Country', 'Venue_Team1', 'Venue_Team2', 'Innings_Team1', 'Innings_Team2']]
data.head()
data_hot = pd.get_dummies(data)
data_hot.head()
from sklearn import preprocessing

le = preprocessing.LabelEncoder()

le.fit(labels)

labels = le.transform(labels) 

# labels = pd.get_dummies(labels)
# labels = labels.reshape(labels.shape[0],1)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(data_hot, labels, test_size=0.1)
from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)

clf.fit(x_train, y_train)

preds = clf.predict(x_test)

len(preds[preds == y_test])/len(preds)
from sklearn.ensemble import RandomForestClassifier as rfc

clf = rfc(n_estimators=100, max_depth=2, random_state=0)

clf.fit(x_train, y_train)

preds = clf.predict(x_test)

len(preds[preds == y_test])/len(preds)
from sklearn.neighbors import KNeighborsClassifier as knn

clf = knn(n_neighbors=10)

clf.fit(x_train, y_train)

preds = clf.predict(x_test)

len(preds[preds == y_test])/len(preds)
data.head()
data = pd.read_csv("../input/ContinousDataset.csv")
data = data[((data['Team 1'] == 'Australia') | (data['Team 1'] == 'West Indies')) & ((data['Team 2'] == 'Australia') | (data['Team 2'] == 'West Indies'))]
labels = data[['Winner']]

data = data[['Team 1', 'Team 2', 'Ground', 'Host_Country', 'Venue_Team1', 'Venue_Team2', 'Innings_Team1', 'Innings_Team2']]
data_hot = pd.get_dummies(data) #one hot encoding our data
# from sklearn import preprocessing

# le = preprocessing.LabelEncoder()

# le.fit(labels)

# labels = le.transform(labels) 

# # labels = pd.get_dummies(labels)
# labels = labels.reshape(labels.shape[0],1)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(data_hot, labels, test_size=0.1)
from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)

clf.fit(x_train, y_train)

preds = clf.predict(x_test)

preds = preds.reshape(preds.shape[0],1)

len(preds[preds == y_test])/len(preds)
from sklearn.ensemble import RandomForestClassifier as rfc

clf = rfc(n_estimators=5, max_depth=2, random_state=0)

clf.fit(x_train, y_train)

preds = clf.predict(x_test)

preds = preds.reshape(preds.shape[0],1)

len(preds[preds == y_test])/len(preds)
from sklearn.tree import DecisionTreeClassifier as DTC

clf = DTC(random_state=0)

clf.fit(x_train, y_train)

preds = clf.predict(x_test)

preds = preds.reshape(preds.shape[0],1)

len(preds[preds == y_test])/len(preds)
from sklearn.neighbors import KNeighborsClassifier as knn

clf = knn(n_neighbors=3)

clf.fit(x_train, y_train)

preds = clf.predict(x_test)

preds = preds.reshape(preds.shape[0],1)

len(preds[preds == y_test])/len(preds)
from sklearn.naive_bayes import GaussianNB as nb

clf = nb()

clf.fit(x_train, y_train)

preds = clf.predict(x_test)

preds = preds.reshape(preds.shape[0],1)

len(preds[preds == y_test])/len(preds)
data = pd.read_csv("../input/ContinousDataset.csv")
data = data[((data['Team 1'] == 'Australia') | (data['Team 1'] == 'India')) & ((data['Team 2'] == 'Australia') | (data['Team 2'] == 'India'))]
data
# recent_match = pd.DataFrame([0,"ODI # 4038", "Aghanistan", "Sri Lanka", "Winning1stInning", "Sep 17, 2018", "Aghanistan", "Dubai", "Neutral", "Neutral", "First", "Second"])
# data = data.append({'Unnamed: 0' : 0 , 'Scorecard' : "ODI # 4038", 'Team 1':"Aghanistan", 'Team 2': "Sri Lanka", 'Margin':"Winning1stInning", 'Ground': "Abu Dubai", 'Match Date': "Sep 17, 2018", 'Winner': "Afghanistan", 'Host_Country': "UAE", 'Venue_Team1':"Neutral", 'Venue_Team2':"Neutral", 'Innings_Team1': "First", 'Innings_Team2': "Second"} , ignore_index=True)
labels = data[['Winner']]

data = data[['Team 1', 'Team 2', 'Ground', 'Host_Country', 'Venue_Team1', 'Venue_Team2', 'Innings_Team1', 'Innings_Team2']]
#making an example entry for our test example

l = [("Australia", "India", "London", "England", "Neutral", "Neutral", "First", "Second"), 

     ("Australia", "India", "London", "England", "Neutral", "Neutral", "Second", "First")

    ]

columns = ["Team 1", "Team 2", "Ground", "Host_Country", "Venue_Team1", "Venue_Team2", "Innings_Team1", "Innings_Team2"]

test_data = pd.DataFrame(l, columns = columns) #creating test example dataframe

data = data.append(test_data) #appending our world cup single match example to the dataset

data_hot = pd.get_dummies(data) #one hot encoding our data
#splitting test and train examples

#only last two examples are for the match we want to predict

x_train = data_hot[:-2]

y_train = labels

x_test = data_hot[-2:]
from sklearn.neighbors import KNeighborsClassifier as knn

clf = knn(n_neighbors=3)

clf.fit(x_train, y_train)

preds = clf.predict_proba(x_test)

# len(preds[preds == y_test])/len(preds)
from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)

clf.fit(x_train, y_train)

preds = clf.predict_proba(x_test)

# preds = preds.reshape(preds.shape[0],1)

# len(preds[preds == y_test])/len(preds)
preds
Team1 = 0.15 * 0.33 + 0.85 * 0.60
Team2 = 0.15 * 0.67 + 0.85 * 0.40
Team1
eng