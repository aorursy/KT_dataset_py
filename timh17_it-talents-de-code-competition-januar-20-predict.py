import numpy as np

import pandas as pd
races = pd.read_csv("../input/races_cleaned.csv") # load data set 
races
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

test = pd.factorize(races.weather)

print(test)

print(len(test[0]))

print(len(races.weather))

races["weather"] = test[0]

X = races[["track_id", "challenger", "opponent", "money", "fuel_consumption", "weather", "race_day"]].loc[races.winner != -1]

X

y = races["winner"].loc[races.winner != -1]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

clf = SVC(gamma='auto', verbose=True)

clf.fit(X_train, y_train)



predicted = clf.predict(X_test)

clf.predict(y_test, predicted)