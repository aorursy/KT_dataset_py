from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

from sklearn.model_selection import train_test_split

import pandas as pd

import numpy as np



#Import data

data = pd.read_csv('../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')

x = data.drop('quality',axis=1).values

y = data['quality'].values

X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2)



#Decision tree classifier

classifier = DecisionTreeClassifier(max_depth=4)

classifier = classifier.fit(X_train, y_train)

print("Score:",classifier.score(X_test,y_test))



#Decision tree regressor

regressor = DecisionTreeRegressor(criterion='mse', max_depth=4)

regressor = regressor.fit(X_train, y_train)

print("Score:",regressor.score(X_test,y_test))