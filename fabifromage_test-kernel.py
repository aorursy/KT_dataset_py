# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import os
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

nba = pd.read_csv('../input/nba-players-stats-20142015/players_stats.csv')
nba
data = nba.dropna(axis=0)


y = data.PTS
data
x = data[['MIN', 'Height','BMI', 'Weight', 'Games Played', 'Age']]
train_x, test_x, train_y, test_y = train_test_split(x, y,random_state = 0)
model = RandomForestRegressor()
model.fit(train_x, train_y)
print(test_y.head())
print(model.predict(test_x.head()))
print(mean_absolute_error(test_y, model.predict(test_x)))

abgabe = pd.DataFrame({'ID': test_x.index, 'Prediction': model.predict(test_x)})
abgabe
abgabe.to_csv('Erg')





