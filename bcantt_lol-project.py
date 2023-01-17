# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
ranked_challanger = pd.read_csv('/kaggle/input/league-of-legends-challenger-ranked-games2020/Challenger_Ranked_Games.csv')

ranked_grandmaster = pd.read_csv('/kaggle/input/league-of-legends-challenger-ranked-games2020/GrandMaster_Ranked_Games.csv')

ranked_master = pd.read_csv('/kaggle/input/league-of-legends-challenger-ranked-games2020/Master_Ranked_Games.csv')
from sklearn.ensemble import RandomForestClassifier

from boruta import BorutaPy

from sklearn.model_selection import train_test_split

from sklearn.metrics import balanced_accuracy_score
ranked_challanger
X = ranked_challanger.drop('blueWins',axis = 1)

y = ranked_challanger['blueWins']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)

feat_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=1)

feat_selector.fit(X_train.values, y_train.values)

X_train_f = feat_selector.transform(X_train.values)

X_test_f = feat_selector.transform(X_test.values)

rf.fit(X_train_f,y_train)

pred = rf.predict(X_test_f)
balanced_accuracy_score(pred, y_test)
X = ranked_grandmaster.drop('blueWins',axis = 1)

y = ranked_grandmaster['blueWins']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)

feat_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=1)

feat_selector.fit(X_train.values, y_train.values)

X_train_f = feat_selector.transform(X_train.values)

X_test_f = feat_selector.transform(X_test.values)

rf.fit(X_train_f,y_train)

pred = rf.predict(X_test_f)
balanced_accuracy_score(pred, y_test)
X = ranked_master.drop('blueWins',axis = 1)

y = ranked_master['blueWins']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)

feat_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=1)

feat_selector.fit(X_train.values, y_train.values)

X_train_f = feat_selector.transform(X_train.values)

X_test_f = feat_selector.transform(X_test.values)
rf.fit(X_train_f,y_train)

pred = rf.predict(X_test_f)
balanced_accuracy_score(pred, y_test)