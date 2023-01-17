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
data = pd.read_csv('/kaggle/input/womens-international-football-results/results.csv')
data.head()
home_wins = np.where(data['home_score']>data['away_score'],1,0)
pd.Series(home_wins).value_counts().plot.bar()
# home percent win rate
pd.Series(home_wins).value_counts()[1] / (pd.Series(home_wins).value_counts()[1] + pd.Series(home_wins).value_counts()[0])
from sklearn.preprocessing import OneHotEncoder

X = data[['home_team','away_team','tournament','country','neutral']]
cat_encoder = OneHotEncoder()
game_cat_1hot = cat_encoder.fit_transform(X)

X_clean = game_cat_1hot.toarray()
X_clean.shape
y = home_wins
from sklearn.model_selection import train_test_split
X_train, X_test,y_train, y_test = train_test_split(X_clean,y, test_size=0.2, random_state=42)
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

log_clf = LogisticRegression(solver="lbfgs", random_state=42)
rnd_clf = RandomForestClassifier(n_estimators=100, random_state=42)
svm_clf = SVC(gamma="scale", random_state=42)

voting_clf = VotingClassifier(
    estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
    voting='hard')
from sklearn.metrics import accuracy_score

for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))
y_pred = log_clf.predict(X_test)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)

