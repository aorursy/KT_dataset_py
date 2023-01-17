# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/winequality-red.csv')
df.head(10)
df.info()
df.isnull().any()
np.round(df.describe())
df.shape
X = df.drop(['quality'], axis=1)
y = df.quality.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=None)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('score: ', model.score(X_test, y_test) * 100)
df.corr()['quality'].sort_values()
X = df[['alcohol', 'sulphates', 'citric acid', 'volatile acidity']]
y = df.quality.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=1000, 
                               random_state=None, 
                               max_depth=1000, 
                               min_samples_leaf=1)
model.fit(X_train, y_train)
model.score(X_test, y_test)

