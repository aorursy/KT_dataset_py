# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load


import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_data = pd.read_csv('/kaggle/input/ghouls-goblins-and-ghosts-boo/train.csv.zip')
test_data = pd.read_csv('/kaggle/input/ghouls-goblins-and-ghosts-boo/test.csv.zip')
print("Colums: ", train_data.columns.values)
print("Shape: ", train_data.shape)
print("Missing values:")
print(train_data.isnull().sum())
train_data.color.unique()
test_data = pd.concat([test_data,
                pd.get_dummies(test_data.color, prefix="color", drop_first = True)
                 ], axis=1)
train_data = pd.concat([train_data,
                pd.get_dummies(train_data.color, prefix="color", drop_first = True)
                 ], axis=1)
test_id = test_data['id'].copy()
test_data.drop(['color','id'], axis=1, inplace=True)
train_data.drop(['color','id'], axis=1, inplace=True)
print("Colums: ", train_data.columns.values)
y=train_data['type']
X=train_data.copy()
del X['type']
print(X)
np.shape(X)
y.unique()
my_map = {'Ghoul': 1, 'Goblin': 2, 'Ghost': 3}
inv_map = {1: 'Ghoul', 2: 'Goblin', 3: 'Ghost'}
y = y.map(my_map)
print(y)

print(X)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
from sklearn.preprocessing import normalize
X_train_norm=normalize(X_train)
X_test_norm=normalize(X_test)
from sklearn.ensemble import GradientBoostingClassifier



clf = GradientBoostingClassifier(learning_rate=0.1,
                                 n_estimators=700,
                                 max_depth=2)

clf.fit(X_train_norm, y_train)
print("RF Accuracy: " + repr(round(clf.score(X_test_norm, y_test) * 100, 2)) + "%")

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(criterion='entropy',
                             n_estimators=700,
                             min_samples_split=5,
                             min_samples_leaf=1,
                             max_features = "auto",
                             oob_score=True,
                             random_state=0,
                             n_jobs=-1)

clf.fit(X_train_norm, y_train)
print("RF Accuracy: " + repr(round(clf.score(X_test_norm, y_test) * 100, 2)) + "%")
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
clf=MLPClassifier(solver='adam',hidden_layer_sizes=350, alpha=1e-04, max_iter =120000)
clf.fit(X_train_norm,y_train)

preds=pd.Series(clf.predict(X_test_norm))
print(accuracy_score(y_test,preds))
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(C=1000000, max_iter=120000)
clf.fit(X_train_norm,y_train)
preds=pd.Series(clf.predict(X_test_norm))
print(accuracy_score(y_test,preds))
from sklearn.neighbors import KNeighborsClassifier

clf= KNeighborsClassifier(n_neighbors=5)
clf.fit(X_train_norm,y_train)
preds=pd.Series(clf.predict(X_test_norm))
print(accuracy_score(y_test,preds))

X_pred=test_data
print(X_pred)
from sklearn.preprocessing import normalize

result = pd.Series(clf.predict(normalize(X_pred)), name='type')
result = result.map(inv_map)
result = pd.concat([test_id,result], axis=1)
df=pd.DataFrame(result)
df.index+=1
print(result.shape)
filename = 'Prediction.csv'
df.to_csv(filename,index=False)
print('Saved file: ' + filename)