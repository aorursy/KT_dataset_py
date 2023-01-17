# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/heart.csv')
df.head()
# stats

df.describe()
df.info()
plt.rcParams['figure.figsize'] = (20,8)

df.hist()
y = df['target']

df.drop('target', axis=1,inplace=True)
from sklearn.tree import DecisionTreeClassifier, export_graphviz

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import accuracy_score
labelEncoder = LabelEncoder()

df['oldpeak'] = labelEncoder.fit_transform(y=df['oldpeak'])
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.3, random_state=17)
tree = DecisionTreeClassifier(random_state=17, max_depth=3, min_samples_leaf=2)

tree.fit(X=X_train, y=y_train)
tree.score(X_test, y_test)
preds = tree.predict(X_test)

accuracy_score(y_true=y_test, y_pred=preds)
export_graphviz(tree, 'tree1.dot', filled=True, feature_names=X_train.columns, rounded=True)

!dot -Tpng 'tree1.dot' -o  'tree1.png'
!ls
from sklearn.model_selection import GridSearchCV, StratifiedKFold
# Let's vary hyperparameters from 2 - 10

best_parameters = {'max_depth': np.arange(2,11), 'min_samples_leaf': np.arange(2,11)}

decision_tree = DecisionTreeClassifier(criterion='entropy') # for information gain and entropy

model = GridSearchCV(estimator=decision_tree, param_grid=best_parameters, n_jobs=-1, verbose=1, cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=17))

model.fit(X_train, y_train)

model.best_params_
model.best_score_
preds_2 = model.predict(X_test)
accuracy_score(y_test, preds_2)
export_graphviz(model.best_estimator_, out_file='tree2.dot', filled=True, feature_names=X_train.columns, rounded=True)
!dot -Tpng 'tree2.dot' -o 'tree2.png'