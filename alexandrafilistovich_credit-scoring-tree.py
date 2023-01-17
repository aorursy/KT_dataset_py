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
import pandas as pd

df_train = pd.read_csv('/kaggle/input/creditscoringds/df_train.csv', sep=',')

tr_id = pd.read_csv('/kaggle/input/mydata/Test.csv', sep=',')

df_train.head()
df_test = pd.read_csv('/kaggle/input/creditscoringds/df_test.csv', sep=',')

df_test = df_test.drop(['ProductCategory_ticket'], axis=1)

df_test.head()
y_test = tr_id[['TransactionId']]

y_test['IsDefaulted'] = 1

y_test.head()
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(max_depth=4, random_state=19,

                             splitter='random')

X = df_train.drop(['IsDefaulted'], axis=1)

y = df_train['IsDefaulted']
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X, y,test_size=0.3, random_state=19)
tree.fit(X_train, y_train)
from sklearn.metrics import accuracy_score

y_pred = tree.predict(X_valid)

accuracy_score(y_valid, y_pred)
tree_all = DecisionTreeClassifier(max_depth=3, random_state=17)

tree_all.fit(X, y)

y_x = tree_all.predict(df_test)
'''from sklearn.tree import export_graphviz

tree_dot = export_graphviz(tree_all)

print(tree_dot)'''
y_test['IsDefaulted'] = y_x

y_test.to_csv('Answer_Tree.csv',index=False)
'''from sklearn.model_selection import GridSearchCV, cross_val_score



tree_params = {'max_depth': range(2, 11)}



tree_grid = GridSearchCV(tree, tree_params,

                         cv=5, n_jobs=-1, verbose=True)



tree_grid.fit(X, y)'''
'''tree_grid.best_params_

best_tree = tree_grid.best_estimator_

pd.DataFrame(tree_grid.cv_results_).T

import matplotlib.pyplot as plt



df_cv = pd.DataFrame(tree_grid.cv_results_)



plt.plot(df_cv['param_max_depth'], df_cv['mean_test_score'])

plt.xlabel("max_depth")

plt.ylabel("accuracy");'''