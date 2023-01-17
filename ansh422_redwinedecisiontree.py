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
dataset=pd.read_csv('/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')



dataset.info()
dataset.describe()
dataset.isnull().sum()
import seaborn as sns

import matplotlib.pyplot as plt
sns.pairplot(dataset)

plt.show()
corr=dataset.corr()

colormap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(corr,cmap=colormap,xticklabels=corr.columns,yticklabels=corr.columns,annot=True)

plt.show()
dataset['quality'] = dataset.quality.apply(lambda x : 1 if x > 6.5 else 0)
sns.countplot(data = dataset, x = 'quality')

plt.show()
X=dataset.drop('quality',1)

y=dataset['quality']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train,y_test= train_test_split(X,y,test_size=0.30, random_state=37)
from sklearn.tree import DecisionTreeClassifier

from sklearn import tree
dt_base=DecisionTreeClassifier(max_depth=10,random_state=4)

dt_base.fit(X_train,y_train)
from sklearn import metrics
y_pred=dt_base.predict(X_test)
acc = metrics.accuracy_score(y_test,y_pred)

print(acc)
tree.plot_tree(dt_base, max_depth=2)
dt_base.tree_.node_count
param_grid = {

    'max_depth' : range(4,20,4),

    'min_samples_leaf' : range(20,200,20),

    'min_samples_split' : range(20,200,20),

    'criterion' : ['gini','entropy'] 

}

n_folds = 5
from sklearn.model_selection import GridSearchCV
dt = DecisionTreeClassifier(random_state=34)

grid = GridSearchCV(dt, param_grid, cv = n_folds, return_train_score=True)
grid.fit(X_train,y_train)
grid.best_params_
best_tree = grid.best_estimator_

best_tree
best_tree.fit(X_train,y_train)

y_pred_best = best_tree.predict(X_test)
acc = metrics.accuracy_score(y_test,y_pred_best)

print(acc)