import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# load the data

df = pd.read_csv('../input/mushrooms.csv')

# There are no null columns in the dataset

df[df.isnull().any(axis=1)]
from sklearn.model_selection import train_test_split





# split the data int x(training data) and y (results)

y = df['class']

x = df.drop(['class'], axis=1)

x = pd.get_dummies(x)

y = pd.get_dummies(y)

x.info()

y.info()

# x.info()

# y.info()

# x.dtypes

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns





# df['habitat_cat'] = df['habitat'].cat.codes

# df['class_cat'] = df['class'].cat.codes

# df.dtypes

# df['habitat_cat'].unique()

# sns.stripplot(x='class', y='habitat_cat', data=df, jitter=True)
from sklearn.model_selection import cross_val_score

from sklearn import tree

from sklearn.model_selection import GridSearchCV



X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)



parameters = {'criterion':('gini', 'entropy'), 

              'min_samples_split':[2,3,4,5], 

              'max_depth':[9,10,11,12],

              'class_weight':('balanced', None),

              'presort':(False,True),

             }





tr = tree.DecisionTreeClassifier()

gsearch = GridSearchCV(tr, parameters)

gsearch.fit(X_train, y_train)

model = gsearch.best_estimator_

model

# gsearch.cv_results_

# scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_macro')

# scores

# model.fit(X_train, y_train)

# The scores are really great, so fit the model and predict

# model.fit(X_train, y_train)

score = model.score(X_test, y_test)

score
import graphviz

dot_data = tree.export_graphviz(model, out_file=None,

                                feature_names=X_test.columns,

                               class_names=y_test.columns,

                               filled=True, rounded=True,

                               special_characters=True)

graph = graphviz.Source(dot_data)

graph