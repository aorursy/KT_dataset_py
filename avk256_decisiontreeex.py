import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import graphviz

import sklearn

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import cross_val_score



import seaborn as sns; sns.set()

import matplotlib.pyplot as plt
data = {'x0': [1, 1, 2, 2, 1, 2, 1, 4, 4, 5, 4, 5], 'x1': [1, 2, 1, 3, 5, 6, 7, 4, 5, 5, 1, 2],

        'class': [0,0,0,0,1,1,1,0,0,0,1,1]}

df = pd.DataFrame.from_dict(data)

print(df)

ax = sns.scatterplot(x="x0", y="x1", data=df, hue='class')
X_train, y_train = df.drop(['class'], axis=1), df['class']

tree_cl = DecisionTreeClassifier(random_state=0, criterion='gini')

tree_cl.fit(X_train, y_train)

print('score',tree_cl.score(X_train, y_train))

with open("classifier.txt", "w") as f:

    f = sklearn.tree.export_graphviz(tree_cl, out_file=f)