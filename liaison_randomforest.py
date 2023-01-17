# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pylab as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df_data = pd.read_csv('../input/mushrooms.csv')

df_data.head().T
def to_factor(serie):
    return pd.factorize(serie)[0]
df_factor = pd.DataFrame([to_factor(df_data[column_name]) for column_name in df_data.columns])
df_factor = df_factor.transpose()

df_factor.columns = df_data.columns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

train_features = list(set(df_data.columns) - set(['class']))

X_train, X_test, y_train, y_test = train_test_split(
    df_factor[train_features], df_factor['class'], test_size=0.33, random_state=42)

model = RandomForestClassifier()

model.fit(X_train, y_train)
print('training accuracy:', model.score(X_train, y_train))
print('testing accuracy:', model.score(X_test, y_test))
varimp = pd.Series(model.feature_importances_, index=train_features)

varimp
varimp.sort_values(ascending=False, inplace=True)
_ = varimp.plot(kind='bar')
y_pos = np.arange(len(train_features))
plt.barh(y_pos, varimp.values)
_ = plt.yticks(y_pos, varimp.index)
df_data['odor'].describe()
df_data.groupby(['odor'])['odor'].count()
np.cov(df_factor['odor'], df_factor['class'])
tree = model.estimators_[0]
from sklearn.tree import export_graphviz
import pydot

export_graphviz(tree, out_file = 'tree.dot')

!cat tree.dot
(graph, ) = pydot.graph_from_dot_file('tree.dot')
graph.write_png('tree.png')
!ls
from IPython.display import Image
Image('tree.png')