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
def plot_dt(tree, feature_names, class_names):
    """Função criada para a visualização da árvore de decisão gerada 
    """
    from sklearn.tree import plot_tree
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    plot_tree(tree, ax = ax, feature_names=feature_names, class_names=class_names)
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
df = pd.read_csv('../input/competicao-curso-verao-inpe-2020/train.csv')
pred = pd.read_csv('../input/competicao-curso-verao-inpe-2020/test.csv')
pred_id = pred['id']
df.head(2)
pred.drop(columns =['id'], inplace  = True)
pred.head(2)
def min_max(x):
    return x.values / x.max()
pred_x = pred.loc[:, 'b1':]
for i in range(1, 10):
    pred_x[f'b{i}'] = min_max(pred_x[f'b{i}'])
pred_x.head()
df.drop(columns =['id'], inplace  = True)
df.head(2)
train_x = df.loc[: , 'b1':]
train_y = df['class']
display(train_x.head(2))
display(train_y.head(2))
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(max_depth = 3)
tree.fit(train_x, train_y)
_class = tree.predict(pred_x)
data_columns = ['b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9']
resp = pd.DataFrame({'id':pred_id.values , 'class': _class})
resp.head()
filename = 'tree3.csv'
resp.to_csv(filename, index = False)
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(train_x, train_y)
_class  = knn.predict(pred)
resp = pd.DataFrame({'id':pred_id.values, 'class': _class})
resp.head()
resp.to_csv('knn7.csv', index = False)
from sklearn import svm
clf = svm.SVC(decision_function_shape='ovo')
clf.fit(train_x, train_y)
clf.predict(pred)
resp = pd.DataFrame({'id':pred_id.values, 'class': clf.predict(pred)})
resp.head()
resp.to_csv('svm(ovo).csv', index = False)