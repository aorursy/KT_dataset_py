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
import pandas as pd
df = pd.read_csv('/kaggle/input/iris-flower-dataset/IRIS.csv')
df
df.info()
df.describe()
df.isnull().sum()
s = df['species'].value_counts()
s
df.hist(bins=30)

import matplotlib.pyplot as plt
plt.tight_layout()
import seaborn as sns
sns.scatterplot(df['sepal_length'],df['sepal_width'])
import seaborn as sns
sns.scatterplot(df['petal_length'],df['petal_width'])
f, axes = plt.subplots(1, 2, figsize=(11,4))

sns.scatterplot(data=df, x="sepal_length", y="sepal_width", hue="species", ax=axes[0])
sns.scatterplot(data=df, x="petal_length", y="petal_width", hue="species", ax=axes[1])

plt.tight_layout()
from sklearn import cluster, datasets
k_means = cluster.KMeans(n_clusters=3)
y_km = k_means.fit_predict(df[['sepal_length','sepal_width','petal_length','petal_width']]) 
y_km
df['cluster'] = y_km
df
k_means.cluster_centers_
f, axes = plt.subplots(2, 2, figsize=(10,6))

sns.scatterplot(data=df, x="sepal_length", y="sepal_width", hue="cluster", ax=axes[0][0])
axes[0][0].scatter(k_means.cluster_centers_[:, 0],
            k_means.cluster_centers_[:, 1],
            s=200, marker='*',
            c='red')
sns.scatterplot(data=df, x="petal_length", y="petal_width", hue="cluster", ax=axes[0][1])
axes[0][1].scatter(k_means.cluster_centers_[:, 2],
            k_means.cluster_centers_[:, 3],
            s=200, marker='*',
            c='red')
sns.scatterplot(data=df, x="sepal_length", y="sepal_width", hue="species", ax=axes[1][0])
sns.scatterplot(data=df, x="petal_length", y="petal_width", hue="species", ax=axes[1][1])

plt.tight_layout()
df
from sklearn.model_selection import train_test_split 
X = df[['sepal_length','sepal_width','petal_length','petal_width']]
y = df[['species']]
Xtr,Xte,ytr,yte = train_test_split(X, y, test_size=0.3)
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix  

tree_clf = DecisionTreeClassifier(max_depth=3)
tree_clf.fit(Xtr, ytr)
y.value_counts()
from sklearn.tree import export_graphviz
with open("tree1.dot", 'w') as f:
     f = export_graphviz(tree_clf,
                              out_file=f,
                              max_depth = 3,
                              impurity = True,
                              feature_names = list(X),
                              class_names = ['Iris-setosa','Iris-versicolor','Iris-virginica'],
                              rounded = True,
                              filled= True )
!dot -Tpng tree1.dot -o tree1.png -Gdpi=300
from IPython.display import Image
Image(filename = 'tree1.png')