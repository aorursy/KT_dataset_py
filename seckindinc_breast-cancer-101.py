import numpy as np

import scipy as sp

import sklearn

#import pydotplus



import matplotlib as mpl

import matplotlib.pyplot as plt

%matplotlib inline

%config InlineBackend.figure_format = 'retina'



import seaborn as sn

sn.set_context("poster")

sn.set(rc={'figure.figsize': (20, 12.)})

sn.set_style("whitegrid")



import pandas as pd

pd.set_option("display.max_rows", 120)

pd.set_option("display.max_columns", 120)
data_set = pd.read_csv('../input//data.csv')
data_set.head()
y = data_set['diagnosis']

del data_set['diagnosis']

del data_set['Unnamed: 32']

del data_set['id']
data_set.describe()
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaled_df = scaler.fit_transform(data_set)

scaled_df = pd.DataFrame(scaled_df,columns=data_set.columns) 
scaled_df.head()
scaled_df.hist(figsize=(15,20),grid=False)

plt.show()
plt.figure(figsize=(15,10)) 

sn.heatmap(scaled_df.corr(),annot=True) 
print('Diagnosis Category Counts\n', y.value_counts())

sn.countplot(y,label="Count") 
y[0:5]
scaled_df.head()
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
X_train, X_test, y_train, y_test = train_test_split(scaled_df, y, test_size = 0.25, random_state = 42)
decision_tree = DecisionTreeClassifier().fit(X_train, y_train)

print('Train model accuracy: {:.4f}'.format(decision_tree.score(X_train, y_train)))

print('Test model accuracy: {:.4f}'.format(decision_tree.score(X_test, y_test)))
from sklearn.tree import export_graphviz

dot_data = export_graphviz(decision_tree, out_file = None, class_names = ["malignant", "bening"],

               feature_names = data_set.columns )



pydot_graph = pydotplus.graph_from_dot_data(dot_data)

pydot_graph.set_size('"5,5!"')

pydot_graph.write_png('tree.png')
n_features = scaled_df.shape[1]

plt.barh(range(n_features), decision_tree.feature_importances_, align = 'center')

plt.yticks(np.arange(n_features), scaled_df.columns)

plt.xlabel('Feature Importance')

plt.ylabel('Feature')

plt.ylim(-1, n_features)
decision_tree = DecisionTreeClassifier(max_depth=4).fit(X_train, y_train)

print('Train model accuracy: {:.4f}'.format(decision_tree.score(X_train, y_train)))

print('Test model accuracy: {:.4f}'.format(decision_tree.score(X_test, y_test)))
from sklearn.tree import export_graphviz

dot_data = export_graphviz(decision_tree, out_file = None, class_names = ["malignant", "bening"],

               feature_names = data_set.columns )



pydot_graph = pydotplus.graph_from_dot_data(dot_data)

pydot_graph.set_size('"5,5!"')

pydot_graph.write_png('tree_max_depth.png')
n_features = scaled_df.shape[1]

plt.barh(range(n_features), decision_tree.feature_importances_, align = 'center')

plt.yticks(np.arange(n_features), scaled_df.columns)

plt.xlabel('Feature Importance')

plt.ylabel('Feature')

plt.ylim(-1, n_features)