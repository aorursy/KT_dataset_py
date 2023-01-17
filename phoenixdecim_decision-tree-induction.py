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
df = pd.read_csv("../input/carseats.csv")
df.shape
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from sklearn.tree import export_graphviz

from sklearn.externals.six import StringIO 

from IPython.display import Image 

from pydot import graph_from_dot_data

from sklearn.tree import export_text

import matplotlib.pyplot as plt
df
from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()

df.ShelveLoc = labelencoder.fit_transform(df.ShelveLoc)

df.Urban = labelencoder.fit_transform(df.Urban)

#df.US = labelencoder.fit_transform(df.US)
df
df.columns
df_x = df[['Sales','CompPrice','Income','Advertising','Population','Price','ShelveLoc','Age','Education','Urban']]
X_train, X_test, y_train, y_test = train_test_split(df_x, df.US, random_state=1)
max_depth_range = list(range(1, 9))

accuracy = []

for depth in max_depth_range:

    clf = DecisionTreeClassifier(max_depth = depth, random_state = 0, criterion = "gini")

    clf.fit(X_train, y_train)

    score = clf.score(X_test, y_test)

    accuracy.append(score)
accuracy
d1 = DecisionTreeClassifier(max_depth = 1, random_state = 0, criterion = "gini")

d1.fit(X_train, y_train)

dot_data = StringIO()

export_graphviz(d1, out_file=dot_data, feature_names=df_x.columns)

(graph, ) = graph_from_dot_data(dot_data.getvalue())

Image(graph.create_png())
d2 = DecisionTreeClassifier(max_depth = 2, random_state = 0, criterion = "gini")

d2.fit(X_train, y_train)

dot_data = StringIO()

export_graphviz(d2, out_file=dot_data, feature_names=df_x.columns)

(graph, ) = graph_from_dot_data(dot_data.getvalue())

Image(graph.create_png())
y_pred = d2.predict(X_test)

label = np.array(y_test)

predictions = np.array(y_pred)

confusion_matrix(label, predictions)
dt = DecisionTreeClassifier(random_state = 0, criterion = "gini")

dt.fit(X_train, y_train)

path = dt.cost_complexity_pruning_path(X_train, y_train)

ccp_alphas, impurities = path.ccp_alphas, path.impurities
fig, ax = plt.subplots()

ax.plot(ccp_alphas[:-1], impurities[:-1], marker='o', drawstyle="steps-post")

ax.set_xlabel("effective alpha")

ax.set_ylabel("total impurity of leaves")

ax.set_title("Total Impurity vs effective alpha for training set")
dts = []

for ccp_alpha in ccp_alphas:

    dt = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)

    dt.fit(X_train, y_train)

    dts.append(dt)

print("Number of nodes in the last tree is: {} with ccp_alpha: {}".format(

     dts[-1].tree_.node_count, ccp_alphas[-1]))
cl = DecisionTreeClassifier(random_state=0, ccp_alpha=0.01)

cl.fit(X_train, y_train)

dot_data = StringIO()

export_graphviz(cl, out_file=dot_data, feature_names=df_x.columns)

(graph, ) = graph_from_dot_data(dot_data.getvalue())

Image(graph.create_png())
tree_rules = export_text(cl, feature_names=list(df_x.columns))

print(tree_rules)
score = cl.score(X_test, y_test)

score
y_pred = cl.predict(X_test)

label = np.array(y_test)

predictions = np.array(y_pred)

confusion_matrix(label, predictions)
#clfs = clfs[:-1]

#ccp_alphas = ccp_alphas[:-1]



node_counts = [dt.tree_.node_count for dt in dts]

depth = [dt.tree_.max_depth for dt in dts]

fig, ax = plt.subplots(2, 1)

ax[0].plot(ccp_alphas, node_counts, marker='o', drawstyle="steps-post")

ax[0].set_xlabel("alpha")

ax[0].set_ylabel("number of nodes")

ax[0].set_title("Number of nodes vs alpha")

ax[1].plot(ccp_alphas, depth, marker='o', drawstyle="steps-post")

ax[1].set_xlabel("alpha")

ax[1].set_ylabel("depth of tree")

ax[1].set_title("Depth vs alpha")

fig.tight_layout()
train_scores = [dt.score(X_train, y_train) for dt in dts]

test_scores = [dt.score(X_test, y_test) for dt in dts]



fig, ax = plt.subplots()

ax.set_xlabel("alpha")

ax.set_ylabel("accuracy")

ax.set_title("Accuracy vs alpha for training and testing sets")

ax.plot(ccp_alphas, train_scores, marker='o', label="train",

        drawstyle="steps-post")

ax.plot(ccp_alphas, test_scores, marker='o', label="test",

        drawstyle="steps-post")

ax.legend()

plt.show()
import matplotlib.pyplot as plt

plt.plot(np.arange(1, 9, 1), accuracy)

plt.plot(np.arange(1, 9, 1), accuracy, 'ro')

plt.show()
importances = pd.DataFrame({'feature':X_train.columns,'importance':np.round(d2.feature_importances_,2)})

importances = importances.sort_values('importance',ascending=False)

importances
tree_rules = export_text(d2, feature_names=list(df_x.columns))

print(tree_rules)