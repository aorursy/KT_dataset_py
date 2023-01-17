import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
%matplotlib inline
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_graphviz
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.externals.six import StringIO  
from IPython.display import Image 
#import pydotplus
import collections
from IPython.display import Image
# import data
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
train.head()
test.head()
train = train.drop(['Ticket', 'Cabin'], axis = 1)
test = test.drop(['Ticket', 'Cabin'], axis = 1)
train.isna().sum()
train = train.dropna()
a4_dims = (20, 5)
fig, ax = plt.subplots(figsize=a4_dims)
sns.barplot(train['Age'], train['Survived'], ax = ax)
train.info()
msk = np.random.rand(len(train)) < 0.8
train2 = train[msk]
test2 = train[~msk]
lb = LabelEncoder()
train2['Embarked'] = lb.fit_transform(train2['Embarked'].astype(str))
train2['Sex'] = lb.fit_transform(train2['Sex'].astype(str))

test2['Embarked'] = lb.fit_transform(test2['Embarked'].astype(str))
test2['Sex'] = lb.fit_transform(test2['Sex'].astype(str))

dt = DecisionTreeClassifier(max_depth = 6, max_leaf_nodes = 10)
x = train2.drop(['Survived', 'PassengerId', 'Name'], axis = 1)
y = train2['Survived']
dt.fit(x, y)
x2 = test2.drop(['Survived', 'PassengerId', 'Name'], axis = 1)
dt.predict(x2)
## calculating accuracy
cross_val_score(dt, x, y, cv = 20).mean()
# this code won't run in Kaggle because Kaggle doesn't have the pydotplus package included so 
# if you want to see the decision tree visual, just download and run the code
#data_feature_names = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

#dot_data = export_graphviz(dt,
#                               feature_names=data_feature_names,
#                                out_file=None,
#                                filled=True,
#                                rounded=True)
#graph = pydotplus.graph_from_dot_data(dot_data)

#colors = ('turquoise', 'orange')
#edges = collections.defaultdict(list)

#for edge in graph.get_edge_list():
#   edges[edge.get_source()].append(int(edge.get_destination()))

#for edge in edges:
#    edges[edge].sort()    
#    for i in range(2):
#        dest = graph.get_node(str(edges[edge][i]))[0]
#        dest.set_fillcolor(colors[i])

#graph.write_png('tree.png')