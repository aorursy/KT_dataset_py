import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import tree # decision tree algorithm



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
## Load the training data into a pandas dataset

data = pd.read_csv("../input/train.csv")
##prep the data

data = pd.get_dummies(data, columns=['Sex','Embarked']) #convert categorical data

data=data.interpolate() #convert nulls
##split into train and test

train=data.sample(frac=0.8,random_state=200)

test=data.drop(train.index)

"Data len:" + str(len(data)) + ", Train len:"+str(len(train)) +", Test len:"+str(len(test))
## create decision treemodel

clf = tree.DecisionTreeRegressor(max_features=3,max_depth=3)

clf = clf.fit(train[['Sex_female','Age','Fare','Embarked_C','Embarked_Q']],train[['Survived']])
##visualise the tree

##tree.export_graphviz(clf,out_file='tree.dot')

from graphviz import Source

Source( tree.export_graphviz(clf, out_file=None, feature_names=('Sex_female','Age','Fare','Embarked_C','Embarked_Q')))
tree.export_graphviz(clf, out_file='titanic_decision_tree', feature_names=('Sex_female','Age','Fare','Embarked_C','Embarked_Q'))