import numpy as np

import pandas as pd

from sklearn.tree import DecisionTreeClassifier

from sklearn import metrics # for score / accuracy
#display input file path

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#Load dataset

dataset = pd.read_csv("/kaggle/input/breast-cancer-wisconsin-data/data.csv", delimiter=",")



# print dataset

dataset[:]
#print all columns (features and lable)

dataset.columns

# len(dataset.columns)
#Select Features 

X = dataset[['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst','perimeter_worst', 'area_worst', 'smoothness_worst','compactness_worst','concavity_worst', 'concave points_worst','symmetry_worst', 'fractal_dimension_worst']].values

X [0:2]
#display Unique Lable along with its count

dataset['diagnosis'].value_counts()
y = dataset['diagnosis'] 

y
# Split dataset into two part train and test dataset 70% training and 30% testset (Random)

from sklearn.model_selection import train_test_split

X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=3)
def MyDecisionTreeClassifier(heuristic, tree_depth = None):

    decision_tree_clfr = DecisionTreeClassifier(criterion = heuristic, max_depth = tree_depth)

    

    #Apply classifier on training dataset

    decision_tree_clfr.fit(X_trainset, y_trainset)

    

    return decision_tree_clfr
heuristic = "entropy"

decision_tree = MyDecisionTreeClassifier(heuristic)



# predict lables using remaining testset

predTree = decision_tree.predict(X_testset)
print("Decision Trees's Accuracy using entropy: ", metrics.accuracy_score(y_testset, predTree))

print("Depth of Decision Tree: ", decision_tree.tree_.max_depth)
!conda install -c conda-forge pydotplus -y

!conda install -c conda-forge python-graphviz -y
import matplotlib.pyplot as plt

from sklearn.externals.six import StringIO

import pydotplus

import matplotlib.image as mpimg

from sklearn import tree

import numpy as np
fileName_entropy = "decision-tree-entropy.png"



dot_data = StringIO()

featureNames = dataset.columns[2:32]

labedNames = dataset["diagnosis"].unique().tolist()

    

# export_graphviz will convert decision tree classifier into dot file

tree.export_graphviz(decision_tree,feature_names = featureNames, out_file = dot_data,

                         class_names = np.unique(y_trainset), filled = True,  special_characters = True,rotate = False) 

    

# Convert dot file int pgn using pydotplus

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

    

#write pgn into file

graph.write_png(fileName_entropy)



#display tree image

img_entropy = mpimg.imread(fileName_entropy)

plt.figure(figsize=(100, 200))

plt.imshow(img_entropy, interpolation='nearest')
heuristic_g = "gini"

decision_tree_g = DecisionTreeClassifier(criterion = heuristic_g)



#Apply classifier on train dataset

decision_tree_g.fit(X_trainset,y_trainset)



# predict lables using remaining testset

predTree_g = decision_tree_g.predict(X_testset)
print("Decision Trees's Accuracy using Gini: ", metrics.accuracy_score(y_testset, predTree_g))

print("Depth of Decision Tree: ", decision_tree_g.tree_.max_depth)
fileName_g = "decision-tree-gini.png"





dot_data = StringIO()

featureNames = dataset.columns[2:32]

labedNames = dataset["diagnosis"].unique().tolist()

    

# export_graphviz will convert decision tree classifier into dot file

tree.export_graphviz(decision_tree_g,feature_names = featureNames, out_file = dot_data,

                         class_names = np.unique(y_trainset), filled = True,  special_characters = True,rotate = False) 

    

# Convert dot file int pgn using pydotplus

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

    

#write pgn into file

graph.write_png(fileName_g)



#display tree image

img_g = mpimg.imread(fileName_g)

plt.figure(figsize=(100, 200))

plt.imshow(img_g, interpolation='nearest')
startingPoint = 2

accuracy_1 = np.zeros((decision_tree.tree_.max_depth - 1))



for x in range(startingPoint, decision_tree.tree_.max_depth + 1):

    heuristic = "entropy"

    decision_tree = MyDecisionTreeClassifier(heuristic)



    # predict lables using remaining testset

    predTree = decision_tree.predict(X_testset)

    

    accuracy_1 [x-startingPoint] = metrics.accuracy_score(y_testset, predTree)

    

    print("Decision Trees's Accuracy (entropy) with depth:", x , " is ", accuracy_1 [x-startingPoint],"")
import matplotlib.pyplot as plt

def ShowAccuracy(_range, data):

    plt.plot(range(2,_range+1),data,'g')

    plt.legend(('Accuracy'))

    plt.ylabel('Accuracy ')

    plt.xlabel('Depth')

    plt.tight_layout()

    plt.show()
ShowAccuracy(decision_tree.tree_.max_depth, accuracy_1)
startingPoint = 2

depth = np.zeros((decision_tree_g.tree_.max_depth - 1))

accuracy_2 = np.zeros((decision_tree_g.tree_.max_depth - 1))



for x in range(startingPoint, decision_tree_g.tree_.max_depth + 1):

    heuristic = "gini"

    decision_tree = MyDecisionTreeClassifier(heuristic)



    # predict lables using remaining testset

    predTree = decision_tree.predict(X_testset)

    

    depth [x-startingPoint] = x;

    accuracy_2 [x-startingPoint] = metrics.accuracy_score(y_testset, predTree)

    

    print("Decision Trees's Accuracy (Gini) with depth:",x, " is ", accuracy_2 [x-startingPoint],"")
ShowAccuracy(decision_tree_g.tree_.max_depth, accuracy_2)