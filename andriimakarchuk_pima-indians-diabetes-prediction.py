import pandas as pd
data = pd.read_csv("../input/pima-indians-diabetes-database/diabetes.csv")

data.head()
from sklearn import tree



X = data.drop(labels="Outcome", axis=1)

y = data["Outcome"]

mainTree = tree.DecisionTreeClassifier(

    min_samples_leaf = 24

)

mainTree.fit(X, y)
print( "R^2 for main decision tree: "+str(mainTree.score(X, y)) )
tree.plot_tree(mainTree)