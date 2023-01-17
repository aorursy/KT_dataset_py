from sklearn.datasets import load_iris

import pandas as pd



iris = load_iris()



# I prefer to work with DataFrame object so I convert the data into a DataFrame

iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)

iris_df['target'] = iris.target
iris_df.head()
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(random_state=42)

model.fit(iris.data, iris.target)
from sklearn.tree import plot_tree

import matplotlib.pyplot as plt



def plot_decision_tree(model, feature_names, class_names):

    # plot_tree function contains a list of all nodes and leaves of the Decision tree

    tree = plot_tree(model, feature_names = feature_names, class_names = class_names,

                     rounded = True, proportion = True, precision = 2, filled = True, fontsize=10)

    

    # I return the tree for the next part

    return tree
fig = plt.figure(figsize=(15, 12))

plot_decision_tree(model, iris.feature_names, iris.target_names)

plt.show()
def plot_decision_path_tree(model, X, class_names=None):

    fig = plt.figure(figsize=(10, 10))

    class_names = model.classes_.astype(str) if type(class_names) == type(None) else class_names

    feature_names = X.index if type(X) == type(pd.Series()) else X.columns

    

    # Getting the tree from the function programmed above

    tree = plot_decision_tree(model, feature_names, class_names)

    

    # Get the decision path of the wanted prediction 

    decision_path = model.decision_path([X])



    # Now remember the tree object contains all nodes and leaves so the logic here

    # is to loop into the tree and change visible attribute for components that 

    # are not in the decision path

    for i in range(0,len(tree)):

        if i not in decision_path.indices:

            plt.setp(tree[i],visible=False)



    plt.show()
from IPython.display import display



display(iris_df.iloc[0,:].to_frame().T)

plot_decision_path_tree(model, iris_df.iloc[3,:-1], class_names=iris.target_names)
display(iris_df.iloc[50,:].to_frame().T)

plot_decision_path_tree(model, iris_df.iloc[50,:-1], class_names=iris.target_names)
display(iris_df.iloc[100,:].to_frame().T)

plot_decision_path_tree(model, iris_df.iloc[100,:-1], class_names=iris.target_names)