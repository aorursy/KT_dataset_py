#import all the necessary libraries

#for decision tree classification and graphviz usage

from sklearn.tree import DecisionTreeClassifier, export_graphviz

from sklearn import tree



#the dataset we want to use

from sklearn.datasets import load_wine



#libraries to play with graphics

from IPython.display import SVG

from graphviz import Source

from IPython.display import display
# load the wine dataset

data = load_wine()
# show its features

# class labels

features = data.feature_names

print("The features are: " + str(features))

# print dataset description

print(data.DESCR)

# feature matrix

X = data.data



# target vector

y = data.target



# Fit the decision tree classifier (using the whole data as the training set)

estimator = DecisionTreeClassifier()

estimator.fit(X, y)



# Export the induced tree to the graphviz format

# When filled option of export_graphviz is set to True each node gets colored according to the majority class.

the_tree= tree.export_graphviz(estimator, out_file=None, feature_names=features, 

                     class_names=['Wine Type 0', 'Wine Type 1', 'Wine Type 2'], filled = True)

graph = Source(the_tree)



# Display the full grown tree

display(SVG(graph.pipe(format='svg')))
# Fit the decision tree classifier (using the whole data as the training set)

estimator = DecisionTreeClassifier(max_depth=2)

estimator.fit(X, y)



# Export the induced tree to the graphviz format

# When filled option of export_graphviz is set to True each node gets colored according to the majority class.

the_tree= tree.export_graphviz(estimator, out_file=None, feature_names=features, 

                     class_names=['Wine Type 0', 'Wine Type 1', 'Wine Type 2'], filled = True)

graph = Source(the_tree)



# Display the full grown tree

display(SVG(graph.pipe(format='svg')))
# Fit the decision tree classifier (using the whole data as the training set)

estimator = DecisionTreeClassifier(max_depth=2, min_samples_split=70)

estimator.fit(X, y)



# Export the induced tree to the graphviz format

# When filled option of export_graphviz is set to True each node gets colored according to the majority class.

the_tree= tree.export_graphviz(estimator, out_file=None, feature_names=features, 

                     class_names=['Wine Type 0', 'Wine Type 1', 'Wine Type 2'], filled = True)

graph = Source(the_tree)



# Display the full grown tree

display(SVG(graph.pipe(format='svg')))
#The same classifier with entropy based splits

estimator = DecisionTreeClassifier(max_depth=2, min_samples_split=70, criterion="entropy")

estimator.fit(X, y)



# Export the induced tree to the graphviz format

# When filled option of export_graphviz is set to True each node gets colored according to the majority class.

the_tree= tree.export_graphviz(estimator, out_file=None, feature_names=features, 

                     class_names=['Wine Type 0', 'Wine Type 1', 'Wine Type 2'], filled = True)

graph = Source(the_tree)



# Display the full grown tree

display(SVG(graph.pipe(format='svg')))