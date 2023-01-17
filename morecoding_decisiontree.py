from sklearn.datasets import load_iris

from sklearn.datasets import load_breast_cancer

from sklearn import tree

from sklearn.model_selection import train_test_split

import graphviz 

import matplotlib.pyplot as plt



data = load_breast_cancer()

X, y = load_breast_cancer(return_X_y=True)

#data = load_iris()

#X, y = load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)



#decision tree built using the training data set only

clf = tree.DecisionTreeClassifier(random_state=0)

clf = clf.fit(X_train, y_train)



#test the predictor using the first test case

q = [X_test[1]]

print("query", q)

print("prediction", clf.predict(q))



#pruning the tree using different alphas, save the pruned trees to clfs

#clfs[0] -- the original tree without pruning; clfs[-1] -- the last tree with only one single node

path = clf.cost_complexity_pruning_path(X_train, y_train)

ccp_alphas, impurities = path.ccp_alphas, path.impurities

clfs = []

for ccp_alpha in ccp_alphas:

    clf = tree.DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)

    clf.fit(X_train, y_train)

    clfs.append(clf)

print("Number of nodes in the last tree is: {} with ccp_alpha: {}".format(clfs[-1].tree_.node_count, ccp_alphas[-1]))

#remove the last tree with only one node

clfs = clfs[:-1]

ccp_alphas = ccp_alphas[:-1]



#compute the accuracy over training dataset and the test dataset

train_scores = [clf.score(X_train, y_train) for clf in clfs]

test_scores = [clf.score(X_test, y_test) for clf in clfs]

#show the accuracy versus alpha (for pruning)

fig, ax = plt.subplots()

ax.set_xlabel("alpha")

ax.set_ylabel("accuracy")

ax.set_title("Accuracy vs alpha for training and testing sets")

ax.plot(ccp_alphas, train_scores, marker='o', label="train", drawstyle="steps-post")

ax.plot(ccp_alphas, test_scores, marker='o', label="test", drawstyle="steps-post")

ax.legend()

plt.show()
#visualize the pruned tree

max_test_accuracy = test_scores[0]

clf_select = 0

for idx in range(1, len(clfs)):

    if test_scores[idx] > max_test_accuracy:

            max_test_accuracy = test_scores[idx]

            clf_select = idx

print("prunned tree, accuracy ", max_test_accuracy)

#print("alpha ", ccp_alpha[clf_select])



#prepare dot file for graphviz

dot_data = tree.export_graphviz(clfs[clf_select], out_file=None, 

                      feature_names=data.feature_names,  

                      class_names=data.target_names,  

                      filled=True, rounded=True,  

                      special_characters=True)  

graph = graphviz.Source(dot_data)  

graph 

#the original tree is a lot more complex

#prepare dot file for graphviz

dot_data = tree.export_graphviz(clfs[0], out_file=None, 

                      feature_names=data.feature_names,  

                      class_names=data.target_names,  

                      filled=True, rounded=True,  

                      special_characters=True)  

graph = graphviz.Source(dot_data)  

graph 


