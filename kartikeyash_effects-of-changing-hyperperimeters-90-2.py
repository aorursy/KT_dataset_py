import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import os



from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import BernoulliNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn import svm

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import VotingClassifier
data = pd.read_csv("../input/heart.csv")

data.info()

data.head()
x_data = data.drop('target', 1)

x_data = x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)).values
X_train, X_test, y_train, y_test = train_test_split(

    x_data, data['target'], test_size=0.2, random_state=0)

print("Number of training examples: {0}".format(X_train.shape[0]))

print("Number of features for a single example in the dataset: {0}".format(X_train.shape[1]))

print("Number of test examples: {0}".format(X_test.shape[0]))
C_values = [0.01, 0.1, 0.5, 1, 5, 10, 100, 1000, 1e42]

accuracies = []

weights_l2 = []



for C in C_values:

    clf_l2 = LogisticRegression(penalty='l2',

                             tol=0.0001,

                             C=C,

                             fit_intercept=True,

                             solver='liblinear',

    )



    # Train

    clf_l2.fit(X_train, y_train)

    

    # Store wieghts for further analysis

    weights_l2.append(clf_l2.coef_[0])

    

    # Calculate the mean accuracy on the given test data and labels.

    accuracies.append(clf_l2.score(X_test, y_test))



fig, ax = plt.subplots(ncols=2)



fig.set_size_inches(14, 4)



ax[0].set_xlabel("Value of C", fontsize=16)

ax[0].set_ylabel("Accuracy", fontsize=16)

ax[0].set_title("L2 Regularization", fontsize=20)

for i, accuracy in enumerate(accuracies):

    ax[0].text(i, accuracy, np.round(accuracies[i], 2), color='black', ha="center", fontsize=14)



sns.barplot(x=C_values, y=accuracies, ax=ax[0])





# For L1 regularization

accuracies = []

weights_l1 = []

for C in C_values:

    clf_l1 = LogisticRegression(penalty='l1',

                             C=C,

                             fit_intercept=True,

                             solver='liblinear',

    )



    # Train

    clf_l1.fit(X_train, y_train)



    # Store wieghts for further analysis

    weights_l1.append(clf_l1.coef_[0])



    # Calculate the mean accuracy on the given test data and labels.

    accuracies.append(clf_l1.score(X_test, y_test))



ax[1].set_xlabel("Value of C", fontsize=16)

ax[1].set_ylabel("Accuracy", fontsize=16)

ax[1].set_title("L1 Regularization", fontsize=20)



for i, accuracy in enumerate(accuracies):

    ax[1].text(i, accuracy, np.round(accuracies[i], 2), color='black', ha="center", fontsize=14)



sns.barplot(x=C_values, y=accuracies, ax=ax[1])
print("Value of C: %s" % C_values[0])

print("Weights:")

print(weights_l2[0])



print("Value of C: %s" % C_values[2])

print("Weights:")

print(weights_l2[2])



print("Value of C: %s" % C_values[-1])

print("Weights:")

print(weights_l2[-1])

print("Value of C: %s" % C_values[0])

print("Weights:")

print(weights_l1[0])



print("Value of C: %s" % C_values[2])

print("Weights:")

print(weights_l1[2])



print("Value of C: %s" % C_values[-1])

print("Weights:")

print(weights_l1[-1])
C = 1.0

solvers = ['newton-cg', 'lbfgs', 'sag', 'saga', 'liblinear']



# For L2 regularization

fig, ax = plt.subplots()

fig.set_size_inches(14, 4)



accuracies = []

for solver in solvers:

    clf = LogisticRegression(penalty='l2',

                             C=C,

                             fit_intercept=True,

                             solver=solver,

                             max_iter=500,

    )



    # Train

    clf.fit(X_train, y_train)



    # Calculate the mean accuracy on the given test data and labels.

    accuracies.append(clf.score(X_test, y_test))



ax.set_title('L2 Regularization', fontsize=20)

for solver in solvers:

    ax.set_xlabel("Solver", fontsize=16)

    ax.set_ylabel("Accuracy", fontsize=16)

    for i, accuracy in enumerate(accuracies):

        ax.text(i, accuracy, np.round(accuracies[i], 2), color='black', ha="center", fontsize=14)



sns.barplot(x=solvers, y=accuracies, ax=ax)



# For L1 regularization

fig, ax = plt.subplots()

fig.set_size_inches(14, 4)



accuracies = []

for solver in solvers[3:]:

    clf = LogisticRegression(penalty='l1',

                             C=C,

                             fit_intercept=True,

                             solver=solver,

                             max_iter=200,

    )



    # Train

    clf.fit(X_train, y_train)



    # Calculate the mean accuracy on the given test data and labels.

    accuracies.append(clf.score(X_test, y_test))



ax.set_title('L1 Regularization', fontsize=20)

for solver in solvers[3:]:

    ax.set_xlabel("Solver", fontsize=16)

    ax.set_ylabel("Accuracy", fontsize=16)

    for i, accuracy in enumerate(accuracies):

        ax.text(i, accuracy, np.round(accuracies[i], 2), color='black', ha="center", fontsize=14)



sns.barplot(x=solvers[3:], y=accuracies, ax=ax)

alpha_values = [0, 0.8, 1.0, 5.0, 100.0, 200.0, 230.0, 300.0, 500.0, 1000.0, 10000.0]



accuracies = []

for alpha in alpha_values:

    clf = BernoulliNB(alpha=alpha)



    # Train

    clf.fit(X_train, y_train)



    # Calculate the mean accuracy on the given test data and labels.

    accuracies.append(clf.score(X_test, y_test))



fig, ax = plt.subplots()

fig.set_size_inches(14, 4)



ax.set_xlabel("Value of $\\alpha$", fontsize=16)

ax.set_ylabel("Accuracy", fontsize=16)

ax.set_title("Naive Bayes", fontsize=20)



for i, accuracy in enumerate(accuracies):

    ax.text(i, accuracy, np.round(accuracies[i], 2), color='black', ha="center", fontsize=14)



sns.barplot(x=alpha_values, y=accuracies, ax=ax)
neighbors = [1, 2, 5, 10, 14, 16, 20, 25, 32, 50, 60]



accuracies = []

for neighbor in neighbors:

    clf = KNeighborsClassifier(

        n_neighbors=neighbor,

        algorithm='brute',

    )



    # Train

    clf.fit(X_train, y_train)



    # Calculate the mean accuracy on the given test data and labels.

    accuracies.append(clf.score(X_test, y_test))

fig, ax = plt.subplots()

fig.set_size_inches(14, 4)



ax.set_xlabel("Value of $K$", fontsize=16)

ax.set_ylabel("Accuracy", fontsize=16)

ax.set_title("K Nearest Neighbors", fontsize=20)



for i, accuracy in enumerate(accuracies):

    ax.text(i, accuracy, np.round(accuracies[i], 2), color='black', ha="center", fontsize=14)



sns.barplot(x=neighbors, y=accuracies, ax=ax)
algorithms = ['ball_tree', 'kd_tree', 'brute']



accuracies = []

for algorithm in algorithms:

    clf = KNeighborsClassifier(

        n_neighbors=5,

        algorithm=algorithm,

    )



    # Train

    clf.fit(X_train, y_train)



    # Calculate the mean accuracy on the given test data and labels.

    accuracies.append(clf.score(X_test, y_test))

fig, ax = plt.subplots()

fig.set_size_inches(14, 4)



ax.set_xlabel("Algorithm", fontsize=16)

ax.set_ylabel("Accuracy", fontsize=16)

ax.set_title("K Nearest Neighbors", fontsize=20)



for i, accuracy in enumerate(accuracies):

    ax.text(i, accuracy, np.round(accuracies[i], 2), color='black', ha="center", fontsize=14)



sns.barplot(x=algorithms, y=accuracies, ax=ax)
kernels = ['linear', 'poly', 'rbf', 'sigmoid']



accuracies = []

for kernel in kernels:

    clf = clf = svm.SVC(

        kernel=kernel,

    )



    # Train

    clf.fit(X_train, y_train)



    # Calculate the mean accuracy on the given test data and labels.

    accuracies.append(clf.score(X_test, y_test))



fig, ax = plt.subplots()

fig.set_size_inches(14, 4)



ax.set_xlabel("Kernel", fontsize=16)

ax.set_ylabel("Accuracy", fontsize=16)

ax.set_title("SVM Classifier", fontsize=20)



for i, accuracy in enumerate(accuracies):

    ax.text(i, accuracy, np.round(accuracies[i], 2), color='black', ha="center", fontsize=14)



sns.barplot(x=kernels, y=accuracies, ax=ax)
C_values = [0.01, 0.1, 0.2, 0.5, 1, 5, 10, 20]



accuracies = []



for C in C_values:

    clf = clf = svm.SVC(

        C=C,

        kernel='linear',

    )



    # Train

    clf.fit(X_train, y_train)



    # Calculate the mean accuracy on the given test data and labels.

    accuracies.append(clf.score(X_test, y_test))



    # Train

    clf.fit(X_train, y_train)



fig, ax = plt.subplots()

fig.set_size_inches(14, 4)



ax.set_xlabel("Value of C", fontsize=16)

ax.set_ylabel("Accuracy", fontsize=16)

ax.set_title("Effect of C SVM", fontsize=20)

for i, accuracy in enumerate(accuracies):

    ax.text(i, accuracy, np.round(accuracies[i], 2), color='black', ha="center", fontsize=14)



sns.barplot(x=C_values, y=accuracies, ax=ax)
gammas = [0.01, 0.1, 0.2, 0.5, 1, 5]



accuracies = []



for gamma in gammas:

    clf = clf = svm.SVC(

        C=5,

        kernel='rbf',

        gamma=gamma,

    )



    # Train

    clf.fit(X_train, y_train)



    # Calculate the mean accuracy on the given test data and labels.

    accuracies.append(clf.score(X_test, y_test))



    # Train

    clf.fit(X_train, y_train)



fig, ax = plt.subplots()

fig.set_size_inches(14, 4)



ax.set_xlabel("Value of $\\gamma$", fontsize=16)

ax.set_ylabel("Accuracy", fontsize=16)

ax.set_title("Effect of $\\gamma$ SVM", fontsize=20)

for i, accuracy in enumerate(accuracies):

    ax.text(i, accuracy, np.round(accuracies[i], 2), color='black', ha="center", fontsize=14)



sns.barplot(x=gammas, y=accuracies, ax=ax)
max_depths = [2, 3, 4, 5, 10, 20, 50, 100]

accuracies = []



# For cross entropy loss

for max_depth in max_depths:

    clf = DecisionTreeClassifier(criterion='entropy',

                                max_depth=max_depth,

    )



    # Train

    clf.fit(X_train, y_train)

    

    # Calculate the mean accuracy on the given test data and labels.

    accuracies.append(clf.score(X_test, y_test))



fig, ax = plt.subplots(ncols=2)

fig.set_size_inches(14, 4)



ax[0].set_xlabel("Max Depth", fontsize=16)

ax[0].set_ylabel("Accuracy", fontsize=16)

ax[0].set_title("Cross Entropy", fontsize=20)

for i, accuracy in enumerate(accuracies):

    ax[0].text(i, accuracy, np.round(accuracies[i], 2), color='black', ha="center", fontsize=14)



sns.barplot(x=max_depths, y=accuracies, ax=ax[0])



accuracies = []



# For Gini loss

for max_depth in max_depths:

    clf = DecisionTreeClassifier(criterion='entropy',

                                max_depth=max_depth,

    )



    # Train

    clf.fit(X_train, y_train)

    

    # Calculate the mean accuracy on the given test data and labels.

    accuracies.append(clf.score(X_test, y_test))



ax[1].set_xlabel("Max Depth", fontsize=16)

ax[1].set_ylabel("Accuracy", fontsize=16)

ax[1].set_title("Gini impurity", fontsize=20)

for i, accuracy in enumerate(accuracies):

    ax[1].text(i, accuracy, np.round(accuracies[i], 2), color='black', ha="center", fontsize=14)



sns.barplot(x=max_depths, y=accuracies, ax=ax[1])

min_samples_splits = [2, 3, 4, 5, 10, 20, 50, 100]



accuracies = []



for min_samples_split in min_samples_splits:

    clf = DecisionTreeClassifier(criterion='gini',

                                 max_depth=3,

                                 min_samples_split=min_samples_split,

    )



    # Train

    clf.fit(X_train, y_train)

    

    # Calculate the mean accuracy on the given test data and labels.

    accuracies.append(clf.score(X_test, y_test))



fig, ax = plt.subplots()

fig.set_size_inches(14, 4)



ax.set_xlabel("Min Samples Split", fontsize=16)

ax.set_ylabel("Accuracy", fontsize=16)

ax.set_title("Min Samples Split vs Accuracy", fontsize=20)

for i, accuracy in enumerate(accuracies):

    ax.text(i, accuracy, np.round(accuracies[i], 2), color='black', ha="center", fontsize=14)



sns.barplot(x=min_samples_splits, y=accuracies, ax=ax)

min_samples_leaves = [2, 3, 4, 5, 10, 20, 50, 100]



accuracies = []



for min_samples_leaf in min_samples_leaves:

    clf = DecisionTreeClassifier(criterion='gini',

                                 max_depth=3,

                                 min_samples_leaf=min_samples_leaf,

    )



    # Train

    clf.fit(X_train, y_train)

    

    # Calculate the mean accuracy on the given test data and labels.

    accuracies.append(clf.score(X_test, y_test))



fig, ax = plt.subplots()

fig.set_size_inches(14, 4)



ax.set_xlabel("Min Samples Leaf", fontsize=16)

ax.set_ylabel("Accuracy", fontsize=16)

ax.set_title("Min Samples Leaf vs Accuracy", fontsize=20)

for i, accuracy in enumerate(accuracies):

    ax.text(i, accuracy, np.round(accuracies[i], 2), color='black', ha="center", fontsize=14)



sns.barplot(x=min_samples_leaves, y=accuracies, ax=ax)
max_leaves_nodes = [2, 3, 4, 5, 6, 7, 10, 20, 50, 100]



accuracies = []



for max_leaf_nodes in max_leaves_nodes:

    clf = DecisionTreeClassifier(criterion='gini',

                                 max_depth=3,

                                 max_leaf_nodes=max_leaf_nodes,

    )



    # Train

    clf.fit(X_train, y_train)

    

    # Calculate the mean accuracy on the given test data and labels.

    accuracies.append(clf.score(X_test, y_test))



fig, ax = plt.subplots()

fig.set_size_inches(14, 4)



ax.set_xlabel("Max Leaf Nodes", fontsize=16)

ax.set_ylabel("Accuracy", fontsize=16)

ax.set_title("Max Leaf Nodes vs Accuracy", fontsize=20)

for i, accuracy in enumerate(accuracies):

    ax.text(i, accuracy, np.round(accuracies[i], 2), color='black', ha="center", fontsize=14)



sns.barplot(x=max_leaves_nodes, y=accuracies, ax=ax)
num_estimators = [2, 3, 4, 5, 10, 20, 50, 100, 200]

accuracies = []



# With bootstraping

for n_estimators in num_estimators:

    clf = RandomForestClassifier(

        criterion='gini',

        max_depth=3,

        min_samples_leaf=4,

        max_leaf_nodes=5,

        n_estimators=n_estimators,

        bootstrap=True,

        random_state=10,

    )



    # Train

    clf.fit(X_train, y_train)



    # Calculate the mean accuracy on the given test data and labels.

    accuracies.append(clf.score(X_test, y_test))



fig, ax = plt.subplots(ncols=2)

fig.set_size_inches(14, 4)



ax[0].set_xlabel("Number of estimators", fontsize=16)

ax[0].set_ylabel("Accuracy", fontsize=16)

ax[0].set_title("With Bootstraping", fontsize=20)

for i, accuracy in enumerate(accuracies):

    ax[0].text(i, accuracy, np.round(accuracies[i], 2), color='black', ha="center", fontsize=14)



sns.barplot(x=num_estimators, y=accuracies, ax=ax[0])



accuracies = []



# Without bootstraping

for n_estimators in num_estimators:

    clf = RandomForestClassifier(

        criterion='gini',

        max_depth=3,

        min_samples_leaf=4,

        max_leaf_nodes=5,

        n_estimators=n_estimators,

        bootstrap=False,

        random_state=10,

    )



    # Train

    clf.fit(X_train, y_train)

    

    # Calculate the mean accuracy on the given test data and labels.

    accuracies.append(clf.score(X_test, y_test))



ax[1].set_xlabel("Number of estimators", fontsize=16)

ax[1].set_ylabel("Accuracy", fontsize=16)

ax[1].set_title("Without Bootstraping", fontsize=20)

for i, accuracy in enumerate(accuracies):

    ax[1].text(i, accuracy, np.round(accuracies[i], 2), color='black', ha="center", fontsize=14)



sns.barplot(x=num_estimators, y=accuracies, ax=ax[1])
learning_rates = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]



accuracies = []



# With deviance loss

for learning_rate in learning_rates:

    clf = GradientBoostingClassifier(

        loss='deviance',

        learning_rate=learning_rate,

        max_depth=3,

        min_samples_leaf=4,

        max_leaf_nodes=5,

        random_state=10,

    )



    # Train

    clf.fit(X_train, y_train)



    # Calculate the mean accuracy on the given test data and labels.

    accuracies.append(clf.score(X_test, y_test))



fig, ax = plt.subplots(ncols=2)

fig.set_size_inches(14, 4)



ax[0].set_xlabel("Learning rate", fontsize=16)

ax[0].set_ylabel("Accuracy", fontsize=16)

ax[0].set_title("With Deviance", fontsize=20)

for i, accuracy in enumerate(accuracies):

    ax[0].text(i, accuracy, np.round(accuracies[i], 2), color='black', ha="center", fontsize=14)



sns.barplot(x=learning_rates, y=accuracies, ax=ax[0])



accuracies = []



# With exponential loss

for learning_rate in learning_rates:

    clf = GradientBoostingClassifier(

        loss='exponential',

        learning_rate=learning_rate,

        max_depth=3,

        min_samples_leaf=4,

        max_leaf_nodes=5,

        random_state=10,

    )



    # Train

    clf.fit(X_train, y_train)

    

    # Calculate the mean accuracy on the given test data and labels.

    accuracies.append(clf.score(X_test, y_test))



ax[1].set_xlabel("Learning rate", fontsize=16)

ax[1].set_ylabel("Accuracy", fontsize=16)

ax[1].set_title("Exponential Loss", fontsize=20)

for i, accuracy in enumerate(accuracies):

    ax[1].text(i, accuracy, np.round(accuracies[i], 2), color='black', ha="center", fontsize=14)



sns.barplot(x=learning_rates, y=accuracies, ax=ax[1])
num_estimators = [2, 3, 4, 5, 10, 20, 50, 100, 200]

accuracies = []



# With bootstraping

for n_estimators in num_estimators:

    clf = GradientBoostingClassifier(

        n_estimators=n_estimators,

        loss='deviance',

        learning_rate=0.1,

        max_depth=3,

        min_samples_leaf=4,

        max_leaf_nodes=5,

        random_state=10,

    )



    # Train

    clf.fit(X_train, y_train)



    # Calculate the mean accuracy on the given test data and labels.

    accuracies.append(clf.score(X_test, y_test))



fig, ax = plt.subplots()

fig.set_size_inches(14, 4)



ax.set_xlabel("Number of estimators", fontsize=16)

ax.set_ylabel("Accuracy", fontsize=16)

ax.set_title("Number of estimators Vs accuracy", fontsize=20)

for i, accuracy in enumerate(accuracies):

    ax.text(i, accuracy, np.round(accuracies[i], 2), color='black', ha="center", fontsize=14)



sns.barplot(x=num_estimators, y=accuracies, ax=ax)
subsampling_values = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

accuracies = []



# With bootstraping

for subsample in subsampling_values:

    clf = GradientBoostingClassifier(

        subsample=subsample,

        n_estimators=50,

        loss='deviance',

        learning_rate=0.1,

        max_depth=3,

        min_samples_leaf=4,

        max_leaf_nodes=5,

        random_state=10,

    )



    # Train

    clf.fit(X_train, y_train)



    # Calculate the mean accuracy on the given test data and labels.

    accuracies.append(clf.score(X_test, y_test))



fig, ax = plt.subplots()

fig.set_size_inches(14, 4)



ax.set_xlabel("Subsampling", fontsize=16)

ax.set_ylabel("Accuracy", fontsize=16)

ax.set_title("Subsampling Vs accuracy", fontsize=20)

for i, accuracy in enumerate(accuracies):

    ax.text(i, accuracy, np.round(accuracies[i], 3), color='black', ha="center", fontsize=14)



sns.barplot(x=subsampling_values, y=accuracies, ax=ax)


classifiers = [

    svm.SVC(probability=True, C=5, kernel='rbf', gamma=0.5,),

    LogisticRegression(penalty='l2', tol=0.0001, C=1.0, fit_intercept=True, solver='liblinear'),

    RandomForestClassifier(criterion='gini', max_depth=3, min_samples_leaf=4, max_leaf_nodes=5,

                           n_estimators=10, bootstrap=True,random_state=10

    ),

    DecisionTreeClassifier(criterion='gini', max_depth=3, max_leaf_nodes=5),

    GradientBoostingClassifier(n_estimators=50, subsample=0.6, loss='deviance', learning_rate=0.1, max_depth=3,

                               min_samples_leaf=4, max_leaf_nodes=5, random_state=10,

    ),

]



classifier_names=['SVM', 'LogisticRegression', 'RandomForest', 'DecisionTree', 'GradientBoosting']



accuracies = []



# With bootstraping

for classifier in classifiers:

    clf = AdaBoostClassifier(

        base_estimator=classifier,

        learning_rate=0.1,

        random_state=10,

    )



    # Train

    clf.fit(X_train, y_train)



    # Calculate the mean accuracy on the given test data and labels.

    accuracies.append(clf.score(X_test, y_test))



fig, ax = plt.subplots()

fig.set_size_inches(14, 4)



ax.set_xlabel("Number of estimators", fontsize=16)

ax.set_ylabel("Accuracy", fontsize=16)

ax.set_title("Number of estimators Vs accuracy", fontsize=20)

for i, accuracy in enumerate(accuracies):

    ax.text(i, accuracy, np.round(accuracies[i], 2), color='black', ha="center", fontsize=14)



sns.barplot(x=classifier_names, y=accuracies, ax=ax)
num_estimators = [2, 3, 4, 5, 10, 20, 50, 100, 200]

accuracies = []



# With bootstraping

for n_estimators in num_estimators:

    clf = AdaBoostClassifier(

        n_estimators=n_estimators,

        learning_rate=0.1,

        random_state=10,

    )



    # Train

    clf.fit(X_train, y_train)



    # Calculate the mean accuracy on the given test data and labels.

    accuracies.append(clf.score(X_test, y_test))



fig, ax = plt.subplots()

fig.set_size_inches(14, 4)



ax.set_xlabel("Number of estimators", fontsize=16)

ax.set_ylabel("Accuracy", fontsize=16)

ax.set_title("Number of estimators Vs accuracy", fontsize=20)

for i, accuracy in enumerate(accuracies):

    ax.text(i, accuracy, np.round(accuracies[i], 2), color='black', ha="center", fontsize=14)



sns.barplot(x=num_estimators, y=accuracies, ax=ax)
algorithms = ['SAMME', 'SAMME.R']

accuracies = []



for algorithm in algorithms:

    clf = AdaBoostClassifier(

        n_estimators=5,

        learning_rate=0.1,

        random_state=10,

        algorithm=algorithm,

    )



    # Train

    clf.fit(X_train, y_train)



    # Calculate the mean accuracy on the given test data and labels.

    accuracies.append(clf.score(X_test, y_test))



fig, ax = plt.subplots()

fig.set_size_inches(14, 4)



ax.set_xlabel("Algorithms", fontsize=16)

ax.set_ylabel("Accuracy", fontsize=16)

ax.set_title("Agorithm Vs accuracy", fontsize=20)

for i, accuracy in enumerate(accuracies):

    ax.text(i, accuracy, np.round(accuracies[i], 2), color='black', ha="center", fontsize=14)



sns.barplot(x=algorithms, y=accuracies, ax=ax)
votings = ['hard', 'soft']



accuracies = []



classifiers = [

    ('SVM',svm.SVC(probability=True, C=5, kernel='rbf', gamma=0.5)),

    ('LR',LogisticRegression(penalty='l2', tol=0.0001, C=1.0, fit_intercept=True, solver='liblinear')),

    ('RF',RandomForestClassifier(criterion='gini', max_depth=3, min_samples_leaf=4, max_leaf_nodes=5,

                           n_estimators=10, bootstrap=True,random_state=10

    )),

    ('DT',DecisionTreeClassifier(criterion='gini', max_depth=3, max_leaf_nodes=5)),

    ('GB',GradientBoostingClassifier(n_estimators=50, subsample=0.6, loss='deviance', learning_rate=0.1, max_depth=3,

                               min_samples_leaf=4, max_leaf_nodes=5, random_state=10,

    )),

    ('AB',AdaBoostClassifier(n_estimators=5, learning_rate=0.1, random_state=10)),

]



for voting in votings:

    clf = VotingClassifier(

        estimators=classifiers,

        voting=voting,

    )



    # Train

    clf.fit(X_train, y_train)



    # Calculate the mean accuracy on the given test data and labels.

    accuracies.append(clf.score(X_test, y_test))



fig, ax = plt.subplots()

fig.set_size_inches(14, 4)



ax.set_xlabel("Voting Strategy", fontsize=16)

ax.set_ylabel("Accuracy", fontsize=16)

ax.set_title("Voting Strategy Vs Accuracy", fontsize=20)

for i, accuracy in enumerate(accuracies):

    ax.text(i, accuracy, np.round(accuracies[i], 3), color='black', ha="center", fontsize=14)



sns.barplot(x=votings, y=accuracies, ax=ax)