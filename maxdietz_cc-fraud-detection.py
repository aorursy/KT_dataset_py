from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.tree._tree import TREE_LEAF



from sklearn.model_selection import learning_curve

from sklearn.model_selection import ShuffleSplit



import matplotlib.pyplot as plt

import numpy as np

import pandas as pd



def prune_index(inner_tree, index, threshold):

    if inner_tree.value[index].max() > threshold:

        # turn node into a leaf by "unlinking" its children

        inner_tree.children_left[index] = TREE_LEAF

        inner_tree.children_right[index] = TREE_LEAF

    # if there are shildren, visit them as well

    if inner_tree.children_left[index] != TREE_LEAF:

        prune_index(inner_tree, inner_tree.children_left[index], threshold)

        prune_index(inner_tree, inner_tree.children_right[index], threshold)





def get_results(data, target):

    cutoff = int(len(data)*0.5)

    training_data = data[:cutoff]

    training_target = target[:cutoff]

    test_data = data[cutoff:]

    test_target = target[cutoff:]

    positive_test_data = [x for x, y in zip(test_data, test_target) if y > 0.9]

    classifiers = [("KNN(3)", KNeighborsClassifier(n_neighbors=3)),

                   ("KNN(5)", KNeighborsClassifier(n_neighbors=5)),

                   ("SVM", SVC(gamma='scale')),

                   ("SVM (rbf)", SVC(kernel='rbf', gamma='scale')),

                   ("NN", MLPClassifier(solver='lbfgs', hidden_layer_sizes=(len(data[0]),), random_state=1))]

    for name, classifier in classifiers:

        classifier.fit(training_data, training_target)

        training_accuracy = len(list(filter(

            None, training_target == classifier.predict(training_data))))

        test_accuracy = len(list(filter(

            None, test_target == classifier.predict(test_data))))

        training_accuracy /= len(training_data)

        test_accuracy /= len(test_data)

        positive_test_count = len(list(filter(lambda t: t > 0.9, classifier.predict(positive_test_data))))

        print("{} (train): {}%".format(name, training_accuracy * 100.0))

        print("{} (test): {}%".format(name, test_accuracy * 100.0))

        print("{} (positive test): {} of {} ({}%)".format(name, positive_test_count, len(positive_test_data), positive_test_count / len(positive_test_data) * 100.))

        plot_learning_curve(name, classifier, data, target)



    # Do algos with pruning after

    classifier = DecisionTreeClassifier()

    classifier.fit(training_data, training_target)

    prune_index(classifier.tree_, 0, 50)

    training_accuracy = len(list(filter(

       None, training_target == classifier.predict(training_data))))

    test_accuracy = len(list(filter(

        None, test_target == classifier.predict(test_data))))

    training_accuracy /= len(training_data)

    test_accuracy /= len(test_data)

    positive_test_count = len(list(filter(lambda t: t > 0.9, classifier.predict(positive_test_data))))

    print("Decision Tree (train): {}%".format(training_accuracy * 100.0))

    print("Decision Tree (test): {}%".format(test_accuracy * 100.0))

    print("Decision Tree (positive test): {} of {} ({}%)".format(positive_test_count, len(positive_test_data), positive_test_count / len(positive_test_data) * 100.))

    plot_learning_curve("Decision Tree", classifier, data, target)



    classifier = AdaBoostClassifier()

    classifier.fit(training_data, training_target)

    for estimator in classifier.estimators_:

        prune_index(estimator.tree_, 0, 50)

    training_accuracy = len(list(filter(

       None, training_target == classifier.predict(training_data))))

    test_accuracy = len(list(filter(

        None, test_target == classifier.predict(test_data))))

    training_accuracy /= len(training_data)

    test_accuracy /= len(test_data)

    positive_test_count = len(list(filter(lambda t: t > 0.9, classifier.predict(positive_test_data))))

    print("Boosted Decision Tree (train): {}%".format(training_accuracy * 100.0))

    print("Boosted Decision Tree (test): {}%".format(test_accuracy * 100.0))

    print("Boosted Decision Tree (positive test): {} of {} ({}%)".format(positive_test_count, len(positive_test_data), positive_test_count / len(positive_test_data) * 100.))

    plot_learning_curve("Boosted Decision Tree", classifier, data, target)



def load_creditcard():

    inputs = []

    outputs = []

    df = pd.read_csv('../input/creditcardfraud/creditcard.csv')

    inputs = df[df.keys()[1:-1]]

    outputs = df[df.keys()[-1]]

    length = int(len(inputs.values) * 0.2)

    return inputs.values[:length], outputs.values[:length]



def load_weather():

    inputs = []

    outputs = []

    df = pd.read_csv('../input/weather-dataset-rattle-package/weatherAUS.csv')

    for bool_field in ('RainTomorrow', 'RainToday',):

        df[bool_field] = df[bool_field].where(df[bool_field].values != 'Yes', 1)

        df[bool_field] = df[bool_field].where(df[bool_field].values == 1, 0)

    df = df.fillna(0)

    inputs = df[["MinTemp", "MaxTemp", "Rainfall", "Evaporation", "Sunshine", "Humidity9am", "Humidity3pm",

                 "Pressure9am", "Pressure3pm", "Temp9am", "Temp3pm", "RainToday"]]

    outputs = df["RainTomorrow"]

    length = int(len(inputs.values) * 0.2)

    return inputs.values[:length], outputs.values[:length]



def plot_learning_curve(title, estimator, data, target):

    # From https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html

    plt.figure()

    plt.title(title)

    plt.xlabel("Training examples")

    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(estimator, data, target, cv=3)

    train_scores_mean = np.mean(train_scores, axis=1)

    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)

    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()



    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,

                     train_scores_mean + train_scores_std, alpha=0.1,

                     color="r")

    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,

                     test_scores_mean + test_scores_std, alpha=0.1, color="g")

    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",

             label="Training score")

    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",

             label="Cross-validation score")



    plt.legend(loc="best")

    return plt
get_results(*load_creditcard())
plt.show()
get_results(*load_weather())
plt.show()