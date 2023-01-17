# define functions for building the classifier



def calculate_distance(point, points):

    """ Calculate the Euclidean distance between a point

    and a set of other points.

    

    Parameters

    ----------

    point : 1D numpy array with p entries

    points: 2D numpy array of shape N x p

    """

    diff = point - points

    distances = np.diag(np.dot(diff, diff.T))

    return np.sqrt(distances)
def find_neighbors(point, points, k):

    """ Find the k neighbors of `point` in `points`.

    Returns the distances and indexes of the neighbors

    """

    distances = calculate_distance(point, points)

    indexes = np.argsort(distances)

    return distances[indexes][:k], indexes[:k]
def calculate_conditional_probability(point, data, g, k):

    """ Implementation of equation 6 """

    points = data[0]

    outputs = data[1]

    distances, indexes = find_neighbors(point, points, k)

    neigh_output = outputs[indexes]

    no_acceptable_points = len(neigh_output[neigh_output == g])

    if no_acceptable_points > 0:

        shortest_distance = distances[neigh_output == g].min()

    else:

        shortest_distance = np.inf # no neighbor corresponds to category g

    return no_acceptable_points/len(indexes), shortest_distance
def k_nn_classifier(point, data, k):

    """ Classify a new input `point` according to available

    pairs of input-output, `data` using the k-NN algorithm

    """

    unique_outputs = np.unique(data[1])

    probabilities = []

    shortest_distances = []

    for g in unique_outputs:

        prob, distance = calculate_conditional_probability(point, data, g, k)

        probabilities.append(prob)

        shortest_distances.append(distance)

    probabilities = np.array(probabilities)

    max_value = probabilities.max()

    max_index = np.argwhere(probabilities == max_value).flatten()

    # in case two different categories give the same probabilities, keep the

    # closest one

    if len(max_index) > 1:

        shortest = np.argmin(shortest_distances)

        return unique_outputs[shortest]

    return unique_outputs[max_index][0]
def calculate_accuracy(y_true, y_pred):

    """ Calculate the accuracy score between actual and predicted values.

    

    Assumes that y_true and y_pred have the same shape and are arrays of

    integers

    """

    score = (y_true == y_pred).sum()

    return score/len(y_true)
# Load needed libraries and Iris dataset

import numpy as np

from sklearn.datasets import load_iris



np.random.seed(10) # keep output notebook constant for every run



iris = load_iris()



# Observations of X and G rvs

X = iris.data

y = iris.target

# Shuffle data

shuffle_indexes = np.random.permutation(len(y))



X = X[shuffle_indexes]

y = y[shuffle_indexes]
# prepare training and test set

ratio = 0.2 # ratio% goes into test set

test_index = int(ratio*len(y))

X_test = X[: test_index]

y_test = y[: test_index]

X_train = X[test_index :]

y_train = y[test_index :]

data = (X_train, y_train)



predictions = []

for x in X_test:

    predictions.append(k_nn_classifier(x, data, 2))
calculate_accuracy(y_test, predictions)
# load the classifier and other libraries



from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection  import cross_val_score, StratifiedShuffleSplit

from sklearn.metrics import accuracy_score



import matplotlib.pyplot as plt
neigh = KNeighborsClassifier(2) # 2 neighbors
neigh.fit(X_train, y_train) # same training and test set
predictions_sk = neigh.predict(X_test) # predictions of KNeighborsClassifier
(predictions_sk != predictions).any()
# make a new training and test set



sample = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=20).split(X, y)

train_ind, test_ind = next(sample)

X_train, X_test, y_train, y_test = X[train_ind], X[test_ind], y[train_ind], y[test_ind]
# Let see how the number of neighbors affects the accuracy of the prediction

# we can look at the accuracy on the training set using cross validation



k_range = np.arange(1, 81)



scores = []

var = []

for k in k_range:

    classifier = KNeighborsClassifier(k)

    cv_scores = cross_val_score(classifier, X_train, y_train, cv=5, scoring='accuracy')

    scores.append(cv_scores.mean())

    var.append(cv_scores.std())

scores = np.array(scores)

var = np.array(var)
plt.plot(k_range, scores)

plt.xlabel('Number of neighbors')

plt.ylabel('Accuracy')
best_score = np.argwhere(scores == scores.max()).flatten()

for i, mean, std in zip(best_score, scores[best_score], var[best_score]):

    print('{} NN accuracy: {:0.3f} (+/- {:0.3f}))'.format(i+1, mean, 2*std))
scores_train = []

scores_test = []

for k in k_range:

    classifier = KNeighborsClassifier(k)

    classifier.fit(X_train, y_train)

    scores_train.append(classifier.score(X_train, y_train))

    scores_test.append(classifier.score(X_test, y_test))

plt.plot(k_range, scores_train, color='black', label='Accuracy on training set')

plt.plot(k_range, scores_test, color='blue', label='Accuracy on test set')

plt.xlabel('Number of neighbors')

plt.ylabel('Accuracy')

plt.legend(loc=3)
import seaborn as sns

import pandas as pd



# load Iris data in a pandas DataFrame

data = np.c_[X_train, y_train]

columns = iris.feature_names

df = pd.DataFrame(data, columns=columns + ['Species'])

df.head()
# convert Iris species from numerical to string value

def iris_name(x):

    return iris.target_names[int(x)]
df['Species'] = df['Species'].apply(iris_name)
# some statistics about the data

df.describe()
from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()



scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
scores = []

var = []

for k in k_range:

    classifier = KNeighborsClassifier(k)

    cv_scores = cross_val_score(classifier, X_train_scaled, y_train, cv=5, scoring='accuracy')

    scores.append(cv_scores.mean())

    var.append(cv_scores.std())

scores = np.array(scores)

var = np.array(var)

best_score = np.argwhere(scores == scores.max()).flatten()

for i, mean, std in zip(best_score, scores[best_score], var[best_score]):

    print('{} NN accuracy: {:0.3f} (+/- {:0.3f}))'.format(i+1, mean, 2*std))
df.var()
scores = []

var = []

for k in k_range:

    classifier = KNeighborsClassifier(k)    

    cv_scores = cross_val_score(classifier, df.drop(columns=['sepal width (cm)', 'Species']),

                                df['Species'], cv=5, scoring='accuracy')

    scores.append(cv_scores.mean())

    var.append(cv_scores.std())

scores = np.array(scores)

var = np.array(var)

best_score = np.argwhere(scores == scores.max()).flatten()

for i, mean, std in zip(best_score, scores[best_score], var[best_score]):

    print('{} NN accuracy: {:0.3f} (+/- {:0.3f}))'.format(i+1, mean, 2*std))
# visualize the data



sns.pairplot(data=df, vars=df.columns[:4],  diag_kind='kde', palette="husl")
# conditional on the output variable

sns.pairplot(data=df, hue='Species', vars=df.columns[:4],  diag_kind='kde', palette="husl")
df.corr()
fig = plt.gcf()

ax = plt.gca()

fig.set_size_inches(9,9)

fig=sns.heatmap(df.corr(), annot=True, vmin=-1, vmax=1, cmap="YlGnBu",

                square=True, cbar=True)

bottom, top = ax.get_ylim()

ax.set_ylim(bottom + 0.5, top - 0.5)
scores = []

var = []

for k in k_range:

    classifier = KNeighborsClassifier(k)    

    cv_scores = cross_val_score(classifier, df.iloc[:, :3], y_train, cv=5, scoring='accuracy')

    scores.append(cv_scores.mean())

    var.append(cv_scores.std())

scores = np.array(scores)

var = np.array(var)

best_score = np.argwhere(scores == scores.max()).flatten()

for i, mean, std in zip(best_score, scores[best_score], var[best_score]):

    print('{} NN accuracy: {:0.3f} (+/- {:0.3f}))'.format(i+1, mean, 2*std))
scores_per_feature = []

var_per_feature = []

for ifeat in range(X.shape[1]):

    feature = df.columns[ifeat]

    scores = []

    var = []

    for k in k_range:

        classifier = KNeighborsClassifier(k)

        data = df[feature].values

        cv_scores = cross_val_score(classifier, data[:, np.newaxis],

                                    df['Species'], cv=5, scoring='accuracy')

        scores.append(cv_scores.mean())

        var.append(cv_scores.std())

    scores = np.array(scores)

    var = np.array(var)

    best_score = np.argwhere(scores == scores.max()).flatten()

    print('Feature: ', feature)

    for i, mean, std in zip(best_score, scores[best_score], var[best_score]):

        print('{} NN accuracy: {:0.3f} (+/- {:0.3f}))'.format(i+1, mean, 2*std))
scores = []

var = []

for k in k_range:

    classifier = KNeighborsClassifier(k)    

    cv_scores = cross_val_score(classifier, df.iloc[:, 2:4], y_train, cv=5, scoring='accuracy')

    scores.append(cv_scores.mean())

    var.append(cv_scores.std())

scores = np.array(scores)

var = np.array(var)

best_score = np.argwhere(scores == scores.max()).flatten()

for i, mean, std in zip(best_score, scores[best_score], var[best_score]):

    print('{} NN accuracy: {:0.3f} (+/- {:0.3f}))'.format(i+1, mean, 2*std))
from scipy.stats import f_oneway # univariate ANOVA

# divide data points in groups according to the species

groups = [X_train[y_train == k] for k in np.unique(y_train)]



f_value, p_value = f_oneway(*groups)

print(f_value, p_value)
from sklearn.feature_selection import SelectKBest, f_classif



selector = SelectKBest(f_classif, 3)

selector.fit(X_train, y_train)

print(selector.scores_, selector.pvalues_)
X_train_new = selector.transform(X_train)



### Let's find the best number of neighbors for this last model and use it to find the accuracy on the test set

scores = []

var = []

for k in k_range:

    classifier = KNeighborsClassifier(k)    

    cv_scores = cross_val_score(classifier, X_train_new, y_train, cv=5, scoring='accuracy')

    scores.append(cv_scores.mean())

    var.append(cv_scores.std())

scores = np.array(scores)

var = np.array(var)

best_score = np.argwhere(scores == scores.max()).flatten()

for i, mean, std in zip(best_score, scores[best_score], var[best_score]):

    print('{} NN accuracy: {:0.3f} (+/- {:0.3f}))'.format(i+1, mean, 2*std))
final_clf = KNeighborsClassifier(9)

final_clf.fit(X_train_new, y_train)

ii = selector.get_support()

X_test_new = X_test[:, ii]

predictions = final_clf.predict(X_test_new)

accuracy_score(predictions, y_test)