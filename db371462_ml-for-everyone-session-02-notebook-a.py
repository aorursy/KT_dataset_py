%matplotlib inline



import numpy as np

import pandas as pd



import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns



import sklearn

from sklearn import tree

from sklearn.tree import DecisionTreeClassifier, export_text

from sklearn.datasets import load_breast_cancer

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score



from matplotlib.colors import ListedColormap



from collections import Counter

from pprint import pprint
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

import sys



# Let's print what versions of the libraries we're using

print(f"python\t\tv {sys.version.split(' ')[0]}\n===")

for lib_ in [np, pd, sns, sklearn, ]:

    sep_ = '\t' if len(lib_.__name__) > 8 else '\t\t'

    print(f"{lib_.__name__}{sep_}v {lib_.__version__}"); del sep_, lib_
np.random.seed(556)



n = 100

n_half = n // 2



X = []

y = []



# Negative targets

for i in range(n_half):

    X.append(

        [

            np.random.normal(loc=-1, scale=0.4),

            np.random.normal(loc=0, scale=0.4),

        ]

    )

    y.append(0)



# Positive targets

for i in range(n_half):

    X.append(

        [

            np.random.normal(loc=1, scale=0.4),

            np.random.normal(loc=0, scale=0.4),

        ]

    )

    y.append(1)

    

X = np.asarray(X)

X.shape
plt.figure(figsize=(6, 6))



x_min, x_max = X[:, 0].min() - 0.2, X[:, 0].max() + 0.2

y_min, y_max = X[:, 1].min() - 0.2, X[:, 1].max() + 0.2



plt.scatter(X[:n_half, 0], X[:n_half, 1], marker="o", s=100, label="Benign")

plt.scatter(X[n_half:, 0], X[n_half:, 1], marker="o", s=100, label="Malignant")

plt.xlim(x_min, x_max)

plt.ylim(y_min, y_max)

plt.xlabel("x0")

plt.ylabel("x1")

plt.legend();
# cmap = sns.diverging_palette(220, 28, s=100.0, l=60, n=15, as_cmap=True)

cmap = sns.diverging_palette(220, 28, l=60, n=15, as_cmap=True)
# Fit a decision tree with a single split

clf = DecisionTreeClassifier(max_depth=1, criterion="entropy")

clf.fit(X, y)





# One figure to plot

plt.figure(figsize=(6, 6))





# Now plot the decision boundary using a fine mesh as input to a

# filled contour plot

x_min, x_max = X[:, 0].min() - 0.2, X[:, 0].max() + 0.2

y_min, y_max = X[:, 1].min() - 0.2, X[:, 1].max() + 0.2

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),

                     np.arange(y_min, y_max, 0.1))





Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

Z = Z.reshape(xx.shape)



plt.contourf(xx, yy, Z, alpha=0.4, cmap=cmap)



# Data set overlay

plt.scatter(X[:n_half, 0], X[:n_half, 1], marker="o", s=100, label="Benign")

plt.scatter(X[n_half:, 0], X[n_half:, 1], marker="o", s=100, label="Malignant")



plt.xlabel("x0")

plt.ylabel("x1")

plt.legend();
tree.plot_tree(clf, feature_names=["x0", "x1"]);
text = export_text(clf, feature_names=["x0", "x1"])

print(text)
def compute_entropy(y):

    count_y = Counter(y)

    N = len(y)

    N_0 = count_y[0]

    N_1 = count_y[1]

    p_0 = N_0 / N

    p_1 = N_1 / N

    

    epsilon = 1e-8

    H = -p_0 * np.log2(p_0 + epsilon) - p_1 * np.log2(p_1 + epsilon)

    

    return H
H_parent = compute_entropy(y)

N = len(y)



col_to_split_val_ig = dict()

for i_col in range(X.shape[1]):

    

    split_val_ig = list()

    

    sorted_x_y = sorted(zip(list(X[:, i_col]), list(y)))

    for i in range(1, len(sorted_x_y)):

        left_x, left_y = zip(*sorted_x_y[:i])

        right_x, right_y = zip(*sorted_x_y[i:])



        split_x_mean = (left_x[-1] + right_x[0]) / 2



        H_left = compute_entropy(left_y)

        H_right = compute_entropy(right_y)

        IG = H_parent - ((len(left_y) / N) * H_left + (len(right_y) / N) * H_right)



        split_val_ig.append((split_x_mean, IG))



    col_to_split_val_ig[i_col] = np.asarray(split_val_ig)
plt.figure(figsize=(6, 4))



plt.plot(col_to_split_val_ig[0][:, 0], col_to_split_val_ig[0][:, 1], label="x0")

plt.plot(col_to_split_val_ig[1][:, 0], col_to_split_val_ig[1][:, 1], label="x1")



plt.xlabel("split value")

plt.ylabel("information gain")

plt.legend();
n = 100

n_half = n // 2



X = []

y = []



# Negative targets

for i in range(n_half):

    X.append(

        [

            np.random.normal(loc=-1, scale=0.4),

            np.random.normal(loc=0, scale=0.8),

        ]

    )

    y.append(0)



# Positive targets

for i in range(n_half):

    X.append(

        [

            np.random.normal(loc=1, scale=0.4),

            np.random.normal(loc=0, scale=0.8),

        ]

    )

    y.append(1)

    

X = np.asarray(X)





theta = np.pi / 4

rot = np.asarray([

    [np.cos(theta), -np.sin(theta)],

    [np.sin(theta), np.cos(theta)]

])



X = X.dot(rot)
plt.figure(figsize=(6, 6))



x_min, x_max = X[:, 0].min() - 0.2, X[:, 0].max() + 0.2

y_min, y_max = X[:, 1].min() - 0.2, X[:, 1].max() + 0.2



plt.scatter(X[:n_half, 0], X[:n_half, 1], marker="o", s=100, label="Benign")

plt.scatter(X[n_half:, 0], X[n_half:, 1], marker="o", s=100, label="Malignant")

plt.xlim(x_min, x_max)

plt.ylim(y_min, y_max)

plt.xlabel("x0")

plt.ylabel("x1")

plt.legend();
# Fit a decision tree with a single split

clf = DecisionTreeClassifier(max_depth=1, criterion="entropy")

clf.fit(X, y)





# One figure to plot

plt.figure(figsize=(6, 6))





# Now plot the decision boundary using a fine mesh as input to a

# filled contour plot

x_min, x_max = X[:, 0].min() - 0.2, X[:, 0].max() + 0.2

y_min, y_max = X[:, 1].min() - 0.2, X[:, 1].max() + 0.2

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),

                     np.arange(y_min, y_max, 0.1))





Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

Z = Z.reshape(xx.shape)



cmap = sns.diverging_palette(220, 28,  l=60, sep=1, as_cmap=True)  # s=100.0,

plt.contourf(xx, yy, Z, alpha=0.4, cmap=cmap)



# Data set overlay

plt.scatter(X[:n_half, 0], X[:n_half, 1], marker="o", s=100, label="Benign")

plt.scatter(X[n_half:, 0], X[n_half:, 1], marker="o", s=100, label="Malignant")



plt.xlabel("x0")

plt.ylabel("x1")

plt.legend();
tree.plot_tree(clf, feature_names=["x0", "x1"]);
# Fit a decision tree with a single split

clf = DecisionTreeClassifier(max_depth=2, criterion="entropy")

clf.fit(X, y)





# One figure to plot

plt.figure(figsize=(6, 6))





# Now plot the decision boundary using a fine mesh as input to a

# filled contour plot

x_min, x_max = X[:, 0].min() - 0.2, X[:, 0].max() + 0.2

y_min, y_max = X[:, 1].min() - 0.2, X[:, 1].max() + 0.2

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),

                     np.arange(y_min, y_max, 0.1))





Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

Z = Z.reshape(xx.shape)



plt.contourf(xx, yy, Z, alpha=0.4, cmap=cmap)



# Data set overlay

plt.scatter(X[:n_half, 0], X[:n_half, 1], marker="o", s=100, label="Benign")

plt.scatter(X[n_half:, 0], X[n_half:, 1], marker="o", s=100, label="Malignant")



plt.xlabel("x0")

plt.ylabel("x1")

plt.legend();
plt.figure(figsize=(8, 8))

tree.plot_tree(clf, feature_names=["x0", "x1"]);
np.random.seed(114)



n = 200

n_half = n // 2

n_quarter = n // 4



X = []

y = []



# Negative targets

for i in range(n_half):

    X.append(

        [

            np.random.normal(loc=-0.8, scale=1),

            np.random.normal(loc=0.5, scale=2.1),

        ]

    )

    y.append(0)



# Positive targets

for i in range(n_quarter):

    X.append(

        [

            np.random.normal(loc=1, scale=0.6),

            np.random.normal(loc=0, scale=2),

        ]

    )

    y.append(1)

    X.append(

        [

            np.random.normal(loc=-1, scale=1.2),

            np.random.normal(loc=-2, scale=0.7),

        ]

    )

    y.append(1)

    

X = np.asarray(X)





theta = np.pi / 3

rot = np.asarray([

    [np.cos(theta), -np.sin(theta)],

    [np.sin(theta), np.cos(theta)]

])



X = X.dot(rot)
plt.figure(figsize=(6, 6))



x_min, x_max = X[:, 0].min() - 0.2, X[:, 0].max() + 0.2

y_min, y_max = X[:, 1].min() - 0.2, X[:, 1].max() + 0.2



plt.scatter(X[:n_half, 0], X[:n_half, 1], marker="o", s=100, label="Benign")

plt.scatter(X[n_half:, 0], X[n_half:, 1], marker="o", s=100, label="Malignant")

plt.xlim(x_min, x_max)

plt.ylim(y_min, y_max)

plt.xlabel("x0")

plt.ylabel("x1")

plt.legend();
# Fit a decision tree with a single split

clf = DecisionTreeClassifier(max_depth=8, criterion="entropy")

clf.fit(X, y)





# One figure to plot

plt.figure(figsize=(6, 6))





# Now plot the decision boundary using a fine mesh as input to a

# filled contour plot

x_min, x_max = X[:, 0].min() - 0.2, X[:, 0].max() + 0.2

y_min, y_max = X[:, 1].min() - 0.2, X[:, 1].max() + 0.2

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),

                     np.arange(y_min, y_max, 0.1))





Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

Z = Z.reshape(xx.shape)



plt.contourf(xx, yy, Z, alpha=0.4, cmap=cmap)



# Data set overlay

plt.scatter(X[:n_half, 0], X[:n_half, 1], marker="o", s=100, label="Benign")

plt.scatter(X[n_half:, 0], X[n_half:, 1], marker="o", s=100, label="Malignant")



plt.xlabel("x0")

plt.ylabel("x1")

plt.legend();
plt.figure(figsize=(16, 16))

tree.plot_tree(clf, feature_names=["x0", "x1"]);
bc_data = load_breast_cancer()

bc_data.keys()
bc_data.target_names
X = bc_data.data

y = bc_data.target



X.shape, y.shape
# swap y values so that malignant is 1

y = y ^ 1
Counter(y)
np.mean(y)
def print_summary(y_true, y_pred):

    print("confusion matrix:")

    for row in confusion_matrix(y_true, y_pred).tolist():

        print(row)



    print()

    print(f"accuracy: {accuracy_score(y_true, y_pred):.4f}"

          f" precision: {precision_score(y_true, y_pred):.4f}"

          f" recall: {recall_score(y_true, y_pred):.4f}")

# Fit on all data

model = DecisionTreeClassifier(

    max_depth=7, 

    criterion="entropy"

)

model.fit(X, y)

y_pred = model.predict(X)

print_summary(y, y_pred)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=0.20

)



model = DecisionTreeClassifier(

    max_depth=7, 

    criterion="entropy"

)

model.fit(X_train, y_train)



print("Training set:")

y_train_pred = model.predict(X_train)

print_summary(y_train, y_train_pred)



print("\nTest set:")

y_test_pred = model.predict(X_test)

print_summary(y_test, y_test_pred)
from sklearn.model_selection import KFold

from collections import defaultdict
train_metrics, test_metrics = defaultdict(list), defaultdict(list)



kf = KFold(n_splits=5)

for train_index, test_index in kf.split(X):

    X_train, X_test = X[train_index], X[test_index]

    y_train, y_test = y[train_index], y[test_index]

    

    model = DecisionTreeClassifier(

        max_depth=7, 

        criterion="entropy"

    )

    model.fit(X_train, y_train)

    

    y_train_pred = model.predict(X_train)

    train_metrics["accuracy"].append(accuracy_score(y_train, y_train_pred))

    train_metrics["precision"].append(precision_score(y_train, y_train_pred))

    train_metrics["recall"].append(recall_score(y_train, y_train_pred))

    

    y_test_pred = model.predict(X_test)

    test_metrics["accuracy"].append(accuracy_score(y_test, y_test_pred))

    test_metrics["precision"].append(precision_score(y_test, y_test_pred))

    test_metrics["recall"].append(recall_score(y_test, y_test_pred))

    

print("Training set stats:")

for metric in ("accuracy", "precision", "recall"):

    print(f"{metric}: {np.mean(train_metrics[metric]):.4f} ({np.std(train_metrics[metric]):.4f})")

    

print("\nTest set stats:")

for metric in ("accuracy", "precision", "recall"):

    print(f"{metric}: {np.mean(test_metrics[metric]):.4f} ({np.std(test_metrics[metric]):.4f})")
from sklearn.model_selection import LeaveOneOut
y_test_preds = list()



loo = LeaveOneOut()

for train_index, test_index in loo.split(X):

    X_train, X_test = X[train_index], X[test_index]

    y_train, y_test = y[train_index], y[test_index]

    

    model = DecisionTreeClassifier(

        max_depth=7, 

        criterion="entropy"

    )

    model.fit(X_train, y_train)

    

    y_test_pred = model.predict(X_test)

    y_test_preds.append(y_test_pred)



print("LOO test set:")

print_summary(y, y_test_preds)