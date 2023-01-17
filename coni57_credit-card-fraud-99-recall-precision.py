import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



from sklearn.decomposition import PCA, IncrementalPCA

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import recall_score, confusion_matrix, precision_score, accuracy_score

from sklearn.linear_model import SGDClassifier

from sklearn.base import clone

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from sklearn.neural_network import MLPClassifier



import warnings

warnings.filterwarnings('ignore')



%matplotlib inline
dataset = pd.read_csv("../input/creditcard.csv")
print(dataset.head())

print(dataset.describe())

print(dataset.info())
dataset['Class'].value_counts()
plt.hist(dataset['Amount'], bins=50)

plt.show()
dataset[(dataset['Amount'] > 3000) & (dataset['Class']==1)]
dataset = dataset[dataset['Amount'] < 3000]
fraud = dataset[dataset['Class']==1]

plt.hist(fraud['Amount'], bins=50)

plt.show()
bins = 50

Amount_max = 3000



Y = []

C = []

X = list(range(0, Amount_max, bins))

for i in X:

    s = fraud[(fraud['Amount'] > i) & (fraud['Amount'] <= i + bins)]['Amount'].sum()

    Y.append(s)

    if len(C) > 0:

        c = C[-1] + s

    else:

        c = s

    C.append(c)

    print("{} => {} $ - {}".format(i, s, c))



plt.bar(X, Y, width=50)

plt.ylabel('Cost')

plt.title('Cost of Frauds per amount')

plt.show()



plt.plot(X, C)

plt.show()
random_seed = 42

n_non_fraud = [100, 1000, 10000, 100000, dataset[dataset["Class"] == 0]["Class"].count()]         # min : 1 - max : 284807-492

n_components = 3

print(n_non_fraud)
for sample_size in n_non_fraud:

    a = dataset[dataset["Class"] == 1]                                                # we keep all frauds

    b = dataset[dataset["Class"] == 0].sample(sample_size, random_state=random_seed)  # we take "sample_size" non fraud to balance the ratio fraud/non_fraud



    dataset_us = pd.concat([a, b]).sample(frac=1, random_state=random_seed)           # merge and shuffle both dataset

    

    y = dataset_us["Class"]

    X = dataset_us.drop(["Time", "Class"], axis=1)

    

    X_scale = StandardScaler().fit_transform(X)

    X_proj = PCA(n_components=n_components).fit_transform(X_scale)

    

    plt.scatter(X_proj[:, 0], X_proj[:, 1], s=X_proj[:, 2], c=y)



    plt.xlabel("PCA1")

    plt.ylabel("PCA2")

    plt.title("{}-points".format(sample_size))

    #plt.savefig("{}-points".format(sample_size), dpi=600)

    plt.show()
# fit the PCA with 100k non-frauds

a = dataset[dataset["Class"] == 1]

b = dataset[dataset["Class"] == 0].sample(100000, random_state=random_seed)



dataset = pd.concat([a, b]).sample(frac=1, random_state=random_seed)



y = dataset["Class"]

X = dataset.drop(["Time", "Class"], axis=1)



X_scale = StandardScaler().fit_transform(dataset)

pca = PCA(n_components=0.95, svd_solver="full")

X_proj = pca.fit(X_scale)



# transform the full dataset with the pca create previously

dataset = pd.read_csv("../input/creditcard.csv")

y = dataset["Class"]

X = dataset.drop(["Time", "Class"], axis=1)



X_scale = StandardScaler().fit_transform(dataset)

X_proj = pca.transform(X_scale)
print(X_proj.shape)
plt.scatter(X_proj[:, 0], X_proj[:, 1], s=X_proj[:, 2], c=y)



plt.xlabel("PCA1")

plt.ylabel("PCA2")

plt.title("{}-points".format(X_proj.shape[0]))

#plt.savefig("{}-points".format(sample_size), dpi=600)

plt.show()
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)  #shuffle is required to avoid having unbalance folds

sgd_clf = SGDClassifier(random_state=random_seed)

for train_index, test_index in skf.split(X_proj, y):

    clone_clf = clone(sgd_clf)

    X_train, X_test = X_proj[train_index], X_proj[test_index]

    y_train, y_test = y[train_index], y[test_index]

    clone_clf.fit(X_train, y_train)

    y_pred = clone_clf.predict(X_test)

    recall = recall_score(y_test, y_pred)

    precision = precision_score(y_test, y_pred)

    print("\nRecall:\t\t {:.4f} \nPrecision:\t {:.4f}".format(recall, precision))

    print(confusion_matrix(y_test, y_pred))
tree_clf = DecisionTreeClassifier(max_depth=7, random_state=random_seed)

for train_index, test_index in skf.split(X_proj, y):

    clone_clf = clone(tree_clf)

    X_train, X_test = X_proj[train_index], X_proj[test_index]

    y_train, y_test = y[train_index], y[test_index]

    clone_clf.fit(X_train, y_train)

    y_pred = clone_clf.predict(X_test)

    recall = recall_score(y_test, y_pred)

    precision = precision_score(y_test, y_pred)

    print(recall, precision)

    print(confusion_matrix(y_test, y_pred))
# svc_clf = SVC(gamma=2, C=1)

# for train_index, test_index in skf.split(X_proj, y):

#     clone_clf = clone(svc_clf)

#     X_train, X_test = X_proj[train_index], X_proj[test_index]

#     y_train, y_test = y[train_index], y[test_index]

#     clone_clf.fit(X_train, y_train)

#     y_pred = clone_clf.predict(X_test)

#     recall = recall_score(y_test, y_pred)

#     precision = precision_score(y_test, y_pred)

#     print(recall, precision)

#     print(confusion_matrix(y_test, y_pred))



#     Usign this model make the computer crach :(
mlp_clf = MLPClassifier(hidden_layer_sizes=(50, 20), random_state=random_seed)

for train_index, test_index in skf.split(X_proj, y):

    clone_clf = clone(mlp_clf)

    X_train, X_test = X_proj[train_index], X_proj[test_index]

    y_train, y_test = y[train_index], y[test_index]

    clone_clf.fit(X_train, y_train)

    y_pred = clone_clf.predict(X_test)

    recall = recall_score(y_test, y_pred)

    precision = precision_score(y_test, y_pred)

    print(recall, precision)

    print(confusion_matrix(y_test, y_pred))
best_model = clone_clf

y_pred = best_model.predict(X_proj)
print("Accuracy score : {}".format(accuracy_score(y, y_pred)))

print("Precision score : {}".format(precision_score(y, y_pred)))

print("Recall score : {}".format(recall_score(y, y_pred)))

print("Confusion Matrix : {}".format(confusion_matrix(y, y_pred)))