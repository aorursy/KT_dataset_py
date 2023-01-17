# load the modules
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn import svm
# read the data
train = pd.read_csv('../input/train.csv')
# test  = pd.read_csv('../input/test.csv')

train.info()
trainY = train.label.values
trainX = train.drop("label",axis=1).values
X_train, X_test, Y_train, Y_test = train_test_split(trainX, trainY, test_size=0.9)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)
random_integer = np.random.randint(2000, size=1)
some_digit = X_train[random_integer]
some_digit_image = some_digit.reshape(28, 28)
plt.imshow(some_digit_image, cmap = matplotlib.cm.binary)
plt.axis("off")
plt.title(int(Y_train[random_integer]))
plt.show()
fig1, ax1 = plt.subplots(1,15, figsize=(15,10))
for i,j in enumerate(np.random.randint(X_train.shape[0], size=15)):
    ax1[i].matshow(X_train[j].reshape((28,28)), cmap="Greys")
    ax1[i].axis('off')
    ax1[i].set_title(Y_train[j])
# Let’s also shuffle the training set;
# this will guarantee that all cross-validation folds will be similar (you don’t want one fold to be missing some digits).
shuffle_index = np.random.permutation(X_train.shape[0])
X_train, Y_train = X_train[shuffle_index], Y_train[shuffle_index]
Y_train_5 = (Y_train == 5) # True for all 5s, False for all other digits.
Y_test_5 = (Y_test == 5)

svc = svm.SVC(kernel='linear').fit(X_train, Y_train_5)
Y_pred = svc.predict(X_test)
n_correct = sum(Y_pred == Y_test_5)
print(n_correct / len(Y_pred))
# Implementing Cross-Validation
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

skfolds = StratifiedKFold(n_splits=3)
for train_index, test_index in skfolds.split(X_train, Y_train_5):
    clone_clf = clone(svc)
    X_train_folds = X_train[train_index]
    Y_train_folds = (Y_train_5[train_index])
    X_test_fold = X_train[test_index]
    Y_test_fold = (Y_train_5[test_index])
    clone_clf.fit(X_train_folds, Y_train_folds)
    Y_pred = clone_clf.predict(X_test_fold)
    n_correct = sum(Y_pred == Y_test_fold)
    print(n_correct / len(Y_pred))
from sklearn.model_selection import cross_val_score
cross_val_score(svc, X_train, Y_train_5, cv=3, scoring="accuracy")
from sklearn.model_selection import cross_val_predict
Y_train_pred = cross_val_predict(svc, X_train, Y_train_5, cv=3)
from sklearn.metrics import confusion_matrix
confusion_matrix(Y_train_5, Y_train_pred)
from sklearn.metrics import precision_score, recall_score
print(precision_score(Y_train_5, Y_train_pred))
print(recall_score(Y_train_5, Y_train_pred))
svc = svm.SVC(kernel='linear').fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
n_correct = sum(Y_pred == Y_test)
print(n_correct / len(Y_pred))
cross_val_score(svc, X_train, Y_train, cv=3, scoring="accuracy")
Y_train_pred = cross_val_predict(svc, X_train_scaled, Y_train, cv=3)
confusion_matrix(Y_train, Y_train_pred)
# from sklearn.ensemble import RandomForestClassifier

# forest_clf = RandomForestClassifier(n_estimators = 100)
# cross_val_score(forest_clf, X_train, Y_train, cv=5)
# n_estimators_array = [10, 20, 50, 100, 200, 500, 700, 1000]
# cv_scores = []
# for n_estimators in n_estimators_array:
#     clf = forest_clf = RandomForestClassifier(n_estimators = n_estimators)
#     cv_scores.append(np.mean(cross_val_score(clf, X_train, Y_train, cv=5)))
    
# cv_scores
# PCA with two components
pca = PCA(n_components=2)
pca.fit(X_train)
transform = pca.transform(X_train)

plt.scatter(transform[:,0],transform[:,1], s=20, c = Y_train, cmap = "nipy_spectral", edgecolor = "None")
plt.colorbar()
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.show()
pca = PCA(n_components=2)
pca.fit(X_train)
print(pca.explained_variance_ratio_)
variances = []
for n_components in range(1,101,10):
    pca = PCA(n_components = n_components)
    pca.fit(X_train)
    variances.append(sum(pca.explained_variance_ratio_))
    
print(variances)
plt.plot(range(1,101, 10), variances)
# PCA with three components
from mpl_toolkits.mplot3d import Axes3D
pca = PCA(n_components=3)
pca.fit(X_train)
transform = pca.transform(X_train)

figure = plt.figure()
axes = Axes3D(figure)

axes.scatter(transform[:,0],transform[:,1], transform[:,2], s=20, c = Y_train, cmap = "nipy_spectral", edgecolor = "None")
# plt.colorbar()
# plt.xlabel("PCA1")
# plt.xlabel("PCA2")
# plt.show()
n_components_array=([1,2,3,4,5,10,20,50,100,200,500])
vr = np.zeros(len(n_components_array))
i=0;
for n_components in n_components_array:
    pca = PCA(n_components=n_components)
    pca.fit(train)
    vr[i] = sum(pca.explained_variance_ratio_)
    i=i+1  
plt.plot(n_components_array, vr)
plt.xlabel("number of PCA components",size=20)
plt.ylabel("variance ratio",size=20)
pca = PCA(n_components=50)
pca.fit(X_train)
transform_X_train = pca.transform(X_train)
transform_X_test = pca.transform(X_test)

clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(transform_X_train, Y_train)
results=clf.predict(transform_X_test)
print(results)
print(Y_test)
n_errors = np.count_nonzero(results - Y_test)
print(n_errors)
print(results.size)
print(n_errors/results.size*100)
clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(X_train, Y_train)
results=clf.predict(X_test)
clf = RandomForestClassifier(n_estimators = 100, n_jobs=1, criterion="gini")
clf.fit(train, target)
results=clf.predict(test)


