# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import datasets

from sklearn.preprocessing import StandardScaler

import iris_helper as H



import warnings

warnings.filterwarnings("ignore")



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
iris = pd.read_csv("../input/iris/Iris.csv")

print(iris.columns)

print(iris.describe())
print(iris.head(1))
iris.drop("Id", axis=1, inplace=True)

iris.head(1)
X = iris.iloc[:,:4]

labels = iris.iloc[:,4].unique()

species = dict()

label = 0

for i in labels:

    species[i] = label

    label+=1

y = iris.iloc[:,4].map(species)
X_train, X_test, y_train, y_test = H.create_test_set(X, y)

X_train, X_dev, y_train, y_dev = H.create_dev_set(X_train, y_train)
sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_dev = sc.transform(X_dev)

X_test = sc.transform(X_test)
# run initial versions on training data



results=list()

columns = ["CVS_Mean", "CVS_Std (x1e-4)", "Time (ms)"]

index = ["Decision Tree", "Gaussian NB", "KNN", "Logistic Reg", "MLP", "Random Forest", "SVM"]

clf, _ = H.build_decision_tree()

results.append(H.initial_run(clf, X_train, y_train))

clf, _ = H.build_gnb()

results.append(H.initial_run(clf, X_train, y_train))

clf, _ = H.build_knn()

results.append(H.initial_run(clf, X_train, y_train))

clf, _ = H.build_log_reg()

results.append(H.initial_run(clf, X_train, y_train))

clf, _ = H.build_mlp()

results.append(H.initial_run(clf, X_train, y_train))

clf, _ = H.build_random_forest()

results.append(H.initial_run(clf, X_train, y_train))

clf, _ = H.build_svm()

results.append(H.initial_run(clf, X_train, y_train))

initial_run = pd.DataFrame(results, columns=columns, index=index)

print(initial_run)
## We run GridSearch on them to fine tune them



best_decision_tree = H.find_best_decision_tree(X_train, y_train)

best_svm = H.find_best_svm(X_train,y_train)

best_mlp = H.find_best_mlp(X_train,y_train)

best_gnb = H.find_best_gnb(X_train,y_train)
#

### Now we see the performance of our fine tuned models on training and CV sets

#

results=list()

del results

results=list()

columns = ["CVS_Mean", "CVS_Std (x1e-4)", "Time (ms)"]

index = ["Decision Tree","SVM", "MLP", "GNB"]

results.append(H.best_cvs(best_decision_tree, X_train, y_train))

results.append(H.best_cvs(best_svm, X_train, y_train))

results.append(H.best_cvs(best_mlp, X_train, y_train))

results.append(H.best_cvs(best_gnb, X_train, y_train))

best_cvs = pd.DataFrame(results, columns=columns, index=index)

print(best_cvs)
#

### We find the accuracy of our models on our dev set

#

columns = ["Accuracy", "Time (us)"]

index = ["SVM", "MLP", "GNB"]

del results

results = list()

results.append(H.find_accuracy(best_svm, X_dev, y_dev))

results.append(H.find_accuracy(best_mlp, X_dev, y_dev))

results.append(H.find_accuracy(best_gnb, X_dev, y_dev))

accuracy = pd.DataFrame(results, columns=columns, index=index)

print(accuracy)
#

### We find the accuracy of our models on our test set

#

columns = ["Accuracy", "Time (us)"]

index = ["SVM", "MLP", "GNB"]

del results

results = list()

results.append(H.find_accuracy(best_svm, X_test, y_test))

results.append(H.find_accuracy(best_mlp, X_test, y_test))

results.append(H.find_accuracy(best_gnb, X_test, y_test))

accuracy = pd.DataFrame(results, columns=columns, index=index)

print(accuracy)