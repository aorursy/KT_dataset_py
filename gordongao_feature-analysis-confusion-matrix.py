import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import warnings

import seaborn as sns

import matplotlib.pyplot as plt



warnings.filterwarnings("ignore")

%matplotlib inline

sns.set(style="white", color_codes=True)



iris = pd.read_csv('../input/Iris.csv')

iris.drop('Id', axis=1, inplace=True)

iris.head()
print(iris["Species"].value_counts())

print()

print(iris.info())

print()

print(iris.describe())
sns.pairplot(iris, hue="Species", size=3, diag_kind="kde")
def _split_iris_dataset(iris):

    from sklearn.model_selection import train_test_split

    from sklearn.preprocessing import StandardScaler

    

    y = iris['Species']

    X = iris.drop('Species', axis=1)



    scaler = StandardScaler()

    X_trans = scaler.fit_transform(X)



    X_train, X_test, y_train, y_test = train_test_split(X_trans, y, test_size=0.33, random_state=42)

    return (X_train, X_test, y_train, y_test)





def _evaluate_iris_classifier(feature, clf, dataset):

    X_train, X_test, y_train, y_test = _split_iris_dataset(dataset)

    

    from sklearn.model_selection import cross_val_predict

    from sklearn.metrics import confusion_matrix

    from sklearn.metrics import precision_score, recall_score

    from sklearn.metrics import f1_score



    y_train_feature = (y_train == feature)

    y_test_feature = (y_test == feature)



    predict = cross_val_predict(clf, X_train, y_train_feature, cv=3)

    print("confusion matrix on", feature)

    print(confusion_matrix(y_train_feature, predict))

    print("percision:", precision_score(y_train_feature, predict))

    print("recall:", recall_score(y_train_feature, predict))

    print("f1 score:", f1_score(y_train_feature, predict))

    
from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=42)



print("---------- Original iris ----------")

_evaluate_iris_classifier('Iris-versicolor', sgd_clf, iris)

print()

print("---------- Enhanced iris ----------")

iris_sgd = iris.copy()

iris_sgd['SepalWidth_PetalLength'] = iris_sgd['SepalWidthCm'] / iris_sgd['PetalLengthCm']

iris_sgd['SepalWidth_PetalWidth'] = iris_sgd['SepalWidthCm'] / iris_sgd['PetalWidthCm']

_evaluate_iris_classifier('Iris-versicolor', sgd_clf, iris_sgd)
from sklearn.ensemble import ExtraTreesClassifier

eXT_clf = ExtraTreesClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)



print("---------- Original iris ----------")

_evaluate_iris_classifier('Iris-versicolor', eXT_clf, iris)

print()

print("---------- Enhanced iris ----------")

iris_eXT = iris.copy()

iris_eXT['SepalWidth_PetalLength'] = iris_eXT['SepalWidthCm'] / iris_eXT['PetalLengthCm']

iris_eXT['SepalWidth_PetalWidth'] = iris_eXT['SepalWidthCm'] / iris_eXT['PetalWidthCm']

iris_eXT.drop('SepalWidthCm', axis=1, inplace=True)

_evaluate_iris_classifier('Iris-versicolor', eXT_clf, iris_eXT)
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import ExtraTreesClassifier





iris = pd.read_csv('../input/Iris.csv')

iris.drop('Id', axis=1, inplace=True)



y = iris['Species']

X = iris.drop('Species', axis=1)



scaler = StandardScaler()

X_trans = scaler.fit_transform(X)



X_train, X_test, y_train, y_test = train_test_split(X_trans, y, test_size=0.30, random_state=42)



eXT_clf = ExtraTreesClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)

eXT_clf.fit(X_train, y_train)

print('The accuracy is {:.2f} out of 1 on training data'.format(eXT_clf.score(X_train, y_train)))

print('The accuracy is {:.2f} out of 1 on test data'.format(eXT_clf.score(X_test, y_test)))
importances = eXT_clf.feature_importances_

std = np.std([tree.feature_importances_ for tree in eXT_clf.estimators_],

             axis=0)

indices = np.argsort(importances)[::-1]

feature_names = list(iris)



# Print the feature ranking

print("Feature ranking:")



for f in range(X.shape[1]):

    print("%d. feature %s (%f)" % (f + 1, feature_names[indices[f]], importances[indices[f]]))



# Plot the feature importances of the forest

plt.figure()

plt.title("Feature importances")

plt.bar(range(X.shape[1]), importances[indices],

       color="r", yerr=std[indices], align="center")

plt.xticks(range(X.shape[1]), indices)

plt.xlim([-1, X.shape[1]])

plt.show()