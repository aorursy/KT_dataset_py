import numpy as np



f = open("../input/diabetes.csv")

f.readline()  # skip the header

data = np.loadtxt(f, delimiter = ',')

X = data[:, :-1]

y = data[:, -1]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
from sklearn.neural_network import MLPClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.gaussian_process.kernels import RBF

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis



names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",

         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",

         "Naive Bayes", "QDA"

        ]



classifiers = [

    KNeighborsClassifier(),

    SVC(kernel="linear"),

    SVC(kernel="rbf"),

    GaussianProcessClassifier(),

    DecisionTreeClassifier(),

    RandomForestClassifier(),

    MLPClassifier(),

    AdaBoostClassifier(),

    GaussianNB(),

    QuadraticDiscriminantAnalysis()

]
from sklearn.model_selection import cross_val_score



# iterate over classifiers

results = {}

for name, clf in zip(names, classifiers):

    scores = cross_val_score(clf, X_train, y_train, cv=5)

    results[name] = scores
for name, scores in results.items():

    print("%20s | Accuracy: %0.2f (+/- %0.2f)" % (name, scores.mean(), scores.std() * 2))
from sklearn.grid_search import GridSearchCV



clf = SVC(kernel="linear")



# prepare a range of values to test

param_grid = [

  {'C': [.01, .1, 1, 10], 'kernel': ['linear']},

 ]



grid = GridSearchCV(estimator=clf, param_grid=param_grid)

grid.fit(X_train, y_train)

print(grid)

# summarize the results of the grid search

print(grid.best_score_)

print(grid.best_estimator_.C)
clf = SVC(kernel="linear", C=0.1)

clf.fit(X_train, y_train)

y_eval = clf.predict(X_test)
sum(y_eval == y_test) / float(len(y_test))