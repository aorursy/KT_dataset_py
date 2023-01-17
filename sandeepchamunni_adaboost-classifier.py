from sklearn.datasets import load_iris

from sklearn import ensemble

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

from sklearn import tree
X, y = load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
stump = tree.DecisionTreeClassifier(max_depth=1)
clf = ensemble.AdaBoostClassifier(base_estimator = stump, algorithm="SAMME", n_estimators=6, random_state=0)

clf = clf.fit(X_train, y_train)
y_test_predicted = clf.predict(X_test)
print(confusion_matrix(y_test, y_test_predicted))

accuracy_score(y_test, y_test_predicted)