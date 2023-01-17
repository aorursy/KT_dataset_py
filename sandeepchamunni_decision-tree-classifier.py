from sklearn.datasets import load_iris

from sklearn import tree

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split
X, y = load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
clf = tree.DecisionTreeClassifier(random_state=0, max_depth=2)

clf = clf.fit(X_train, y_train)

tree.plot_tree(clf) 
y_test_predicted = clf.predict(X_test)
print(confusion_matrix(y_test, y_test_predicted))

accuracy_score(y_test, y_test_predicted)