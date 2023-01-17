from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
clf = MLPClassifier(random_state=1, max_iter=300)
# .fit(X_train, y_train)

X_train = [
    [0, 123, 1, 456, 2, 777],
    [0, 123, 1, 999, 2, 789],
    [0, 111, 1, 456, 2, 789],
    [0, 111, 1, 456, 2, 777],
    [0, 111, 1, 999, 2, 789]
]
y_train = [
    0,1,0,0,1
]
X_test = [
    [0, 123, 1, 456, 2, 777],
    [0, 111, 1, 999, 2, 789]
]
y_test  = [
    0, 1
]

# choices = ["APPLE","ORANGE","PEAR"]
# index = choices.index("PEAR")
# print(index)
# print(index / len(choices))

clf.fit(X_train, y_train)
y_test = clf.predict(X_test)
clf.score(X_test, y_test)

