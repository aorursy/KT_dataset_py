import numpy as np

X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])

Y = np.array([1, 1, 1, 2, 2, 2])



from sklearn.naive_bayes import GaussianNB

clf = GaussianNB()

clf.fit(X, Y)

print(clf.predict([[-0.8, -1]]))



clf_pf = GaussianNB()

clf_pf.partial_fit(X, Y, np.unique(Y))

print(clf_pf.predict([[-0.8, -1]]))



X = np.random.randint(5, size=(6, 100))

y = np.array([1, 2, 3, 4, 5, 6])



from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB()

clf.fit(X, y)

print(clf.predict(X[2:3]))
from sklearn.ensemble import RandomForestClassifier

from sklearn.datasets import make_classification



# make_classification : Generate a random n-class classification problem

X, y = make_classification(n_samples=1000, n_features=4, n_informative=2, n_redundant=0, random_state=0, shuffle=False)

clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)

clf.fit(X, y)  

print(clf.feature_importances_)

print(clf.predict([[0, 0, 0, 0]]))