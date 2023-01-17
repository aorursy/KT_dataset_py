from sklearn.ensemble import RandomForestClassifier



X = [[0, 0], [1, 0], [0, 1], [1, 1]]

y = [0, 0, 1, 1]



clf = RandomForestClassifier()

clf.fit(X, y)



clf.predict([[0.5, 0.01]])