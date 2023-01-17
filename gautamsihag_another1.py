from sklearn.neural_network import MLPClassifier
#>>> clf.fit(X, y)
import pandas as pd
import numpy as np
X = [[0, 0], [10, 1], [8, 9], [11, 1], [12, 3]]
X = pd.DataFrame(X)
y1 = [0, 1, 3, 7, 9]
y = pd.DataFrame(y)
y = y.values.ravel()
clf = MLPClassifier(algorithm='l-bfgs', alpha=1e-5, hidden_layer_sizes=(15, 2), random_state=1)
clf.fit(X, y) 
clf.predict([[2., 2.], [-1., -2.]])
classes.classes_
test1 = [[2, 2],[4,7],[12,3]]
test1 = pd.DataFrame(test1)
y_pr = clf.predict_proba(test1)
y_pr[1,:]
#print(y_pr[0,:])
print(y_pr.argsort())
print(y_pr.argsort()[::-1])
print(y_pr.argsort()[:2])
y_pr[1].argsort() #.values[-3:]
y1 = [0, 1, 3, 7, 9]
y1 = pd.DataFrame(y1)
a = y1.values
a[y_pr.argsort()[::-1]]
import ml_metrics as metrics
def find_top_5(row):
    return list(row.nlargest(5).index)
np.array_str(a[y_pr.argsort()[::-1][:2]])[1:-1]
most_popular