import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline 



from sklearn.metrics import log_loss

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.datasets import make_hastie_10_2

from sklearn.model_selection import train_test_split



def sigmoid(x):

    return 1 / (1 + np.exp(-x))
X_all = np.random.randn(5000, 1)

y_all = (X_all[:, 0] > 0)*2 - 1



X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.5, random_state=42)
clf = DecisionTreeClassifier(max_depth=1)

clf.fit(X_train, y_train)



print ('Accuracy for a single decision stump: {}'.format(clf.score(X_test, y_test)))
# For convenience we will use sklearn's GBM, the situation will be similar with XGBoost and others

clf = GradientBoostingClassifier(n_estimators=5000, learning_rate=0.01, max_depth=3, random_state=0)

clf.fit(X_train, y_train)



y_pred = clf.predict_proba(X_test)[:, 1]

print("Test logloss: {}".format(log_loss(y_test, y_pred)))
def compute_loss(y_true, scores_pred):

    '''

        Since we use raw scores we will wrap log_loss 

        and apply sigmoid to our predictions before computing log_loss itself

    '''

    return log_loss(y_true, sigmoid(scores_pred))

    



'''

    Get cummulative sum of *decision function* for trees. i-th element is a sum of trees 0...i-1.

    We cannot use staged_predict_proba, since we want to maniputate raw scores

    (not probabilities). And only in the end convert the scores to probabilities using sigmoid

'''

cum_preds = np.array([x for x in clf.staged_decision_function(X_test)])[:, :, 0] 



print ("Logloss using all trees:           {}".format(compute_loss(y_test, cum_preds[-1, :])))

print ("Logloss using all trees but last:  {}".format(compute_loss(y_test, cum_preds[-2, :])))

print ("Logloss using all trees but first: {}".format(compute_loss(y_test, cum_preds[-1, :] - cum_preds[0, :])))
# Pick an object of class 1 for visualisation

plt.plot(cum_preds[:, y_test == 1][:, 0])



plt.xlabel('n_trees')

plt.ylabel('Cumulative decision score');
clf = GradientBoostingClassifier(n_estimators=5000, learning_rate=8, max_depth=3, random_state=0)

clf.fit(X_train, y_train)



y_pred = clf.predict_proba(X_test)[:, 1]

print("Test logloss: {}".format(log_loss(y_test, y_pred)))
cum_preds = np.array([x for x in clf.staged_decision_function(X_test)])[:, :, 0] 



print ("Logloss using all trees:           {}".format(compute_loss(y_test, cum_preds[-1, :])))

print ("Logloss using all trees but last:  {}".format(compute_loss(y_test, cum_preds[-2, :])))

print ("Logloss using all trees but first: {}".format(compute_loss(y_test, cum_preds[-1, :] - cum_preds[0, :])))