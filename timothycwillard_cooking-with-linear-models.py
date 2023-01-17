import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier, RidgeClassifierCV
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

sns.set(style = "darkgrid")
xsize = 12.0
ysize = 8.0

import os
print(os.listdir("../input"))
train_df = pd.read_json("../input/train.json").set_index("id")
print(train_df.isnull().sum(axis = 0))
train_df.head()
X = train_df.drop(columns = ["cuisine"])
y = train_df["cuisine"]

x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size=0.66)
tfidf_vectorizer = TfidfVectorizer(analyzer='word', tokenizer = lambda x: x, 
                                   preprocessor = lambda x: x, token_pattern = None)
tfidf_train = tfidf_vectorizer.fit_transform(x_train["ingredients"], y_train)
tfidf_valid = tfidf_vectorizer.transform(x_valid["ingredients"])
tfidf_train
ridge_clf = RidgeClassifier()
ridge_clf.fit(tfidf_train, y_train)
pred_train = ridge_clf.predict(tfidf_train)
pred_valid = ridge_clf.predict(tfidf_valid)
print("Train Accuracy: "+str(accuracy_score(y_train, pred_train)))
print("Valid Accuracy: "+str(accuracy_score(y_valid, pred_valid)))
alphas = np.geomspace(1e-3, 1e3, 50)
train_accuracy = np.zeros(len(alphas))
valid_accuracy = np.zeros(len(alphas))

for i, alpha in enumerate(alphas):
    ridge_clf = RidgeClassifier(alpha = alpha)
    ridge_clf.fit(tfidf_train, y_train)
    pred_train = ridge_clf.predict(tfidf_train)
    pred_valid = ridge_clf.predict(tfidf_valid)
    train_accuracy[i] = accuracy_score(y_train, pred_train)
    valid_accuracy[i] = accuracy_score(y_valid, pred_valid)

train_valid_diff = np.absolute(train_accuracy - valid_accuracy)
fig, ax = plt.subplots()
fig.set_size_inches(xsize, ysize)

ax.semilogx(alphas, train_accuracy, "o:", label = "Train")
ax.semilogx(alphas, valid_accuracy, "o:", label = "Valid")
ax.semilogx(alphas, train_valid_diff, "o:", label = "Difference")
ax.set_title("Various Accuracy Metrics vs Regularization of Strength on a Log Scale")
ax.set_xlabel("Regularization of Strength")
ax.set_ylabel("Accuracy Metrics")
ax.legend()

plt.show()
ridge_clfcv = RidgeClassifierCV(alphas = alphas, cv = 5)
ridge_clfcv.fit(tfidf_train, y_train)
pred_train = ridge_clfcv.predict(tfidf_train)
pred_valid = ridge_clfcv.predict(tfidf_valid)
print("Train Accuracy: "+str(accuracy_score(y_train, pred_train)))
print("Valid Accuracy: "+str(accuracy_score(y_valid, pred_valid)))
print("Estimated Alpha: "+str(ridge_clfcv.alpha_))
logreg = LogisticRegression(solver = "newton-cg")
logreg.fit(tfidf_train, y_train)
pred_train = logreg.predict(tfidf_train)
pred_valid = logreg.predict(tfidf_valid)
print("Train Accuracy: "+str(accuracy_score(y_train, pred_train)))
print("Valid Accuracy: "+str(accuracy_score(y_valid, pred_valid)))
Cs = np.geomspace(1e-3, 1e3, 50)
train_accuracy = np.zeros(len(alphas))
valid_accuracy = np.zeros(len(alphas))

for i, C in enumerate(Cs):
    logreg = LogisticRegression(C = C, solver = "newton-cg")
    logreg.fit(tfidf_train, y_train)
    pred_train = logreg.predict(tfidf_train)
    pred_valid = logreg.predict(tfidf_valid)
    train_accuracy[i] = accuracy_score(y_train, pred_train)
    valid_accuracy[i] = accuracy_score(y_valid, pred_valid)

train_valid_diff = np.absolute(train_accuracy - valid_accuracy)
fig, ax = plt.subplots()
fig.set_size_inches(xsize, ysize)

ax.semilogx(alphas, train_accuracy, "o:", label = "Train")
ax.semilogx(alphas, valid_accuracy, "o:", label = "Valid")
ax.semilogx(alphas, train_valid_diff, "o:", label = "Difference")
ax.set_title("Various Accuracy Metrics vs Inverse Regularization of Strength on a Log Scale")
ax.set_xlabel("Inverse Regularization of Strength")
ax.set_ylabel("Accuracy Metrics")
ax.legend()

plt.show()
logregcv = LogisticRegressionCV(Cs = Cs, cv = 5, solver = "newton-cg", refit = False)
logregcv.fit(tfidf_train, y_train)
pred_train = logregcv.predict(tfidf_train)
pred_valid = logregcv.predict(tfidf_valid)
print("Train Accuracy: "+str(accuracy_score(y_train, pred_train)))
print("Valid Accuracy: "+str(accuracy_score(y_valid, pred_valid)))
print("Estimated C: "+str(logregcv.C_))
sgdc = SGDClassifier(loss = "hinge", penalty = "L2")
sgdc.fit(tfidf_train, y_train)
pred_train = sgdc.predict(tfidf_train)
pred_valid = sgdc.predict(tfidf_valid)
print("Train Accuracy: "+str(accuracy_score(y_train, pred_train)))
print("Valid Accuracy: "+str(accuracy_score(y_valid, pred_valid)))
alphas = np.geomspace(1e-6, 1e6, 50)
penalties = ["L1", "L2", "elasticnet"]
losses = ["hinge", "log", "modified_huber", "squared_hinge", "perceptron"]
train_accuracies = {}
valid_accuracies = {}
train_valid_diffs = {}

for i, penalty in enumerate(penalties):
    for j, loss in enumerate(losses):
        train_accuracies[penalty+"_"+loss] = np.zeros(len(alphas))
        valid_accuracies[penalty+"_"+loss] = np.zeros(len(alphas))
        for k, alpha in enumerate(alphas):
            sgdc = SGDClassifier(loss = loss, penalty = penalty, alpha = alpha)
            sgdc.fit(tfidf_train, y_train)
            pred_train = sgdc.predict(tfidf_train)
            pred_valid = sgdc.predict(tfidf_valid)
            train_accuracies[penalty+"_"+loss][k] = accuracy_score(y_train, pred_train)
            valid_accuracies[penalty+"_"+loss][k] = accuracy_score(y_valid, pred_valid)
        train_valid_diffs[penalty+"_"+loss] = np.absolute(train_accuracies[penalty+"_"+loss] - valid_accuracies[penalty+"_"+loss])
fig, axes = plt.subplots(ncols = len(penalties), nrows = len(losses))
fig.set_size_inches(len(penalties)*xsize, len(losses)*ysize)

for i, penalty in enumerate(penalties):
    for j, loss in enumerate(losses):
        ax = axes[j][i]
        ax.semilogx(alphas, train_accuracies[penalty+"_"+loss], "o:", label = "Train")
        ax.semilogx(alphas, valid_accuracies[penalty+"_"+loss], "o:", label = "Valid")
        ax.semilogx(alphas, train_valid_diffs[penalty+"_"+loss], "o:", label = "Difference")
        ax.set_title("Various Accuracy Metrics vs Regularization Strength on a Log Scale with "+penalty+" Penalty and "+loss+" Loss")
        ax.set_ylabel("Accuracy Metrics")
        ax.set_xlabel("Regularization Strength")
        ix = np.argmax(valid_accuracies[penalty+"_"+loss])
        valid_accuracy_max = valid_accuracies[penalty+"_"+loss][ix]
        alpha_max = alphas[ix]
        ax.annotate("Valid Max of "+str(valid_accuracy_max)+"\n"+" at "+r"$\alpha=$"+str(alpha_max), xy = (1e-1, 0.8))
        ax.legend()

plt.show()
sgdc_optimal_alpha = alphas[np.argmax(valid_accuracies["L2_modified_huber"])]
sgdc = SGDClassifier(penalty = "L2", loss = "modified_huber", alpha = sgdc_optimal_alpha)
sgdc.fit(tfidf_train, y_train)
pred_train = sgdc.predict(tfidf_train)
pred_valid = sgdc.predict(tfidf_valid)
print("Train Accuracy: "+str(accuracy_score(y_train, pred_train)))
print("Valid Accuracy: "+str(accuracy_score(y_valid, pred_valid)))
le = LabelEncoder()
le.fit(np.unique(y_train))
def voting_ensemble(X):
    estimators = [sgdc, logregcv, ridge_clfcv]
    Y = np.zeros([X.shape[0], len(estimators)], dtype = int)
    for i, clf in enumerate(estimators):
        Y[:, i] = le.transform(clf.predict(X))
    y = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        y[i] = np.argmax(np.bincount(Y[i,:]))
    return le.inverse_transform(y.astype(int))
pred_train = voting_ensemble(tfidf_train)
pred_valid = voting_ensemble(tfidf_valid)
print("Train Accuracy: "+str(accuracy_score(y_train, pred_train)))
print("Valid Accuracy: "+str(accuracy_score(y_valid, pred_valid)))
test_df = pd.read_json("../input/test.json")
tfidf_test = tfidf_vectorizer.transform(test_df["ingredients"])
test_df["cuisine"] = voting_ensemble(tfidf_test)
test_df = test_df.drop(columns = ["ingredients"])
test_df.head()
test_df.to_csv("linearmodels_submission.csv", header = ["id", "cuisine"], index = False)
