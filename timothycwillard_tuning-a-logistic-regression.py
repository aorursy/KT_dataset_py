import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import RandomizedSearchCV
from scipy import stats
import time

sns.set(style = "darkgrid")
xsize = 18.0
ysize = 12.0

import os
print(os.listdir("../input"))
train_df = pd.read_json("../input/train.json").set_index("id")
train_df.head()
le = LabelEncoder()
y = le.fit_transform(train_df["cuisine"].values)
X = train_df["ingredients"].values

x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.33)
tfidf_vectorizer = TfidfVectorizer(analyzer = "word", preprocessor = lambda x: x, tokenizer = lambda x: x, token_pattern = None)
tfidf_train = tfidf_vectorizer.fit_transform(x_train)
tfidf_valid = tfidf_vectorizer.transform(x_valid)
tfidf_train
LogisticRegression().get_params().keys()
logreg = LogisticRegression(penalty = "L2", dual = False, tol = 1e-5, C = 1.0, fit_intercept = True, intercept_scaling = 1, 
                            class_weight = None, random_state = None, solver = "saga", max_iter = 250, multi_class = "ovr", 
                            verbose = 0, warm_start = False, n_jobs = 1)
logreg.fit(tfidf_train, y_train)
train_pred = logreg.predict(tfidf_train)
valid_pred = logreg.predict(tfidf_valid)
print("Train Accuracy: "+str(accuracy_score(y_train, train_pred)))
print("Valid Accuracy: "+str(accuracy_score(y_valid, valid_pred)))
print("Train F1 Score: "+str(f1_score(y_train, train_pred, average = "weighted")))
print("Valid F1 Score: "+str(f1_score(y_valid, valid_pred, average = "weighted")))
print("Train Precision: "+str(precision_score(y_train, train_pred, average = "weighted")))
print("Valid Precision: "+str(precision_score(y_valid, valid_pred, average = "weighted")))
print("Train Recall: "+str(recall_score(y_train, train_pred, average = "weighted")))
print("Valid Recall: "+str(recall_score(y_valid, valid_pred, average = "weighted")))
Cs = np.geomspace(1e-8, 1e8, 30)
penalties = ["L1", "L2"]

accuracy_train = {}
accuracy_valid = {}
accuracy_diff = {}
f1_train = {}
f1_valid = {}
f1_diff = {}
precision_train = {}
precision_valid = {}
precision_diff = {}
recall_train = {}
recall_valid = {}
recall_diff = {}

for i, penalty in enumerate(penalties):
    accuracy_train[penalty] = np.zeros(len(Cs))
    f1_train[penalty] = np.zeros(len(Cs))
    precision_train[penalty] = np.zeros(len(Cs))
    recall_train[penalty] = np.zeros(len(Cs))
    accuracy_valid[penalty] = np.zeros(len(Cs))
    f1_valid[penalty] = np.zeros(len(Cs))
    precision_valid[penalty] = np.zeros(len(Cs))
    recall_valid[penalty] = np.zeros(len(Cs))
    for j, C in enumerate(Cs):
        start_time = time.time()
        print("("+str((i*30)+(j+1))+") Starting penalty="+penalty+", C="+str(C))
        logreg = LogisticRegression(penalty = penalty, dual = False, tol = 1e-5, C = C, fit_intercept = True, intercept_scaling = 1, 
                                    class_weight = None, random_state = None, solver = "saga", max_iter = 250, multi_class = "ovr", 
                                    verbose = 0, warm_start = False, n_jobs = 1)
        logreg.fit(tfidf_train, y_train)
        train_pred = logreg.predict(tfidf_train)
        valid_pred = logreg.predict(tfidf_valid)
        accuracy_train[penalty][j] = accuracy_score(y_train, train_pred)
        accuracy_valid[penalty][j] = accuracy_score(y_valid, valid_pred)
        f1_train[penalty][j] = f1_score(y_train, train_pred, average = "weighted")
        f1_valid[penalty][j] = f1_score(y_valid, valid_pred, average = "weighted")
        precision_train[penalty][j] = precision_score(y_train, train_pred, average = "weighted")
        precision_valid[penalty][j] = precision_score(y_valid, valid_pred, average = "weighted")
        recall_train[penalty][j] = recall_score(y_train, train_pred, average = "weighted")
        recall_valid[penalty][j] = recall_score(y_valid, valid_pred, average = "weighted")
        end_time = time.time()
        print("("+str((i*30)+(j+1))+") Finished penalty="+penalty+", C="+str(C)+" in "+str(end_time - start_time)+"s")
    accuracy_diff[penalty] = np.absolute(accuracy_train[penalty] - accuracy_valid[penalty])
    f1_diff[penalty] = np.absolute(f1_train[penalty] - f1_valid[penalty])
    precision_diff[penalty] = np.absolute(precision_train[penalty] - precision_valid[penalty])
    recall_diff[penalty] = np.absolute(recall_train[penalty] - recall_valid[penalty])
fig, axes = plt.subplots(nrows = 4, ncols = 2)
fig.set_size_inches(2.0*xsize, 4.0*ysize)

axes = np.array(axes).flatten()

axes[0].semilogx(Cs, accuracy_train["L1"], "o:", label = "Train")
axes[0].semilogx(Cs, accuracy_valid["L1"], "o:", label = "Valid")
axes[0].semilogx(Cs, accuracy_diff["L1"], "o:", label = "Difference")
axes[0].set_title("Accuracy Metrics vs Inverse Regularization of Strength with L1 Penalty")
axes[0].set_ylabel("Accuracy Metrics")
axes[0].set_xlabel("Inverse Regularization of Strength")
axes[0].legend()

axes[1].semilogx(Cs, accuracy_train["L2"], "o:", label = "Train")
axes[1].semilogx(Cs, accuracy_valid["L2"], "o:", label = "Valid")
axes[1].semilogx(Cs, accuracy_diff["L2"], "o:", label = "Difference")
axes[1].set_title("Accuracy Metrics vs Inverse Regularization of Strength with L2 Penalty")
axes[1].set_ylabel("Accuracy Metrics")
axes[1].set_xlabel("Inverse Regularization of Strength")
axes[1].legend()

axes[2].semilogx(Cs, f1_train["L1"], "o:", label = "Train")
axes[2].semilogx(Cs, f1_valid["L1"], "o:", label = "Valid")
axes[2].semilogx(Cs, f1_diff["L1"], "o:", label = "Difference")
axes[2].set_title("F1 Metrics vs Inverse Regularization of Strength with L1 Penalty")
axes[2].set_ylabel("F1 Metrics")
axes[2].set_xlabel("Inverse Regularization of Strength")
axes[2].legend()

axes[3].semilogx(Cs, f1_train["L2"], "o:", label = "Train")
axes[3].semilogx(Cs, f1_valid["L2"], "o:", label = "Valid")
axes[3].semilogx(Cs, f1_diff["L2"], "o:", label = "Difference")
axes[3].set_title("F1 Metrics vs Inverse Regularization of Strength with L2 Penalty")
axes[3].set_ylabel("F1 Metrics")
axes[3].set_xlabel("Inverse Regularization of Strength")
axes[3].legend()

axes[4].semilogx(Cs, precision_train["L1"], "o:", label = "Train")
axes[4].semilogx(Cs, precision_valid["L1"], "o:", label = "Valid")
axes[4].semilogx(Cs, precision_diff["L1"], "o:", label = "Difference")
axes[4].set_title("Precision Metrics vs Inverse Regularization of Strength with L1 Penalty")
axes[4].set_ylabel("Precision Metrics")
axes[4].set_xlabel("Inverse Regularization of Strength")
axes[4].legend()

axes[5].semilogx(Cs, precision_train["L2"], "o:", label = "Train")
axes[5].semilogx(Cs, precision_valid["L2"], "o:", label = "Valid")
axes[5].semilogx(Cs, precision_diff["L2"], "o:", label = "Difference")
axes[5].set_title("Precision Metrics vs Inverse Regularization of Strength with L2 Penalty")
axes[5].set_ylabel("Precision Metrics")
axes[5].set_xlabel("Inverse Regularization of Strength")
axes[5].legend()

axes[6].semilogx(Cs, recall_train["L1"], "o:", label = "Train")
axes[6].semilogx(Cs, recall_valid["L1"], "o:", label = "Valid")
axes[6].semilogx(Cs, recall_diff["L1"], "o:", label = "Difference")
axes[6].set_title("Recall Metrics vs Inverse Regularization of Strength with L1 Penalty")
axes[6].set_ylabel("Recall Metrics")
axes[6].set_xlabel("Inverse Regularization of Strength")
axes[6].legend()

axes[7].semilogx(Cs, recall_train["L2"], "o:", label = "Train")
axes[7].semilogx(Cs, recall_valid["L2"], "o:", label = "Valid")
axes[7].semilogx(Cs, recall_diff["L2"], "o:", label = "Difference")
axes[7].set_title("Recall Metrics vs Inverse Regularization of Strength with L2 Penalty")
axes[7].set_ylabel("Recall Metrics")
axes[7].set_xlabel("Inverse Regularization of Strength")
axes[7].legend()

plt.show()
xx = np.argmax(accuracy_valid["L1"])
yx = np.argmax(accuracy_valid["L2"])
print("Max L1 validation accuracy is at C="+str(Cs[xx]))
print("Train Accuracy : "+str(accuracy_train["L1"][xx]))
print("Valid Accuracy : "+str(accuracy_valid["L1"][xx]))
print("Difference Accuracy : "+str(accuracy_diff["L1"][xx]))
print("Max L2 validation accuracy is at C="+str(Cs[yx]))
print("Train Accuracy : "+str(accuracy_train["L2"][yx]))
print("Valid Accuracy : "+str(accuracy_valid["L2"][yx]))
print("Difference Accuracy : "+str(accuracy_diff["L2"][yx]))
logreg = LogisticRegression(dual = False, tol = 1e-5, fit_intercept = True, intercept_scaling = 1, class_weight = None, 
                            random_state = None, solver = "saga", max_iter = 1000, multi_class = "ovr", 
                            verbose = 0, warm_start = False, n_jobs = 1)
params = {
    "penalty": ["L1", "L2"],
    "C": stats.lognorm(0.75, scale = Cs[xx])
}
clf = RandomizedSearchCV(logreg, params, scoring = "accuracy", cv = 5)
clf.fit(tfidf_train, y_train)
print(clf.best_params_)
print(clf.best_params_)
C_best = clf.best_params_["C"]
penalty_best = clf.best_params_["penalty"]

tfidf_vectorizer = TfidfVectorizer(analyzer = "word", preprocessor = lambda x: x, tokenizer = lambda x: x, token_pattern = None)
tfidf = tfidf_vectorizer.fit_transform(train_df["ingredients"])

le = LabelEncoder()
y = le.fit_transform(train_df["cuisine"])

logreg = LogisticRegression(dual = False, tol = 1e-5, fit_intercept = True, intercept_scaling = 1, class_weight = None, 
                            random_state = None, solver = "saga", max_iter = 100000, multi_class = "ovr", 
                            verbose = 0, warm_start = False, n_jobs = 1, C = C_best, penalty = penalty_best)
logreg.fit(tfidf, y)
pred = logreg.predict(tfidf)
print("Accuracy: "+str(accuracy_score(y, pred)))
test_df = pd.read_json("../input/test.json").set_index("id")
test_tfidf = tfidf_vectorizer.transform(test_df["ingredients"])
pred_test = logreg.predict(test_tfidf)
test_df["cuisine"] = le.inverse_transform(pred_test)
test_df.drop(columns = ["ingredients"], inplace = True)
test_df.to_csv("tuned_logistic_regression_submission.csv")