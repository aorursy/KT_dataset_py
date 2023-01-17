import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeClassifier, RidgeClassifierCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
from scipy import stats
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from decimal import Decimal as D

sns.set(style = "darkgrid")
xsize = 18.0
ysize = 12.0

import os
print(os.listdir("../input"))
train_df = pd.read_json("../input/train.json").set_index("id")
train_df.head()
le = LabelEncoder()
train_df["cuisine"] = le.fit_transform(train_df["cuisine"])

x_train, x_valid, y_train, y_valid = train_test_split(train_df["ingredients"], 
                                                      train_df["cuisine"], test_size = 0.33)
tfidf_vectorizer = TfidfVectorizer(analyzer = "word", tokenizer = lambda x: x,
                                  preprocessor = lambda x: x, token_pattern = None)
tfidf_train = tfidf_vectorizer.fit_transform(x_train)
tfidf_valid = tfidf_vectorizer.transform(x_valid)
ridge_clf = RidgeClassifier(alpha = 1.0, fit_intercept = True, normalize = False, 
                            copy_X = True, max_iter = None, tol = 0.001, class_weight = None, 
                            solver = "auto", random_state = None)
ridge_clf.fit(tfidf_train, y_train)
pred_train = ridge_clf.predict(tfidf_train)
pred_valid = ridge_clf.predict(tfidf_valid)
print("Train Accuracy: "+str(accuracy_score(y_train, pred_train)))
print("Valid Accuracy: "+str(accuracy_score(y_valid, pred_valid)))
alphas_array = np.geomspace(1e-6, 1e6, 50)
normalize_array = [False, True]

train_accuracies = {}
valid_accuracies = {}
diff_accuracies = {}

for i, normalize in enumerate(normalize_array):
    start_time = time.time()
    print("("+str(i+1)+") Starting normalize="+str(normalize))

    train_accuracies[str(normalize)] = np.zeros(len(alphas_array))
    valid_accuracies[str(normalize)] = np.zeros(len(alphas_array))

    for k, alpha in enumerate(alphas_array):
        start_time2 = time.time()
        ridge_clf = RidgeClassifier(alpha = alpha, fit_intercept = True, normalize = normalize, 
                                    copy_X = True, max_iter = None, tol = 0.001, class_weight = None, 
                                    solver = "sag", random_state = None)
        ridge_clf.fit(tfidf_train, y_train)
        pred_train = ridge_clf.predict(tfidf_train)
        pred_valid = ridge_clf.predict(tfidf_valid)
        train_accuracies[str(normalize)][k] = accuracy_score(y_train, pred_train)
        valid_accuracies[str(normalize)][k] = accuracy_score(y_valid, pred_valid)
        end_time2 = time.time()
        time_diff2 = end_time2 - start_time2
        print("("+str(i+1)+"."+str(k+1)+") Finished alpha="+str("%.2E"%D(alpha))+" in "+str("%.2f"%time_diff2)+"s")

    diff_accuracies[str(normalize)] = np.absolute(train_accuracies[str(normalize)] - valid_accuracies[str(normalize)])

    end_time = time.time()
    time_diff = end_time - start_time
    print("("+str(i+1)+") Finished normalize="+str("%.2f"%normalize)+" in "+str("%.2f"%time_diff)+"s")
fig, axes = plt.subplots(nrows = len(normalize_array))
fig.set_size_inches(xsize, len(normalize_array)*ysize)

axes = np.array(axes).flatten()

for i, normalize in enumerate(normalize_array):
    axes[i].semilogx(alphas_array, train_accuracies[str(normalize)], "o:", label = "Train")
    axes[i].semilogx(alphas_array, train_accuracies[str(normalize)], "o:", label = "Valid")
    axes[i].semilogx(alphas_array, diff_accuracies[str(normalize)], "o:", label = "Difference")
    axes[i].set_title("Accuracy Metrics vs Regularization Strength for normalization="+str(normalize))
    axes[i].set_xlabel("Regularization Strength")
    axes[i].set_ylabel("Accuracy Metrics")
    axes[i].legend()

plt.show()
false_xx = np.argmax(valid_accuracies[str(False)])
true_xx = np.argmax(valid_accuracies[str(True)])

alpha_best_false = alphas_array[false_xx]
train_best_false = train_accuracies[str(False)][false_xx]
valid_best_false = valid_accuracies[str(False)][false_xx]

alpha_best_true = alphas_array[true_xx]
train_best_true = train_accuracies[str(True)][true_xx]
valid_best_true = valid_accuracies[str(True)][true_xx]

print("For normalize="+str(False)+" alpha="+str(alpha_best_false))
print("Train Accuracy: "+str(train_best_false))
print("Valid Accuracy: "+str(valid_best_false))
print("\n")
print("For normalize="+str(True)+" alpha="+str(alpha_best_true))
print("Train Accuracy: "+str(train_best_true))
print("Valid Accuracy: "+str(valid_best_true))
ridge_clf_normalize_false = RidgeClassifier(fit_intercept = True, normalize = False, copy_X = True, 
                                            max_iter = None, tol = 0.001, class_weight = None, 
                                            solver = "auto", random_state = None)
params = {
    "alpha": stats.lognorm(0.8, scale = alpha_best_false)
}
false_clf = RandomizedSearchCV(ridge_clf_normalize_false, params, scoring = "accuracy", cv = 5)
false_clf.fit(tfidf_train, y_train)
print(false_clf.best_params_)

ridge_clf_normalize_true = RidgeClassifier(fit_intercept = True, normalize = True, copy_X = True, 
                                            max_iter = None, tol = 0.001, class_weight = None, 
                                            solver = "auto", random_state = None)
params = {
    "alpha": stats.lognorm(0.8, scale = alpha_best_true)
}
true_clf = RandomizedSearchCV(ridge_clf_normalize_true, params, scoring = "accuracy", cv = 5)
true_clf.fit(tfidf_train, y_train)
print(true_clf.best_params_)
print("For normalize="+str(False)+" alpha="+str(false_clf.best_params_["alpha"]))
print("Train Accuracy: "+str(false_clf.score(tfidf_train, y_train)))
print("Valid Accuracy: "+str(false_clf.score(tfidf_valid, y_valid)))
print("\n")
print("For normalize="+str(True)+" alpha="+str(true_clf.best_params_["alpha"]))
print("Train Accuracy: "+str(true_clf.score(tfidf_train, y_train)))
print("Valid Accuracy: "+str(true_clf.score(tfidf_valid, y_valid)))
norm = True if true_clf.score(tfidf_valid, y_valid) > false_clf.score(tfidf_valid, y_valid) else False
alpha = true_clf.best_params_["alpha"] if true_clf.score(tfidf_valid, y_valid) > false_clf.score(tfidf_valid, y_valid) else false_clf.best_params_["alpha"]
tfidf_vectorizer = TfidfVectorizer(analyzer = "word", tokenizer = lambda x: x,
                                  preprocessor = lambda x: x, token_pattern = None)
tfidf_train = tfidf_vectorizer.fit_transform(train_df["ingredients"])
y_train = train_df["cuisine"]

ridge_clf = RidgeClassifier(alpha = 1.0, fit_intercept = True, normalize = norm, 
                            copy_X = True, max_iter = None, tol = 0.001, class_weight = None, 
                            solver = "auto", random_state = None)
ridge_clf.fit(tfidf_train, y_train)

test_df = pd.read_json("../input/test.json").set_index("id")
tfidf_test = tfidf_vectorizer.transform(test_df["ingredients"])
test_df["cuisine"] = ridge_clf.predict(tfidf_test)
test_df["cuisine"] = le.inverse_transform(test_df["cuisine"])
test_df.drop(columns = ["ingredients"], inplace = True)
test_df.to_csv("optimized_ridge_classifier_submission.csv")
