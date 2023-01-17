# Load dataset

import pandas as pd

creditcard = pd.read_csv("../input/creditcardfraud/creditcard.csv")
# Understand the data

print(creditcard["Class"].value_counts())

creditcard.head()
# Split train/test using StratifiedShuffleSplit

from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)



for train_index, test_index in split.split(creditcard, creditcard["Class"]):

  strat_train_set = creditcard.loc[train_index]

  strat_test_set = creditcard.loc[test_index]



strat_test_set["Class"].value_counts() / len(strat_test_set)
# Separate labels from data

creditcard_train = strat_train_set.drop("Class", axis=1)

creditcard_train_labels = strat_train_set["Class"].copy()
# Train a RandomForestClassifier model and perform cross-validation (might take a while)

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_predict



forest_clf = RandomForestClassifier(random_state=42)



# See how it performs in the test set

creditcard_test = strat_test_set.drop("Class", axis=1)

creditcard_test_labels = strat_test_set["Class"].copy()



y_probas_forest = cross_val_predict(forest_clf, creditcard_test, creditcard_test_labels, cv=3, method="predict_proba")
# Plot ROC curve

import matplotlib.pyplot as plt

def plot_roc_curve(fpr, tpr, label=None):

    plt.plot(fpr, tpr, "g-", linewidth=2, label=label)

    plt.plot([0, 1], [0, 1], "k--", label="Random Estimator")

    plt.axis([0, 1, 0, 1])

    plt.xlabel("False Positive Rate")

    plt.ylabel("True Positive Rate (Recall)")

    

from sklearn.metrics import roc_curve

y_scores_forest = y_probas_forest[:, 1] # probability of positive class (fraud)

fpr_forest, tpr_forest, thresholds_forest = roc_curve(creditcard_test_labels, y_scores_forest)



plot_roc_curve(fpr_forest, tpr_forest, "RandomForestClassifier")

plt.legend(loc="lower right")

plt.show()
# Print AUC score and cross-validation scores

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import cross_val_score

auc_score = roc_auc_score(creditcard_test_labels, y_scores_forest)

cross_val_s = cross_val_score(forest_clf, creditcard_test, creditcard_test_labels, cv=3, scoring="accuracy")

print(auc_score, cross_val_s)