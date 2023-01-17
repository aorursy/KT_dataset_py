# Loading the packages

import numpy as np

import pandas as pd 

from sklearn.dummy import DummyClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score

from sklearn.tree import DecisionTreeClassifier

#from sklearn.model_selection import cross_validate

from sklearn.model_selection import StratifiedKFold

import matplotlib.pyplot as plt

# Loading the training dataset

df_train = pd.read_csv("../input/train.csv")
y = df_train["target"]

# We exclude the target and id columns from the training dataset

df_train.pop("target");

df_train.pop("id")

X = df_train 

del df_train

X = X.values # Converting pandas dataframe to numpy array 

y = y.values # Converting pandas series to numpy array 

skf = StratifiedKFold(n_splits=10)

skf.get_n_splits(X, y)
def fit_decision_tree(max_depth=1, nbins=5):

    train_auc = []

    test_auc = []

    

    for train_index, test_index in skf.split(X, y):

        model = DecisionTreeClassifier(max_depth=max_depth)

        model.fit(X[train_index], y[train_index])

        y_train = y[train_index]

        y_test = y[test_index]

    

        y_train_predict = model.predict_proba(X[train_index])

        y_train_predict = y_train_predict[:,1]

        y_test_predict = model.predict_proba(X[test_index], )

        y_test_predict = y_test_predict[:,1]        

        train_auc.append(roc_auc_score(y_train, y_train_predict))

        test_auc.append(roc_auc_score(y_test, y_test_predict))

        

    n_bins = 5



    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, tight_layout=True);



    ax1.hist(train_auc, bins=n_bins);

    ax1.set_title("Histogram of AUC training")

    ax2.hist(test_auc, bins=n_bins);

    ax2.set_title("Histogram of AUC validation")        

    
fit_decision_tree(1, 5)
fit_decision_tree(2, 5)
fit_decision_tree(3, 5)
model = DecisionTreeClassifier(max_depth=1, class_weight='balanced')

model.fit(X, y)
df_test = pd.read_csv("../input/test.csv")

df_test.pop("id");

X = df_test 

del df_test

y_pred = model.predict_proba(X)

y_pred = y_pred[:,1]
# submit prediction

smpsb_df = pd.read_csv("../input/sample_submission.csv")

smpsb_df["target"] = y_pred

smpsb_df.to_csv("decision_tree_improved.csv", index=None)
