import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

!pip install --upgrade wandb

import wandb



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



import scipy

import pandas as pd

from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import Ridge

from sklearn.experimental import enable_hist_gradient_boosting

from sklearn.ensemble import HistGradientBoostingClassifier

!pip install cuml

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn import naive_bayes

from sklearn.model_selection import train_test_split



# Any results you write to the current directory are saved as output.
df = pd.read_csv("/kaggle/input/cat-in-the-dat-ii/train.csv", index_col="id")

df_test = pd.read_csv("/kaggle/input/cat-in-the-dat-ii/test.csv", index_col="id")



y = df["target"]

D = df.drop(columns="target")

features = D.columns

test_ids = df_test.index



D_all = pd.concat([D, df_test])

num_train = len(D)



print(f"D_all.shape = {D_all.shape}")
# Map value in train xor test

for col in D.columns.difference(["id"]):

    train_vals = set(D[col].dropna().unique())

    test_vals = set(df_test[col].dropna().unique())



    xor_cat_vals = train_vals ^ test_vals

    if xor_cat_vals:

        print(f"Replacing {len(xor_cat_vals)} values in {col}, {xor_cat_vals}")

        D_all.loc[D_all[col].isin(xor_cat_vals), col] = "xor"
# Ordinal encoding

ord_maps = {

    "ord_0": {val: i for i, val in enumerate([1, 2, 3])},

    "ord_1": {

        val: i

        for i, val in enumerate(

            ["Novice", "Contributor", "Expert", "Master", "Grandmaster"]

        )

    },

    "ord_2": {

        val: i

        for i, val in enumerate(

            ["Freezing", "Cold", "Warm", "Hot", "Boiling Hot", "Lava Hot"]

        )

    },

    **{col: {val: i for i, val in enumerate(sorted(D_all[col].dropna().unique()))} for col in ["ord_3", "ord_4", "ord_5", "day", "month"]},

}
# OneHot encoding

oh_cols = D_all.columns.difference(ord_maps.keys() - {"day", "month"})



print(f"OneHot encoding {len(oh_cols)} columns")



one_hot = pd.get_dummies(

    D_all[oh_cols],

    columns=oh_cols,

    drop_first=True,

    dummy_na=True,

    sparse=True,

    dtype="int8",

).sparse.to_coo()
# Ordinal encoding

ord_cols = pd.concat([D_all[col].map(ord_map).fillna(max(ord_map.values())//2).astype("float32") for col, ord_map in ord_maps.items()], axis=1)

ord_cols /= ord_cols.max()  # for convergence



ord_cols_sqr = 4*(ord_cols - 0.5)**2
# Combine data

X = scipy.sparse.hstack([one_hot, ord_cols, ord_cols_sqr]).tocsr()

print(f"X.shape = {X.shape}")



# Split into training and validation sets

X_train, X_test, y_train, y_test = train_test_split(X[:num_train], y, test_size=0.1, random_state=42, shuffle=False)

X_train = X_train[:10000]

y_train = y_train[:10000]

X_test = X_test[:2000]

y_test = y_test[:2000]
# Classification - predict pulsar

# Train a model, get predictions

log = LogisticRegression(C=0.05, solver="lbfgs", max_iter=5000)

dtree = DecisionTreeClassifier(random_state=4)

rtree = RandomForestClassifier(n_estimators=100, random_state=4)

svm = SVC(random_state=4, probability=True)

nb = GaussianNB()

gbc = GradientBoostingClassifier()

knn = KNeighborsClassifier(n_neighbors=400)

adaboost = AdaBoostClassifier(n_estimators=500, learning_rate=0.01, random_state=42,

                             base_estimator=DecisionTreeClassifier(max_depth=8,

                             min_samples_leaf=10, random_state=42))

labels = [0,1]



def model_algorithm(clf, X_train, y_train, X_test, y_test, name, labels, features):

    clf.fit(X_train, y_train)

    y_probas = clf.predict_proba(X_test)

    y_pred = clf.predict(X_test)

    wandb.init(anonymous='allow', project="kaggle-feature-encoding", name=name, reinit=True)

    # wandb.sklearn.plot_roc(y_test, y_probas, labels, reinit = True)

    wandb.termlog('\nPlotting %s.'%name)

    wandb.sklearn.plot_learning_curve(clf, X_train, y_train)

    wandb.termlog('Logged learning curve.')

    wandb.sklearn.plot_confusion_matrix(y_test, y_pred, labels)

    wandb.termlog('Logged confusion matrix.')

    wandb.sklearn.plot_summary_metrics(clf, X=X_train, y=y_train, X_test=X_test, y_test=y_test)

    wandb.termlog('Logged summary metrics.')

    wandb.sklearn.plot_class_proportions(y_train, y_test, labels)

    wandb.termlog('Logged class proportions.')

    if(not isinstance(clf, naive_bayes.MultinomialNB)):

        wandb.sklearn.plot_calibration_curve(clf, X_train, y_train, name)

    wandb.termlog('Logged calibration curve.')

    wandb.sklearn.plot_roc(y_test, y_probas, labels)

    wandb.termlog('Logged roc curve.')

    wandb.sklearn.plot_precision_recall(y_test, y_probas, labels)

    wandb.termlog('Logged precision recall curve.')

    csv_name = "submission_"+name+".csv"

    # Create submission file

    # pd.DataFrame({"id": test_ids, "target": y_pred}).to_csv(csv_name, index=False)
model_algorithm(log, X_train, y_train, X_test, y_test, 'LogisticRegression', labels, features)
model_algorithm(svm, X_train, y_train, X_test, y_test, 'SVM', labels, features)
model_algorithm(knn, X_train, y_train, X_test, y_test, 'KNearestNeighbor', labels, features)
model_algorithm(adaboost, X_train, y_train, X_test, y_test, 'AdaBoost', labels, features)
model_algorithm(gbc, X_train, y_train, X_test, y_test, 'GradientBoosting', labels, features)
model_algorithm(dtree, X_train, y_train, X_test, y_test, 'DecisionTree', labels, None)
model_algorithm(rtree, X_train, y_train, X_test, y_test, 'RandomForest', labels, features)
clf=LogisticRegression(C=0.05, solver="lbfgs", max_iter=5000)

clf.fit(X_train, y_train)

pred = clf.predict_proba(X_test)[:, 1]

pd.DataFrame({"id": test_ids, "target": pred}).to_csv("submission_lr.csv", index=False)