import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
puls = pd.read_csv("/kaggle/input/predicting-a-pulsar-star/pulsar_stars.csv")
puls.describe()
puls.shape
puls.dtypes
puls.info()
puls.isnull().sum()
puls.target_class.value_counts()
puls.target_class.plot(kind="hist", bins=3);
plt.figure(figsize=(12, 8))

sns.heatmap(puls.describe()[1:].T,

           annot=True, linecolor="w",

           cmap=sns.color_palette("Set3"))

plt.title("Data Summary")
puls.hist(bins=50, figsize=(15, 9));
puls.target_class.value_counts() / len(puls)
from sklearn.model_selection import StratifiedShuffleSplit



split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(puls, puls.target_class):

    puls_train_set = puls.loc[train_index]

    puls_test_set = puls.loc[test_index]

    

puls_test_set["target_class"].value_counts() / len(puls_test_set)
len(puls_test_set), len(puls_train_set)
puls = puls_train_set.copy()
plt.figure(figsize=(10, 8))

corr = puls.corr()

sns.heatmap(corr, annot=True,

           cmap=sns.color_palette("GnBu_d"),

           xticklabels=corr.columns,

           yticklabels=corr.columns

);

plt.title("Feature Correlation Heatmap");
corr["target_class"].sort_values(ascending=False)
from pandas.plotting import scatter_matrix



attributes = ["target_class", 

              " Excess kurtosis of the integrated profile", 

              " Skewness of the integrated profile",

              " Mean of the integrated profile",

              " Standard deviation of the DM-SNR curve"

             ]



scatter_matrix(puls[attributes], figsize=(12, 9));
X_train = puls.drop(["target_class"], axis=1)

y_train = puls.target_class

X_test = puls_test_set.drop(["target_class"], axis=1)

y_test = puls_test_set.target_class
X_train.shape, y_train.shape, X_test.shape
from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()



X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)
from sklearn.linear_model import SGDClassifier, LogisticRegression

from sklearn.ensemble import RandomForestClassifier



sgd = SGDClassifier(tol=-np.infty)

rf = RandomForestClassifier(n_jobs=-1, random_state=42)

lr = LogisticRegression(solver='liblinear', random_state=42)

from sklearn.model_selection import GridSearchCV
sgd.get_params()
sgd_params = {

    'alpha': np.logspace(-4, -2, 5),

    'max_iter': np.linspace(100, 1000, 10),

    'penalty': ['l1', 'l2']

}



sgd_grid = GridSearchCV(sgd, sgd_params, cv=5, scoring="f1")

sgd_grid.fit(X_train_scaled, y_train)
sgd_grid.best_params_
sgd_clf = sgd_grid.best_estimator_
rf.get_params()
rf_params = {

    'n_estimators': [30, 50, 100],

    'min_samples_leaf': [1, 3, 5],

    'min_samples_split': [2, 5, 10],

    'max_depth': [5, 10, 30]

}



rf_grid = GridSearchCV(rf, rf_params, cv=3, scoring="f1")

rf_grid.fit(X_train, y_train)
rf_grid.best_params_
rf_clf = rf_grid.best_estimator_
lr.get_params()
lr_params = {

    'C': np.logspace(-2, 2, 9),

    'penalty': ['l1', 'l2']

}



lr_grid = GridSearchCV(lr, lr_params, cv=5, scoring='f1')

lr_grid.fit(X_train_scaled, y_train)
lr_grid.best_params_
lr_clf = lr_grid.best_estimator_
from sklearn.model_selection import cross_val_score, cross_val_predict



print("SGD classifier precision: \t", cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="precision"))

print("SGD classifier recall: \t\t", cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="recall"))

print("Random Forest precision: \t", cross_val_score(rf_clf, X_train, y_train, cv=3, scoring="precision"))

print("Random Forest recall: \t\t", cross_val_score(rf_clf, X_train, y_train, cv=3, scoring="recall"))

print("Log Regression precision: \t", cross_val_score(lr_clf, X_train_scaled, y_train, cv=3, scoring="precision"))

print("Log Regression recall: \t\t", cross_val_score(lr_clf, X_train_scaled, y_train, cv=3, scoring="recall"))
# since the Random Forrest classifier doesn't require scaling, let's use the original data

y_scores_rf = cross_val_predict(rf_clf, X_train, y_train, cv=3, method="predict_proba")

y_scores_rf[:, 1]
y_scores_sgd = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3, method="decision_function")

y_scores_sgd
y_scores_lr = cross_val_predict(lr_clf, X_train_scaled, y_train, cv=3, method="decision_function")

y_scores_lr
from sklearn.metrics import confusion_matrix, precision_recall_curve
prec_sgd, rec_sgd, thresholds_sgd = precision_recall_curve(y_train, y_scores_sgd)

# score is a predict probability for a positive class

prec_rf, rec_rf, thresholds_rf = precision_recall_curve(y_train, y_scores_rf[:, 1])

prec_lr, rec_lr, thresholds_lr = precision_recall_curve(y_train, y_scores_lr)
def plot_precision_recall(precision, recall, style, label=None):

    plt.plot(recall, precision, style, label=label)

    plt.xlabel("recall")

    plt.ylabel("precision")

    plt.title("Precision versus recall");

    plt.axis([0, 1, 0, 1]);
plt.figure(figsize=(8, 6))

plot_precision_recall(prec_sgd, rec_sgd, "b-", "SGD");

plot_precision_recall(prec_rf, rec_rf, "r:", "Random Forest");

plot_precision_recall(prec_lr, rec_lr, "g--", "Logistic Regression");

plt.legend(loc="lower left");
from sklearn.metrics import roc_curve



fpr_sgd, tpr_sgd, thresholds_sgd = roc_curve(y_train, y_scores_sgd)

# score is a probability of a positive class

fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_train, y_scores_rf[:, 1])

fpr_lr, tpr_lr, thresholds_lr = roc_curve(y_train, y_scores_lr)

    
def plot_roc_curve(fpr, tpr, style, label=None):

    plt.plot(fpr, tpr, style, label=label)

    plt.plot([0, 1], [0, 1], "k--", linewidth=1)

    plt.xlabel("False Positive Rate")

    plt.ylabel("True Positive Rate")

    plt.axis([0, 1, 0, 1]);
plt.figure(figsize=(8, 6))

plot_roc_curve(fpr_sgd, tpr_sgd, "b-", "SGD")

plot_roc_curve(fpr_rf, tpr_rf, "r:", "Random Forest")

plot_roc_curve(fpr_lr, tpr_lr, "g--", "Logistic Regression")

plt.legend(loc="lower right", fontsize=14);
from sklearn.metrics import roc_auc_score



print("SGD Classifier AUC: \t", roc_auc_score(y_train, y_scores_sgd))

print("Random Forest AUC: \t", roc_auc_score(y_train, y_scores_rf[:, 1]))

print("Log Regression AUC: \t", roc_auc_score(y_train, y_scores_lr))
from sklearn.metrics import confusion_matrix, f1_score
sgd_pred = sgd_clf.predict(X_test_scaled)

confusion_matrix(y_test, sgd_pred)
rf_pred = rf_clf.predict(X_test)

confusion_matrix(y_test, rf_pred)
lr_pred = lr_clf.predict(X_test_scaled)

confusion_matrix(y_test, lr_pred)
print("SGD Classifier F1 score: \t", f1_score(y_test, sgd_pred))

print("Random Forest F1 score: \t", f1_score(y_test, rf_pred))

print("Log Regression F1 score: \t", f1_score(y_test, lr_pred))
from sklearn.metrics import accuracy_score



print("SGD Classifier F1 score: \t", accuracy_score(y_test, sgd_pred))

print("Random Forest F1 score: \t", accuracy_score(y_test, rf_pred))

print("Log Regression F1 score: \t", accuracy_score(y_test, lr_pred))