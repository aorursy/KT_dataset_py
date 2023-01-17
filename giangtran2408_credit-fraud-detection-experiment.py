# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import sklearn
df = pd.read_csv("../input/creditcardfraud/creditcard.csv")
df.head()
df.describe()
df.columns
from sklearn.preprocessing import StandardScaler



std_scaler = StandardScaler()



std_time = std_scaler.fit_transform(df.loc[:, "Time"].values.reshape((-1, 1)))

std_amount = std_scaler.fit_transform(df.loc[:, "Amount"].values.reshape((-1, 1)))



df["Time"] = std_time

df["Amount"] = std_amount
df.head()
num_fraud = df[df["Class"].values == 1].shape[0]

num_normal = df[df["Class"].values == 0].shape[0]
print("Number of Fraud transaction: %d" % num_fraud)

print("Number of Normal transaction: %d" % num_normal)
print("Ratio of Fraud Transaction: %.5f" % (num_fraud/(num_fraud+num_normal)))

print("Ratio of Normal Transaction: %.5f" % (num_normal/(num_fraud+num_normal)))
from sklearn.model_selection import train_test_split



df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
print("Ratio training data of Fraud transaction after train/test split: %.5f" % (df_train["Class"].value_counts().values[1] / df_train["Class"].value_counts().values.sum()))

print("Ratio testing data of Fraud transaction after train/test split: %.5f" % (df_test["Class"].value_counts().values[1] / df_test["Class"].value_counts().values.sum()))
X_train = df_train.drop("Class", axis=1)

y_train = df_train["Class"]



X_test = df_test.drop("Class", axis=1)

y_test = df_test["Class"]
from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.neural_network import MLPClassifier



classifiers = {

    "Decision Tree": DecisionTreeClassifier(),

    "Logistic Regression": LogisticRegression(),

    "SVM": SVC(),

    "KNN": KNeighborsClassifier(),

    "Neural Network": MLPClassifier(hidden_layer_sizes=(26, 14))

}



from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support



def fit_models(classifiers, X_train, y_train):

    for k, classifier in classifiers.items():

        classifier.fit(X_train, y_train)

        y_pred = classifier.predict(X_test)

        print(k + ": ")

        conf_mat = confusion_matrix(y_test, y_pred)

        print(conf_mat)

        prf = precision_recall_fscore_support(y_test, y_pred, average="binary")

        print("Accuracy: %.5f" % ((conf_mat[0, 0] + conf_mat[1, 1])/conf_mat.sum() ))

        print("Precision: %.5f" % prf[0])

        print("Recall: %.5f" % prf[1])

        print("F1-score: %.5f" % prf[2])

        print("-"*50)
fit_models(classifiers, X_train, y_train)
df_train = df_train.sample(frac=1)



# amount of fraud classes 492 rows.

fraud_df = df_train.loc[df_train['Class'] == 1]

non_fraud_df = df_train.loc[df_train['Class'] == 0][:fraud_df.shape[0]]



under_sampling_df = pd.concat([fraud_df, non_fraud_df])



# Shuffle dataframe rows

under_sampling_df = under_sampling_df.sample(frac=1, random_state=42)



under_sampling_df.head()
under_sampling_df["Class"].value_counts()
X_train = under_sampling_df.drop("Class", axis=1)

y_train = under_sampling_df["Class"]



X_test = df_test.drop("Class", axis=1)

y_test = df_test["Class"]
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt



def tsne_plot(x1, y1, name="graph.png"):

    tsne = TSNE(n_components=2, random_state=0)

    X_t = tsne.fit_transform(x1)



    plt.figure(figsize=(12, 8))

    plt.scatter(X_t[np.where(y1 == 0), 0], X_t[np.where(y1 == 0), 1], marker='o', color='g', linewidth='1', alpha=0.8, label='Non Fraud')

    plt.scatter(X_t[np.where(y1 == 1), 0], X_t[np.where(y1 == 1), 1], marker='o', color='r', linewidth='1', alpha=0.8, label='Fraud')



    plt.legend(loc='best');

    plt.savefig(name);

    plt.show();
fit_models(classifiers, X_train, y_train)
# Use grid search to find the best parameters

from sklearn.model_selection import GridSearchCV



def do_grid_search(X_train, y_train):



    # Logistic Regression 

    log_reg_params = {"penalty": ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}



    grid_log_reg = GridSearchCV(LogisticRegression(), log_reg_params, scoring="f1")

    grid_log_reg.fit(X_train, y_train)

    # We automatically get the logistic regression with the best parameters.

    log_reg = grid_log_reg.best_estimator_



    knears_params = {"n_neighbors": list(range(2,5,1)), 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}



    grid_knears = GridSearchCV(KNeighborsClassifier(), knears_params, scoring="f1")

    grid_knears.fit(X_train, y_train)

    # KNears best estimator

    knears_neighbors = grid_knears.best_estimator_



    # Support Vector Classifier

    svc_params = {'C': [0.5, 0.7, 0.9, 1], 'kernel': ['rbf', 'poly', 'sigmoid', 'linear']}

    grid_svc = GridSearchCV(SVC(), svc_params, scoring="f1")

    grid_svc.fit(X_train, y_train)



    # SVC best estimator

    svc = grid_svc.best_estimator_



    # DecisionTree Classifier

    tree_params = {"criterion": ["gini", "entropy"], "max_depth": list(range(2,4,1)), 

                  "min_samples_leaf": list(range(5,7,1))}

    grid_tree = GridSearchCV(DecisionTreeClassifier(), tree_params, scoring="f1")

    grid_tree.fit(X_train, y_train)



    # tree best estimator

    tree_clf = grid_tree.best_estimator_

    

    nn_params = {"hidden_layer_sizes": [(50, ), (32, 16), (100, ), (50, 64, 32, 16)], "activation": ["logistic", "tanh", "relu"], "solver": ["lbfgs", "sgd", "adam"], "alpha": [0.01, 0.1, 0.001, 0.0001]}

    grid_nn = GridSearchCV(MLPClassifier(), nn_params, scoring="f1")

    

    return log_reg, knears_neighbors, svc, tree_clf, grid_nn
log_reg, knears_neighbors, svc, tree_clf, nn = do_grid_search(X_train, y_train)



best_classifiers = {

    "Logistic Regression": log_reg, 

    "KNN": knears_neighbors, 

    "SVM": svc, 

    "Decision Tree": tree_clf,

    "Neural Network": nn

}



fit_models(best_classifiers, X_train, y_train)
from imblearn.under_sampling import NearMiss, ClusterCentroids
X_train = df_train.drop("Class", axis=1)

y_train = df_train["Class"]



X_test = df_test.drop("Class", axis=1)

y_test = df_test["Class"]
df_train["Class"].value_counts()
nm = NearMiss(sampling_strategy="auto", version=2)

X_train, y_train = nm.fit_sample(X_train, y_train)
print("X_train.shape = %s" % str(X_train.shape))

print("y_train.shape = %s" % str(y_train.shape))
fit_models(classifiers, X_train, y_train)
log_reg, knears_neighbors, svc, tree_clf, nn = do_grid_search(X_train, y_train)



best_classifiers = {

    "Logistic Regression": log_reg, 

    "KNN": knears_neighbors, 

    "SVM": svc, 

    "Decision Tree": tree_clf,

    "Neural Network": nn

}



fit_models(best_classifiers, X_train, y_train)
X_train = df_train.drop("Class", axis=1)

y_train = df_train["Class"]



X_test = df_test.drop("Class", axis=1)

y_test = df_test["Class"]
from imblearn.over_sampling import SMOTE



sm = SMOTE(sampling_strategy="auto", random_state=42)



X_train, y_train = sm.fit_sample(X_train, y_train)
print("Number of Normal Transaction after Over-sampling using SMOTE technique: %d" % len(y_train[y_train == 0]))

print("Number of Fraud Transaction after Over-sampling using SMOTE technique: %d" % len(y_train[y_train == 1]))
classifiers = {

    "Decision Tree": DecisionTreeClassifier(),

    "Logistic Regression": LogisticRegression(),

    "Neural Network": MLPClassifier(hidden_layer_sizes=(50, 64, 32, 16))

}



fit_models(classifiers, X_train, y_train)
X_train = df_train.drop("Class", axis=1)

y_train = df_train["Class"]



X_test = df_test.drop("Class", axis=1)

y_test = df_test["Class"]
from imblearn.over_sampling import RandomOverSampler



ros = RandomOverSampler(sampling_strategy="auto", random_state=42)

X_train, y_train = sm.fit_sample(X_train, y_train)
print("Number of Normal Transaction after Over-sampling using random technique: %d" % len(y_train[y_train == 0]))

print("Number of Fraud Transaction after Over-sampling using random technique: %d" % len(y_train[y_train == 1]))
fit_models(classifiers, X_train, y_train)