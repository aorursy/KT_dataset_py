! pip install -q dabl
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

from colorama import Fore, Style



import dabl



from sklearn.model_selection import train_test_split

from sklearn.neural_network import MLPClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.gaussian_process.kernels import RBF

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.metrics import roc_auc_score

from xgboost import XGBClassifier

from catboost import CatBoostClassifier



plt.style.use("classic")

warnings.filterwarnings('ignore')
def cout(string: str, color=Fore.RED):

    """

    Saves some work 😅

    """

    print(color+string+Style.RESET_ALL)
data = pd.read_csv("../input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv")

data.head()
dabl.plot(data, target_col='DEATH_EVENT')
# First, we split the data

data = data[['age', 'ejection_fraction', 'serum_creatinine', 'serum_sodium', 'time', 'DEATH_EVENT']]



trainX, testX, trainY, testY = train_test_split(data.drop(['DEATH_EVENT'], axis=1), data['DEATH_EVENT'], test_size=0.2, random_state=1)



cout(f"Training Data Shape is: {trainX.shape}", Fore.RED)

cout(f"Testing Data Shape is: {testX.shape}", Fore.BLUE)
# Define all models

names = ["Logistic Regression", "Nearest Neighbors", "RBF SVM", "Gaussian Process",

         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",

         "Naive Bayes", "QDA", "XGBoost", "CatBoost"]



classifiers = [

    LogisticRegression(),

    KNeighborsClassifier(4),

    SVC(kernel="rbf", random_state=0),

    GaussianProcessClassifier(2.0 * RBF(1.0)),

    DecisionTreeClassifier(max_depth=20),

    RandomForestClassifier(n_estimators = 17, criterion='gini', random_state=0),

    MLPClassifier(alpha=3, max_iter=2000),

    AdaBoostClassifier(),

    GaussianNB(),

    QuadraticDiscriminantAnalysis(),

    XGBClassifier(),

    CatBoostClassifier()

]
# Train all models



clf_results_roc = {}

clf_results_acc = {}



for name, clf in zip(names, classifiers):    

    # Fit on the traning data

    clf.fit(trainX, trainY)

    

    # Get the test time prediction

    preds = clf.predict(testX)

    

    # Calculate Test ROC_AUC

    score = roc_auc_score(testY, preds)

    

    # Calculate the val accuracy

    val_acc = clf.score(testX, testY)

    

    # Store the results in a dictionary

    clf_results_roc[name] = score

    clf_results_acc[name] = val_acc

    

    cout(f"Classifier: {name}", Fore.YELLOW)

    cout(f"\nval_acc: {val_acc:.3f} | roc_auc: {score:.3f}\n", Fore.BLUE)
# Sort the Model Accuracies based on the test score

sort_clf = dict(sorted(clf_results_acc.items(), key=lambda x: x[1], reverse=True))



# Get the names and the corresponding scores

clf_names = list(sort_clf.keys())[::-1]

clf_scores = list(sort_clf.values())[::-1]



# Plot the results

plt.figure(figsize=(14, 8))

sns.barplot(x=clf_names, y=clf_scores)

plt.xlabel("Models")

plt.ylabel("Validation Accuracy")

plt.xticks(rotation=45)

plt.title("Model Comparison - Validation Accuracy")

plt.show()
# Sort the Model Accuracies based on the roc-auc score

sort_clf = dict(sorted(clf_results_roc.items(), key=lambda x: x[1], reverse=True))



# Get the names and the corresponding scores

clf_names = list(sort_clf.keys())[::-1]

clf_scores = list(sort_clf.values())[::-1]



# Plot the results

plt.figure(figsize=(14, 8))

sns.barplot(x=clf_names, y=clf_scores)

plt.xlabel("Models")

plt.ylabel("ROC-AUC Score")

plt.xticks(rotation=45)

plt.title("Model Comparison - ROC-AUC Scores")

plt.show()