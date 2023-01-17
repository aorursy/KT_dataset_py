# Data manipulation and plotting modules

import numpy as np

import pandas as pd
# Data pre-processing

# z = (x-mean)/stdev

from sklearn.preprocessing import RobustScaler
# Working with imbalanced data

# http://contrib.scikit-learn.org/imbalanced-learn/stable/generated/imblearn.over_sampling.SMOTE.html

# Check imblearn version number as:

#   import imblearn;  imblearn.__version__

# Install as:  conda install -c conda-forge imbalanced-learn

from imblearn.over_sampling import SMOTE, ADASYN



# Also using NearMiss for undersampling

from imblearn.under_sampling import NearMiss
from sklearn.model_selection import StratifiedKFold
# Classifier Libraries

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier
# libraries for ROC graphs & metrics

import scikitplot as skplt

from sklearn.metrics import confusion_matrix

from sklearn.metrics import average_precision_score

import sklearn.metrics as metrics

from sklearn.metrics import precision_score, recall_score, f1_score, auc, roc_curve, roc_auc_score, accuracy_score, classification_report, precision_recall_curve

from sklearn.model_selection import cross_val_score
# Modules for graph plotting

import matplotlib.pyplot as plt

import seaborn as sns
# Importing GridSearchCV

from sklearn.model_selection import GridSearchCV, train_test_split
# Importing RandomizedSearchCV

from sklearn.model_selection import RandomizedSearchCV
# Importing some supporting modules

import time

import random

import os

from scipy.stats import uniform

import warnings

warnings.filterwarnings("ignore")
print(os.listdir("../input/"))

data = pd.read_csv("../input/creditcard.csv")
# Explore Data

data.info()
# Data Contents

data.head(10)
data.describe()
# Class is the target for this problem, 

# 1 indicates fraudulent transaction 

# and 0 as normal transaction

# lets check if data is balanced



print(data.Class.value_counts())
# Check if any column in dataset is all zeroes

# We need to remove these columns

# Sum each row, and check in which case sum is 0

# axis = 1 ==> Across columns



x = np.sum(data, axis = 1)

v = x.index[x == 0]     # Get index of the row which meets a condition



if len(v) > 0:

    print("Column Index: ", v)

else:

    print("Good - No Column in Dataset has all rows zeroes.")
fig, ax = plt.subplots(1, 2, figsize=(18,4))



amount_val = data['Amount'].values

time_val = data['Time'].values



sns.distplot(amount_val, ax=ax[0], color='r')

ax[0].set_title('Distribution of Transaction Amount', fontsize=14)

ax[0].set_xlim([min(amount_val), max(amount_val)])



sns.distplot(time_val, ax=ax[1], color='b')

ax[1].set_title('Distribution of Transaction Time', fontsize=14)

ax[1].set_xlim([min(time_val), max(time_val)])
# Since most of the data has already been scaled we should scale the columns that are left to scale (Amount and Time)

# RobustScaler is less prone to outliers.



rob_scaler = RobustScaler()



data['scaled_amount'] = rob_scaler.fit_transform(data['Amount'].values.reshape(-1,1))

data['scaled_time'] = rob_scaler.fit_transform(data['Time'].values.reshape(-1,1))



data.drop(['Time','Amount'], axis=1, inplace=True)
# Lets check the class values where 1 indicates frauds and 0 shows normal transactions.

print('No Frauds: Label 0 - {0} % of the dataset.'.format(round(data['Class'].value_counts()[0]/len(data) * 100,2)))

print('Frauds   : Label 1 - {0} % of the dataset.'.format(round(data['Class'].value_counts()[1]/len(data) * 100,2)))
colors = ["#0101DF", "#DF0101"]



sns.countplot(x = 'Class', data = data, palette= colors)

plt.title('Class Distributions \n (0: No Fraud || 1: Fraud)', fontsize=14)

plt.show()
# Separate data set into Original_X and Original_y 

Original_X = data.drop('Class', axis=1)

Original_y = data['Class']



print("Original X Shape: ", Original_X.shape)

print("\nOriginal  Y Shape: ", Original_y.shape)

print("\nOriginal X Columns: \n", Original_X.columns)

print("\n\n")

Original_X.head()
def printvalues(section, X_train, X_test, y_train, y_test, X):

    print(section, "_X Train Shape: ", X_train.shape)

    print(section, "_y Train Shape: ", y_train.shape)

    print(section, "_X Test Shape: ", X_test.shape)

    print(section, "_y Test Shape: ", y_test.shape)

    print("\n\n", section, " data columns: \n")

    for i in range(0, len(X.columns)):

        print(i, " ", X.columns[i])

        

def printscore(model, y_test, y_pred):

    # Best Score and best parameters

    print("Best score Train:", model.best_score_ * 100)

    print("Best parameter set :", model.best_params_)



    # Check Accuracy

    print("Accuracy: ", accuracy_score(y_test, y_pred) * 100)

    print("Precision: ", precision_score(y_test, y_pred))

    print("Recall: ", recall_score(y_test, y_pred))

    print("F1: ", f1_score(y_test, y_pred))

    print("ROC_AUC: ", roc_auc_score(y_test, y_pred))
# Modelling on the given data without using under / over sampling first.



Original_X_train, Original_X_test, Original_y_train, Original_y_test = train_test_split(Original_X, Original_y, test_size=0.25, random_state=42)



printvalues("Original", Original_X_train, Original_X_test, Original_y_train, Original_y_test, Original_X)
# Since our classes are highly skewed we should make them equivalent in order to have a normal distribution of the classes.



# Lets shuffle the data before creating the subsamples



data = data.sample(frac=1)



# amount of fraud classes 492 rows.

fraud_data = data.loc[data['Class'] == 1]

non_fraud_data = data.loc[data['Class'] == 0][:492]



normal_distributed_data = pd.concat([fraud_data, non_fraud_data])



# Shuffle dataframe rows

new_data = normal_distributed_data.sample(frac=1, random_state=42)

print("\nNew Data Shape", new_data.shape,"\n\n")

new_data.head()
colors = ["#0101DF", "#DF0101"]



sns.countplot(x = 'Class', data = new_data, palette= colors)

plt.title('Class Distributions \n (0: No Fraud || 1: Fraud)', fontsize=14)

plt.show()
# Undersampling before cross validating (prone to overfit)

new_X = new_data.drop('Class', axis=1)

new_y = new_data['Class']



# This is explicitly used for undersampling.

Under_X_train, Under_X_test, Under_y_train, Under_y_test = train_test_split(new_X, new_y, test_size=0.25, random_state=42)



printvalues("Under", Under_X_train, Under_X_test, Under_y_train, Under_y_test, new_X)
nr = NearMiss()

NearMiss_X, NearMiss_y = nr.fit_sample(Original_X, Original_y)



NearMiss_X_train, NearMiss_X_test, NearMiss_y_train, NearMiss_y_test = train_test_split(NearMiss_X, NearMiss_y, test_size=0.25, random_state=42)



#let us check the amount of records in each category

print("NearMiss_X_train Shape: ", NearMiss_X_train.shape)

print("NearMiss_y_train Shape: ", NearMiss_y_train.shape)
colors = ["#0101DF", "#DF0101"]



sns.countplot(x = NearMiss_y_train, palette= colors)

plt.title('Class Distributions \n (0: No Fraud || 1: Fraud)', fontsize=14)

plt.show()
sm = SMOTE(sampling_strategy='minority')

SMOTE_X, SMOTE_y = sm.fit_sample(Original_X, Original_y)



SMOTE_X_train, SMOTE_X_test, SMOTE_y_train, SMOTE_y_test = train_test_split(SMOTE_X, SMOTE_y, test_size=0.25, random_state=42)



#let us check the amount of records in each category

print("SMOTE_X_train Shape: ", SMOTE_X_train.shape)

print("SMOTE_y_train Shape: ", SMOTE_y_train.shape)

print("SMOTE_Classes: ", np.bincount(SMOTE_y_train))
colors = ["#0101DF", "#DF0101"]



sns.countplot(x = SMOTE_y_train, palette= colors)

plt.title('Class Distributions \n (0: No Fraud || 1: Fraud)', fontsize=14)

plt.show()
ad = ADASYN(sampling_strategy='minority')

ADASYN_X, ADASYN_y = ad.fit_sample(Original_X, Original_y)



ADASYN_X_train, ADASYN_X_test, ADASYN_y_train, ADASYN_y_test = train_test_split(ADASYN_X, ADASYN_y, test_size=0.25, random_state=42)



#let us check the amount of records in each category

print("ADASYN_X_train Shape: ", ADASYN_X_train.shape)

print("ADASYN_y_train Shape: ", ADASYN_y_train.shape)

print("ADASYN_Classes: ", np.bincount(ADASYN_y_train))
colors = ["#0101DF", "#DF0101"]



sns.countplot(x = ADASYN_y_train, palette= colors)

plt.title('Class Distributions \n (0: No Fraud || 1: Fraud)', fontsize=14)

plt.show()
def cGridSearchCV(X_train, y_train):

    

    parameters = {'learning_rate': [1], 'n_estimators': [100], 'max_depth': [1]}

    workers = 6

    

    clf = GridSearchCV(XGBClassifier(silent = False, n_jobs= workers),

                       parameters,

                       n_jobs = workers,

                       cv = 3,

                       verbose = 1,

                       scoring = ['accuracy', 'roc_auc'],

                       refit = 'roc_auc'

                       )

    clf.fit(X_train, y_train)

    return clf
##################### Randomized Search #################



def cRandomizedSearchCV(X_train, y_train):



    parameters = {'learning_rate': [1], 'n_estimators': [10], 'max_depth': [1]}

    workers = 6



    clf = RandomizedSearchCV(XGBClassifier(silent = False, n_jobs=workers),

                            param_distributions=parameters,

                            scoring= ['roc_auc', 'accuracy'],

                            n_iter=15,          # Max combination of parameter to try. Default = 10

                            verbose = 1,

                            refit = 'roc_auc',

                            n_jobs = workers,       # Use parallel cpu threads

                            cv = 3               # No of folds, so n_iter * cv combinations

                           )

    clf.fit(X_train, y_train)

    return clf
def MiscClassifiers(X_train, y_train):

    # Let's implement simple classifiers



    classifiers = {

        "LogisiticRegression": LogisticRegression(n_jobs = 6),

        "KNearest": KNeighborsClassifier(n_jobs = 6),

        "DecisionTreeClassifier": DecisionTreeClassifier(max_depth = 2)

    }



    # 100% accuracy score, these could be due to overfitting.

    # we shall use under and over sampling in next few sections to fix this.



    for key, classifier in classifiers.items():

        classifier.fit(X_train, y_train)

        training_score = cross_val_score(classifier, X_train, y_train, cv=5)

        print("Classifiers: ", classifier.__class__.__name__, "has a training score of", round(training_score.mean(), 2) * 100, "% accuracy score")
MiscClassifiers(Original_X_train, Original_y_train)
GridSearch_result_without_sampling = cGridSearchCV(Original_X_train, Original_y_train)
# Best Score and best parameters

GridSearch_y_pred_without_sampling = GridSearch_result_without_sampling.predict(Original_X_test)



printscore(GridSearch_result_without_sampling, Original_y_test, GridSearch_y_pred_without_sampling)
RandomizedSearch_result_without_sampling = cRandomizedSearchCV(Original_X_train, Original_y_train)
# Best Score and best parameters

RandomizedSearch_y_pred_without_sampling = RandomizedSearch_result_without_sampling.predict(Original_X_test)



printscore(RandomizedSearch_result_without_sampling, Original_y_test, RandomizedSearch_y_pred_without_sampling)
MiscClassifiers(Under_X_train, Under_y_train)
GridSearch_result_under_sampling = cGridSearchCV(Under_X_train, Under_y_train)
# Best Score and best parameters

GridSearch_y_pred_under_sampling = GridSearch_result_under_sampling.predict(Under_X_test)



printscore(GridSearch_result_under_sampling, Under_y_test, GridSearch_y_pred_under_sampling)
RandomizedSearch_result_under_sampling = cRandomizedSearchCV(Under_X_train, Under_y_train)
# Best Score and best parameters

RandomizedSearch_y_pred_under_sampling = RandomizedSearch_result_under_sampling.predict(Under_X_test)



printscore(RandomizedSearch_result_under_sampling, Under_y_test, RandomizedSearch_y_pred_under_sampling)
MiscClassifiers(NearMiss_X_train, NearMiss_y_train)
GridSearch_result_NearMiss = cGridSearchCV(NearMiss_X_train, NearMiss_y_train)



# Best Score and best parameters

GridSearch_y_pred_NearMiss = GridSearch_result_NearMiss.predict(NearMiss_X_test)



printscore(GridSearch_result_NearMiss, NearMiss_y_test, GridSearch_y_pred_NearMiss)
GridSearch_result_NearMiss = cGridSearchCV(NearMiss_X_train, NearMiss_y_train)



# Best Score and best parameters

GridSearch_y_pred_NearMiss = GridSearch_result_NearMiss.predict(NearMiss_X_test)



printscore(GridSearch_result_NearMiss, NearMiss_y_test, GridSearch_y_pred_NearMiss)
RandomizedSearch_result_NearMiss = cRandomizedSearchCV(NearMiss_X_train, NearMiss_y_train)

time.sleep(2)



# Best Score and best parameters

RandomizedSearch_y_pred_NearMiss = RandomizedSearch_result_NearMiss.predict(NearMiss_X_test)



printscore(RandomizedSearch_result_NearMiss, NearMiss_y_test, RandomizedSearch_y_pred_NearMiss)
MiscClassifiers(SMOTE_X_train, SMOTE_y_train)
sss = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)



# List to append the score and then find the average

accuracy_lst = []

precision_lst = []

recall_lst = []

f1_lst = []

auc_lst = []



# Implementing SMOTE Technique 

# Cross Validating parameters



rand_log_reg = ""

best_est = ""

prediction = ""





parameters = {"penalty": ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10], 'learning_rate': [1], 'n_estimators': [10], 'max_depth': [1]}



rand_log_reg = RandomizedSearchCV(XGBClassifier(silent = False, n_jobs=6),

                            param_distributions=parameters,

                            scoring= ['roc_auc', 'accuracy'],

                            n_iter=15,          # Max combination of parameter to try. Default = 10

                            verbose = 1,

                            refit = 'roc_auc',

                            cv = 3,

                            n_jobs = 6, 

                            )



for train, test in sss.split(SMOTE_X, SMOTE_y):

    rand_log_reg.fit(SMOTE_X[train], SMOTE_y[train])    

    best_est = rand_log_reg.best_estimator_

    prediction = best_est.predict(SMOTE_X[test])

    

    accuracy_lst.append(best_est.score(SMOTE_X[test], SMOTE_y[test]))

    precision_lst.append(precision_score(SMOTE_y[test], prediction))

    recall_lst.append(recall_score(SMOTE_y[test], prediction))

    f1_lst.append(f1_score(SMOTE_y[test], prediction))

    auc_lst.append(roc_auc_score(SMOTE_y[test], prediction))

    

print('---' * 10)

print('')

print("accuracy: {}".format(np.mean(accuracy_lst)))

print("precision: {}".format(np.mean(precision_lst)))

print("recall: {}".format(np.mean(recall_lst)))

print("f1: {}".format(np.mean(f1_lst)))

print('---' * 10)
GridSearch_result_SMOTE = cGridSearchCV(SMOTE_X_train, SMOTE_y_train)



# Best Score and best parameters

GridSearch_y_pred_SMOTE = GridSearch_result_SMOTE.predict(SMOTE_X_test)



printscore(GridSearch_result_SMOTE, SMOTE_y_test, GridSearch_y_pred_SMOTE)
RandomizedSearch_result_SMOTE = cRandomizedSearchCV(SMOTE_X_train, SMOTE_y_train)



# Best Score and best parameters

RandomizedSearch_y_pred_SMOTE = RandomizedSearch_result_SMOTE.predict(SMOTE_X_test)



printscore(RandomizedSearch_result_SMOTE, SMOTE_y_test, RandomizedSearch_y_pred_SMOTE)
MiscClassifiers(ADASYN_X_train, ADASYN_y_train)
# List to append the score and then find the average

accuracy_lst = []

precision_lst = []

recall_lst = []

f1_lst = []

auc_lst = []



# Implementing ADASYN Technique 

# Cross Validating parameters



log_reg_params = {"penalty": ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10], 'learning_rate': [1], 'n_estimators': [10], 'max_depth': [1]}



ADASYN_rand_log_reg = RandomizedSearchCV(XGBClassifier(silent = False, n_jobs=6),

                            param_distributions=parameters,

                            scoring= ['roc_auc', 'accuracy'],

                            n_iter=15,

                            verbose = 1,

                            refit = 'roc_auc',

                            cv = 3,

                            n_jobs = 6, 

                            )



for train, test in sss.split(ADASYN_X_train, ADASYN_y_train):

    ADASYN_rand_log_reg.fit(ADASYN_X_train[train], ADASYN_y_train[train])    

    best_est = ADASYN_rand_log_reg.best_estimator_

    ADASYN_prediction = best_est.predict(ADASYN_X_train[test])

    

    accuracy_lst.append(best_est.score(ADASYN_X_train[test], ADASYN_y_train[test]))

    precision_lst.append(precision_score(ADASYN_y_train[test], ADASYN_prediction))

    recall_lst.append(recall_score(ADASYN_y_train[test], ADASYN_prediction))

    f1_lst.append(f1_score(ADASYN_y_train[test], ADASYN_prediction))

    auc_lst.append(roc_auc_score(ADASYN_y_train[test], ADASYN_prediction))

    

print('---' * 10)

print('')

print("accuracy: {}".format(np.mean(accuracy_lst)))

print("precision: {}".format(np.mean(precision_lst)))

print("recall: {}".format(np.mean(recall_lst)))

print("f1: {}".format(np.mean(f1_lst)))

print('---' * 10)
GridSearch_result_ADASYN = cGridSearchCV(ADASYN_X_train, ADASYN_y_train)



# Best Score and best parameters

GridSearch_y_pred_ADASYN = GridSearch_result_ADASYN.predict(ADASYN_X_test)



printscore(GridSearch_result_ADASYN, ADASYN_y_test, GridSearch_y_pred_ADASYN)
RandomizedSearch_result_ADASYN = cRandomizedSearchCV(ADASYN_X_train, ADASYN_y_train)



# Best Score and best parameters

RandomizedSearch_y_pred_ADASYN = RandomizedSearch_result_ADASYN.predict(ADASYN_X_test)



printscore(RandomizedSearch_result_ADASYN, ADASYN_y_test, RandomizedSearch_y_pred_ADASYN)
def conf_matrix(axis, y_test, y_pred, title):

    ##### Confusion matrix

    cm = confusion_matrix(y_test,y_pred)

    axis.set_title(title)

    sns.heatmap(cm, annot=True, fmt='g', annot_kws={"size": 14}, ax = axis)
fig, axis = plt.subplots(5,2,figsize=(10,10))



conf_matrix(axis[0,0], Original_y_test, GridSearch_y_pred_without_sampling, "GridSearcCV\nWithout Sampling")

conf_matrix(axis[0,1], Original_y_test, RandomizedSearch_y_pred_without_sampling, "RandomizedSearcCV\nWithout Sampling")



conf_matrix(axis[1,0], Under_y_test, GridSearch_y_pred_under_sampling, "GridSearcCV\nUnder Sampling")

conf_matrix(axis[1,1], Under_y_test, RandomizedSearch_y_pred_under_sampling, "RandomizedSearcCV\nUnder Sampling")



conf_matrix(axis[2,0], NearMiss_y_test, GridSearch_y_pred_NearMiss, "GridSearcCV\nNearMiss Sampling")

conf_matrix(axis[2,1], NearMiss_y_test, RandomizedSearch_y_pred_NearMiss, "RandomizedSearcCV\nNearMiss Sampling")



conf_matrix(axis[3,0], SMOTE_y_test, GridSearch_y_pred_SMOTE, "GridSearcCV\nSMOTE Sampling")

conf_matrix(axis[3,1], SMOTE_y_test, RandomizedSearch_y_pred_SMOTE, "RandomizedSearcCV\nSMOTE Sampling")



conf_matrix(axis[4,0], ADASYN_y_test, GridSearch_y_pred_ADASYN, "GridSearcCV\nADASYN Sampling")

conf_matrix(axis[4,1], ADASYN_y_test, RandomizedSearch_y_pred_ADASYN, "RandomizedSearcCV\nADASYN Sampling")



fig.tight_layout()
def ROC_Curve_Graph(axis, model, X_test, y_test, title):

    # probbaility of occurrence of each class

    y_pred_prob = model.predict_proba(X_test)



    #print("y_pred_prob shape : ", y_pred_prob.shape)

    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob[: , 0], pos_label = 0)



    # Plot the ROC curve

    fig = plt.figure(figsize=(10,10))          # Create window frame

    ax = axis   # Create axes

    ax.plot(fpr, tpr)           # Plot on the axes

    # Also connect diagonals

    ax.plot([0, 1], [0, 1], ls="--")   # Dashed diagonal line

    # Labels etc

    ax.set_xlabel('False Positive Rate')  # Final plot decorations

    ax.set_ylabel('True Positive Rate')

    ax.set_title(title)



    ax.set_xlim([0.0, 1.0])

    ax.set_ylim([0.0, 1.0])
fig, axis = plt.subplots(5,2,figsize=(10,10))



ROC_Curve_Graph(axis[0, 0], GridSearch_result_without_sampling, Original_X_test, Original_y_test, "GridSearcCV\nWithout Sampling")

ROC_Curve_Graph(axis[0, 1], RandomizedSearch_result_without_sampling, Original_X_test, Original_y_test, "RandomizedSearcCV\nWithout Sampling")



ROC_Curve_Graph(axis[1, 0], GridSearch_result_under_sampling, Under_X_test, Under_y_test, "GridSearcCV\nUnder Sampling")

ROC_Curve_Graph(axis[1, 1], RandomizedSearch_result_under_sampling, Under_X_test, Under_y_test, "RandomizedSearcCV\nUnder Sampling")



ROC_Curve_Graph(axis[2, 0], GridSearch_result_NearMiss, NearMiss_X_test, NearMiss_y_test, "GridSearcCV\nNearMiss Sampling")

ROC_Curve_Graph(axis[2, 1], RandomizedSearch_result_NearMiss, NearMiss_X_test, NearMiss_y_test, "RandomizedSearcCV\nNearMiss Sampling")



ROC_Curve_Graph(axis[3, 0], GridSearch_result_SMOTE, SMOTE_X_test, SMOTE_y_test, "GridSearcCV\nSMOTE Sampling")

ROC_Curve_Graph(axis[3, 1], RandomizedSearch_result_SMOTE, SMOTE_X_test, SMOTE_y_test, "RandomizedSearcCV\nSMOTE Sampling")



ROC_Curve_Graph(axis[4, 0], GridSearch_result_ADASYN, ADASYN_X_test, ADASYN_y_test, "GridSearcCV\nADASYN Sampling")

ROC_Curve_Graph(axis[4, 1], RandomizedSearch_result_ADASYN, ADASYN_X_test, ADASYN_y_test, "RandomizedSearcCV\nADASYN Sampling")

    

fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)