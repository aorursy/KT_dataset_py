# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# Imported Libraries



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.manifold import TSNE

from sklearn.decomposition import PCA, TruncatedSVD

import matplotlib.patches as mpatches

import time



# Classifier Libraries

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

import collections





# Other Libraries

from sklearn.model_selection import train_test_split

from sklearn.pipeline import make_pipeline

from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline

from imblearn.over_sampling import SMOTE

from imblearn.under_sampling import NearMiss

from imblearn.metrics import classification_report_imbalanced

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report

from collections import Counter

from sklearn.model_selection import KFold, StratifiedKFold

import warnings

warnings.filterwarnings("ignore")





df = pd.read_csv('../input/creditcard.csv')

df.head()
df.describe()
# Good No Null Values!

df.isnull().sum().max()
df.columns
# The classes are heavily skewed we need to solve this issue later.

print('No Frauds', round(df['Class'].value_counts()[0]/len(df) * 100,2), '% of the dataset')

print('Frauds', round(df['Class'].value_counts()[1]/len(df) * 100,2), '% of the dataset')
colors = ["#0101DF", "#DF0101"]



sns.countplot('Class', data=df, palette=colors)

plt.title('Class Distributions \n (0: No Fraud || 1: Fraud)', fontsize=14)
fig, ax = plt.subplots(1, 2, figsize=(18,4))



amount_val = df['Amount'].values

time_val = df['Time'].values



sns.distplot(amount_val, ax=ax[0], color='r')

ax[0].set_title('Distribution of Transaction Amount', fontsize=14)

ax[0].set_xlim([min(amount_val), max(amount_val)])



sns.distplot(time_val, ax=ax[1], color='b')

ax[1].set_title('Distribution of Transaction Time', fontsize=14)

ax[1].set_xlim([min(time_val), max(time_val)])







plt.show()
# Since most of our data has already been scaled we should scale the columns that are left to scale (Amount and Time)

from sklearn.preprocessing import StandardScaler, RobustScaler



# RobustScaler is less prone to outliers.



std_scaler = StandardScaler()

rob_scaler = RobustScaler()



df['scaled_amount'] = rob_scaler.fit_transform(df['Amount'].values.reshape(-1,1))

df['scaled_time'] = rob_scaler.fit_transform(df['Time'].values.reshape(-1,1))



df.drop(['Time','Amount'], axis=1, inplace=True)
scaled_amount = df['scaled_amount']

scaled_time = df['scaled_time']



df.drop(['scaled_amount', 'scaled_time'], axis=1, inplace=True)

df.insert(0, 'scaled_amount', scaled_amount)

df.insert(1, 'scaled_time', scaled_time)



# Amount and Time are Scaled!



df.head()
from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedShuffleSplit



print('No Frauds', round(df['Class'].value_counts()[0]/len(df) * 100,2), '% of the dataset')

print('Frauds', round(df['Class'].value_counts()[1]/len(df) * 100,2), '% of the dataset')



X = df.drop('Class', axis=1)

y = df['Class']



sss = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)



for train_index, test_index in sss.split(X, y):

    print("Train:", train_index, "Test:", test_index)

    original_Xtrain, original_Xtest = X.iloc[train_index], X.iloc[test_index]

    original_ytrain, original_ytest = y.iloc[train_index], y.iloc[test_index]



# We already have X_train and y_train for undersample data thats why I am using original to distinguish and to not overwrite these variables.

# original_Xtrain, original_Xtest, original_ytrain, original_ytest = train_test_split(X, y, test_size=0.2, random_state=42)



# Check the Distribution of the labels





# Turn into an array

original_Xtrain = original_Xtrain.values

original_Xtest = original_Xtest.values

original_ytrain = original_ytrain.values

original_ytest = original_ytest.values



# See if both the train and test label distribution are similarly distributed

train_unique_label, train_counts_label = np.unique(original_ytrain, return_counts=True)

test_unique_label, test_counts_label = np.unique(original_ytest, return_counts=True)

print('-' * 100)



print('Label Distributions: \n')

print(train_counts_label/ len(original_ytrain))

print(test_counts_label/ len(original_ytest))
# Since our classes are highly skewed we should make them equivalent in order to have a normal distribution of the classes.



# Lets shuffle the data before creating the subsamples



df = df.sample(frac=1)



# amount of fraud classes 492 rows.

fraud_df = df.loc[df['Class'] == 1]

non_fraud_df = df.loc[df['Class'] == 0][:492]



normal_distributed_df = pd.concat([fraud_df, non_fraud_df])



# Shuffle dataframe rows

new_df = normal_distributed_df.sample(frac=1, random_state=42)



new_df.head()
print('Distribution of the Classes in the subsample dataset')

print(new_df['Class'].value_counts()/len(new_df))







sns.countplot('Class', data=new_df, palette=colors)

plt.title('Equally Distributed Classes', fontsize=14)

plt.show()
# Undersampling before cross validating (prone to overfit)

X = new_df.drop('Class', axis=1)

y = new_df['Class']
# Our data is already scaled we should split our training and test sets

from sklearn.model_selection import train_test_split



# This is explicitly used for undersampling.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Turn the values into an array for feeding the classification algorithms.

X_train = X_train.values

X_test = X_test.values

y_train = y_train.values

y_test = y_test.values
# Let's implement simple classifiers



classifiers = {

    "LogisiticRegression": LogisticRegression(),

    "KNearest": KNeighborsClassifier(),

    "Support Vector Classifier": SVC(),

    "DecisionTreeClassifier": DecisionTreeClassifier()

}
# Wow our scores are getting even high scores even when applying cross validation.

from sklearn.model_selection import cross_val_score





for key, classifier in classifiers.items():

    classifier.fit(X_train, y_train)

    training_score = cross_val_score(classifier, X_train, y_train, cv=5)

    print("Classifiers: ", classifier.__class__.__name__, "Has a training score of", round(training_score.mean(), 2) * 100, "% accuracy score")
# Use GridSearchCV to find the best parameters.

from sklearn.model_selection import GridSearchCV





# Logistic Regression 

log_reg_params = {"penalty": ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}







grid_log_reg = GridSearchCV(LogisticRegression(), log_reg_params)

grid_log_reg.fit(X_train, y_train)

# We automatically get the logistic regression with the best parameters.

log_reg = grid_log_reg.best_estimator_



knears_params = {"n_neighbors": list(range(2,5,1)), 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}



grid_knears = GridSearchCV(KNeighborsClassifier(), knears_params)

grid_knears.fit(X_train, y_train)

# KNears best estimator

knears_neighbors = grid_knears.best_estimator_



# Support Vector Classifier

svc_params = {'C': [0.5, 0.7, 0.9, 1], 'kernel': ['rbf', 'poly', 'sigmoid', 'linear']}

grid_svc = GridSearchCV(SVC(), svc_params)

grid_svc.fit(X_train, y_train)



# SVC best estimator

svc = grid_svc.best_estimator_



# DecisionTree Classifier

tree_params = {"criterion": ["gini", "entropy"], "max_depth": list(range(2,4,1)), 

              "min_samples_leaf": list(range(5,7,1))}

grid_tree = GridSearchCV(DecisionTreeClassifier(), tree_params)

grid_tree.fit(X_train, y_train)



# tree best estimator

tree_clf = grid_tree.best_estimator_
# Overfitting Case



log_reg_score = cross_val_score(log_reg, X_train, y_train, cv=5)

print('Logistic Regression Cross Validation Score: ', round(log_reg_score.mean() * 100, 2).astype(str) + '%')





knears_score = cross_val_score(knears_neighbors, X_train, y_train, cv=5)

print('Knears Neighbors Cross Validation Score', round(knears_score.mean() * 100, 2).astype(str) + '%')



svc_score = cross_val_score(svc, X_train, y_train, cv=5)

print('Support Vector Classifier Cross Validation Score', round(svc_score.mean() * 100, 2).astype(str) + '%')



tree_score = cross_val_score(tree_clf, X_train, y_train, cv=5)

print('DecisionTree Classifier Cross Validation Score', round(tree_score.mean() * 100, 2).astype(str) + '%')
# We will undersample during cross validating

undersample_X = df.drop('Class', axis=1)

undersample_y = df['Class']



for train_index, test_index in sss.split(undersample_X, undersample_y):

    print("Train:", train_index, "Test:", test_index)

    undersample_Xtrain, undersample_Xtest = undersample_X.iloc[train_index], undersample_X.iloc[test_index]

    undersample_ytrain, undersample_ytest = undersample_y.iloc[train_index], undersample_y.iloc[test_index]

    

undersample_Xtrain = undersample_Xtrain.values

undersample_Xtest = undersample_Xtest.values

undersample_ytrain = undersample_ytrain.values

undersample_ytest = undersample_ytest.values 



undersample_accuracy = []

undersample_precision = []

undersample_recall = []

undersample_f1 = []

undersample_auc = []



# Implementing NearMiss Technique 

# Distribution of NearMiss (Just to see how it distributes the labels we won't use these variables)

X_nearmiss, y_nearmiss = NearMiss().fit_sample(undersample_X.values, undersample_y.values)

print('NearMiss Label Distribution: {}'.format(Counter(y_nearmiss)))

# Cross Validating the right way



for train, test in sss.split(undersample_Xtrain, undersample_ytrain):

    undersample_pipeline = imbalanced_make_pipeline(NearMiss(sampling_strategy='majority'), log_reg) # SMOTE happens during Cross Validation not before..

    undersample_model = undersample_pipeline.fit(undersample_Xtrain[train], undersample_ytrain[train])

    undersample_prediction = undersample_model.predict(undersample_Xtrain[test])

    

    undersample_accuracy.append(undersample_pipeline.score(original_Xtrain[test], original_ytrain[test]))

    undersample_precision.append(precision_score(original_ytrain[test], undersample_prediction))

    undersample_recall.append(recall_score(original_ytrain[test], undersample_prediction))

    undersample_f1.append(f1_score(original_ytrain[test], undersample_prediction))

    undersample_auc.append(roc_auc_score(original_ytrain[test], undersample_prediction))
# Let's Plot LogisticRegression Learning Curve

from sklearn.model_selection import ShuffleSplit

from sklearn.model_selection import learning_curve



def plot_learning_curve(estimator1, estimator2, estimator3, estimator4, X, y, ylim=None, cv=None,

                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):

    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(20,14), sharey=True)

    if ylim is not None:

        plt.ylim(*ylim)

    # First Estimator

    train_sizes, train_scores, test_scores = learning_curve(

        estimator1, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)

    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)

    test_scores_std = np.std(test_scores, axis=1)

    ax1.fill_between(train_sizes, train_scores_mean - train_scores_std,

                     train_scores_mean + train_scores_std, alpha=0.1,

                     color="#ff9124")

    ax1.fill_between(train_sizes, test_scores_mean - test_scores_std,

                     test_scores_mean + test_scores_std, alpha=0.1, color="#2492ff")

    ax1.plot(train_sizes, train_scores_mean, 'o-', color="#ff9124",

             label="Training score")

    ax1.plot(train_sizes, test_scores_mean, 'o-', color="#2492ff",

             label="Cross-validation score")

    ax1.set_title("Logistic Regression Learning Curve", fontsize=14)

    ax1.set_xlabel('Training size (m)')

    ax1.set_ylabel('Score')

    ax1.grid(True)

    ax1.legend(loc="best")

    

    # Second Estimator 

    train_sizes, train_scores, test_scores = learning_curve(

        estimator2, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)

    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)

    test_scores_std = np.std(test_scores, axis=1)

    ax2.fill_between(train_sizes, train_scores_mean - train_scores_std,

                     train_scores_mean + train_scores_std, alpha=0.1,

                     color="#ff9124")

    ax2.fill_between(train_sizes, test_scores_mean - test_scores_std,

                     test_scores_mean + test_scores_std, alpha=0.1, color="#2492ff")

    ax2.plot(train_sizes, train_scores_mean, 'o-', color="#ff9124",

             label="Training score")

    ax2.plot(train_sizes, test_scores_mean, 'o-', color="#2492ff",

             label="Cross-validation score")

    ax2.set_title("Knears Neighbors Learning Curve", fontsize=14)

    ax2.set_xlabel('Training size (m)')

    ax2.set_ylabel('Score')

    ax2.grid(True)

    ax2.legend(loc="best")

    

    # Third Estimator

    train_sizes, train_scores, test_scores = learning_curve(

        estimator3, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)

    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)

    test_scores_std = np.std(test_scores, axis=1)

    ax3.fill_between(train_sizes, train_scores_mean - train_scores_std,

                     train_scores_mean + train_scores_std, alpha=0.1,

                     color="#ff9124")

    ax3.fill_between(train_sizes, test_scores_mean - test_scores_std,

                     test_scores_mean + test_scores_std, alpha=0.1, color="#2492ff")

    ax3.plot(train_sizes, train_scores_mean, 'o-', color="#ff9124",

             label="Training score")

    ax3.plot(train_sizes, test_scores_mean, 'o-', color="#2492ff",

             label="Cross-validation score")

    ax3.set_title("Support Vector Classifier \n Learning Curve", fontsize=14)

    ax3.set_xlabel('Training size (m)')

    ax3.set_ylabel('Score')

    ax3.grid(True)

    ax3.legend(loc="best")

    

    # Fourth Estimator

    train_sizes, train_scores, test_scores = learning_curve(

        estimator4, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)

    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)

    test_scores_std = np.std(test_scores, axis=1)

    ax4.fill_between(train_sizes, train_scores_mean - train_scores_std,

                     train_scores_mean + train_scores_std, alpha=0.1,

                     color="#ff9124")

    ax4.fill_between(train_sizes, test_scores_mean - test_scores_std,

                     test_scores_mean + test_scores_std, alpha=0.1, color="#2492ff")

    ax4.plot(train_sizes, train_scores_mean, 'o-', color="#ff9124",

             label="Training score")

    ax4.plot(train_sizes, test_scores_mean, 'o-', color="#2492ff",

             label="Cross-validation score")

    ax4.set_title("Decision Tree Classifier \n Learning Curve", fontsize=14)

    ax4.set_xlabel('Training size (m)')

    ax4.set_ylabel('Score')

    ax4.grid(True)

    ax4.legend(loc="best")

    return plt
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=42)

plot_learning_curve(log_reg, knears_neighbors, svc, tree_clf, X_train, y_train, (0.87, 1.01), cv=cv, n_jobs=4)
from sklearn.metrics import roc_curve

from sklearn.model_selection import cross_val_predict

# Create a DataFrame with all the scores and the classifiers names.



log_reg_pred = cross_val_predict(log_reg, X_train, y_train, cv=5,

                             method="decision_function")



knears_pred = cross_val_predict(knears_neighbors, X_train, y_train, cv=5)



svc_pred = cross_val_predict(svc, X_train, y_train, cv=5,

                             method="decision_function")



tree_pred = cross_val_predict(tree_clf, X_train, y_train, cv=5)
from sklearn.metrics import roc_auc_score



print('Logistic Regression: ', roc_auc_score(y_train, log_reg_pred))

print('KNears Neighbors: ', roc_auc_score(y_train, knears_pred))

print('Support Vector Classifier: ', roc_auc_score(y_train, svc_pred))

print('Decision Tree Classifier: ', roc_auc_score(y_train, tree_pred))
log_fpr, log_tpr, log_thresold = roc_curve(y_train, log_reg_pred)

knear_fpr, knear_tpr, knear_threshold = roc_curve(y_train, knears_pred)

svc_fpr, svc_tpr, svc_threshold = roc_curve(y_train, svc_pred)

tree_fpr, tree_tpr, tree_threshold = roc_curve(y_train, tree_pred)





def graph_roc_curve_multiple(log_fpr, log_tpr, knear_fpr, knear_tpr, svc_fpr, svc_tpr, tree_fpr, tree_tpr):

    plt.figure(figsize=(16,8))

    plt.title('ROC Curve \n Top 4 Classifiers', fontsize=18)

    plt.plot(log_fpr, log_tpr, label='Logistic Regression Classifier Score: {:.4f}'.format(roc_auc_score(y_train, log_reg_pred)))

    plt.plot(knear_fpr, knear_tpr, label='KNears Neighbors Classifier Score: {:.4f}'.format(roc_auc_score(y_train, knears_pred)))

    plt.plot(svc_fpr, svc_tpr, label='Support Vector Classifier Score: {:.4f}'.format(roc_auc_score(y_train, svc_pred)))

    plt.plot(tree_fpr, tree_tpr, label='Decision Tree Classifier Score: {:.4f}'.format(roc_auc_score(y_train, tree_pred)))

    plt.plot([0, 1], [0, 1], 'k--')

    plt.axis([-0.01, 1, 0, 1])

    plt.xlabel('False Positive Rate', fontsize=16)

    plt.ylabel('True Positive Rate', fontsize=16)

    plt.annotate('Minimum ROC Score of 50% \n (This is the minimum score to get)', xy=(0.5, 0.5), xytext=(0.6, 0.3),

                arrowprops=dict(facecolor='#6E726D', shrink=0.05),

                )

    plt.legend()

    

graph_roc_curve_multiple(log_fpr, log_tpr, knear_fpr, knear_tpr, svc_fpr, svc_tpr, tree_fpr, tree_tpr)

plt.show()
def logistic_roc_curve(log_fpr, log_tpr):

    plt.figure(figsize=(12,8))

    plt.title('Logistic Regression ROC Curve', fontsize=16)

    plt.plot(log_fpr, log_tpr, 'b-', linewidth=2)

    plt.plot([0, 1], [0, 1], 'r--')

    plt.xlabel('False Positive Rate', fontsize=16)

    plt.ylabel('True Positive Rate', fontsize=16)

    plt.axis([-0.01,1,0,1])

    

    

logistic_roc_curve(log_fpr, log_tpr)

plt.show()
from sklearn.metrics import precision_recall_curve



precision, recall, threshold = precision_recall_curve(y_train, log_reg_pred)
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score

y_pred = log_reg.predict(X_train)



# Overfitting Case

print('---' * 45)

print('Overfitting: \n')

print('Recall Score: {:.2f}'.format(recall_score(y_train, y_pred)))

print('Precision Score: {:.2f}'.format(precision_score(y_train, y_pred)))

print('F1 Score: {:.2f}'.format(f1_score(y_train, y_pred)))

print('Accuracy Score: {:.2f}'.format(accuracy_score(y_train, y_pred)))

print('---' * 45)



# How it should look like

print('---' * 45)

print('How it should be:\n')

print("Accuracy Score: {:.2f}".format(np.mean(undersample_accuracy)))

print("Precision Score: {:.2f}".format(np.mean(undersample_precision)))

print("Recall Score: {:.2f}".format(np.mean(undersample_recall)))

print("F1 Score: {:.2f}".format(np.mean(undersample_f1)))

print('---' * 45)
undersample_y_score = log_reg.decision_function(original_Xtest)
from sklearn.metrics import average_precision_score



undersample_average_precision = average_precision_score(original_ytest, undersample_y_score)



print('Average precision-recall score: {0:0.2f}'.format(

      undersample_average_precision))
from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import BaggingClassifier

from sklearn.ensemble import ExtraTreesClassifier



ensemble = {

    "RandomForest": RandomForestClassifier(),

    "GradientBoosting": GradientBoostingClassifier(),

    "Bagging": BaggingClassifier(),

    "ExtraTree": ExtraTreesClassifier()

}
from sklearn.model_selection import cross_val_score





for key, ensemble in ensemble.items():

    ensemble.fit(X_train, y_train)

    training_score = cross_val_score(classifier, X_train, y_train, cv=5)

    print("Ensemble: ", ensemble.__class__.__name__, "Has a training score of", round(training_score.mean(), 2) * 100, "% accuracy score")
# Random Forest Classifier 

param_rfc={

            'max_depth': [3, None],

            'n_estimators': (10, 30, 50, 100, 200, 400, 600, 800, 1000),

            'max_features': (2,4,6)

}

grid_rfc = GridSearchCV(RandomForestClassifier(), param_rfc)

grid_rfc.fit(X_train,y_train)

rfc = grid_rfc.best_estimator_



# Gradient Boosting Classifier

param_gbc = {

    'n_estimators': (10, 30, 50, 100, 200, 400, 600, 800, 1000),

    'max_depth': [1,2,3,4,5]

}

grid_gbc = GridSearchCV(GradientBoostingClassifier(), param_gbc)

grid_gbc.fit(X_train, y_train)

gbc = grid_gbc.best_estimator_



#Bagging Classifer

param_bagg = {

    'base_estimator__max_depth' : [1, 2, 3, 4, 5],

    'max_samples' : [0.05, 0.1, 0.2, 0.5]

}



grid_bagg = GridSearchCV(BaggingClassifier(DecisionTreeClassifier(),

                                     n_estimators = 100, max_features = 0.5),

                   param_bagg)

grid_bagg.fit(X_train, y_train)

bagg = grid_bagg.best_estimator_
# Overfitting Case



rfc_score = cross_val_score(rfc, X_train, y_train, cv=5)

rfc_score.mean()

gbc_score = cross_val_score(gbc, X_train, y_train, cv=5)

gbc_score.mean()
bagg_score = cross_val_score(bagg, X_train, y_train, cv=5)

bagg_score.mean()