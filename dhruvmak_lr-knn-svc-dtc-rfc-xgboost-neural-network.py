#Importing Libraries



import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import MultinomialNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import BaggingClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split

from imblearn.over_sampling import SMOTE

from sklearn.preprocessing import StandardScaler, RobustScaler

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from xgboost import XGBClassifier

from sklearn.model_selection import cross_val_score

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report

import warnings

import itertools

import keras

from keras import backend as K

from keras.models import Sequential

from keras.layers import Activation

from keras.layers.core import Dense

from keras.optimizers import Adam

from keras.metrics import categorical_crossentropy

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split, RandomizedSearchCV

from sklearn.model_selection import StratifiedKFold

from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline

from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve

from sklearn.model_selection import cross_val_predict

from sklearn.metrics import average_precision_score

from sklearn.metrics import precision_recall_curve

from sklearn.model_selection import ShuffleSplit

from sklearn.model_selection import learning_curve

import time



warnings.filterwarnings("ignore")
#loading data



df = pd.read_csv('../input/creditcard.csv')

df.head()
#descriptive statistics

df.describe()
#shape of data

df.shape
#checking for null values

df.isnull().values.any()
#printing column names

df.columns
#printing number of fraud entries vs number of no fraud entries

print('Number of Entries with class = No Frauds is {}'.format(df['Class'].value_counts()[0]))

print('Number of Entries with class = Frauds is {}'.format(df['Class'].value_counts()[1]))
#printing percentage of fraud entries and no fraud entries

print('Percentage of Entries with class = No Frauds is {}'.format(round(df['Class'].value_counts()[0]/len(df) * 100,2), '% of the dataset'))

print('Percentage of Entries with class = Frauds is {}'.format(round(df['Class'].value_counts()[1]/len(df) * 100,2), '% of the dataset'))
#count plot of class column

sns.countplot('Class', data=df, palette="Set3")

plt.title('Class Distributions \n (0: No Fraud || 1: Fraud)')
#distribution plot of Time 

sns.distplot(df["Time"], hist = False, color="y")

plt.title('Distribution of Transaction Time')

plt.xlim([min(df["Time"]), max(df["Time"])])

plt.show()
#distribution plot of Amount 

sns.distplot(df["Amount"], hist = False, color="b")

plt.title('Distribution of Transaction Amount')

plt.xlim([min(df["Amount"]), max(df["Amount"])])

plt.show()
#Scaling amount and time column and replace original column with scaled column



rob_scaler = RobustScaler()



df['Amount'] = rob_scaler.fit_transform(df['Amount'].values.reshape(-1,1))

df['Time'] = rob_scaler.fit_transform(df['Time'].values.reshape(-1,1))
#printing 5 rows after scaling to check if we have scaled perfectly 

df.head()
#seperating original dataframe for testing purposes

X = df.drop('Class', axis=1)

Y = df['Class']



ss = StratifiedKFold(n_splits = 5, random_state = None, shuffle = False)



for train_index, test_index in ss.split(X, Y):

    print("Train:", train_index, "Test:", test_index)

    original_Xtrain, original_Xtest = X.iloc[train_index], X.iloc[test_index]

    original_ytrain, original_ytest = Y.iloc[train_index], Y.iloc[test_index]

    

#Turning values into array

original_Xtrain = original_Xtrain.values

original_Xtest = original_Xtest.values

original_ytrain = original_ytrain.values

original_ytest = original_ytest.values
#Before performing undersampling or oversampling let's check accuracy with default parameters

#Our data is already scaled we should split our training and test sets now



X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
#Turn the values into an array

X_train = X_train.values

X_test = X_test.values

y_train = y_train.values

y_test = y_test.values
#Initialising classifiers

svc = SVC()

knc = KNeighborsClassifier()

dtc = DecisionTreeClassifier()

lrc = LogisticRegression()

rfc = RandomForestClassifier()

etc = ExtraTreesClassifier()

xgb = XGBClassifier()
clfs = {'SVC' : svc,'KN' : knc, 'DT': dtc, 'LR': lrc, 'RF': rfc, 'ETC': etc, "XGB": xgb}
#function for classifier

def train_classifier(clf, train_X, train_y):

    clf.fit(train_X, train_y)
#function for predictions

def predict_labels(clf, test_X):

    return (clf.predict(test_X))
#calling training and predict function for every classifier

pred_scores = []

for k,v in clfs.items():

    train_classifier(v, X_train, y_train)

    pred = predict_labels(v,X_test)

    pred_scores.append((k, [accuracy_score(y_test,pred)]))
#printing testing accuracy for every classifier

score = pd.DataFrame.from_items(pred_scores,orient='index', columns=['Score'])

score
#Random undersampling of data(Take only 492 Entries of non_fraud data)



fraud_df = df.loc[df['Class'] == 1]

non_fraud_df = df.loc[df['Class'] == 0][:492]



normal_distributed_df = pd.concat([fraud_df, non_fraud_df])



#Shuffle dataframe rows

new_df = normal_distributed_df.sample(frac=1, random_state=42)



#printing first 5 rows after undersampling

new_df.head()
#printing shape of new dataframe

new_df.shape
#printing number of fraud entries vs number of no fraud entries

print('Number of Entries with class = No Frauds is {}'.format(new_df['Class'].value_counts()[0]))

print('Number of Entries with class = Frauds is {}'.format(new_df['Class'].value_counts()[1]))
#Correlation with heat map

corr = new_df.corr()

sns.set_context("notebook", font_scale=1.0, rc={"lines.linewidth": 2.5})

plt.figure(figsize=(23,17))



#create a mask so we only see the correlation values once

mask = np.zeros_like(corr)

mask[np.triu_indices_from(mask, 1)] = True

a = sns.heatmap(corr,mask=mask, annot=True, fmt='.2f')

rotx = a.set_xticklabels(a.get_xticklabels(), rotation=90)

roty = a.set_yticklabels(a.get_yticklabels(), rotation=30)
# -----> V17 removing outliers from fraud transactions

v17_fraud = new_df['V17'].loc[new_df['Class'] == 1].values

q25, q75 = np.percentile(v17_fraud, 25), np.percentile(v17_fraud, 75)

v17_iqr = q75 - q25



v17_cut_off = v17_iqr * 1.5

v17_lower, v17_upper = q25 - v17_cut_off, q75 + v17_cut_off

print('V17 Lower: {}'.format(v17_lower))

print('V17 Upper: {}'.format(v17_upper))

outliers = [x for x in v17_fraud if x < v17_lower or x > v17_upper]

print('V17 outliers: {}'.format(outliers))

print('Feature V17 Outliers for Fraud Cases: {}'.format(len(outliers)))

new_df = new_df.drop(new_df[(new_df['V17'] > v17_upper) | (new_df['V17'] < v17_lower)].index)

print('Number of Instances after outliers removal: {}'.format(len(new_df)))

print('----' * 44)



# -----> V14 Removing Outliers (Highest Negative Correlated with Labels)

v14_fraud = new_df['V14'].loc[new_df['Class'] == 1].values

q25, q75 = np.percentile(v14_fraud, 25), np.percentile(v14_fraud, 75)

print('Quartile 25: {} | Quartile 75: {}'.format(q25, q75))

v14_iqr = q75 - q25

print('iqr: {}'.format(v14_iqr))



v14_cut_off = v14_iqr * 1.5

v14_lower, v14_upper = q25 - v14_cut_off, q75 + v14_cut_off

print('Cut Off: {}'.format(v14_cut_off))

print('V14 Lower: {}'.format(v14_lower))

print('V14 Upper: {}'.format(v14_upper))



outliers = [x for x in v14_fraud if x < v14_lower or x > v14_upper]

print('Feature V14 Outliers for Fraud Cases: {}'.format(len(outliers)))

print('V10 outliers:{}'.format(outliers))



new_df = new_df.drop(new_df[(new_df['V14'] > v14_upper) | (new_df['V14'] < v14_lower)].index)

print('----' * 44)



# -----> V12 removing outliers from fraud transactions

v12_fraud = new_df['V12'].loc[new_df['Class'] == 1].values

q25, q75 = np.percentile(v12_fraud, 25), np.percentile(v12_fraud, 75)

v12_iqr = q75 - q25



v12_cut_off = v12_iqr * 1.5

v12_lower, v12_upper = q25 - v12_cut_off, q75 + v12_cut_off

print('V12 Lower: {}'.format(v12_lower))

print('V12 Upper: {}'.format(v12_upper))

outliers = [x for x in v12_fraud if x < v12_lower or x > v12_upper]

print('V12 outliers: {}'.format(outliers))

print('Feature V12 Outliers for Fraud Cases: {}'.format(len(outliers)))

new_df = new_df.drop(new_df[(new_df['V12'] > v12_upper) | (new_df['V12'] < v12_lower)].index)

print('Number of Instances after outliers removal: {}'.format(len(new_df)))

print('----' * 44)





# Removing outliers V10 Feature

v10_fraud = new_df['V10'].loc[new_df['Class'] == 1].values

q25, q75 = np.percentile(v10_fraud, 25), np.percentile(v10_fraud, 75)

v10_iqr = q75 - q25



v10_cut_off = v10_iqr * 1.5

v10_lower, v10_upper = q25 - v10_cut_off, q75 + v10_cut_off

print('V10 Lower: {}'.format(v10_lower))

print('V10 Upper: {}'.format(v10_upper))

outliers = [x for x in v10_fraud if x < v10_lower or x > v10_upper]

print('V10 outliers: {}'.format(outliers))

print('Feature V10 Outliers for Fraud Cases: {}'.format(len(outliers)))

new_df = new_df.drop(new_df[(new_df['V10'] > v10_upper) | (new_df['V10'] < v10_lower)].index)

print('Number of Instances after outliers removal: {}'.format(len(new_df)))
#Undersampling before cross validating (prone to overfit)

X = new_df.drop('Class', axis=1)

y = new_df['Class']
#Our data is already scaled we should split our training and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#Turn the values into an array

X_train = X_train.values

X_test = X_test.values

y_train = y_train.values

y_test = y_test.values
clf = {'SVC' : svc,'KN' : knc, 'DT': dtc, 'LR': lrc, 'RF': rfc, 'ETC': etc, 'XGB':xgb}
#function for training classifier

def train_classifier(clf, train_X, train_y):    

    clf.fit(train_X, train_y)
#function for predictions

def predict_labels(clf, features):

    return (clf.predict(features))
#confusion matrix for train data

def train_cf(clf, X_train, y_train):

    predi = clf.predict(X_train)

    conf_mat = confusion_matrix(y_train, predi)

    class_label = ["No fraud", "Fraud"]

    df = pd.DataFrame(conf_mat, index = class_label, columns = class_label)

    sns.heatmap(df, annot = True, fmt="d")

    plt.title("Confusion Matrix for train data with {}".format(clf))

    plt.xlabel("Predicted Label")

    plt.ylabel("True Label")

    plt.show()
#confusion matrix for test data

def test_cf(clf, X_test, y_test):

    predi = clf.predict(X_test)

    conf_mat = confusion_matrix(y_test, predi)

    class_label = ["No fraud", "Fraud"]

    df = pd.DataFrame(conf_mat, index = class_label, columns = class_label)

    sns.heatmap(df, annot = True, fmt="d")

    plt.title("Confusion Matrix for test data with {}".format(clf))

    plt.xlabel("Predicted Label")

    plt.ylabel("True Label")

    plt.show()
#function for calling training, ,testing and confusion matrics

pred_scores = []

for k,v in clf.items():

    train_classifier(v, X_train, y_train)

    pred = predict_labels(v,X_test)

    pred_scores.append((k, [accuracy_score(y_test,pred)]))

    train_cf(v, X_train, y_train)

    test_cf(v, X_test, y_test)
#printing testing accuracy

score = pd.DataFrame.from_items(pred_scores,orient='index', columns=['Score'])

score
#Use GridSearchCV to find the best parameters.



#Logistic Regression 

log_reg_params = {"penalty": ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}



grid_log_reg = GridSearchCV(LogisticRegression(), log_reg_params)

grid_log_reg.fit(X_train, y_train)

log_reg = grid_log_reg.best_estimator_



#KNearestNeighbour

knears_params = {"n_neighbors": list(range(1,50,2)), 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}



grid_knears = GridSearchCV(KNeighborsClassifier(), knears_params)

grid_knears.fit(X_train, y_train)

knears_neighbors = grid_knears.best_estimator_



#Support Vector Classifier

svc_params = {'C': [0.5, 0.7, 0.9, 1], 'kernel': ['rbf', 'poly', 'sigmoid', 'linear']}

grid_svc = GridSearchCV(SVC(), svc_params)

grid_svc.fit(X_train, y_train)



svc = grid_svc.best_estimator_



#DecisionTree Classifier

depths = [1, 5, 10, 15]

min_samples_leaf = [2, 5, 10, 15]

tree_params = {"criterion": ["gini", "entropy"], "max_depth": depths, 

              "min_samples_leaf": min_samples_leaf}

grid_tree = GridSearchCV(DecisionTreeClassifier(), tree_params)

grid_tree.fit(X_train, y_train)



tree_clf = grid_tree.best_estimator_



#Random Forest Classifier

base_learners = [20, 40, 60]

depths = [1, 5, 10, 15]



forest_params = {'n_estimators': base_learners, 'max_depth':depths}

grid_forest = GridSearchCV(RandomForestClassifier(max_features='sqrt'), forest_params)

grid_forest.fit(X_train, y_train)



forest_clf = grid_forest.best_estimator_



#GBDt

base_learners = [20, 40, 60]

depths = [1, 5, 10, 15]



gbdt_params = {'n_estimators': base_learners, 'max_depth':depths}

grid_gbdt = GridSearchCV(XGBClassifier(booster = 'gbtree'), gbdt_params)

grid_gbdt.fit(X_train, y_train)



gbdt_clf = grid_gbdt.best_estimator_

# Overfitting Case

log_reg_score = cross_val_score(log_reg, X_train, y_train, cv=5)

print('Logistic Regression Cross Validation Score: ', round(log_reg_score.mean() * 100, 2).astype(str) + '%')



knears_score = cross_val_score(knears_neighbors, X_train, y_train, cv=5)

print('Knears Neighbors Cross Validation Score', round(knears_score.mean() * 100, 2).astype(str) + '%')



svc_score = cross_val_score(svc, X_train, y_train, cv=5)

print('Support Vector Classifier Cross Validation Score', round(svc_score.mean() * 100, 2).astype(str) + '%')



tree_score = cross_val_score(tree_clf, X_train, y_train, cv=5)

print('DecisionTree Classifier Cross Validation Score', round(tree_score.mean() * 100, 2).astype(str) + '%')



forest_score = cross_val_score(forest_clf, X_train, y_train, cv=5)

print('Random Forest Classifier Cross Validation Score', round(forest_score.mean() * 100, 2).astype(str) + '%')



gbdt_score = cross_val_score(gbdt_clf, X_train, y_train, cv=5)

print('gbdt Classifier Cross Validation Score', round(gbdt_score.mean() * 100, 2).astype(str) + '%')
#Plotting Learning Curve



def plot_learning_curve(estimator1, estimator2, estimator3, estimator4, estimator5, estimator6, X, y, ylim=None, cv=None,

                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):

    f, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3,2, figsize=(20,18), sharey=True)

    if ylim is not None:

        plt.ylim(*ylim)

        

    #First Estimator (Logistic Regression)

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

    

    #Second Estimator (K Nearest Neighbour)

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

    

    #Third Estimator (Support Vector Classifier)

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

    ax3.set_title("Support Vector Classifier Learning Curve", fontsize=14)

    ax3.set_xlabel('Training size (m)')

    ax3.set_ylabel('Score')

    ax3.grid(True)

    ax3.legend(loc="best")

    

    #Fourth Estimator (Decision Tree Classifier)

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

    ax4.set_title("Decision Tree Classifier Learning Curve", fontsize=14)

    ax4.set_xlabel('Training size (m)')

    ax4.set_ylabel('Score')

    ax4.grid(True)

    ax4.legend(loc="best")

    

    #Fifth Estimator (Random Forest Classifier)

    train_sizes, train_scores, test_scores = learning_curve(

        estimator5, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)

    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)

    test_scores_std = np.std(test_scores, axis=1)

    ax5.fill_between(train_sizes, train_scores_mean - train_scores_std,

                     train_scores_mean + train_scores_std, alpha=0.1,

                     color="#ff9124")

    ax5.fill_between(train_sizes, test_scores_mean - test_scores_std,

                     test_scores_mean + test_scores_std, alpha=0.1, color="#2492ff")

    ax5.plot(train_sizes, train_scores_mean, 'o-', color="#ff9124",

             label="Training score")

    ax5.plot(train_sizes, test_scores_mean, 'o-', color="#2492ff",

             label="Cross-validation score")

    ax5.set_title("Random Forest Classifier Learning Curve", fontsize=14)

    ax5.set_xlabel('Training size (m)')

    ax5.set_ylabel('Score')

    ax5.grid(True)

    ax5.legend(loc="best")

    

    #Sixth Estimator (XGBoost)

    train_sizes, train_scores, test_scores = learning_curve(

        estimator6, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)

    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)

    test_scores_std = np.std(test_scores, axis=1)

    ax6.fill_between(train_sizes, train_scores_mean - train_scores_std,

                     train_scores_mean + train_scores_std, alpha=0.1,

                     color="#ff9124")

    ax6.fill_between(train_sizes, test_scores_mean - test_scores_std,

                     test_scores_mean + test_scores_std, alpha=0.1, color="#2492ff")

    ax6.plot(train_sizes, train_scores_mean, 'o-', color="#ff9124",

             label="Training score")

    ax6.plot(train_sizes, test_scores_mean, 'o-', color="#2492ff",

             label="Cross-validation score")

    ax6.set_title("GBDT Learning Curve", fontsize=14)

    ax6.set_xlabel('Training size (m)')

    ax6.set_ylabel('Score')

    ax6.grid(True)

    ax6.legend(loc="best")



    return plt
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=42)

plot_learning_curve(log_reg, knears_neighbors, svc, tree_clf, forest_clf, gbdt_clf, X_train, y_train, (0.87, 1.01), cv=cv, n_jobs=4)
#Create a DataFrame with all the scores and the classifiers names.

log_reg_pred = cross_val_predict(log_reg, X_train, y_train, cv=5, method="decision_function")

knears_pred = cross_val_predict(knears_neighbors, X_train, y_train, cv=5)

svc_pred = cross_val_predict(svc, X_train, y_train, cv=5, method="decision_function")

tree_pred = cross_val_predict(tree_clf, X_train, y_train, cv=5)

forest_pred = cross_val_predict(forest_clf, X_train, y_train, cv=5)

gbdt_pred = cross_val_predict(gbdt_clf, X_train, y_train, cv=5)
print('Logistic Regression: ', roc_auc_score(y_train, log_reg_pred))

print('KNears Neighbors: ', roc_auc_score(y_train, knears_pred))

print('Support Vector Classifier: ', roc_auc_score(y_train, svc_pred))

print('Decision Tree Classifier: ', roc_auc_score(y_train, tree_pred))

print("Random Forest Classifier: ",roc_auc_score(y_train, tree_pred))

print("GBDT: ",roc_auc_score(y_train, gbdt_pred))
print('Length of X (train): {} | Length of y (train): {}'.format(len(original_Xtrain), len(original_ytrain)))

print('Length of X (test): {} | Length of y (test): {}'.format(len(original_Xtest), len(original_ytest)))



#List to append the score and then find the average

accuracy_lst = []

precision_lst = []

recall_lst = []

f1_lst = []

auc_lst = []



#Classifier with optimal parameters

#log_reg_sm = grid_log_reg.best_estimator_

log_reg_sm = LogisticRegression()

rand_log_reg = RandomizedSearchCV(LogisticRegression(), log_reg_params, n_iter=4)



#Implementing SMOTE Technique 

#Cross Validating the right way

#Parameters

log_reg_params = {"penalty": ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

for train, test in ss.split(original_Xtrain, original_ytrain):

    pipeline = imbalanced_make_pipeline(SMOTE(sampling_strategy='minority'), rand_log_reg) # SMOTE happens during Cross Validation not before..

    model = pipeline.fit(original_Xtrain[train], original_ytrain[train])

    best_est = rand_log_reg.best_estimator_

    prediction = best_est.predict(original_Xtrain[test])

    

    accuracy_lst.append(pipeline.score(original_Xtrain[test], original_ytrain[test]))

    precision_lst.append(precision_score(original_ytrain[test], prediction))

    recall_lst.append(recall_score(original_ytrain[test], prediction))

    f1_lst.append(f1_score(original_ytrain[test], prediction))

    auc_lst.append(roc_auc_score(original_ytrain[test], prediction))

    

print('---' * 45)

print('')

print("accuracy: {}".format(np.mean(accuracy_lst)))

print("precision: {}".format(np.mean(precision_lst)))

print("recall: {}".format(np.mean(recall_lst)))

print("f1: {}".format(np.mean(f1_lst)))

print('---' * 45)
labels = ['No Fraud', 'Fraud']

smote_prediction = best_est.predict(original_Xtest)

print(classification_report(original_ytest, smote_prediction, target_names=labels))
y_score = best_est.decision_function(original_Xtest)
average_precision = average_precision_score(original_ytest, y_score)

print('Average precision-recall score: {0:0.2f}'.format(average_precision))
fig = plt.figure(figsize=(12,6))

precision, recall, _ = precision_recall_curve(original_ytest, y_score)

plt.step(recall, precision, color='r', alpha=0.2, where='post')

plt.fill_between(recall, precision, step='post', alpha=0.2, color='#F59B00')

plt.xlabel('Recall')

plt.ylabel('Precision')

plt.ylim([0.0, 1.05])

plt.xlim([0.0, 1.0])

plt.title('OverSampling Precision-Recall curve: \n Average Precision-Recall Score ={0:0.2f}'.format(

    average_precision), fontsize=16)
#SMOTE Technique (OverSampling) After splitting and Cross Validating

sm = SMOTE(ratio='minority', random_state=42)



#This will be the data were we are going to 

Xsm_train, ysm_train = sm.fit_sample(original_Xtrain, original_ytrain)
#We Improve the score by 2% points approximately Implement GridSearchCV and the other models.

#Logistic Regression

t0 = time.time()

log_reg_sm = grid_log_reg.best_estimator_

log_reg_sm.fit(Xsm_train, ysm_train)

t1 = time.time()

print("Fitting oversample data took :{} sec".format(t1 - t0))
#Logistic Regression fitted using SMOTE technique

y_pred_log_reg = log_reg_sm.predict(X_test)



#Other models fitted with UnderSampling

y_pred_knear = knears_neighbors.predict(X_test)

y_pred_svc = svc.predict(X_test)

y_pred_tree = tree_clf.predict(X_test)



log_reg_cf = confusion_matrix(y_test, y_pred_log_reg)

kneighbors_cf = confusion_matrix(y_test, y_pred_knear)

svc_cf = confusion_matrix(y_test, y_pred_svc)

tree_cf = confusion_matrix(y_test, y_pred_tree)



fig, ax = plt.subplots(2, 2,figsize=(22,12))





sns.heatmap(log_reg_cf, ax=ax[0][0], annot=True, cmap=plt.cm.copper)

ax[0, 0].set_title("Logistic Regression \n Confusion Matrix", fontsize=14)

ax[0, 0].set_xticklabels(['', ''], fontsize=14, rotation=90)

ax[0, 0].set_yticklabels(['', ''], fontsize=14, rotation=360)



sns.heatmap(kneighbors_cf, ax=ax[0][1], annot=True, cmap=plt.cm.copper)

ax[0][1].set_title("KNearsNeighbors \n Confusion Matrix", fontsize=14)

ax[0][1].set_xticklabels(['', ''], fontsize=14, rotation=90)

ax[0][1].set_yticklabels(['', ''], fontsize=14, rotation=360)



sns.heatmap(svc_cf, ax=ax[1][0], annot=True, cmap=plt.cm.copper)

ax[1][0].set_title("Suppor Vector Classifier \n Confusion Matrix", fontsize=14)

ax[1][0].set_xticklabels(['', ''], fontsize=14, rotation=90)

ax[1][0].set_yticklabels(['', ''], fontsize=14, rotation=360)



sns.heatmap(tree_cf, ax=ax[1][1], annot=True, cmap=plt.cm.copper)

ax[1][1].set_title("DecisionTree Classifier \n Confusion Matrix", fontsize=14)

ax[1][1].set_xticklabels(['', ''], fontsize=14, rotation=90)

ax[1][1].set_yticklabels(['', ''], fontsize=14, rotation=360)





plt.show()
print('Logistic Regression:')

print(classification_report(y_test, y_pred_log_reg))



print('KNears Neighbors:')

print(classification_report(y_test, y_pred_knear))



print('Support Vector Classifier:')

print(classification_report(y_test, y_pred_svc))



print('Decision Tree Classifier:')

print(classification_report(y_test, y_pred_tree))
#Final Score in the test set of logistic regression



#Logistic Regression with Under-Sampling

y_pred = log_reg.predict(X_test)

undersample_score = accuracy_score(y_test, y_pred)







#Logistic Regression with SMOTE Technique (Better accuracy with SMOTE t)

y_pred_sm = best_est.predict(original_Xtest)

oversample_score = accuracy_score(original_ytest, y_pred_sm)





d = {'Technique': ['Random UnderSampling', 'Oversampling (SMOTE)'], 'Score': [undersample_score, oversample_score]}

final_df = pd.DataFrame(data=d)



#Move column

score = final_df['Score']

final_df.drop('Score', axis=1, inplace=True)

final_df.insert(1, 'Score', score)



#Note how high is accuracy score it can be misleading! 

final_df
#Neural Network

n_inputs = X_train.shape[1]



undersample_model = Sequential([

    Dense(n_inputs, input_shape=(n_inputs, ), activation='relu'),

    Dense(32, activation='relu'),

    Dense(2, activation='softmax')

])
undersample_model.summary()
undersample_model.compile(Adam(lr=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
undersample_model.fit(X_train, y_train, validation_split=0.2, batch_size=25, epochs=20, shuffle=True, verbose=1)
n_inputs = Xsm_train.shape[1]



oversample_model = Sequential([

    Dense(n_inputs, input_shape=(n_inputs, ), activation='relu'),

    Dense(32, activation='relu'),

    Dense(2, activation='softmax')

])
oversample_model.compile(Adam(lr=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
oversample_model.fit(Xsm_train, ysm_train, validation_split=0.2, batch_size=300, epochs=20, shuffle=True, verbose=1)