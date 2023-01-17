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

from xgboost import XGBClassifier

import collections

import imblearn

from sklearn.ensemble import IsolationForest

from sklearn.metrics import make_scorer, f1_score

from sklearn import model_selection

from sklearn.datasets import make_classification

from sklearn.neighbors import LocalOutlierFactor

from pylab import rcParams

rcParams['figure.figsize'] = 14, 8

RANDOM_SEED = 42

LABELS = ["Normal", "Fraud"]



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
import pandas as pd

data = pd.read_csv("../input/creditcardfraud/creditcard.csv")

data.head()

data.isnull().sum().max()

print('Legitimate transactions are', round(data['Class'].value_counts()[0]/len(data) * 100,2), '% of the dataset')

print('Fraud transactions are', round(data['Class'].value_counts()[1]/len(data) * 100,2), '% of the dataset')
#colors = ["#DF0101", "#0101DF"]

colors = ['red', 'gold']

explode = (0.1, 0)  # explode 1st slice



#sns.countplot('Class', data=data, palette=colors)

labels = ['Fraud (0.17)', 'Legit (99.83)']

sizes = [492,284315]

plt.pie(sizes,explode = explode, colors=colors, labels = labels,shadow=True, startangle=90)

#plt.title('Distribution of 2 classes \n (0: Legit || 1: Fraud)', fontsize=10)





# Scaling the Time and Amount features

from sklearn.preprocessing import StandardScaler, RobustScaler



std_scaler = StandardScaler()

rob_scaler = RobustScaler()



data['scaled_amount'] = std_scaler.fit_transform(data['Amount'].values.reshape(-1,1))

data['scaled_time'] = std_scaler.fit_transform(data['Time'].values.reshape(-1,1))



data.drop(['Time','Amount'], axis=1, inplace=True)

scaled_amount = data['scaled_amount']

scaled_time = data['scaled_time']



data.drop(['scaled_amount', 'scaled_time'], axis=1, inplace=True)

data.insert(0, 'scaled_amount', scaled_amount)

data.insert(1, 'scaled_time', scaled_time)



# Amount and Time are Scaled!



data.head()
from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedShuffleSplit



print('No Frauds', round(data['Class'].value_counts()[0]/len(data) * 100,2), '% of the dataset')

print('Frauds', round(data['Class'].value_counts()[1]/len(data) * 100,2), '% of the dataset')



X = data.drop('Class', axis=1)

y = data['Class']



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
# To improve the accuracy of the model, we can remove those features that are highly

# correlated with the class and are extreme outliers. We can change the threshold 

# to detect the outliers

from scipy.stats import norm



f, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(20, 6))



v14_fraud_dist = data['V14'].loc[data['Class'] == 1].values

sns.distplot(v14_fraud_dist,ax=ax1, fit=norm, color='#FB8861')

ax1.set_title('V14 Distribution \n (Fraud Transactions)', fontsize=14)



v12_fraud_dist = data['V12'].loc[data['Class'] == 1].values

sns.distplot(v12_fraud_dist,ax=ax2, fit=norm, color='#56F9BB')

ax2.set_title('V12 Distribution \n (Fraud Transactions)', fontsize=14)





v17_fraud_dist = data['V17'].loc[data['Class'] == 1].values

sns.distplot(v17_fraud_dist,ax=ax3, fit=norm, color='#C5B3F9')

ax3.set_title('V17 Distribution \n (Fraud Transactions)', fontsize=14)



plt.show()
# # -----> V14 Removing Outliers (Highest Negative Correlated with Labels)

v14_fraud = data['V14'].loc[data['Class'] == 1].values

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



data = data.drop(data[(data['V14'] > v14_upper) | (data['V14'] < v14_lower)].index)

print('----' * 44)



# -----> V12 removing outliers from fraud transactions

v12_fraud = data['V12'].loc[data['Class'] == 1].values

q25, q75 = np.percentile(v12_fraud, 25), np.percentile(v12_fraud, 75)

v12_iqr = q75 - q25



v12_cut_off = v12_iqr * 1.5

v12_lower, v12_upper = q25 - v12_cut_off, q75 + v12_cut_off

print('V12 Lower: {}'.format(v12_lower))

print('V12 Upper: {}'.format(v12_upper))

outliers = [x for x in v12_fraud if x < v12_lower or x > v12_upper]

print('V12 outliers: {}'.format(outliers))

print('Feature V12 Outliers for Fraud Cases: {}'.format(len(outliers)))

data = data.drop(data[(data['V12'] > v12_upper) | (data['V12'] < v12_lower)].index)

print('Number of Instances after outliers removal: {}'.format(len(data)))

print('----' * 44)





# Removing outliers V17 Feature

v17_fraud = data['V17'].loc[data['Class'] == 1].values

q25, q75 = np.percentile(v17_fraud, 25), np.percentile(v17_fraud, 75)

v17_iqr = q75 - q25



v17_cut_off = v17_iqr * 1.5

v17_lower, v17_upper = q25 - v17_cut_off, q75 + v17_cut_off

print('V17 Lower: {}'.format(v17_lower))

print('V17 Upper: {}'.format(v17_upper))

outliers = [x for x in v17_fraud if x < v17_lower or x > v17_upper]

print('V17 outliers: {}'.format(outliers))

print('Feature V17 Outliers for Fraud Cases: {}'.format(len(outliers)))

data = data.drop(data[(data['V17'] > v17_upper) | (data['V17'] < v17_lower)].index)

print('Number of Instances after outliers removal: {}'.format(len(data)))



# Boxplots with outliers removed



f,(ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20,6))



colors = ['#B3F9C5', '#f9c5b3']

# Feature V14

sns.boxplot(x="Class", y="V14", data=data,ax=ax1, palette=colors)

ax1.set_title("V14 Feature \n Reduction of outliers", fontsize=14)

ax1.annotate('Fewer extreme \n outliers', xy=(0.98, -17.5), xytext=(0, -12),

            arrowprops=dict(facecolor='black'),

            fontsize=14)



# Feature 12

sns.boxplot(x="Class", y="V12", data=data, ax=ax2, palette=colors)

ax2.set_title("V12 Feature \n Reduction of outliers", fontsize=14)

ax2.annotate('Fewer extreme \n outliers', xy=(0.98, -17.3), xytext=(0, -12),

            arrowprops=dict(facecolor='black'),

            fontsize=14)



# Feature V17

sns.boxplot(x="Class", y="V17", data=data, ax=ax3, palette=colors)

ax3.set_title("V17 Feature \n Reduction of outliers", fontsize=14)

ax3.annotate('Fewer extreme \n outliers', xy=(0.95, -16.5), xytext=(0, -12),

            arrowprops=dict(facecolor='black'),

            fontsize=14)





plt.show()
# Handling imbalanced structure of the data with undersampling the overpresented class 

# Just for building the model not for testing!(avoiding overfitting)

# I shuffle the data before creating the sub sample

data = data.sample(frac=1)



# amount of fraud classes 492 rows.

fraud_data = data.loc[data['Class'] == 1]

legit_data = data.loc[data['Class'] == 0][:492]



normal_distributed_data = pd.concat([fraud_data, legit_data])



# Shuffle dataframe rows

new_data = normal_distributed_data.sample(frac=1, random_state=42)



new_data.head()



sns.countplot('Class', data=new_data, palette=colors)

plt.title('Equally Distributed Classes', fontsize=10)

plt.show()
# Visualizing the original dataset using PCA  after removing 2 features outliers

X = data.drop('Class', axis=1)

y = data['Class']





# PCA Implementation

X_reduced_pca = PCA(n_components=2, random_state=42).fit_transform(X.values)



f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24,6))

# labels = ['No Fraud', 'Fraud']

f.suptitle('Clusters using Dimensionality Reduction', fontsize=14)





blue_patch = mpatches.Patch(color='#0101DF', label='No Fraud')

red_patch = mpatches.Patch(color='#DF0101', label='Fraud')



# PCA scatter plot

ax2.scatter(X_reduced_pca[:,0], X_reduced_pca[:,1], c=(y == 0), cmap='coolwarm', label='No Fraud', linewidths=2)

ax2.scatter(X_reduced_pca[:,0], X_reduced_pca[:,1], c=(y == 1), cmap='coolwarm', label='Fraud', linewidths=2)

ax2.set_title('PCA', fontsize=14)



ax2.grid(True)



ax2.legend(handles=[blue_patch, red_patch])



plt.show()
# Classifier Selection

X = new_data.drop('Class', axis=1)

y = new_data['Class']



from sklearn.model_selection import train_test_split



# This is explicitly used for undersampling.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Turn the values into an array for feeding the classification algorithms.

X_train = X_train.values

X_test = X_test.values

y_train = y_train.values

y_test = y_test.values



# Simple classifiers 



classifiers = {

    "LogisiticRegression": LogisticRegression(),

    "KNearest": KNeighborsClassifier(),

    "Support Vector Classifier": SVC(),

    "Random Forest Classifier": RandomForestClassifier(),

    "XGBoost Classifier": XGBClassifier()    

}





# Cross validation score for different classifiers

from sklearn.model_selection import cross_val_score



for key, classifier in classifiers.items():

    classifier.fit(X_train, y_train)

    training_score = cross_val_score(classifier, X_train, y_train, cv=5)

    print("Classifiers: ", classifier.__class__.__name__, "Has a training score of", round(training_score.mean(), 2) * 100, "% accuracy score")
# The use of anomaly detection algorithms with original and undersampled dataset

from sklearn.metrics import confusion_matrix



# Isolation Forest 



outliers = new_data.loc[data['Class']==1]

normal = new_data.loc[data['Class']==0]

outliers = outliers.drop(['Class'] , axis=1)

normal = normal.drop(['Class'] , axis=1)

normal_=np.array(normal)

outliers_=np.array(outliers)

X_outliers = np.array(outliers)

X_lof=np.r_[normal_+2,normal_-2,outliers_]



clf = IsolationForest(max_samples=100)

clf.fit(X_train)



y_pred_train = clf.predict(X_train)

y_pred_test = clf.predict(X_test)

y_pred_outliers = clf.predict(X_outliers)



# plot the line, the samples, and the nearest vectors to the plane

x_train=X_train[:10000,[1,28]]

x_test=X_test[:1000,[1,28]]

x_outliers=X_outliers[:100,[1,28]]



x_train=np.array(x_train)

x_test=np.array(x_test)

x_outliers=np.array(x_outliers)



clf_test = IsolationForest(max_samples=100)

clf_test.fit(x_train)



xx, yy = np.meshgrid(np.linspace(-5, 5, 50), np.linspace(-5, 5, 50))

Z = clf_test.decision_function(np.c_[xx.ravel(), yy.ravel()])

Z = Z.reshape(xx.shape)



plt.title("IsolationForest")

plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)



b1 = plt.scatter(x_train[:, 0], x_train[:, 1], c='red',

                 s=20, edgecolor='k')

b2 = plt.scatter(x_test[:, 0], x_test[:, 1], c='green',

                 s=20, edgecolor='k')

c = plt.scatter(x_outliers[:, 0], x_outliers[:, 1], c='white',

                s=20, edgecolor='k')

plt.axis('tight')

plt.xlim((-5, 5))

plt.ylim((-5, 5))

plt.legend([b1, b2, c],

           ["training observations",

            "new regular observations", "new abnormal observations"],

           loc="upper left")

plt.show()



# Isolation Forest Evaluation



print("Test Accuracy score for IF   :", list(y_pred_test).count(1)/y_pred_test.shape[0])

print("Outliers Accuracy for IF :", list(y_pred_outliers).count(-1)/y_pred_outliers.shape[0])



#cnf_IF = confusion_matrix(y_test,y_pred_test)

#fig, ax = plt.subplots(2, 3,figsize=(22,12))



#cnf_IF

#a=sns.heatmap(cnf_IF, annot=True, cmap=plt.cm.copper)

#plt.title("Isolation Forest with undersampling \n Confusion Matrix", fontsize=14)

# Finding the best optimizer using GridSearchCV cross validation

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



# Random Forest Classifier

param_grid = { 

    'n_estimators': [200, 500],

    'max_features': ['auto', 'sqrt', 'log2'],

    'max_depth' : [4,5,6,7,8],

    'criterion' :['gini', 'entropy']

}



grid_forest = GridSearchCV(RandomForestClassifier(), param_grid)

grid_forest.fit(X_train, y_train)



# RF best estimator

RF = grid_forest.best_estimator_

# XGBoost Classifier

parameters = {'nthread':[4], #when use hyperthread, xgboost may become slower

              'objective':['binary:logistic'],

              'learning_rate': [0.05], #so called `eta` value

              'max_depth': [6],

              'min_child_weight': [11],

              'silent': [1],

              'subsample': [0.8],

              'colsample_bytree': [0.7],

              'n_estimators': [10], #number of trees, change it to 1000 for better results

              'missing':[-999],

              'seed': [1337]}

grid_xgb = GridSearchCV(XGBClassifier(), parameters)

grid_xgb.fit(X_train, y_train)

# XGB best estimator

xgb = grid_xgb.best_estimator_



# Isolation Forest 

#f1sc = make_scorer(f1_score(y_test,y_pred,average='micro'))



#param_IF = {'n_estimators': list(range(100, 800, 5)), 

              #'max_samples': list(range(100, 500, 5)), 

              #'contamination': [0.1, 0.2, 0.3, 0.4, 0.5], 

              #'max_features': [5,10,15], 

              #'bootstrap': [True, False], 

              #'n_jobs': [5, 10, 20, 30]}

#grid_IF = GridSearchCV(IsolationForest(),param_IF,cv=5,scoring=f1sc)

#grid_IF.fit(X_train,y_train)

# Isolation Forest best estimator

#IF = grid_IF.best_estimator_

# Overfitting Case



log_reg_score = cross_val_score(log_reg, X_train, y_train, cv=5)

print('Overfitting : Logistic Regression Cross Validation Score: ', round(log_reg_score.mean() * 100, 2).astype(str) + '%')



knears_score = cross_val_score(knears_neighbors, X_train, y_train, cv=5)

print('Overfitting: Knears Neighbors Cross Validation Score', round(knears_score.mean() * 100, 2).astype(str) + '%')

svc_score = cross_val_score(svc, X_train, y_train, cv=5)

print('Overfitting: Support Vector Classifier Cross Validation Score', round(svc_score.mean() * 100, 2).astype(str) + '%')



RF_score = cross_val_score(RF, X_train, y_train, cv=5)

print('Overfitting: RandomForest Classifier Cross Validation Score', round(RF_score.mean() * 100, 2).astype(str) + '%')

xgb_score = cross_val_score(xgb, X_train, y_train, cv=5)

print('Overfitting: XGBoost Classifier Cross Validation Score', round(xgb_score.mean() * 100, 2).astype(str) + '%')

#IF_score = cross_val_score(IF, X_train, y_train, cv=5)

#print('Overfitting: Isolation Forest Cross Validation Score', round(IF_score.mean() * 100, 2).astype(str) + '%')
# We will undersample during cross validating

undersample_X = new_data.drop('Class', axis=1)

undersample_y = new_data['Class']



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
# Let's Plot the Learning Curves

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

    ax4.set_title("XGBoost Classifier \n Learning Curve", fontsize=14)

    ax4.set_xlabel('Training size (m)')

    ax4.set_ylabel('Score')

    ax4.grid(True)

    ax4.legend(loc="best")

    

    

    

    

    return plt  



cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=42)

plot_learning_curve(log_reg, knears_neighbors, svc, xgb, X_train, y_train, (0.87, 1.01), cv=cv, n_jobs=4)

from sklearn.metrics import roc_curve

from sklearn.model_selection import cross_val_predict

# Create a DataFrame with all the scores and the classifiers names.



log_reg_pred = cross_val_predict(log_reg, X_train, y_train, cv=5,

                             method="decision_function")

knears_pred = cross_val_predict(knears_neighbors, X_train, y_train, cv=5)



svc_pred = cross_val_predict(svc, X_train, y_train, cv=5,

                             method="decision_function")



RF_pred = cross_val_predict(RF, X_train, y_train, cv=5)



xgb_pred = cross_val_predict(xgb, X_train, y_train, cv=5)









from sklearn.metrics import roc_auc_score



print('Logistic Regression: ', roc_auc_score(y_train, log_reg_pred))

print('Knears Neighbors Classifier: ', roc_auc_score(y_train, knears_pred))

print('Support Vector Classifier: ', roc_auc_score(y_train, svc_pred))

print('Random Forest Classifier: ', roc_auc_score(y_train, RF_pred))

print('XGBoost Classifier: ', roc_auc_score(y_train, xgb_pred))

# Plotting the roc curve

log_fpr, log_tpr, log_thresold = roc_curve(y_train, log_reg_pred)

knear_fpr, knear_tpr, knear_threshold = roc_curve(y_train, knears_pred)

svc_fpr, svc_tpr, svc_threshold = roc_curve(y_train, svc_pred)

tree_fpr, tree_tpr, tree_threshold = roc_curve(y_train, RF_pred)

xgb_fpr, xgb_tpr, xgb_threshold = roc_curve(y_train, xgb_pred)







def graph_roc_curve_multiple(log_fpr, log_tpr,knear_fpr, knear_tpr, svc_fpr, svc_tpr, tree_fpr, tree_tpr,xgb_fpr,xgb_tpr):

    plt.figure(figsize=(16,8))

    plt.title('ROC Curve \n 5 Classifiers', fontsize=18)

    plt.plot(log_fpr, log_tpr, label='Logistic Regression Classifier Score: {:.4f}'.format(roc_auc_score(y_train, log_reg_pred)))

    plt.plot(knear_fpr, knear_tpr, label='KNears Neighbors Classifier Score: {:.4f}'.format(roc_auc_score(y_train, knears_pred)))

    plt.plot(svc_fpr, svc_tpr, label='Support Vector Classifier Score: {:.4f}'.format(roc_auc_score(y_train, svc_pred)))

    plt.plot(tree_fpr, tree_tpr, label='Random Forest Classifier Score: {:.4f}'.format(roc_auc_score(y_train, RF_pred)))

    plt.plot(xgb_fpr, xgb_tpr, label='XGBoost Classifier Score: {:.4f}'.format(roc_auc_score(y_train, xgb_pred)))



    plt.plot([0, 1], [0, 1], 'k--')

    plt.axis([-0.01, 1, 0, 1])

    plt.xlabel('False Positive Rate', fontsize=16)

    plt.ylabel('True Positive Rate', fontsize=16)

    plt.annotate('Minimum ROC Score of 50% \n (This is the minimum score to get)', xy=(0.5, 0.5), xytext=(0.6, 0.3),

                arrowprops=dict(facecolor='#6E726D', shrink=0.05),

                )

    plt.legend()

    

graph_roc_curve_multiple(log_fpr, log_tpr,knear_fpr, knear_tpr, svc_fpr, svc_tpr, tree_fpr, tree_tpr,xgb_fpr,xgb_tpr)

plt.show()
#%% Logistic Regression seems to be the winner!

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

from sklearn.metrics import precision_recall_curve

import matplotlib.pyplot as plt



fig = plt.figure(figsize=(12,6))



precision, recall, _ = precision_recall_curve(original_ytest, undersample_y_score)



plt.step(recall, precision, color='#004a93', alpha=0.2,

         where='post')

plt.fill_between(recall, precision, step='post', alpha=0.2,

                 color='#48a6ff')



plt.xlabel('Recall')

plt.ylabel('Precision')

plt.ylim([0.0, 1.05])

plt.xlim([0.0, 1.0])

plt.title('UnderSampling Precision-Recall curve: \n Average Precision-Recall Score ={0:0.2f}'.format(

          undersample_average_precision), fontsize=16)
#%% Logistic Regression with SMOTE

from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split, RandomizedSearchCV





print('Length of X (train): {} | Length of y (train): {}'.format(len(original_Xtrain), len(original_ytrain)))

print('Length of X (test): {} | Length of y (test): {}'.format(len(original_Xtest), len(original_ytest)))



# List to append the score and then find the average

accuracy_lst = []

precision_lst = []

recall_lst = []

f1_lst = []

auc_lst = []



# Classifier with optimal parameters

# log_reg_sm = grid_log_reg.best_estimator_

log_reg_sm = LogisticRegression()





rand_log_reg = RandomizedSearchCV(LogisticRegression(), log_reg_params, n_iter=4)





# Implementing SMOTE Technique 

# Cross Validating the right way

# Parameters

log_reg_params = {"penalty": ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

for train, test in sss.split(original_Xtrain, original_ytrain):

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

# Prediction

labels = ['No Fraud', 'Fraud']

smote_prediction = best_est.predict(original_Xtest)

print(classification_report(original_ytest, smote_prediction, target_names=labels))

y_score = best_est.decision_function(original_Xtest)

average_precision = average_precision_score(original_ytest, y_score)



print('Average precision-recall score: {0:0.2f}'.format(

      average_precision))

fig = plt.figure(figsize=(12,6))



precision, recall, _ = precision_recall_curve(original_ytest, y_score)



plt.step(recall, precision, color='r', alpha=0.2,

         where='post')

plt.fill_between(recall, precision, step='post', alpha=0.2,

                 color='#F59B00')



plt.xlabel('Recall')

plt.ylabel('Precision')

plt.ylim([0.0, 1.05])

plt.xlim([0.0, 1.0])

plt.title('OverSampling Precision-Recall curve: \n Average Precision-Recall Score ={0:0.2f}'.format(

          average_precision), fontsize=16)
#%% # SMOTE Technique (OverSampling) After splitting and Cross Validating

sm = SMOTE(ratio='minority', random_state=42)

# Xsm_train, ysm_train = sm.fit_sample(X_train, y_train)





Xsm_train, ysm_train = sm.fit_sample(original_Xtrain, original_ytrain)

# We Improve the score by 2% points approximately 

# Implement GridSearchCV and the other models.



# Logistic Regression

t0 = time.time()

log_reg_sm = grid_log_reg.best_estimator_

log_reg_sm.fit(Xsm_train, ysm_train)

t1 = time.time()

print("Fitting oversample data took :{} sec".format(t1 - t0))
#%% TESTING CLASSIFIERS WITH UNDERSAMPLING

from sklearn.metrics import confusion_matrix



# Logistic Regression fitted using SMOTE technique

y_pred_log_reg = log_reg_sm.predict(original_Xtest)



# Other models fitted with UnderSampling

y_pred_knear = knears_neighbors.predict(original_Xtest)

y_pred_svc = svc.predict(original_Xtest)

y_pred_RF = RF.predict(original_Xtest)

y_pred_xgb = xgb.predict(original_Xtest)

y_pred_lr = log_reg.predict(original_Xtest)







#log_reg_cf = confusion_matrix(y_test, y_pred_log_reg)

log_reg_cf = confusion_matrix(original_ytest, y_pred_lr)

kneighbors_cf = confusion_matrix(original_ytest, y_pred_knear)

svc_cf = confusion_matrix(original_ytest, y_pred_svc)

tree_cf = confusion_matrix(original_ytest, y_pred_RF)

xgb_cf = confusion_matrix(original_ytest, y_pred_xgb)









fig, ax = plt.subplots(2, 3,figsize=(22,12))





sns.heatmap(log_reg_cf, ax=ax[0][0], annot=True, cmap=plt.cm.copper)

ax[0, 0].set_title("Logistic Regression with undersampling \n Confusion Matrix", fontsize=14)

ax[0, 0].set_xticklabels(['', ''], fontsize=14, rotation=90)

ax[0, 0].set_yticklabels(['', ''], fontsize=14, rotation=360)



sns.heatmap(kneighbors_cf, ax=ax[0][1], annot=True, cmap=plt.cm.copper)

ax[0][1].set_title("KNearsNeighbors with undersampling \n Confusion Matrix", fontsize=14)

ax[0][1].set_xticklabels(['', ''], fontsize=14, rotation=90)

ax[0][1].set_yticklabels(['', ''], fontsize=14, rotation=360)



sns.heatmap(svc_cf, ax=ax[1][0], annot=True, cmap=plt.cm.copper)

ax[1][0].set_title("Suppor Vector Classifier with undersampling \n Confusion Matrix", fontsize=14)

ax[1][0].set_xticklabels(['', ''], fontsize=14, rotation=90)

ax[1][0].set_yticklabels(['', ''], fontsize=14, rotation=360)



sns.heatmap(tree_cf, ax=ax[1][1], annot=True, cmap=plt.cm.copper)

ax[1][1].set_title("Random Forest Classifier with undersampling \n Confusion Matrix", fontsize=14)

ax[1][1].set_xticklabels(['', ''], fontsize=14, rotation=90)

ax[1][1].set_yticklabels(['', ''], fontsize=14, rotation=360)





sns.heatmap(xgb_cf, ax=ax[1][2], annot=True, cmap=plt.cm.copper)

ax[1][2].set_title("XGBoost Classifier with undersampling \n Confusion Matrix", fontsize=14)

ax[1][2].set_xticklabels(['', ''], fontsize=20, rotation=90)

ax[1][2].set_yticklabels(['', ''], fontsize=20, rotation=360)



actual_cmu = confusion_matrix(original_ytest,original_ytest)



sns.heatmap(actual_cmu, ax=ax[0][2], annot=True, cmap=plt.cm.copper)

ax[0][2].set_title("100% Accuracy Confusion Matrix for undersampling", fontsize=14)

ax[0][2].set_xticklabels(['', ''], fontsize=20, rotation=90)

ax[0][2].set_yticklabels(['', ''], fontsize=20, rotation=360)



plt.show()



plot_confusion_matrix(log_reg_cf, labels, title="Logistic Regression \n Confusion Matrix")

plot_confusion_matrix(kneighbors_cf, labels, title="KNN \n Confusion Matrix")

plot_confusion_matrix(svc_cf, labels, title="SVM \n Confusion Matrix")

plot_confusion_matrix(tree_cf, labels, title="RF \n Confusion Matrix")

plot_confusion_matrix(xgb_cf, labels, title="XGBoost \n Confusion Matrix")

plot_confusion_matrix(actual_cmu, labels, title="100% Accuracy \n Confusion Matrix")







from sklearn.metrics import classification_report





print('Logistic Regression:')

print(classification_report(original_ytest, y_pred_lr))



print('KNears Neighbors:')

print(classification_report(original_ytest, y_pred_knear))



print('Support Vector Classifier:')

print(classification_report(original_ytest, y_pred_svc))



print('Random Forest Classifier:')

print(classification_report(original_ytest, y_pred_RF))



print('XGBoost Classifier:')

print(classification_report(original_ytest, y_pred_xgb))
#%% # Final Score in the test set of logistic regression

from sklearn.metrics import accuracy_score



# Logistic Regression with Under-Sampling

y_pred = log_reg.predict(original_Xtest)

undersample_score = accuracy_score(original_ytest, y_pred)







# Logistic Regression with SMOTE Technique (Better accuracy with SMOTE t)

y_pred_sm = best_est.predict(original_Xtest)

oversample_score = accuracy_score(original_ytest, y_pred_sm)



###NEW

# confusion matrix for oversampling

log_reg_cf_up = confusion_matrix(original_ytest, y_pred_sm)



fig, ax = plt.subplots(2, 3,figsize=(22,12))





sns.heatmap(log_reg_cf_up, ax=ax[0][0], annot=True, cmap=plt.cm.copper)

ax[0, 0].set_title("Logistic Regression with oversampling \n Confusion Matrix", fontsize=14)

ax[0, 0].set_xticklabels(['', ''], fontsize=14, rotation=90)

ax[0, 0].set_yticklabels(['', ''], fontsize=14, rotation=360)





actual_cm_up = confusion_matrix(original_ytest,original_ytest)



sns.heatmap(actual_cm_up, ax=ax[0][2], annot=True, cmap=plt.cm.copper)

ax[0][2].set_title("100% Accuracy Confusion Matrix for oversampling", fontsize=14)

ax[0][2].set_xticklabels(['', ''], fontsize=20, rotation=90)

ax[0][2].set_yticklabels(['', ''], fontsize=20, rotation=360)



plt.show()

###

d = {'Technique': ['Random UnderSampling', 'Oversampling (SMOTE)'], 'Score': [undersample_score, oversample_score]}

final_df = pd.DataFrame(data=d)



# Move column

score = final_df['Score']

final_df.drop('Score', axis=1, inplace=True)

final_df.insert(1, 'Score', score)



# Note how high is accuracy score it can be misleading! 

final_df
#%% Neural Networks with undersampling and oversampling

import keras

from keras import backend as K

from keras.models import Sequential

from keras.layers import Activation

from keras.layers.core import Dense

from keras.optimizers import Adam

from keras.metrics import categorical_crossentropy

from keras.layers import Dropout



n_inputs = X_train.shape[1]



# undersampling

undersample_model = Sequential([

    Dense(n_inputs, input_shape=(n_inputs, ), activation='relu'),

    Dense(32, activation='relu'),

    Dense(2, activation='softmax')

])



undersample_model.summary()

undersample_model.compile(Adam(lr=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

undersample_model.fit(X_train, y_train, validation_split=0.2, batch_size=25, epochs=20, shuffle=True, verbose=2)

undersample_predictions = undersample_model.predict(original_Xtest, batch_size=200, verbose=0)

undersample_fraud_predictions = undersample_model.predict_classes(original_Xtest, batch_size=200, verbose=0)



import itertools



# Create a confusion matrix

def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    print(cm)



    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title, fontsize=14)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, format(cm[i, j], fmt),

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

undersample_cm = confusion_matrix(original_ytest, undersample_fraud_predictions)

actual_cm = confusion_matrix(original_ytest, original_ytest)

labels = ['No Fraud', 'Fraud']



fig = plt.figure(figsize=(16,8))



fig.add_subplot(221)

plot_confusion_matrix(undersample_cm, labels, title="Random UnderSampling \n Confusion Matrix", cmap=plt.cm.Reds)



fig.add_subplot(222)

plot_confusion_matrix(actual_cm, labels, title="Confusion Matrix \n (with 100% accuracy)", cmap=plt.cm.Greens)





print('Neural Networks with undersampling:')

print(classification_report(original_ytest, undersample_fraud_predictions))



# Oversampling

n_inputs = Xsm_train.shape[1]



oversample_model = Sequential([

    Dense(n_inputs, input_shape=(n_inputs, ), activation='relu'),

    Dense(32, activation='relu'),

    Dense(2, activation='softmax')

])

oversample_model.compile(Adam(lr=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

oversample_model.fit(Xsm_train, ysm_train, validation_split=0.2, batch_size=300, epochs=20, shuffle=True, verbose=2)

oversample_predictions = oversample_model.predict(original_Xtest, batch_size=200, verbose=0)

oversample_fraud_predictions = oversample_model.predict_classes(original_Xtest, batch_size=200, verbose=0)



oversample_smote = confusion_matrix(original_ytest, oversample_fraud_predictions)

actual_cm = confusion_matrix(original_ytest, original_ytest)

labels = ['No Fraud', 'Fraud']



fig = plt.figure(figsize=(16,8))



fig.add_subplot(221)

plot_confusion_matrix(oversample_smote, labels, title="OverSampling (SMOTE) \n Confusion Matrix", cmap=plt.cm.Oranges)



fig.add_subplot(222)

plot_confusion_matrix(actual_cm, labels, title="Confusion Matrix \n (with 100% accuracy)", cmap=plt.cm.Greens)    



print('Neural Networks with SMOTE:')

print(classification_report(original_ytest, oversample_fraud_predictions))