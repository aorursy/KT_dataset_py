# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



%matplotlib inline

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import os as os

import sys

import warnings

warnings.filterwarnings('ignore')



# From Scikit Learn

from sklearn import preprocessing, decomposition, tree

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import scale

from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix, classification_report

from astropy.table import Table, Column

# Set DEBUG = True to produce debug results

DEBUG = False
print("The Python version is %s.%s.%s." % sys.version_info[:3])
%pwd
# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

churn = pd.read_csv("../input/WA_Fn-UseC_-Telco-Customer-Churn.csv", header=0, sep=",")
churn.describe(include='all')
# Source: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.dropna.html

churn = churn.dropna(axis = 0, how = 'all')

if DEBUG:

    #Dimensions of dataset

    print("Shape of Data", churn.shape)

    #Colum names

    print("Colums Names", churn.columns)

    #See bottol few rows of dataset

    print(churn.tail())
# designate target variable name

targetName = 'Churn'

targetSeries = churn[targetName]

#remove target from current location and insert in collum 0

del churn[targetName]

churn.insert(0, targetName, targetSeries)

#reprint dataframe and see target is in position 0

churn.head()
churn.info()
#Basic bar chart since the target is binominal

groupby = churn.groupby(targetName)

targetEDA=groupby[targetName].aggregate(len)

plt.figure()

targetEDA.plot(kind='bar', grid=False)

plt.axhline(0, color='k')

plt.title('Bar Chart of Churn')
groupby.mean()
# Source: https://pandas.pydata.org/pandas-docs/version/0.17.0/generated/pandas.crosstab.html

churn_ac_cross = pd.crosstab(churn['PaperlessBilling'], churn['Churn'])

if DEBUG:

    print(churn_ac_cross)
# Source: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.plot.html

churn_ac_cross.plot(kind='bar', stacked=True)

plt.title('Churn by Paperless Billing')
# Source: https://pandas.pydata.org/pandas-docs/version/0.17.0/generated/pandas.crosstab.html

churn_ac_cross = pd.crosstab(churn['Contract'], churn['Churn'])

if DEBUG:

    print(churn_ac_cross)
# Source: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.plot.html

churn_ac_cross.plot(kind='bar', stacked=True)

plt.title('Churn by Contract')
# Source: https://pandas.pydata.org/pandas-docs/version/0.17.0/generated/pandas.crosstab.html

churn_st_cross = pd.crosstab(churn['OnlineSecurity'], churn['Churn'])

if DEBUG:

    print(churn_st_cross)
# Source: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.plot.html

#plt.rcParams["figure.figsize"] = (20,3)

churn_st_cross.plot(kind='bar', stacked=True)

plt.title('Churn by Online Security')
print(churn.info())
from sklearn import preprocessing

le_dep = preprocessing.LabelEncoder()

#to convert into numbers

churn['Churn'] = le_dep.fit_transform(churn['Churn'])

churn['gender'] = le_dep.fit_transform(churn['gender'])

churn['Partner'] = le_dep.fit_transform(churn['Partner'])

churn['Dependents'] = le_dep.fit_transform(churn['Dependents'])

churn['PhoneService'] = le_dep.fit_transform(churn['PhoneService'])

churn['MultipleLines'] = le_dep.fit_transform(churn['MultipleLines'])

churn['InternetService'] = le_dep.fit_transform(churn['InternetService'])

churn['OnlineSecurity'] = le_dep.fit_transform(churn['OnlineSecurity'])

churn['OnlineBackup'] = le_dep.fit_transform(churn['OnlineBackup'])

churn['DeviceProtection'] = le_dep.fit_transform(churn['DeviceProtection'])

churn['TechSupport'] = le_dep.fit_transform(churn['TechSupport'])

churn['StreamingTV'] = le_dep.fit_transform(churn['StreamingTV'])

churn['StreamingMovies'] = le_dep.fit_transform(churn['StreamingMovies'])

churn['Contract'] = le_dep.fit_transform(churn['Contract'])

churn['PaperlessBilling'] = le_dep.fit_transform(churn['PaperlessBilling'])

churn['PaymentMethod'] = le_dep.fit_transform(churn['PaymentMethod'])
#Note: axis=1 denotes that we are referring to a column, not a row

churn=churn.drop(['customerID'],axis=1)

if DEBUG:

    print(churn.head())
churn.info()
churn['TotalCharges'] = churn['TotalCharges'].convert_objects(convert_numeric=True)
print(churn['TotalCharges'].describe())
churn['TotalCharges'].fillna(churn['TotalCharges'].median(), inplace=True)

print(churn['TotalCharges'].describe())
if DEBUG:

    print(churn.shape)

    print(churn.info())

    print(churn.head())
groupby = churn.groupby(targetName)

print(groupby.mean())
# Source: https://etav.github.io/python/scikit_pca.html

churn_numeric = churn.select_dtypes(include=['number'])

features = list(churn_numeric)

X = churn_numeric.loc[:, features].values

X = scale(X)
pca = PCA(n_components=0.99, whiten=True)
pca.fit(X)

variance = pca.explained_variance_ratio_ #calculate variance ratios

X_pca = pca.fit_transform(X)

var=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=3)*100)

var #cumulative sum of variance explained with [n] features
# Show results

print('Original number of features:', X.shape[1])

print('Reduced number of features:', X_pca.shape[1])
plt.ylabel('% Variance Explained')

plt.xlabel('# of Features')

plt.title('PCA Analysis')

plt.ylim(30,100.5)

plt.style.context('seaborn-whitegrid')



plt.plot(var)
# split dataset into testing and training

features_train, features_test, target_train, target_test = train_test_split(churn.iloc[:,1:].values, churn.iloc[:,0].values, test_size=0.33, random_state=0)
from sklearn.preprocessing import MinMaxScaler

scaler = preprocessing.MinMaxScaler().fit(features_train)

features_train = scaler.transform(features_train)

features_test = scaler.transform(features_test)
train_results = []

test_results = []

# search for an optimal value of max_depth for decision tree

k_range = list(range(25,50))

k_scores = []

for k in k_range:

    knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1, algorithm='auto', weights='uniform')

    scores = cross_val_score(knn, features_train, target_train, cv=10, scoring='recall', n_jobs=-1)

    k_scores.append(scores.mean())

    # Code for plotting results

    knn.fit(features_train, target_train)

    train_pred = knn.predict(features_train)

    false_positive_rate, true_positive_rate, thresholds = roc_curve(target_train, train_pred)

    roc_auc = auc(false_positive_rate, true_positive_rate)

    # Add auc score to previous train results

    train_results.append(roc_auc)

    y_pred = knn.predict(features_test)

    false_positive_rate, true_positive_rate, thresholds = roc_curve(target_test, y_pred)

    roc_auc = auc(false_positive_rate, true_positive_rate)

    # Add auc score to previous test results

    test_results.append(roc_auc)

if DEBUG:

    print(k_scores)
if DEBUG:

    scores = pd.DataFrame(k_scores)

    print(scores)
from matplotlib.legend_handler import HandlerLine2D

line1, = plt.plot(k_range, train_results, 'b', label="Train AUC")

line2, = plt.plot(k_range, test_results, 'r', label="Test AUC")

plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})

plt.ylabel('AUC score')

plt.xlabel('KNN Neighbors')

plt.show()
#KNN train model. Call up my model and name it clf

clf_knn = KNeighborsClassifier(n_neighbors=27, n_jobs=-1, algorithm='auto', weights='uniform')

#Call up the model to see the parameters you can tune (and their default setting)

print(clf_knn)

#Fit clf to the training data

clf_knn = clf_knn.fit(features_train, target_train)

#Predict clf DT model again test data

target_predicted_knn = clf_knn.predict(features_test)
acc_knn = accuracy_score(target_test, target_predicted_knn)

prec_knn = precision_score(target_test, target_predicted_knn)

recall_knn = recall_score(target_test, target_predicted_knn)

f1_knn = f1_score(target_test, target_predicted_knn)

cm_knn = confusion_matrix(target_test, target_predicted_knn)

print("KNN Accuracy Score", acc_knn)

print(classification_report(target_test, target_predicted_knn))

if DEBUG:

    print(confusion_matrix(target_test, target_predicted_knn))
# Source: https://seaborn.pydata.org/examples/heatmap_annotation.html

plt.figure(figsize=(5.5,4))

sns.heatmap(cm_knn, annot=True, fmt='d')

plt.title('KNN Model 1 Confusion Matrix \nAccuracy:{0:.3f}'.format(acc_knn))

plt.ylabel('True label')

plt.xlabel('Predicted label')

plt.show()
#verify KNN with Cross Validation

scores_knn = cross_val_score(clf_knn, features_train, target_train, cv=10, n_jobs=-1)

print("Cross Validation Score for each K",scores_knn)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores_knn.mean(), scores_knn.std() * 2))
# Determine the false positive and true positive rates

fpr, tpr, _ = roc_curve(target_test, clf_knn.predict_proba(features_test)[:,1]) 

    

# Calculate the AUC

roc_auc_knn = auc(fpr, tpr)

print('ROC AUC: %0.3f' % roc_auc_knn)

 

# Plot of a ROC curve for a specific class

plt.figure()

plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc_knn)

plt.plot([0, 1], [0, 1], 'k--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('KNN Model 1 ROC Curve')

plt.legend(loc="lower right")

plt.show()
#KNN train model. Call up my model and name it clf

clf_knn1 = KNeighborsClassifier(n_neighbors=27, n_jobs=-1, algorithm='auto', weights='distance')

#Call up the model to see the parameters you can tune (and their default setting)

print(clf_knn1)

#Fit clf to the training data

clf_knn1 = clf_knn1.fit(features_train, target_train)

#Predict clf DT model again test data

target_predicted_knn1 = clf_knn1.predict(features_test)
acc_knn1 = accuracy_score(target_test, target_predicted_knn1)

prec_knn1 = precision_score(target_test, target_predicted_knn1)

recall_knn1 = recall_score(target_test, target_predicted_knn1)

f1_knn1 = f1_score(target_test, target_predicted_knn1)

cm_knn1 = confusion_matrix(target_test, target_predicted_knn1)

print("KNN Accuracy Score", acc_knn1)

print(classification_report(target_test, target_predicted_knn1))

if DEBUG:

    print(confusion_matrix(target_test, target_predicted_knn1))
# Source: https://seaborn.pydata.org/examples/heatmap_annotation.html

plt.figure(figsize=(5.5,4))

sns.heatmap(cm_knn1, annot=True, fmt='d')

plt.title('KNN Model 2 Confusion Matrix \nAccuracy:{0:.3f}'.format(acc_knn1))

plt.ylabel('True label')

plt.xlabel('Predicted label')

plt.show()
#verify KNN with Cross Validation

scores_knn1 = cross_val_score(clf_knn1, features_train, target_train, cv=10, n_jobs=-1)

print("Cross Validation Score for each K",scores_knn1)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores_knn1.mean(), scores_knn1.std() * 2))
# Determine the false positive and true positive rates

fpr, tpr, _ = roc_curve(target_test, clf_knn1.predict_proba(features_test)[:,1]) 

    

# Calculate the AUC

roc_auc_knn1 = auc(fpr, tpr)

print('ROC AUC: %0.3f' % roc_auc_knn1)

 

# Plot of a ROC curve for a specific class

plt.figure()

plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc_knn1)

plt.plot([0, 1], [0, 1], 'k--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('KNN Model 2 ROC Curve')

plt.legend(loc="lower right")

plt.show()
#KNN train model. Call up my model and name it clf

clf_knn2 = KNeighborsClassifier(p=1, n_neighbors=27, n_jobs=-1, algorithm='kd_tree', weights='uniform')

#Call up the model to see the parameters you can tune (and their default setting)

print(clf_knn2)

#Fit clf to the training data

clf_knn2 = clf_knn2.fit(features_train, target_train)

#Predict clf DT model again test data

target_predicted_knn2 = clf_knn2.predict(features_test)
acc_knn2 = accuracy_score(target_test, target_predicted_knn2)

prec_knn2 = precision_score(target_test, target_predicted_knn2)

recall_knn2 = recall_score(target_test, target_predicted_knn2)

f1_knn2 = f1_score(target_test, target_predicted_knn2)

cm_knn2 = confusion_matrix(target_test, target_predicted_knn2)

print("KNN Accuracy Score", acc_knn2)

print(classification_report(target_test, target_predicted_knn2))

if DEBUG:

    print(confusion_matrix(target_test, target_predicted_knn2))
# Source: https://seaborn.pydata.org/examples/heatmap_annotation.html

plt.figure(figsize=(5.5,4))

sns.heatmap(cm_knn2, annot=True, fmt='d')

plt.title('KNN Model 3 Confusion Matrix \nAccuracy:{0:.3f}'.format(acc_knn2))

plt.ylabel('True label')

plt.xlabel('Predicted label')

plt.show()
#verify KNN with Cross Validation

scores_knn2 = cross_val_score(clf_knn2, features_train, target_train, cv=10, n_jobs=-1)

print("Cross Validation Score for each K",scores_knn2)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores_knn2.mean(), scores_knn2.std() * 2))
# Determine the false positive and true positive rates

fpr, tpr, _ = roc_curve(target_test, clf_knn2.predict_proba(features_test)[:,1]) 

    

# Calculate the AUC

roc_auc_knn2 = auc(fpr, tpr)

print('ROC AUC: %0.3f' % roc_auc_knn2)

 

# Plot of a ROC curve for a specific class

plt.figure()

plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc_knn2)

plt.plot([0, 1], [0, 1], 'k--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('KNN Model 3 ROC Curve')

plt.legend(loc="lower right")

plt.show()
# Source: http://docs.astropy.org/en/stable/table/construct_table.html

t = Table()

t[''] = ['KNN Model 1','KNN Model 2','KNN Model 3']

t['Cross Validation Score'] = [round(scores_knn.mean(),4),round(scores_knn1.mean(),4),round(scores_knn2.mean(),4)]

t['Accuracy Score'] = [round(acc_knn,4),round(acc_knn1,4),round(acc_knn2,4)]

t['Precision'] = [round(prec_knn,4),round(prec_knn1,4),round(prec_knn2,4)]

t['Recall'] = [round(recall_knn,4),round(recall_knn1,4),round(recall_knn2,4)]

t['F1 Score'] = [round(f1_knn,4),round(f1_knn1,4),round(f1_knn2,4)]

t['ROC AUC'] = [round(roc_auc_knn,4),round(roc_auc_knn1,4),round(roc_auc_knn2,4)]

t
train_results = []

test_results = []

# search for an optimal value of max_depth for decision tree

k_range = list(range(1,20))

k_scores = []

for k in k_range:

    dt = tree.DecisionTreeClassifier(max_depth=k)

    scores = cross_val_score(dt, features_train, target_train, cv=10, scoring='recall', n_jobs=-1)

    k_scores.append(scores.mean())

    # Code for plotting results

    dt.fit(features_train, target_train)

    train_pred = dt.predict(features_train)

    false_positive_rate, true_positive_rate, thresholds = roc_curve(target_train, train_pred)

    roc_auc = auc(false_positive_rate, true_positive_rate)

    # Add auc score to previous train results

    train_results.append(roc_auc)

    y_pred = dt.predict(features_test)

    false_positive_rate, true_positive_rate, thresholds = roc_curve(target_test, y_pred)

    roc_auc = auc(false_positive_rate, true_positive_rate)

    # Add auc score to previous test results

    test_results.append(roc_auc)

if DEBUG:

    print(k_scores)
if DEBUG:

    scores = pd.DataFrame(k_scores)

    print(scores)
from matplotlib.legend_handler import HandlerLine2D

line1, = plt.plot(k_range, train_results, 'b', label="Train AUC")

line2, = plt.plot(k_range, test_results, 'r', label="Test AUC")

plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})

plt.ylabel('AUC score')

plt.xlabel('Tree depth')

plt.show()
#Decision Tree train model. Call up my model and name it clf 

clf_dt = tree.DecisionTreeClassifier(max_depth=5)

#Call up the model to see the parameters you can tune (and their default setting)

print(clf_dt)

#Fit clf to the training data

clf_dt = clf_dt.fit(features_train, target_train)

#Predict clf DT model again test data

target_predicted_dt = clf_dt.predict(features_test)
acc_dt = accuracy_score(target_test, target_predicted_dt)

prec_dt = precision_score(target_test, target_predicted_dt)

recall_dt = recall_score(target_test, target_predicted_dt)

f1_dt = f1_score(target_test, target_predicted_dt)

cm_dt = confusion_matrix(target_test, target_predicted_dt)

print("DT Accuracy Score", acc_dt)

print(classification_report(target_test, target_predicted_dt))

if DEBUG:

    print(confusion_matrix(target_test, target_predicted_dt))
# Source: https://seaborn.pydata.org/examples/heatmap_annotation.html

plt.figure(figsize=(5.5,4))

sns.heatmap(cm_dt, annot=True, fmt='d')

plt.title('Decision Tree Model 1 Confusion Matrix \nAccuracy:{0:.3f}'.format(acc_dt))

plt.ylabel('True label')

plt.xlabel('Predicted label')

plt.show()
#verify DT with Cross Validation

scores_dt = cross_val_score(clf_dt, features_train, target_train, cv=10, n_jobs=-1)

print("Cross Validation Score for each K",scores_dt)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores_dt.mean(), scores_dt.std() * 2))
# Determine the false positive and true positive rates

fpr, tpr, _ = roc_curve(target_test, clf_dt.predict_proba(features_test)[:,1]) 

    

# Calculate the AUC

roc_auc_dt = auc(fpr, tpr)

print('ROC AUC: %0.3f' % roc_auc_dt)

 

# Plot of a ROC curve for a specific class

plt.figure()

plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc_dt)

plt.plot([0, 1], [0, 1], 'k--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Decision Tree Model 1 ROC Curve')

plt.legend(loc="lower right")

plt.show()
train_results = []

test_results = []

# search for an optimal value of max_depth for decision tree

#k_range = list(range(2,40))

k_range = np.linspace(0.1, 0.3, 20, endpoint=True)

k_scores = []

for k in k_range:

    dt = tree.DecisionTreeClassifier(max_depth=5, min_samples_leaf=k)

    scores = cross_val_score(dt, features_train, target_train, cv=10, scoring='recall', n_jobs=-1)

    k_scores.append(scores.mean())

    # Code for plotting results

    dt.fit(features_train, target_train)

    train_pred = dt.predict(features_train)

    false_positive_rate, true_positive_rate, thresholds = roc_curve(target_train, train_pred)

    roc_auc = auc(false_positive_rate, true_positive_rate)

    # Add auc score to previous train results

    train_results.append(roc_auc)

    y_pred = dt.predict(features_test)

    false_positive_rate, true_positive_rate, thresholds = roc_curve(target_test, y_pred)

    roc_auc = auc(false_positive_rate, true_positive_rate)

    # Add auc score to previous test results

    test_results.append(roc_auc)

if DEBUG:

    print(k_scores)
if DEBUG:

    scores = pd.DataFrame(k_scores)

    print(scores)
#from matplotlib.legend_handler import HandlerLine2D

line1, = plt.plot(k_range, train_results, 'b', label="Train AUC")

line2, = plt.plot(k_range, test_results, 'r', label="Test AUC")

plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})

plt.ylabel('AUC score')

plt.xlabel('Min Samples Leaf')

plt.show()
#Decision Tree train model. Call up my model and name it clf 

clf_dt1 = tree.DecisionTreeClassifier(max_depth=5, min_samples_leaf=0.225)

#Call up the model to see the parameters you can tune (and their default setting)

print(clf_dt1)

#Fit clf to the training data

clf_dt1 = clf_dt1.fit(features_train, target_train)

#Predict clf DT model again test data

target_predicted_dt1 = clf_dt1.predict(features_test)
acc_dt1 = accuracy_score(target_test, target_predicted_dt1)

prec_dt1 = precision_score(target_test, target_predicted_dt1)

recall_dt1 = recall_score(target_test, target_predicted_dt1)

f1_dt1 = f1_score(target_test, target_predicted_dt1)

cm_dt1 = confusion_matrix(target_test, target_predicted_dt1)

print("DT Accuracy Score", acc_dt1)

print(classification_report(target_test, target_predicted_dt1))

if DEBUG:

    print(confusion_matrix(target_test, target_predicted_dt1))
# Source: https://seaborn.pydata.org/examples/heatmap_annotation.html

plt.figure(figsize=(5.5,4))

sns.heatmap(cm_dt1, annot=True, fmt='d')

plt.title('Decision Tree Model 2 Confusion Matrix \nAccuracy:{0:.3f}'.format(acc_dt1))

plt.ylabel('True label')

plt.xlabel('Predicted label')

plt.show()
#verify DT with Cross Validation

scores_dt1 = cross_val_score(clf_dt1, features_train, target_train, cv=10, n_jobs=-1)

print("Cross Validation Score for each K",scores_dt1)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores_dt1.mean(), scores_dt1.std() * 2))
# Determine the false positive and true positive rates

fpr, tpr, _ = roc_curve(target_test, clf_dt1.predict_proba(features_test)[:,1]) 

    

# Calculate the AUC

roc_auc_dt1 = auc(fpr, tpr)

print('ROC AUC: %0.3f' % roc_auc_dt1)

 

# Plot of a ROC curve for a specific class

plt.figure()

plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc_dt1)

plt.plot([0, 1], [0, 1], 'k--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Decision Tree Model 2 ROC Curve')

plt.legend(loc="lower right")

plt.show()
train_results = []

test_results = []

# search for an optimal value of max_depth for decision tree

#k_range = list(range(2,40))

k_range = np.linspace(0.1, 1.0, 20, endpoint=True)

k_scores = []

for k in k_range:

    dt = tree.DecisionTreeClassifier(max_depth=5, min_samples_split=k)

    scores = cross_val_score(dt, features_train, target_train, cv=10, scoring='recall', n_jobs=-1)

    k_scores.append(scores.mean())

    # Code for plotting results

    dt.fit(features_train, target_train)

    train_pred = dt.predict(features_train)

    false_positive_rate, true_positive_rate, thresholds = roc_curve(target_train, train_pred)

    roc_auc = auc(false_positive_rate, true_positive_rate)

    # Add auc score to previous train results

    train_results.append(roc_auc)

    y_pred = dt.predict(features_test)

    false_positive_rate, true_positive_rate, thresholds = roc_curve(target_test, y_pred)

    roc_auc = auc(false_positive_rate, true_positive_rate)

    # Add auc score to previous test results

    test_results.append(roc_auc)

if DEBUG:

    print(k_scores)
#from matplotlib.legend_handler import HandlerLine2D

line1, = plt.plot(k_range, train_results, 'b', label="Train AUC")

line2, = plt.plot(k_range, test_results, 'r', label="Test AUC")

plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})

plt.ylabel('AUC score')

plt.xlabel('Min Samples Split')

plt.show()
#Decision Tree train model. Call up my model and name it clf 

clf_dt2 = tree.DecisionTreeClassifier(max_depth=5, min_samples_split=0.5)

#Call up the model to see the parameters you can tune (and their default setting)

print(clf_dt2)

#Fit clf to the training data

clf_dt2 = clf_dt2.fit(features_train, target_train)

#Predict clf DT model again test data

target_predicted_dt2 = clf_dt2.predict(features_test)
acc_dt2 = accuracy_score(target_test, target_predicted_dt2)

prec_dt2 = precision_score(target_test, target_predicted_dt2)

recall_dt2 = recall_score(target_test, target_predicted_dt2)

f1_dt2 = f1_score(target_test, target_predicted_dt2)

cm_dt2 = confusion_matrix(target_test, target_predicted_dt2)

print("DT Accuracy Score", acc_dt2)

print(classification_report(target_test, target_predicted_dt2))

if DEBUG:

    print(confusion_matrix(target_test, target_predicted_dt2))
# Source: https://seaborn.pydata.org/examples/heatmap_annotation.html

plt.figure(figsize=(5.5,4))

sns.heatmap(cm_dt2, annot=True, fmt='d')

plt.title('Decision Tree Model 3 Confusion Matrix \nAccuracy:{0:.3f}'.format(acc_dt2))

plt.ylabel('True label')

plt.xlabel('Predicted label')

plt.show()
#verify DT with Cross Validation

scores_dt2 = cross_val_score(clf_dt2, features_train, target_train, cv=10, n_jobs=-1)

print("Cross Validation Score for each K",scores_dt2)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores_dt2.mean(), scores_dt2.std() * 2))
# Determine the false positive and true positive rates

fpr, tpr, _ = roc_curve(target_test, clf_dt2.predict_proba(features_test)[:,1]) 

    

# Calculate the AUC

roc_auc_dt2 = auc(fpr, tpr)

print('ROC AUC: %0.3f' % roc_auc_dt2)

 

# Plot of a ROC curve for a specific class

plt.figure()

plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc_dt2)

plt.plot([0, 1], [0, 1], 'k--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Decision Tree Model 3 ROC Curve')

plt.legend(loc="lower right")

plt.show()
t = Table()

t[''] = ['Decision Tree Model 1','Decision Tree Model 2','Decision Tree Model 3']

t['Cross Validation Score'] = [round(scores_dt.mean(),4),round(scores_dt1.mean(),4),round(scores_dt2.mean(),4)]

t['Accuracy Score'] = [round(acc_dt,4),round(acc_dt1,4),round(acc_dt2,4)]

t['Precision'] = [round(prec_dt,4),round(prec_dt1,4),round(prec_dt2,4)]

t['Recall'] = [round(recall_dt,4),round(recall_dt1,4),round(recall_dt2,4)]

t['F1 Score'] = [round(f1_dt,4),round(f1_dt1,4),round(f1_dt2,4)]

t['ROC AUC'] = [round(roc_auc_dt,4),round(roc_auc_dt1,4),round(roc_auc_dt2,4)]

t
train_results = []

test_results = []

# search for an optimal value of max_depth for decision tree

k_range = list(range(1,20))

k_scores = []

for k in k_range:

    rf = RandomForestClassifier(max_depth=k, n_jobs=-1)

    scores = cross_val_score(rf, features_train, target_train, cv=10, scoring='recall', n_jobs=-1)

    k_scores.append(scores.mean())

    # Code for plotting results

    rf.fit(features_train, target_train)

    train_pred = rf.predict(features_train)

    false_positive_rate, true_positive_rate, thresholds = roc_curve(target_train, train_pred)

    roc_auc = auc(false_positive_rate, true_positive_rate)

    # Add auc score to previous train results

    train_results.append(roc_auc)

    y_pred = rf.predict(features_test)

    false_positive_rate, true_positive_rate, thresholds = roc_curve(target_test, y_pred)

    roc_auc = auc(false_positive_rate, true_positive_rate)

    # Add auc score to previous test results

    test_results.append(roc_auc)

if DEBUG:

    print(k_scores)
from matplotlib.legend_handler import HandlerLine2D

line1, = plt.plot(k_range, train_results, 'b', label="Train AUC")

line2, = plt.plot(k_range, test_results, 'r', label="Test AUC")

plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})

plt.ylabel('AUC score')

plt.xlabel('Max Depth')

plt.show()
# Random Forest train model. Call up my model and name it clf

clf_rf = RandomForestClassifier(max_depth=7, n_jobs=-1)

#Call up the model to see the parameters you can tune (and their default setting)

print(clf_rf)

#Fit clf to the training data

clf_rf = clf_rf.fit(features_train, target_train)

#Predict clf DT model again test data

target_predicted_rf = clf_rf.predict(features_test)
acc_rf = accuracy_score(target_test, target_predicted_rf)

prec_rf = precision_score(target_test, target_predicted_rf)

recall_rf = recall_score(target_test, target_predicted_rf)

f1_rf = f1_score(target_test, target_predicted_rf)

cm_rf = confusion_matrix(target_test, target_predicted_rf)

print("RF Accuracy Score", acc_rf)

print(classification_report(target_test, target_predicted_rf))

if DEBUG:

    print(confusion_matrix(target_test, target_predicted_rf))
# Source: https://seaborn.pydata.org/examples/heatmap_annotation.html

plt.figure(figsize=(5.5,4))

sns.heatmap(cm_rf, annot=True, fmt='d')

plt.title('Random Forest Model 1 Confusion Matrix \nAccuracy:{0:.3f}'.format(acc_rf))

plt.ylabel('True label')

plt.xlabel('Predicted label')

plt.show()
#verify RF with Cross Validation

scores_rf = cross_val_score(clf_rf, features_train, target_train, cv=10, n_jobs=-1)

print("Cross Validation Score for each K",scores_rf)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores_rf.mean(), scores_rf.std() * 2))
# Determine the false positive and true positive rates

fpr, tpr, _ = roc_curve(target_test, clf_rf.predict_proba(features_test)[:,1]) 

    

# Calculate the AUC

roc_auc_rf = auc(fpr, tpr)

print('ROC AUC: %0.3f' % roc_auc_rf)

 

# Plot of a ROC curve for a specific class

plt.figure()

plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc_rf)

plt.plot([0, 1], [0, 1], 'k--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Random Forest Model 1 ROC Curve')

plt.legend(loc="lower right")

plt.show()
train_results = []

test_results = []

# search for an optimal value of max_depth for decision tree

k_range = list(range(1,20))

k_scores = []

for k in k_range:

    rf = RandomForestClassifier(n_estimators=k, n_jobs=-1, max_depth=7)

    scores = cross_val_score(rf, features_train, target_train, cv=10, scoring='recall', n_jobs=-1)

    k_scores.append(scores.mean())

    # Code for plotting results

    rf.fit(features_train, target_train)

    train_pred = rf.predict(features_train)

    false_positive_rate, true_positive_rate, thresholds = roc_curve(target_train, train_pred)

    roc_auc = auc(false_positive_rate, true_positive_rate)

    # Add auc score to previous train results

    train_results.append(roc_auc)

    y_pred = rf.predict(features_test)

    false_positive_rate, true_positive_rate, thresholds = roc_curve(target_test, y_pred)

    roc_auc = auc(false_positive_rate, true_positive_rate)

    # Add auc score to previous test results

    test_results.append(roc_auc)

if DEBUG:

    print(k_scores)
from matplotlib.legend_handler import HandlerLine2D

line1, = plt.plot(k_range, train_results, 'b', label="Train AUC")

line2, = plt.plot(k_range, test_results, 'r', label="Test AUC")

plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})

plt.ylabel('AUC score')

plt.xlabel('N Estimators')

plt.show()
# Random Forest train model. Call up my model and name it clf

clf_rf1 = RandomForestClassifier(n_estimators=17, n_jobs=-1, max_depth=7)

#Call up the model to see the parameters you can tune (and their default setting)

print(clf_rf1)

#Fit clf to the training data

clf_rf1 = clf_rf1.fit(features_train, target_train)

#Predict clf DT model again test data

target_predicted_rf1 = clf_rf1.predict(features_test)
acc_rf1 = accuracy_score(target_test, target_predicted_rf1)

prec_rf1 = precision_score(target_test, target_predicted_rf1)

recall_rf1 = recall_score(target_test, target_predicted_rf1)

f1_rf1 = f1_score(target_test, target_predicted_rf1)

cm_rf1 = confusion_matrix(target_test, target_predicted_rf1)

print("RF Accuracy Score", acc_rf1)

print(classification_report(target_test, target_predicted_rf1))

if DEBUG:

    print(confusion_matrix(target_test, target_predicted_rf1))
# Source: https://seaborn.pydata.org/examples/heatmap_annotation.html

plt.figure(figsize=(5.5,4))

sns.heatmap(cm_rf1, annot=True, fmt='d')

plt.title('Random Forest Model 2 Confusion Matrix \nAccuracy:{0:.3f}'.format(acc_rf1))

plt.ylabel('True label')

plt.xlabel('Predicted label')

plt.show()
#verify RF with Cross Validation

scores_rf1 = cross_val_score(clf_rf1, features_train, target_train, cv=10, n_jobs=-1)

print("Cross Validation Score for each K",scores_rf1)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores_rf1.mean(), scores_rf1.std() * 2))
# Determine the false positive and true positive rates

fpr, tpr, _ = roc_curve(target_test, clf_rf1.predict_proba(features_test)[:,1]) 

    

# Calculate the AUC

roc_auc_rf1 = auc(fpr, tpr)

print('ROC AUC: %0.3f' % roc_auc_rf1)

 

# Plot of a ROC curve for a specific class

plt.figure()

plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc_rf1)

plt.plot([0, 1], [0, 1], 'k--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Random Forest Model 2 ROC Curve')

plt.legend(loc="lower right")

plt.show()
train_results = []

test_results = []

# search for an optimal value of max_depth for decision tree

k_range = list(range(1,20))

k_scores = []

for k in k_range:

    rf = RandomForestClassifier(n_estimators=17, n_jobs=-1, max_depth=7, min_samples_leaf=k)

    scores = cross_val_score(rf, features_train, target_train, cv=10, scoring='recall', n_jobs=-1)

    k_scores.append(scores.mean())

    # Code for plotting results

    rf.fit(features_train, target_train)

    train_pred = rf.predict(features_train)

    false_positive_rate, true_positive_rate, thresholds = roc_curve(target_train, train_pred)

    roc_auc = auc(false_positive_rate, true_positive_rate)

    # Add auc score to previous train results

    train_results.append(roc_auc)

    y_pred = rf.predict(features_test)

    false_positive_rate, true_positive_rate, thresholds = roc_curve(target_test, y_pred)

    roc_auc = auc(false_positive_rate, true_positive_rate)

    # Add auc score to previous test results

    test_results.append(roc_auc)

if DEBUG:

    print(k_scores)
from matplotlib.legend_handler import HandlerLine2D

line1, = plt.plot(k_range, train_results, 'b', label="Train AUC")

line2, = plt.plot(k_range, test_results, 'r', label="Test AUC")

plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})

plt.ylabel('AUC score')

plt.xlabel('Min Samples Leaf')

plt.show()
# Random Forest train model. Call up my model and name it clf

clf_rf2 = RandomForestClassifier(n_estimators=17, n_jobs=-1, max_depth=7, min_samples_leaf=14)

#Call up the model to see the parameters you can tune (and their default setting)

print(clf_rf2)

#Fit clf to the training data

clf_rf2 = clf_rf2.fit(features_train, target_train)

#Predict clf DT model again test data

target_predicted_rf2 = clf_rf2.predict(features_test)
acc_rf2 = accuracy_score(target_test, target_predicted_rf2)

prec_rf2 = precision_score(target_test, target_predicted_rf2)

recall_rf2 = recall_score(target_test, target_predicted_rf2)

f1_rf2 = f1_score(target_test, target_predicted_rf2)

cm_rf2 = confusion_matrix(target_test, target_predicted_rf2)

print("RF Accuracy Score", acc_rf2)

print(classification_report(target_test, target_predicted_rf2))

if DEBUG:

    print(confusion_matrix(target_test, target_predicted_rf2))
# Source: https://seaborn.pydata.org/examples/heatmap_annotation.html

plt.figure(figsize=(5.5,4))

sns.heatmap(cm_rf2, annot=True, fmt='d')

plt.title('Random Forest Model 3 Confusion Matrix \nAccuracy:{0:.3f}'.format(acc_rf2))

plt.ylabel('True label')

plt.xlabel('Predicted label')

plt.show()
#verify RF with Cross Validation

scores_rf2 = cross_val_score(clf_rf2, features_train, target_train, cv=10, n_jobs=-1)

print("Cross Validation Score for each K",scores_rf2)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores_rf2.mean(), scores_rf2.std() * 2))
# Determine the false positive and true positive rates

fpr, tpr, _ = roc_curve(target_test, clf_rf2.predict_proba(features_test)[:,1]) 

    

# Calculate the AUC

roc_auc_rf2 = auc(fpr, tpr)

print('ROC AUC: %0.3f' % roc_auc_rf2)

 

# Plot of a ROC curve for a specific class

plt.figure()

plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc_rf2)

plt.plot([0, 1], [0, 1], 'k--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Random Forest Model 3 ROC Curve')

plt.legend(loc="lower right")

plt.show()
t = Table()

t[''] = ['RF Model 1','RF Model 2','RF Model 3']

t['Cross Validation Score'] = [round(scores_rf.mean(),4),round(scores_rf1.mean(),4),round(scores_rf2.mean(),4)]

t['Accuracy Score'] = [round(acc_rf,4),round(acc_rf1,4),round(acc_rf2,4)]

t['Precision'] = [round(prec_rf,4),round(prec_rf1,4),round(prec_rf2,4)]

t['Recall'] = [round(recall_rf,4),round(recall_rf1,4),round(recall_rf2,4)]

t['F1 Score'] = [round(f1_rf,4),round(f1_rf1,4),round(f1_rf2,4)]

t['ROC AUC'] = [round(roc_auc_rf,4),round(roc_auc_rf1,4),round(roc_auc_rf2,4)]

t
from sklearn.ensemble import BaggingClassifier

# Random Forest train model. Call up my model and name it clf

clf_bag = BaggingClassifier(base_estimator=clf_knn2, n_jobs=-1)

#Call up the model to see the parameters you can tune (and their default setting)

print(clf_bag)

#Fit clf to the training data

clf_bag = clf_bag.fit(features_train, target_train)

#Predict clf DT model again test data

target_predicted_bag = clf_bag.predict(features_test)
acc_bag = accuracy_score(target_test, target_predicted_bag)

prec_bag = precision_score(target_test, target_predicted_bag)

recall_bag = recall_score(target_test, target_predicted_bag)

f1_bag = f1_score(target_test, target_predicted_bag)

cm_bag = confusion_matrix(target_test, target_predicted_bag)

print("Bag Accuracy Score", acc_bag)

print(classification_report(target_test, target_predicted_bag))

if DEBUG:

    print(confusion_matrix(target_test, target_predicted_bag))
# Source: https://seaborn.pydata.org/examples/heatmap_annotation.html

plt.figure(figsize=(5.5,4))

sns.heatmap(cm_bag, annot=True, fmt='d')

plt.title('Bagging Classifier Model 1 Confusion Matrix \nAccuracy:{0:.3f}'.format(acc_bag))

plt.ylabel('True label')

plt.xlabel('Predicted label')

plt.show()
#verify RF with Cross Validation

scores_bag = cross_val_score(clf_bag, features_train, target_train, cv=10, n_jobs=-1)

print("Cross Validation Score for each K",scores_bag)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores_bag.mean(), scores_bag.std() * 2))
# Determine the false positive and true positive rates

fpr, tpr, _ = roc_curve(target_test, clf_bag.predict_proba(features_test)[:,1]) 

    

# Calculate the AUC

roc_auc_bag = auc(fpr, tpr)

print('ROC AUC: %0.3f' % roc_auc_bag)

 

# Plot of a ROC curve for a specific class

plt.figure()

plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc_bag)

plt.plot([0, 1], [0, 1], 'k--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Bagging Classifier Model 1 ROC Curve')

plt.legend(loc="lower right")

plt.show()
train_results = []

test_results = []

# search for an optimal value of max_depth for decision tree

k_range = list(range(1,20))

k_scores = []

for k in k_range:

    bag = BaggingClassifier(base_estimator=clf_dt, n_estimators=k, n_jobs=-1)

    scores = cross_val_score(bag, features_train, target_train, cv=10, scoring='recall', n_jobs=-1)

    k_scores.append(scores.mean())

    # Code for plotting results

    bag.fit(features_train, target_train)

    train_pred = bag.predict(features_train)

    false_positive_rate, true_positive_rate, thresholds = roc_curve(target_train, train_pred)

    roc_auc = auc(false_positive_rate, true_positive_rate)

    # Add auc score to previous train results

    train_results.append(roc_auc)

    y_pred = bag.predict(features_test)

    false_positive_rate, true_positive_rate, thresholds = roc_curve(target_test, y_pred)

    roc_auc = auc(false_positive_rate, true_positive_rate)

    # Add auc score to previous test results

    test_results.append(roc_auc)

if DEBUG:

    print(k_scores)
from matplotlib.legend_handler import HandlerLine2D

line1, = plt.plot(k_range, train_results, 'b', label="Train AUC")

line2, = plt.plot(k_range, test_results, 'r', label="Test AUC")

plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})

plt.ylabel('AUC score')

plt.xlabel('N Estimators')

plt.show()
# Random Forest train model. Call up my model and name it clf

clf_bag1 = BaggingClassifier(base_estimator=clf_dt, n_estimators=4, n_jobs=-1)

#Call up the model to see the parameters you can tune (and their default setting)

print(clf_bag1)

#Fit clf to the training data

clf_bag1 = clf_bag1.fit(features_train, target_train)

#Predict clf DT model again test data

target_predicted_bag1 = clf_bag1.predict(features_test)
acc_bag1 = accuracy_score(target_test, target_predicted_bag1)

prec_bag1 = precision_score(target_test, target_predicted_bag1)

recall_bag1 = recall_score(target_test, target_predicted_bag1)

f1_bag1 = f1_score(target_test, target_predicted_bag1)

cm_bag1 = confusion_matrix(target_test, target_predicted_bag1)

print("Bag Accuracy Score", acc_bag1)

print(classification_report(target_test, target_predicted_bag1))

if DEBUG:

    print(confusion_matrix(target_test, target_predicted_bag1))
# Source: https://seaborn.pydata.org/examples/heatmap_annotation.html

plt.figure(figsize=(5.5,4))

sns.heatmap(cm_bag1, annot=True, fmt='d')

plt.title('Bagging Classifier Model 2 Confusion Matrix \nAccuracy:{0:.3f}'.format(acc_bag1))

plt.ylabel('True label')

plt.xlabel('Predicted label')

plt.show()
#verify RF with Cross Validation

scores_bag1 = cross_val_score(clf_bag1, features_train, target_train, cv=10, n_jobs=-1)

print("Cross Validation Score for each K",scores_bag1)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores_bag1.mean(), scores_bag1.std() * 2))
# Determine the false positive and true positive rates

fpr, tpr, _ = roc_curve(target_test, clf_bag1.predict_proba(features_test)[:,1]) 

    

# Calculate the AUC

roc_auc_bag1 = auc(fpr, tpr)

print('ROC AUC: %0.3f' % roc_auc_bag1)

 

# Plot of a ROC curve for a specific class

plt.figure()

plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc_bag1)

plt.plot([0, 1], [0, 1], 'k--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Bagging Classifier Model 2 ROC Curve')

plt.legend(loc="lower right")

plt.show()
train_results = []

test_results = []

# search for an optimal value of max_depth for decision tree

k_range = list(range(1,20))

k_scores = []

for k in k_range:

    bag = BaggingClassifier(base_estimator=clf_rf, n_estimators=k, n_jobs=-1)

    scores = cross_val_score(bag, features_train, target_train, cv=10, scoring='recall', n_jobs=-1)

    k_scores.append(scores.mean())

    # Code for plotting results

    bag.fit(features_train, target_train)

    train_pred = bag.predict(features_train)

    false_positive_rate, true_positive_rate, thresholds = roc_curve(target_train, train_pred)

    roc_auc = auc(false_positive_rate, true_positive_rate)

    # Add auc score to previous train results

    train_results.append(roc_auc)

    y_pred = bag.predict(features_test)

    false_positive_rate, true_positive_rate, thresholds = roc_curve(target_test, y_pred)

    roc_auc = auc(false_positive_rate, true_positive_rate)

    # Add auc score to previous test results

    test_results.append(roc_auc)

if DEBUG:

    print(k_scores)
from matplotlib.legend_handler import HandlerLine2D

line1, = plt.plot(k_range, train_results, 'b', label="Train AUC")

line2, = plt.plot(k_range, test_results, 'r', label="Test AUC")

plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})

plt.ylabel('AUC score')

plt.xlabel('N Estimators')

plt.show()
# Random Forest train model. Call up my model and name it clf

clf_bag2 = BaggingClassifier(base_estimator=clf_rf, n_estimators=10, n_jobs=-1)

#Call up the model to see the parameters you can tune (and their default setting)

print(clf_bag2)

#Fit clf to the training data

clf_bag2 = clf_bag2.fit(features_train, target_train)

#Predict clf DT model again test data

target_predicted_bag2 = clf_bag2.predict(features_test)
acc_bag2 = accuracy_score(target_test, target_predicted_bag2)

prec_bag2 = precision_score(target_test, target_predicted_bag2)

recall_bag2 = recall_score(target_test, target_predicted_bag2)

f1_bag2 = f1_score(target_test, target_predicted_bag2)

cm_bag2 = confusion_matrix(target_test, target_predicted_bag2)

print("Bag Accuracy Score", acc_bag2)

print(classification_report(target_test, target_predicted_bag2))

if DEBUG:

    print(confusion_matrix(target_test, target_predicted_bag2))
# Source: https://seaborn.pydata.org/examples/heatmap_annotation.html

plt.figure(figsize=(5.5,4))

sns.heatmap(cm_bag2, annot=True, fmt='d')

plt.title('Bagging Classifier Model 3 Confusion Matrix \nAccuracy:{0:.3f}'.format(acc_bag2))

plt.ylabel('True label')

plt.xlabel('Predicted label')

plt.show()
#verify RF with Cross Validation

scores_bag2 = cross_val_score(clf_bag2, features_train, target_train, cv=10, n_jobs=-1)

print("Cross Validation Score for each K",scores_bag2)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores_bag2.mean(), scores_bag2.std() * 2))
# Determine the false positive and true positive rates

fpr, tpr, _ = roc_curve(target_test, clf_bag2.predict_proba(features_test)[:,1]) 

    

# Calculate the AUC

roc_auc_bag2 = auc(fpr, tpr)

print('ROC AUC: %0.3f' % roc_auc_bag2)

 

# Plot of a ROC curve for a specific class

plt.figure()

plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc_bag2)

plt.plot([0, 1], [0, 1], 'k--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Bagging Classifier Model 3 ROC Curve')

plt.legend(loc="lower right")

plt.show()
t = Table()

t[''] = ['Bagging Classifier Model 1','Bagging Classifier Model 2','Bagging Classifier Model 3']

t['Cross Validation Score'] = [round(scores_bag.mean(),4),round(scores_bag1.mean(),4),round(scores_bag2.mean(),4)]

t['Accuracy Score'] = [round(acc_bag,4),round(acc_bag1,4),round(acc_bag2,4)]

t['Precision'] = [round(prec_bag,4),round(prec_bag1,4),round(prec_bag2,4)]

t['Recall'] = [round(recall_bag,4),round(recall_bag1,4),round(recall_bag2,4)]

t['F1 Score'] = [round(f1_bag,4),round(f1_bag1,4),round(f1_bag2,4)]

t['ROC AUC'] = [round(roc_auc_bag,4),round(roc_auc_bag1,4),round(roc_auc_bag2,4)]

t
from sklearn.ensemble import ExtraTreesClassifier

train_results = []

test_results = []

# search for an optimal value of max_depth for decision tree

k_range = list(range(1,15))

k_scores = []

for k in k_range:

    xdt = ExtraTreesClassifier(max_depth=k, n_jobs=-1)

    scores = cross_val_score(xdt, features_train, target_train, cv=10, scoring='recall', n_jobs=-1)

    k_scores.append(scores.mean())

    # Code for plotting results

    xdt.fit(features_train, target_train)

    train_pred = xdt.predict(features_train)

    false_positive_rate, true_positive_rate, thresholds = roc_curve(target_train, train_pred)

    roc_auc = auc(false_positive_rate, true_positive_rate)

    # Add auc score to previous train results

    train_results.append(roc_auc)

    y_pred = xdt.predict(features_test)

    false_positive_rate, true_positive_rate, thresholds = roc_curve(target_test, y_pred)

    roc_auc = auc(false_positive_rate, true_positive_rate)

    # Add auc score to previous test results

    test_results.append(roc_auc)

if DEBUG:

    print(k_scores)
from matplotlib.legend_handler import HandlerLine2D

line1, = plt.plot(k_range, train_results, 'b', label="Train AUC")

line2, = plt.plot(k_range, test_results, 'r', label="Test AUC")

plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})

plt.ylabel('AUC score')

plt.xlabel('Max Depth')

plt.show()
# Random Forest train model. Call up my model and name it clf

clf_xdt = ExtraTreesClassifier(max_depth=8, n_jobs=-1)

#Call up the model to see the parameters you can tune (and their default setting)

print(clf_xdt)

#Fit clf to the training data

clf_xdt = clf_xdt.fit(features_train, target_train)

#Predict clf DT model again test data

target_predicted_xdt = clf_xdt.predict(features_test)
acc_xdt = accuracy_score(target_test, target_predicted_xdt)

prec_xdt = precision_score(target_test, target_predicted_xdt)

recall_xdt = recall_score(target_test, target_predicted_xdt)

f1_xdt = f1_score(target_test, target_predicted_xdt)

cm_xdt = confusion_matrix(target_test, target_predicted_xdt)

print("XDT Accuracy Score", acc_xdt)

print(classification_report(target_test, target_predicted_xdt))

if DEBUG:

    print(confusion_matrix(target_test, target_predicted_xdt))
# Source: https://seaborn.pydata.org/examples/heatmap_annotation.html

plt.figure(figsize=(5.5,4))

sns.heatmap(cm_xdt, annot=True, fmt='d')

plt.title('Extra Trees Model 1 Confusion Matrix \nAccuracy:{0:.3f}'.format(acc_xdt))

plt.ylabel('True label')

plt.xlabel('Predicted label')

plt.show()
#verify RF with Cross Validation

scores_xdt = cross_val_score(clf_xdt, features_train, target_train, cv=10, n_jobs=-1)

print("Cross Validation Score for each K",scores_xdt)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores_xdt.mean(), scores_xdt.std() * 2))
# Determine the false positive and true positive rates

fpr, tpr, _ = roc_curve(target_test, clf_xdt.predict_proba(features_test)[:,1]) 

    

# Calculate the AUC

roc_auc_xdt = auc(fpr, tpr)

print('ROC AUC: %0.3f' % roc_auc_xdt)

 

# Plot of a ROC curve for a specific class

plt.figure()

plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc_xdt)

plt.plot([0, 1], [0, 1], 'k--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Extra Trees Model 1 ROC Curve')

plt.legend(loc="lower right")

plt.show()
train_results = []

test_results = []

# search for an optimal value of max_depth for decision tree

k_range = list(range(1,20))

k_scores = []

for k in k_range:

    xdt = ExtraTreesClassifier(min_samples_leaf=k, max_depth=8, n_jobs=-1)

    scores = cross_val_score(xdt, features_train, target_train, cv=10, scoring='recall', n_jobs=-1)

    k_scores.append(scores.mean())

    # Code for plotting results

    xdt.fit(features_train, target_train)

    train_pred = xdt.predict(features_train)

    false_positive_rate, true_positive_rate, thresholds = roc_curve(target_train, train_pred)

    roc_auc = auc(false_positive_rate, true_positive_rate)

    # Add auc score to previous train results

    train_results.append(roc_auc)

    y_pred = xdt.predict(features_test)

    false_positive_rate, true_positive_rate, thresholds = roc_curve(target_test, y_pred)

    roc_auc = auc(false_positive_rate, true_positive_rate)

    # Add auc score to previous test results

    test_results.append(roc_auc)

if DEBUG:

    print(k_scores)
from matplotlib.legend_handler import HandlerLine2D

line1, = plt.plot(k_range, train_results, 'b', label="Train AUC")

line2, = plt.plot(k_range, test_results, 'r', label="Test AUC")

plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})

plt.ylabel('AUC score')

plt.xlabel('Min Samples Leaf')

plt.show()
# Random Forest train model. Call up my model and name it clf

clf_xdt1 = ExtraTreesClassifier(min_samples_leaf=7, max_depth=8, n_jobs=-1)

#Call up the model to see the parameters you can tune (and their default setting)

print(clf_xdt1)

#Fit clf to the training data

clf_xdt1 = clf_xdt1.fit(features_train, target_train)

#Predict clf DT model again test data

target_predicted_xdt1 = clf_xdt1.predict(features_test)
acc_xdt1 = accuracy_score(target_test, target_predicted_xdt1)

prec_xdt1 = precision_score(target_test, target_predicted_xdt1)

recall_xdt1 = recall_score(target_test, target_predicted_xdt1)

f1_xdt1 = f1_score(target_test, target_predicted_xdt1)

cm_xdt1 = confusion_matrix(target_test, target_predicted_xdt1)

print("XDT Accuracy Score", acc_xdt1)

print(classification_report(target_test, target_predicted_xdt1))

if DEBUG:

    print(confusion_matrix(target_test, target_predicted_xdt1))
# Source: https://seaborn.pydata.org/examples/heatmap_annotation.html

plt.figure(figsize=(5.5,4))

sns.heatmap(cm_xdt1, annot=True, fmt='d')

plt.title('Extra Trees Model 2 Confusion Matrix \nAccuracy:{0:.3f}'.format(acc_xdt1))

plt.ylabel('True label')

plt.xlabel('Predicted label')

plt.show()
#verify RF with Cross Validation

scores_xdt1 = cross_val_score(clf_xdt1, features_train, target_train, cv=10, n_jobs=-1)

print("Cross Validation Score for each K",scores_xdt1)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores_xdt1.mean(), scores_xdt1.std() * 2))
# Determine the false positive and true positive rates

fpr, tpr, _ = roc_curve(target_test, clf_xdt1.predict_proba(features_test)[:,1]) 

    

# Calculate the AUC

roc_auc_xdt1 = auc(fpr, tpr)

print('ROC AUC: %0.3f' % roc_auc_xdt1)

 

# Plot of a ROC curve for a specific class

plt.figure()

plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc_xdt1)

plt.plot([0, 1], [0, 1], 'k--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Extra Trees Model 2 ROC Curve')

plt.legend(loc="lower right")

plt.show()
train_results = []

test_results = []

# search for an optimal value of max_depth for decision tree

k_range = list(range(50,70))

k_scores = []

for k in k_range:

    xdt = ExtraTreesClassifier(max_leaf_nodes=k, min_samples_leaf=7, max_depth=8, n_jobs=-1)

    scores = cross_val_score(xdt, features_train, target_train, cv=10, scoring='recall', n_jobs=-1)

    k_scores.append(scores.mean())

    # Code for plotting results

    xdt.fit(features_train, target_train)

    train_pred = xdt.predict(features_train)

    false_positive_rate, true_positive_rate, thresholds = roc_curve(target_train, train_pred)

    roc_auc = auc(false_positive_rate, true_positive_rate)

    # Add auc score to previous train results

    train_results.append(roc_auc)

    y_pred = xdt.predict(features_test)

    false_positive_rate, true_positive_rate, thresholds = roc_curve(target_test, y_pred)

    roc_auc = auc(false_positive_rate, true_positive_rate)

    # Add auc score to previous test results

    test_results.append(roc_auc)

if DEBUG:

    print(k_scores)
from matplotlib.legend_handler import HandlerLine2D

line1, = plt.plot(k_range, train_results, 'b', label="Train AUC")

line2, = plt.plot(k_range, test_results, 'r', label="Test AUC")

plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})

plt.ylabel('AUC score')

plt.xlabel('Max Leaf Nodes')

plt.show()
# Random Forest train model. Call up my model and name it clf

clf_xdt2 = ExtraTreesClassifier(max_leaf_nodes=66, min_samples_leaf=7, max_depth=8, n_jobs=-1)

#Call up the model to see the parameters you can tune (and their default setting)

print(clf_xdt2)

#Fit clf to the training data

clf_xdt2 = clf_xdt2.fit(features_train, target_train)

#Predict clf DT model again test data

target_predicted_xdt2 = clf_xdt2.predict(features_test)
acc_xdt2 = accuracy_score(target_test, target_predicted_xdt2)

prec_xdt2 = precision_score(target_test, target_predicted_xdt2)

recall_xdt2 = recall_score(target_test, target_predicted_xdt2)

f1_xdt2 = f1_score(target_test, target_predicted_xdt2)

cm_xdt2 = confusion_matrix(target_test, target_predicted_xdt2)

print("XDT Accuracy Score", acc_xdt2)

print(classification_report(target_test, target_predicted_xdt2))

if DEBUG:

    print(confusion_matrix(target_test, target_predicted_xdt2))
# Source: https://seaborn.pydata.org/examples/heatmap_annotation.html

plt.figure(figsize=(5.5,4))

sns.heatmap(cm_xdt2, annot=True, fmt='d')

plt.title('Extra Trees Model 3 Confusion Matrix \nAccuracy:{0:.3f}'.format(acc_xdt2))

plt.ylabel('True label')

plt.xlabel('Predicted label')

plt.show()
#verify RF with Cross Validation

scores_xdt2 = cross_val_score(clf_xdt2, features_train, target_train, cv=10, n_jobs=-1)

print("Cross Validation Score for each K",scores_xdt2)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores_xdt2.mean(), scores_xdt2.std() * 2))
# Determine the false positive and true positive rates

fpr, tpr, _ = roc_curve(target_test, clf_xdt2.predict_proba(features_test)[:,1]) 

    

# Calculate the AUC

roc_auc_xdt2 = auc(fpr, tpr)

print('ROC AUC: %0.3f' % roc_auc_xdt2)

 

# Plot of a ROC curve for a specific class

plt.figure()

plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc_xdt2)

plt.plot([0, 1], [0, 1], 'k--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Extra Trees Model 3 ROC Curve')

plt.legend(loc="lower right")

plt.show()
t = Table()

t[''] = ['XDT Model 1','XDT Model 2','XDT Model 3']

t['Cross Validation Score'] = [round(scores_xdt.mean(),4),round(scores_xdt1.mean(),4),round(scores_xdt2.mean(),4)]

t['Accuracy Score'] = [round(acc_xdt,4),round(acc_xdt1,4),round(acc_xdt2,4)]

t['Precision'] = [round(prec_xdt,4),round(prec_xdt1,4),round(prec_xdt2,4)]

t['Recall'] = [round(recall_xdt,4),round(recall_xdt1,4),round(recall_xdt2,4)]

t['F1 Score'] = [round(f1_xdt,4),round(f1_xdt1,4),round(f1_xdt2,4)]

t['ROC AUC'] = [round(roc_auc_xdt,4),round(roc_auc_xdt1,4),round(roc_auc_xdt2,4)]

t
from sklearn.ensemble import GradientBoostingClassifier

# Random Forest train model. Call up my model and name it clf

clf_gbc = GradientBoostingClassifier()

#Call up the model to see the parameters you can tune (and their default setting)

print(clf_gbc)

#Fit clf to the training data

clf_gbc = clf_gbc.fit(features_train, target_train)

#Predict clf DT model again test data

target_predicted_gbc = clf_gbc.predict(features_test)
acc_gbc = accuracy_score(target_test, target_predicted_gbc)

prec_gbc = precision_score(target_test, target_predicted_gbc)

recall_gbc = recall_score(target_test, target_predicted_gbc)

f1_gbc = f1_score(target_test, target_predicted_gbc)

cm_gbc = confusion_matrix(target_test, target_predicted_gbc)

print("GBC Accuracy Score", acc_gbc)

print(classification_report(target_test, target_predicted_gbc))

if DEBUG:

    print(confusion_matrix(target_test, target_predicted_gbc))
# Source: https://seaborn.pydata.org/examples/heatmap_annotation.html

plt.figure(figsize=(5.5,4))

sns.heatmap(cm_gbc, annot=True, fmt='d')

plt.title('Gradient Boost Model 1 Confusion Matrix \nAccuracy:{0:.3f}'.format(acc_gbc))

plt.ylabel('True label')

plt.xlabel('Predicted label')

plt.show()
#verify RF with Cross Validation

scores_gbc = cross_val_score(clf_gbc, features_train, target_train, cv=10, n_jobs=-1)

print("Cross Validation Score for each K",scores_gbc)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores_gbc.mean(), scores_gbc.std() * 2))
# Determine the false positive and true positive rates

fpr, tpr, _ = roc_curve(target_test, clf_gbc.predict_proba(features_test)[:,1]) 

    

# Calculate the AUC

roc_auc_gbc = auc(fpr, tpr)

print('ROC AUC: %0.3f' % roc_auc_gbc)

 

# Plot of a ROC curve for a specific class

plt.figure()

plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc_gbc)

plt.plot([0, 1], [0, 1], 'k--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Gradient Boost Model 1 ROC Curve')

plt.legend(loc="lower right")

plt.show()
train_results = []

test_results = []

# search for an optimal value of max_depth for decision tree

k_range = list(range(1,10))

k_scores = []

for k in k_range:

    gbc = GradientBoostingClassifier(max_depth=k)

    scores = cross_val_score(gbc, features_train, target_train, cv=10, scoring='recall', n_jobs=-1)

    k_scores.append(scores.mean())

    # Code for plotting results

    gbc.fit(features_train, target_train)

    train_pred = gbc.predict(features_train)

    false_positive_rate, true_positive_rate, thresholds = roc_curve(target_train, train_pred)

    roc_auc = auc(false_positive_rate, true_positive_rate)

    # Add auc score to previous train results

    train_results.append(roc_auc)

    y_pred = gbc.predict(features_test)

    false_positive_rate, true_positive_rate, thresholds = roc_curve(target_test, y_pred)

    roc_auc = auc(false_positive_rate, true_positive_rate)

    # Add auc score to previous test results

    test_results.append(roc_auc)

if DEBUG:

    print(k_scores)
from matplotlib.legend_handler import HandlerLine2D

line1, = plt.plot(k_range, train_results, 'b', label="Train AUC")

line2, = plt.plot(k_range, test_results, 'r', label="Test AUC")

plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})

plt.ylabel('AUC score')

plt.xlabel('Max Depth')

plt.show()
# Random Forest train model. Call up my model and name it clf

clf_gbc1 = GradientBoostingClassifier(max_depth=2)

#Call up the model to see the parameters you can tune (and their default setting)

print(clf_gbc1)

#Fit clf to the training data

clf_gbc1 = clf_gbc1.fit(features_train, target_train)

#Predict clf DT model again test data

target_predicted_gbc1 = clf_gbc1.predict(features_test)
acc_gbc1 = accuracy_score(target_test, target_predicted_gbc1)

prec_gbc1 = precision_score(target_test, target_predicted_gbc1)

recall_gbc1 = recall_score(target_test, target_predicted_gbc1)

f1_gbc1 = f1_score(target_test, target_predicted_gbc1)

cm_gbc1 = confusion_matrix(target_test, target_predicted_gbc1)

print("GBC Accuracy Score", acc_gbc1)

print(classification_report(target_test, target_predicted_gbc1))

if DEBUG:

    print(confusion_matrix(target_test, target_predicted_gbc1))
# Source: https://seaborn.pydata.org/examples/heatmap_annotation.html

plt.figure(figsize=(5.5,4))

sns.heatmap(cm_gbc1, annot=True, fmt='d')

plt.title('Gradient Boost Model 2 Confusion Matrix \nAccuracy:{0:.3f}'.format(acc_gbc1))

plt.ylabel('True label')

plt.xlabel('Predicted label')

plt.show()
#verify RF with Cross Validation

scores_gbc1 = cross_val_score(clf_gbc1, features_train, target_train, cv=10, n_jobs=-1)

print("Cross Validation Score for each K",scores_gbc1)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores_gbc1.mean(), scores_gbc1.std() * 2))
# Determine the false positive and true positive rates

fpr, tpr, _ = roc_curve(target_test, clf_gbc1.predict_proba(features_test)[:,1]) 

    

# Calculate the AUC

roc_auc_gbc1 = auc(fpr, tpr)

print('ROC AUC: %0.3f' % roc_auc_gbc1)

 

# Plot of a ROC curve for a specific class

plt.figure()

plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc_gbc1)

plt.plot([0, 1], [0, 1], 'k--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Gradient Boost Model 2 ROC Curve')

plt.legend(loc="lower right")

plt.show()
train_results = []

test_results = []

# search for an optimal value of max_depth for decision tree

k_range = list(range(100,200))

k_scores = []

for k in k_range:

    gbc = GradientBoostingClassifier(n_estimators=k, max_depth=2)

    scores = cross_val_score(gbc, features_train, target_train, cv=10, scoring='recall', n_jobs=-1)

    k_scores.append(scores.mean())

    # Code for plotting results

    gbc.fit(features_train, target_train)

    train_pred = gbc.predict(features_train)

    false_positive_rate, true_positive_rate, thresholds = roc_curve(target_train, train_pred)

    roc_auc = auc(false_positive_rate, true_positive_rate)

    # Add auc score to previous train results

    train_results.append(roc_auc)

    y_pred = gbc.predict(features_test)

    false_positive_rate, true_positive_rate, thresholds = roc_curve(target_test, y_pred)

    roc_auc = auc(false_positive_rate, true_positive_rate)

    # Add auc score to previous test results

    test_results.append(roc_auc)

if DEBUG:

    print(k_scores)
from matplotlib.legend_handler import HandlerLine2D

line1, = plt.plot(k_range, train_results, 'b', label="Train AUC")

line2, = plt.plot(k_range, test_results, 'r', label="Test AUC")

plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})

plt.ylabel('AUC score')

plt.xlabel('N Estimators')

plt.show()
# Random Forest train model. Call up my model and name it clf

clf_gbc2 = GradientBoostingClassifier(n_estimators=140, max_depth=2)

#Call up the model to see the parameters you can tune (and their default setting)

print(clf_gbc2)

#Fit clf to the training data

clf_gbc2 = clf_gbc2.fit(features_train, target_train)

#Predict clf DT model again test data

target_predicted_gbc2 = clf_gbc2.predict(features_test)
acc_gbc2 = accuracy_score(target_test, target_predicted_gbc2)

prec_gbc2 = precision_score(target_test, target_predicted_gbc2)

recall_gbc2 = recall_score(target_test, target_predicted_gbc2)

f1_gbc2 = f1_score(target_test, target_predicted_gbc2)

cm_gbc2 = confusion_matrix(target_test, target_predicted_gbc2)

print("GBC Accuracy Score", acc_gbc2)

print(classification_report(target_test, target_predicted_gbc2))

if DEBUG:

    print(confusion_matrix(target_test, target_predicted_gbc2))
# Source: https://seaborn.pydata.org/examples/heatmap_annotation.html

plt.figure(figsize=(5.5,4))

sns.heatmap(cm_gbc2, annot=True, fmt='d')

plt.title('Gradient Boost Model 3 Confusion Matrix \nAccuracy:{0:.3f}'.format(acc_gbc2))

plt.ylabel('True label')

plt.xlabel('Predicted label')

plt.show()
#verify RF with Cross Validation

scores_gbc2 = cross_val_score(clf_gbc2, features_train, target_train, cv=10, n_jobs=-1)

print("Cross Validation Score for each K",scores_gbc2)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores_gbc2.mean(), scores_gbc2.std() * 2))
# Determine the false positive and true positive rates

fpr, tpr, _ = roc_curve(target_test, clf_gbc2.predict_proba(features_test)[:,1]) 

    

# Calculate the AUC

roc_auc_gbc2 = auc(fpr, tpr)

print('ROC AUC: %0.3f' % roc_auc_gbc2)

 

# Plot of a ROC curve for a specific class

plt.figure()

plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc_gbc2)

plt.plot([0, 1], [0, 1], 'k--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Gradient Boost Model 3 ROC Curve')

plt.legend(loc="lower right")

plt.show()
t = Table()

t[''] = ['Gradient Boast Model 1','Gradient Boast Model 2','Gradient Boast Model 3']

t['Cross Validation Score'] = [round(scores_gbc.mean(),4),round(scores_gbc1.mean(),4),round(scores_gbc2.mean(),4)]

t['Accuracy Score'] = [round(acc_gbc,4),round(acc_gbc1,4),round(acc_gbc2,4)]

t['Precision'] = [round(prec_gbc,4),round(prec_gbc1,4),round(prec_gbc2,4)]

t['Recall'] = [round(recall_gbc,4),round(recall_gbc1,4),round(recall_gbc2,4)]

t['F1 Score'] = [round(f1_gbc,4),round(f1_gbc1,4),round(f1_gbc2,4)]

t['ROC AUC'] = [round(roc_auc_gbc,4),round(roc_auc_gbc1,4),round(roc_auc_gbc2,4)]

t
from sklearn.linear_model import SGDClassifier

# Random Forest train model. Call up my model and name it clf

clf_sgd_huber = SGDClassifier(loss='modified_huber', penalty='l2', n_jobs=-1, max_iter=1000)

#Call up the model to see the parameters you can tune (and their default setting)

print(clf_sgd_huber)

#Fit clf to the training data

clf_sgd_huber = clf_sgd_huber.fit(features_train, target_train)

#Predict clf DT model again test data

target_predicted_sgd_huber = clf_sgd_huber.predict(features_test)
acc_sgd_huber = accuracy_score(target_test, target_predicted_sgd_huber)

prec_sgd_huber = precision_score(target_test, target_predicted_sgd_huber)

recall_sgd_huber = recall_score(target_test, target_predicted_sgd_huber)

f1_sgd_huber = f1_score(target_test, target_predicted_sgd_huber)

cm_sgd_huber = confusion_matrix(target_test, target_predicted_sgd_huber)

print("SGD Accuracy Score", acc_sgd_huber)

print(classification_report(target_test, target_predicted_sgd_huber))

if DEBUG:

    print(confusion_matrix(target_test, target_predicted_sgd_huber))
# Source: https://seaborn.pydata.org/examples/heatmap_annotation.html

plt.figure(figsize=(5.5,4))

sns.heatmap(cm_sgd_huber, annot=True, fmt='d')

plt.title('Stochastic Gradient Descent "Modified Huber" Model Confusion Matrix \nAccuracy:{0:.3f}'.format(acc_sgd_huber))

plt.ylabel('True label')

plt.xlabel('Predicted label')

plt.show()
#verify RF with Cross Validation

scores_sgd_huber = cross_val_score(clf_sgd_huber, features_train, target_train, cv=10, n_jobs=-1)

print("Cross Validation Score for each K",scores_sgd_huber)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores_sgd_huber.mean(), scores_sgd_huber.std() * 2))
# Determine the false positive and true positive rates

fpr, tpr, _ = roc_curve(target_test, clf_sgd_huber.predict_proba(features_test)[:,1]) 

    

# Calculate the AUC

roc_auc_sgd_huber = auc(fpr, tpr)

print('ROC AUC: %0.3f' % roc_auc_sgd_huber)

 

# Plot of a ROC curve for a specific class

plt.figure()

plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc_sgd_huber)

plt.plot([0, 1], [0, 1], 'k--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Stochastic Gradient Descent "Modified Huber" Model ROC Curve')

plt.legend(loc="lower right")

plt.show()
# Random Forest train model. Call up my model and name it clf

clf_sgd_log = SGDClassifier(loss='log', n_jobs=-1, max_iter=1000)

#Call up the model to see the parameters you can tune (and their default setting)

print(clf_sgd_log)

#Fit clf to the training data

clf_sgd_log = clf_sgd_log.fit(features_train, target_train)

#Predict clf DT model again test data

target_predicted_sgd_log = clf_sgd_log.predict(features_test)
acc_sgd_log = accuracy_score(target_test, target_predicted_sgd_log)

prec_sgd_log = precision_score(target_test, target_predicted_sgd_log)

recall_sgd_log = recall_score(target_test, target_predicted_sgd_log)

f1_sgd_log = f1_score(target_test, target_predicted_sgd_log)

cm_sgd_log = confusion_matrix(target_test, target_predicted_sgd_log)

print("SGD Accuracy Score", acc_sgd_log)

print(classification_report(target_test, target_predicted_sgd_log))

if DEBUG:

    print(confusion_matrix(target_test, target_predicted_sgd_log))
# Source: https://seaborn.pydata.org/examples/heatmap_annotation.html

plt.figure(figsize=(5.5,4))

sns.heatmap(cm_sgd_log, annot=True, fmt='d')

plt.title('Stochastic Gradient Descent "Log" Model Confusion Matrix \nAccuracy:{0:.3f}'.format(acc_sgd_log))

plt.ylabel('True label')

plt.xlabel('Predicted label')

plt.show()
#verify RF with Cross Validation

scores_sgd_log = cross_val_score(clf_sgd_log, features_train, target_train, cv=10, n_jobs=-1)

print("Cross Validation Score for each K",scores_sgd_log)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores_sgd_log.mean(), scores_sgd_log.std() * 2))
# Determine the false positive and true positive rates

fpr, tpr, _ = roc_curve(target_test, clf_sgd_log.predict_proba(features_test)[:,1]) 

    

# Calculate the AUC

roc_auc_sgd_log = auc(fpr, tpr)

print('ROC AUC: %0.3f' % roc_auc_sgd_log)

 

# Plot of a ROC curve for a specific class

plt.figure()

plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc_sgd_log)

plt.plot([0, 1], [0, 1], 'k--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Stochastic Gradient Descent "Log" Model ROC Curve')

plt.legend(loc="lower right")

plt.show()
# Random Forest train model. Call up my model and name it clf

clf_sgd_logl1 = SGDClassifier(loss='log', penalty='l1', n_jobs=-1, max_iter=1000)

#Call up the model to see the parameters you can tune (and their default setting)

print(clf_sgd_logl1)

#Fit clf to the training data

clf_sgd_logl1 = clf_sgd_logl1.fit(features_train, target_train)

#Predict clf DT model again test data

target_predicted_sgd_logl1 = clf_sgd_logl1.predict(features_test)
acc_sgd_logl1 = accuracy_score(target_test, target_predicted_sgd_logl1)

prec_sgd_logl1 = precision_score(target_test, target_predicted_sgd_logl1)

recall_sgd_logl1 = recall_score(target_test, target_predicted_sgd_logl1)

f1_sgd_logl1 = f1_score(target_test, target_predicted_sgd_logl1)

cm_sgd_logl1 = confusion_matrix(target_test, target_predicted_sgd_logl1)

print("SGD Accuracy Score", acc_sgd_logl1)

print(classification_report(target_test, target_predicted_sgd_logl1))

if DEBUG:

    print(confusion_matrix(target_test, target_predicted_sgd_logl1))
# Source: https://seaborn.pydata.org/examples/heatmap_annotation.html

plt.figure(figsize=(5.5,4))

sns.heatmap(cm_sgd_logl1, annot=True, fmt='d')

plt.title('Stochastic Gradient Descent "Log L1" Model Confusion Matrix \nAccuracy:{0:.3f}'.format(acc_sgd_logl1))

plt.ylabel('True label')

plt.xlabel('Predicted label')

plt.show()
#verify RF with Cross Validation

scores_sgd_logl1 = cross_val_score(clf_sgd_logl1, features_train, target_train, cv=10, n_jobs=-1)

print("Cross Validation Score for each K",scores_sgd_logl1)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores_sgd_logl1.mean(), scores_sgd_logl1.std() * 2))
# Determine the false positive and true positive rates

fpr, tpr, _ = roc_curve(target_test, clf_sgd_logl1.predict_proba(features_test)[:,1]) 

    

# Calculate the AUC

roc_auc_sgd_logl1 = auc(fpr, tpr)

print('ROC AUC: %0.3f' % roc_auc_sgd_logl1)

 

# Plot of a ROC curve for a specific class

plt.figure()

plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc_sgd_logl1)

plt.plot([0, 1], [0, 1], 'k--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Stochastic Gradient Descent "Log L1" Model ROC Curve')

plt.legend(loc="lower right")

plt.show()
t = Table()

t[''] = ['SGD Modified Huber Model','SGD Log Model','SGD Log L1 Model']

t['Cross Validation Score'] = [round(scores_sgd_huber.mean(),4),round(scores_sgd_log.mean(),4),round(scores_sgd_logl1.mean(),4)]

t['Accuracy Score'] = [round(acc_sgd_huber,4),round(acc_sgd_log,4),round(acc_sgd_logl1,4)]

t['Precision'] = [round(prec_sgd_huber,4),round(prec_sgd_log,4),round(prec_sgd_logl1,4)]

t['Recall'] = [round(recall_sgd_huber,4),round(recall_sgd_log,4),round(recall_sgd_logl1,4)]

t['F1 Score'] = [round(f1_sgd_huber,4),round(f1_sgd_log,4),round(f1_sgd_logl1,4)]

t['ROC AUC'] = [round(roc_auc_sgd_huber,4),round(roc_auc_sgd_log,4),round(roc_auc_sgd_logl1,4)]

t
from sklearn.svm import LinearSVC

# Random Forest train model. Call up my model and name it clf

clf_lsvm = LinearSVC()

#Call up the model to see the parameters you can tune (and their default setting)

print(clf_lsvm)

#Fit clf to the training data

clf_lsvm = clf_lsvm.fit(features_train, target_train)

#Predict clf DT model again test data

target_predicted_lsvm = clf_lsvm.predict(features_test)
acc_lsvm = accuracy_score(target_test, target_predicted_lsvm)

prec_lsvm = precision_score(target_test, target_predicted_lsvm)

recall_lsvm = recall_score(target_test, target_predicted_lsvm)

f1_lsvm = f1_score(target_test, target_predicted_lsvm)

cm_lsvm = confusion_matrix(target_test, target_predicted_lsvm)

print("LSVM Accuracy Score", acc_lsvm)

print(classification_report(target_test, target_predicted_lsvm))

if DEBUG:

    print(confusion_matrix(target_test, target_predicted_lsvm))
# Source: https://seaborn.pydata.org/examples/heatmap_annotation.html

plt.figure(figsize=(5.5,4))

sns.heatmap(cm_lsvm, annot=True, fmt='d')

plt.title('Linear Support Vector Classification Model 1 Confusion Matrix \nAccuracy:{0:.3f}'.format(acc_lsvm))

plt.ylabel('True label')

plt.xlabel('Predicted label')

plt.show()
#verify RF with Cross Validation

scores_lsvm = cross_val_score(clf_lsvm, features_train, target_train, cv=10, n_jobs=-1)

print("Cross Validation Score for each K",scores_lsvm)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores_lsvm.mean(), scores_lsvm.std() * 2))
# Random Forest train model. Call up my model and name it clf

clf_lsvml1 = LinearSVC(penalty='l1',dual=False)

#Call up the model to see the parameters you can tune (and their default setting)

print(clf_lsvml1)

#Fit clf to the training data

clf_lsvml1 = clf_lsvml1.fit(features_train, target_train)

#Predict clf DT model again test data

target_predicted_lsvml1 = clf_lsvml1.predict(features_test)
acc_lsvml1 = accuracy_score(target_test, target_predicted_lsvml1)

prec_lsvml1 = precision_score(target_test, target_predicted_lsvml1)

recall_lsvml1 = recall_score(target_test, target_predicted_lsvml1)

f1_lsvml1 = f1_score(target_test, target_predicted_lsvml1)

cm_lsvml1 = confusion_matrix(target_test, target_predicted_lsvml1)

print("LSVM Accuracy Score", acc_lsvml1)

print(classification_report(target_test, target_predicted_lsvml1))

if DEBUG:

    print(confusion_matrix(target_test, target_predicted_lsvml1))
# Source: https://seaborn.pydata.org/examples/heatmap_annotation.html

plt.figure(figsize=(5.5,4))

sns.heatmap(cm_lsvml1, annot=True, fmt='d')

plt.title('Linear Support Vector Classification Model 2 Confusion Matrix \nAccuracy:{0:.3f}'.format(acc_lsvml1))

plt.ylabel('True label')

plt.xlabel('Predicted label')

plt.show()
#verify RF with Cross Validation

scores_lsvml1 = cross_val_score(clf_lsvml1, features_train, target_train, cv=10, n_jobs=-1)

print("Cross Validation Score for each K",scores_lsvml1)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores_lsvml1.mean(), scores_lsvml1.std() * 2))
train_results = []

test_results = []

# search for an optimal value of max_depth for decision tree

k_range = list(range(1,10))

k_scores = []

for k in k_range:

    lsvm = LinearSVC(penalty='l2',C=k,dual=False,fit_intercept=False)

    scores = cross_val_score(lsvm, features_train, target_train, cv=10, scoring='recall', n_jobs=-1)

    k_scores.append(scores.mean())

    # Code for plotting results

    lsvm.fit(features_train, target_train)

    train_pred = lsvm.predict(features_train)

    false_positive_rate, true_positive_rate, thresholds = roc_curve(target_train, train_pred)

    roc_auc = auc(false_positive_rate, true_positive_rate)

    # Add auc score to previous train results

    train_results.append(roc_auc)

    y_pred = lsvm.predict(features_test)

    false_positive_rate, true_positive_rate, thresholds = roc_curve(target_test, y_pred)

    roc_auc = auc(false_positive_rate, true_positive_rate)

    # Add auc score to previous test results

    test_results.append(roc_auc)

if DEBUG:

    print(k_scores)
from matplotlib.legend_handler import HandlerLine2D

line1, = plt.plot(k_range, train_results, 'b', label="Train AUC")

line2, = plt.plot(k_range, test_results, 'r', label="Test AUC")

plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})

plt.ylabel('AUC score')

plt.xlabel('C Value')

plt.show()
# Random Forest train model. Call up my model and name it clf

clf_lsvml2 = LinearSVC(penalty='l2',C=3,dual=False,fit_intercept=False)

#Call up the model to see the parameters you can tune (and their default setting)

print(clf_lsvml2)

#Fit clf to the training data

clf_lsvml2 = clf_lsvml2.fit(features_train, target_train)

#Predict clf DT model again test data

target_predicted_lsvml2 = clf_lsvml2.predict(features_test)
acc_lsvml2 = accuracy_score(target_test, target_predicted_lsvml2)

prec_lsvml2 = precision_score(target_test, target_predicted_lsvml2)

recall_lsvml2 = recall_score(target_test, target_predicted_lsvml2)

f1_lsvml2 = f1_score(target_test, target_predicted_lsvml2)

cm_lsvml2 = confusion_matrix(target_test, target_predicted_lsvml2)

print("LSVM Accuracy Score", acc_lsvml2)

print(classification_report(target_test, target_predicted_lsvml2))

if DEBUG:

    print(confusion_matrix(target_test, target_predicted_lsvml2))
# Source: https://seaborn.pydata.org/examples/heatmap_annotation.html

plt.figure(figsize=(5.5,4))

sns.heatmap(cm_lsvml2, annot=True, fmt='d')

plt.title('Linear Support Vector Classification Model 3 Confusion Matrix \nAccuracy:{0:.3f}'.format(acc_lsvml2))

plt.ylabel('True label')

plt.xlabel('Predicted label')

plt.show()
#verify RF with Cross Validation

scores_lsvml2 = cross_val_score(clf_lsvml2, features_train, target_train, cv=10, n_jobs=-1)

print("Cross Validation Score for each K",scores_lsvml2)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores_lsvml2.mean(), scores_lsvml2.std() * 2))
t = Table()

t[''] = ['Linear SVC Model 1','Linear SVC Model 2','Linear SVC Model 3']

t['Cross Validation Score'] = [round(scores_lsvm.mean(),4),round(scores_lsvml1.mean(),4),round(scores_lsvml2.mean(),4)]

t['Accuracy Score'] = [round(acc_lsvm,4),round(acc_lsvml1,4),round(acc_lsvml2,4)]

t['Precision'] = [round(prec_lsvm,4),round(prec_lsvml1,4),round(prec_lsvml2,4)]

t['Recall'] = [round(recall_lsvm,4),round(recall_lsvml1,4),round(recall_lsvml2,4)]

t['F1 Score'] = [round(f1_lsvm,4),round(f1_lsvml1,4),round(f1_lsvml2,4)]

t
from sklearn.svm import SVC

# Random Forest train model. Call up my model and name it clf

clf_svc = SVC(probability=True, gamma='auto', max_iter=-1)

#Call up the model to see the parameters you can tune (and their default setting)

print(clf_svc)

#Fit clf to the training data

clf_svc = clf_svc.fit(features_train, target_train)

#Predict clf DT model again test data

target_predicted_svc = clf_svc.predict(features_test)
acc_svc = accuracy_score(target_test, target_predicted_svc)

prec_svc = precision_score(target_test, target_predicted_svc)

recall_svc = recall_score(target_test, target_predicted_svc)

f1_svc = f1_score(target_test, target_predicted_svc)

cm_svc = confusion_matrix(target_test, target_predicted_svc)

print("SVC Accuracy Score", acc_svc)

print(classification_report(target_test, target_predicted_svc))

if DEBUG:

    print(confusion_matrix(target_test, target_predicted_svc))
# Source: https://seaborn.pydata.org/examples/heatmap_annotation.html

plt.figure(figsize=(5.5,4))

sns.heatmap(cm_svc, annot=True, fmt='d')

plt.title('SVC Mode 1 Confusion Matrix \nAccuracy:{0:.3f}'.format(acc_svc))

plt.ylabel('True label')

plt.xlabel('Predicted label')

plt.show()
#verify RF with Cross Validation

scores_svc = cross_val_score(clf_svc, features_train, target_train, cv=10, n_jobs=-1)

print("Cross Validation Score for each K",scores_svc)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores_svc.mean(), scores_svc.std() * 2))
# Determine the false positive and true positive rates

fpr, tpr, _ = roc_curve(target_test, clf_svc.predict_proba(features_test)[:,1]) 

    

# Calculate the AUC

roc_auc_svc = auc(fpr, tpr)

print('ROC AUC: %0.3f' % roc_auc_svc)

 

# Plot of a ROC curve for a specific class

plt.figure()

plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc_svc)

plt.plot([0, 1], [0, 1], 'k--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('SVM Model 1 ROC Curve')

plt.legend(loc="lower right")

plt.show()
# Random Forest train model. Call up my model and name it clf

clf_svc1 = SVC(probability=True, gamma='scale', max_iter=-1)

#Call up the model to see the parameters you can tune (and their default setting)

print(clf_svc1)

#Fit clf to the training data

clf_svc1 = clf_svc1.fit(features_train, target_train)

#Predict clf DT model again test data

target_predicted_svc1 = clf_svc1.predict(features_test)
acc_svc1 = accuracy_score(target_test, target_predicted_svc1)

prec_svc1 = precision_score(target_test, target_predicted_svc1)

recall_svc1 = recall_score(target_test, target_predicted_svc1)

f1_svc1 = f1_score(target_test, target_predicted_svc1)

cm_svc1 = confusion_matrix(target_test, target_predicted_svc1)

print("SVC Accuracy Score", acc_svc1)

print(classification_report(target_test, target_predicted_svc1))

if DEBUG:

    print(confusion_matrix(target_test, target_predicted_svc1))
# Source: https://seaborn.pydata.org/examples/heatmap_annotation.html

plt.figure(figsize=(5.5,4))

sns.heatmap(cm_svc1, annot=True, fmt='d')

plt.title('SVC Model 2 Confusion Matrix \nAccuracy:{0:.3f}'.format(acc_svc1))

plt.ylabel('True label')

plt.xlabel('Predicted label')

plt.show()
#verify RF with Cross Validation

scores_svc1 = cross_val_score(clf_svc1, features_train, target_train, cv=10, n_jobs=-1)

print("Cross Validation Score for each K",scores_svc1)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores_svc1.mean(), scores_svc1.std() * 2))
# Determine the false positive and true positive rates

fpr, tpr, _ = roc_curve(target_test, clf_svc1.predict_proba(features_test)[:,1]) 

    

# Calculate the AUC

roc_auc_svc1 = auc(fpr, tpr)

print('ROC AUC: %0.3f' % roc_auc_svc1)

 

# Plot of a ROC curve for a specific class

plt.figure()

plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc_svc1)

plt.plot([0, 1], [0, 1], 'k--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Support Vector RBF Model 2 ROC Curve')

plt.legend(loc="lower right")

plt.show()
train_results = []

test_results = []

# search for an optimal value of max_depth for decision tree

#k_range = list(range(1,100))

k_range = np.linspace(0.1, 2.0, 20, endpoint=True)

k_scores = []

for k in k_range:

    svc = SVC(C=k, probability=True, gamma='scale', max_iter=-1)

    scores = cross_val_score(svc, features_train, target_train, cv=10, scoring='recall', n_jobs=-1)

    k_scores.append(scores.mean())

    # Code for plotting results

    svc.fit(features_train, target_train)

    train_pred = svc.predict(features_train)

    false_positive_rate, true_positive_rate, thresholds = roc_curve(target_train, train_pred)

    roc_auc = auc(false_positive_rate, true_positive_rate)

    # Add auc score to previous train results

    train_results.append(roc_auc)

    y_pred = svc.predict(features_test)

    false_positive_rate, true_positive_rate, thresholds = roc_curve(target_test, y_pred)

    roc_auc = auc(false_positive_rate, true_positive_rate)

    # Add auc score to previous test results

    test_results.append(roc_auc)

if DEBUG:

    print(k_scores)
from matplotlib.legend_handler import HandlerLine2D

line1, = plt.plot(k_range, train_results, 'b', label="Train AUC")

line2, = plt.plot(k_range, test_results, 'r', label="Test AUC")

plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})

plt.ylabel('AUC score')

plt.xlabel('C Value')

plt.show()
# Random Forest train model. Call up my model and name it clf

clf_svc2 = SVC(C=1.25, probability=True, gamma='scale', max_iter=-1)

#Call up the model to see the parameters you can tune (and their default setting)

print(clf_svc2)

#Fit clf to the training data

clf_svc2 = clf_svc2.fit(features_train, target_train)

#Predict clf DT model again test data

target_predicted_svc2 = clf_svc2.predict(features_test)
acc_svc2 = accuracy_score(target_test, target_predicted_svc2)

prec_svc2 = precision_score(target_test, target_predicted_svc2)

recall_svc2 = recall_score(target_test, target_predicted_svc2)

f1_svc2 = f1_score(target_test, target_predicted_svc2)

cm_svc2 = confusion_matrix(target_test, target_predicted_svc2)

print("SVC Accuracy Score", acc_svc2)

print(classification_report(target_test, target_predicted_svc2))

if DEBUG:

    print(confusion_matrix(target_test, target_predicted_svc2))
# Source: https://seaborn.pydata.org/examples/heatmap_annotation.html

plt.figure(figsize=(5.5,4))

sns.heatmap(cm_svc2, annot=True, fmt='d')

plt.title('Support Vector RBF Model 3 Confusion Matrix \nAccuracy:{0:.3f}'.format(acc_svc2))

plt.ylabel('True label')

plt.xlabel('Predicted label')

plt.show()
#verify RF with Cross Validation

scores_svc2 = cross_val_score(clf_svc2, features_train, target_train, cv=10, n_jobs=-1)

print("Cross Validation Score for each K",scores_svc2)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores_svc2.mean(), scores_svc2.std() * 2))
# Determine the false positive and true positive rates

fpr, tpr, _ = roc_curve(target_test, clf_svc2.predict_proba(features_test)[:,1]) 

    

# Calculate the AUC

roc_auc_svc2 = auc(fpr, tpr)

print('ROC AUC: %0.3f' % roc_auc_svc2)

 

# Plot of a ROC curve for a specific class

plt.figure()

plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc_svc2)

plt.plot([0, 1], [0, 1], 'k--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Support Vector RBF Model 3 ROC Curve')

plt.legend(loc="lower right")

plt.show()
t = Table()

t[''] = ['Support Vector RBF Model 1','Support Vector RBF Model 2','Support Vector RBF Model 3']

t['Cross Validation Score'] = [round(scores_svc.mean(),4),round(scores_svc1.mean(),4),round(scores_svc2.mean(),4)]

t['Accuracy Score'] = [round(acc_svc,4),round(acc_svc1,4),round(acc_svc2,4)]

t['Precision'] = [round(prec_svc,4),round(prec_svc1,4),round(prec_svc2,4)]

t['Recall'] = [round(recall_svc,4),round(recall_svc1,4),round(recall_svc2,4)]

t['F1 Score'] = [round(f1_svc,4),round(f1_svc1,4),round(f1_svc2,4)]

t
from sklearn.neural_network import MLPClassifier

# Random Forest train model. Call up my model and name it clf

clf_NN = MLPClassifier(activation='tanh', solver='adam', hidden_layer_sizes=(20,8), max_iter=1000)

#Call up the model to see the parameters you can tune (and their default setting)

print(clf_NN)

#Fit clf to the training data

clf_NN = clf_NN.fit(features_train, target_train)

#Predict clf DT model again test data

target_predicted_NN = clf_NN.predict(features_test)
acc_NN = accuracy_score(target_test, target_predicted_NN)

prec_NN = precision_score(target_test, target_predicted_NN)

recall_NN = recall_score(target_test, target_predicted_NN)

f1_NN = f1_score(target_test, target_predicted_NN)

cm_NN = confusion_matrix(target_test, target_predicted_NN)

print("MLP Accuracy Score", acc_NN)

print(classification_report(target_test, target_predicted_NN))

if DEBUG:

    print(confusion_matrix(target_test, target_predicted_NN))
# Source: https://seaborn.pydata.org/examples/heatmap_annotation.html

plt.figure(figsize=(5.5,4))

sns.heatmap(cm_NN, annot=True, fmt='d')

plt.title('MLP Model 1 Confusion Matrix \nAccuracy:{0:.3f}'.format(acc_NN))

plt.ylabel('True label')

plt.xlabel('Predicted label')

plt.show()
#verify RF with Cross Validation

scores_NN = cross_val_score(clf_NN, features_train, target_train, cv=10, n_jobs=-1)

print("Cross Validation Score for each K",scores_NN)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores_NN.mean(), scores_NN.std() * 2))
# Determine the false positive and true positive rates

fpr, tpr, _ = roc_curve(target_test, clf_NN.predict_proba(features_test)[:,1]) 

    

# Calculate the AUC

roc_auc_NN = auc(fpr, tpr)

print('ROC AUC: %0.3f' % roc_auc_NN)

 

# Plot of a ROC curve for a specific class

plt.figure()

plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc_NN)

plt.plot([0, 1], [0, 1], 'k--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('MLP Model 1 ROC Curve')

plt.legend(loc="lower right")

plt.show()
# Random Forest train model. Call up my model and name it clf

clf_NN1 =MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',

       beta_1=0.9, beta_2=0.999, early_stopping=False,

       epsilon=1e-08, hidden_layer_sizes=(15,), learning_rate='constant',

       learning_rate_init=0.001, max_iter=1000, momentum=0.9,

       nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,

       solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,

       warm_start=False)

#Call up the model to see the parameters you can tune (and their default setting)

print(clf_NN1)

#Fit clf to the training data

clf_NN1 = clf_NN1.fit(features_train, target_train)

#Predict clf DT model again test data

target_predicted_NN1 = clf_NN1.predict(features_test)
acc_NN1 = accuracy_score(target_test, target_predicted_NN1)

prec_NN1 = precision_score(target_test, target_predicted_NN1)

recall_NN1 = recall_score(target_test, target_predicted_NN1)

f1_NN1 = f1_score(target_test, target_predicted_NN1)

cm_NN1 = confusion_matrix(target_test, target_predicted_NN1)

print("MLP Accuracy Score", acc_NN1)

print(classification_report(target_test, target_predicted_NN1))

if DEBUG:

    print(confusion_matrix(target_test, target_predicted_NN1))
# Source: https://seaborn.pydata.org/examples/heatmap_annotation.html

plt.figure(figsize=(5.5,4))

sns.heatmap(cm_NN1, annot=True, fmt='d')

plt.title('MLP Model 2 Confusion Matrix \nAccuracy:{0:.3f}'.format(acc_NN1))

plt.ylabel('True label')

plt.xlabel('Predicted label')

plt.show()
#verify RF with Cross Validation

scores_NN1 = cross_val_score(clf_NN1, features_train, target_train, cv=10, n_jobs=-1)

print("Cross Validation Score for each K",scores_NN1)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores_NN1.mean(), scores_NN1.std() * 2))
# Determine the false positive and true positive rates

fpr, tpr, _ = roc_curve(target_test, clf_NN1.predict_proba(features_test)[:,1]) 

    

# Calculate the AUC

roc_auc_NN1 = auc(fpr, tpr)

print('ROC AUC: %0.3f' % roc_auc_NN1)

 

# Plot of a ROC curve for a specific class

plt.figure()

plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc_NN1)

plt.plot([0, 1], [0, 1], 'k--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('MLP Model 2 ROC Curve')

plt.legend(loc="lower right")

plt.show()
# Random Forest train model. Call up my model and name it clf

clf_NN2 = MLPClassifier(solver='sgd', hidden_layer_sizes=(30,30,30),max_iter=1000)

#Call up the model to see the parameters you can tune (and their default setting)

print(clf_NN2)

#Fit clf to the training data

clf_NN2 = clf_NN2.fit(features_train, target_train)

#Predict clf DT model again test data

target_predicted_NN2 = clf_NN2.predict(features_test)
acc_NN2 = accuracy_score(target_test, target_predicted_NN2)

prec_NN2 = precision_score(target_test, target_predicted_NN2)

recall_NN2 = recall_score(target_test, target_predicted_NN2)

f1_NN2 = f1_score(target_test, target_predicted_NN2)

cm_NN2 = confusion_matrix(target_test, target_predicted_NN2)

print("MLP Accuracy Score", acc_NN2)

print(classification_report(target_test, target_predicted_NN2))

if DEBUG:

    print(confusion_matrix(target_test, target_predicted_NN2))
# Source: https://seaborn.pydata.org/examples/heatmap_annotation.html

plt.figure(figsize=(5.5,4))

sns.heatmap(cm_NN2, annot=True, fmt='d')

plt.title('MLP Model 3 Confusion Matrix \nAccuracy:{0:.3f}'.format(acc_NN2))

plt.ylabel('True label')

plt.xlabel('Predicted label')

plt.show()
#verify RF with Cross Validation

scores_NN2 = cross_val_score(clf_NN2, features_train, target_train, cv=10, n_jobs=-1)

print("Cross Validation Score for each K",scores_NN2)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores_NN2.mean(), scores_NN2.std() * 2))
# Determine the false positive and true positive rates

fpr, tpr, _ = roc_curve(target_test, clf_NN2.predict_proba(features_test)[:,1]) 

    

# Calculate the AUC

roc_auc_NN2 = auc(fpr, tpr)

print('ROC AUC: %0.3f' % roc_auc_NN2)

 

# Plot of a ROC curve for a specific class

plt.figure()

plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc_NN2)

plt.plot([0, 1], [0, 1], 'k--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('MLP Model 3 ROC Curve')

plt.legend(loc="lower right")

plt.show()
t = Table()

t[''] = ['MLP Classifier Model 1','MLP Classifier Model 2','MLP Classifier Model 3']

t['Cross Validation Score'] = [round(scores_NN.mean(),4),round(scores_NN1.mean(),4),round(scores_NN2.mean(),4)]

t['Accuracy Score'] = [round(acc_NN,4),round(acc_NN1,4),round(acc_NN2,4)]

t['Precision'] = [round(prec_NN,4),round(prec_NN1,4),round(prec_NN2,4)]

t['Recall'] = [round(recall_NN,4),round(recall_NN1,4),round(recall_NN2,4)]

t['F1 Score'] = [round(f1_NN,4),round(f1_NN1,4),round(f1_NN2,4)]

t['ROC AUC'] = [round(roc_auc_NN,4),round(roc_auc_NN1,4),round(roc_auc_NN2,4)]

t
from sklearn.ensemble import AdaBoostClassifier

# Random Forest train model. Call up my model and name it clf

clf_ada = AdaBoostClassifier(base_estimator=clf_dt2)

#Call up the model to see the parameters you can tune (and their default setting)

print(clf_ada)

#Fit clf to the training data

clf_ada = clf_ada.fit(features_train, target_train)

#Predict clf DT model again test data

target_predicted_ada = clf_ada.predict(features_test)
acc_ada = accuracy_score(target_test, target_predicted_ada)

prec_ada = precision_score(target_test, target_predicted_ada)

recall_ada = recall_score(target_test, target_predicted_ada)

f1_ada = f1_score(target_test, target_predicted_ada)

cm_ada = confusion_matrix(target_test, target_predicted_ada)

print("Ada Accuracy Score", acc_ada)

print(classification_report(target_test, target_predicted_ada))

if DEBUG:

    print(confusion_matrix(target_test, target_predicted_ada))
# Source: https://seaborn.pydata.org/examples/heatmap_annotation.html

plt.figure(figsize=(5.5,4))

sns.heatmap(cm_ada, annot=True, fmt='d')

plt.title('AdaBoost Classifier Model 1 Confusion Matrix \nAccuracy:{0:.3f}'.format(acc_ada))

plt.ylabel('True label')

plt.xlabel('Predicted label')

plt.show()
#verify RF with Cross Validation

scores_ada = cross_val_score(clf_ada, features_train, target_train, cv=10, n_jobs=-1)

print("Cross Validation Score for each K",scores_ada)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores_ada.mean(), scores_ada.std() * 2))
# Determine the false positive and true positive rates

fpr, tpr, _ = roc_curve(target_test, clf_ada.predict_proba(features_test)[:,1]) 

    

# Calculate the AUC

roc_auc_ada = auc(fpr, tpr)

print('ROC AUC: %0.3f' % roc_auc_ada)

 

# Plot of a ROC curve for a specific class

plt.figure()

plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc_ada)

plt.plot([0, 1], [0, 1], 'k--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('AdaBoost Classifier Model 1 ROC Curve')

plt.legend(loc="lower right")

plt.show()
# Random Forest train model. Call up my model and name it clf

clf_ada1 = AdaBoostClassifier(base_estimator=clf_gbc2)

#Call up the model to see the parameters you can tune (and their default setting)

print(clf_ada1)

#Fit clf to the training data

clf_ada1 = clf_ada1.fit(features_train, target_train)

#Predict clf DT model again test data

target_predicted_ada1 = clf_ada1.predict(features_test)
acc_ada1 = accuracy_score(target_test, target_predicted_ada1)

prec_ada1 = precision_score(target_test, target_predicted_ada1)

recall_ada1 = recall_score(target_test, target_predicted_ada1)

f1_ada1 = f1_score(target_test, target_predicted_ada1)

cm_ada1 = confusion_matrix(target_test, target_predicted_ada1)

print("Ada Accuracy Score", acc_ada1)

print(classification_report(target_test, target_predicted_ada1))

if DEBUG:

    print(confusion_matrix(target_test, target_predicted_ada1))
# Source: https://seaborn.pydata.org/examples/heatmap_annotation.html

plt.figure(figsize=(5.5,4))

sns.heatmap(cm_ada1, annot=True, fmt='d')

plt.title('AdaBoost Classifier Model 2 Confusion Matrix \nAccuracy:{0:.3f}'.format(acc_ada1))

plt.ylabel('True label')

plt.xlabel('Predicted label')

plt.show()
#verify RF with Cross Validation

scores_ada1 = cross_val_score(clf_ada1, features_train, target_train, cv=10, n_jobs=-1)

print("Cross Validation Score for each K",scores_ada1)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores_ada1.mean(), scores_ada1.std() * 2))
# Determine the false positive and true positive rates

fpr, tpr, _ = roc_curve(target_test, clf_ada1.predict_proba(features_test)[:,1]) 

    

# Calculate the AUC

roc_auc_ada1 = auc(fpr, tpr)

print('ROC AUC: %0.3f' % roc_auc_ada1)

 

# Plot of a ROC curve for a specific class

plt.figure()

plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc_ada1)

plt.plot([0, 1], [0, 1], 'k--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('AdaBoost Classifier Model 2 ROC Curve')

plt.legend(loc="lower right")

plt.show()
# Random Forest train model. Call up my model and name it clf

clf_ada2 = AdaBoostClassifier(base_estimator=clf_svc)

#Call up the model to see the parameters you can tune (and their default setting)

print(clf_ada2)

#Fit clf to the training data

clf_ada2 = clf_ada2.fit(features_train, target_train)

#Predict clf DT model again test data

target_predicted_ada2 = clf_ada2.predict(features_test)
acc_ada2 = accuracy_score(target_test, target_predicted_ada2)

prec_ada2 = precision_score(target_test, target_predicted_ada2)

recall_ada2 = recall_score(target_test, target_predicted_ada2)

f1_ada2 = f1_score(target_test, target_predicted_ada2)

cm_ada2 = confusion_matrix(target_test, target_predicted_ada2)

print("Ada Accuracy Score", acc_ada2)

print(classification_report(target_test, target_predicted_ada2))

if DEBUG:

    print(confusion_matrix(target_test, target_predicted_ada2))
# Source: https://seaborn.pydata.org/examples/heatmap_annotation.html

plt.figure(figsize=(5.5,4))

sns.heatmap(cm_ada2, annot=True, fmt='d')

plt.title('AdaBoost Classifier Model 3 Confusion Matrix \nAccuracy:{0:.3f}'.format(acc_ada2))

plt.ylabel('True label')

plt.xlabel('Predicted label')

plt.show()
#verify RF with Cross Validation

scores_ada2 = cross_val_score(clf_ada2, features_train, target_train, cv=10, n_jobs=-1)

print("Cross Validation Score for each K",scores_ada2)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores_ada2.mean(), scores_ada2.std() * 2))
# Determine the false positive and true positive rates

fpr, tpr, _ = roc_curve(target_test, clf_ada2.predict_proba(features_test)[:,1]) 

    

# Calculate the AUC

roc_auc_ada2 = auc(fpr, tpr)

print('ROC AUC: %0.3f' % roc_auc_ada2)

 

# Plot of a ROC curve for a specific class

plt.figure()

plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc_ada2)

plt.plot([0, 1], [0, 1], 'k--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('AdaBoost Classifier Model 3 ROC Curve')

plt.legend(loc="lower right")

plt.show()
t = Table()

t[''] = ['AdaBoost Classifier Model 1','AdaBoost Classifier Model 2','AdaBoost Classifier Model 3']

t['Cross Validation Score'] = [round(scores_ada.mean(),4),round(scores_ada1.mean(),4),round(scores_ada2.mean(),4)]

t['Accuracy Score'] = [round(acc_ada,4),round(acc_ada1,4),round(acc_ada2,4)]

t['Precision'] = [round(prec_ada,4),round(prec_ada1,4),round(prec_ada2,4)]

t['Recall'] = [round(recall_ada,4),round(recall_ada1,4),round(recall_ada2,4)]

t['F1 Score'] = [round(f1_ada,4),round(f1_ada1,4),round(f1_ada2,4)]

t
from sklearn.ensemble import VotingClassifier

# Random Forest train model. Call up my model and name it clf

clf1 = clf_knn2

clf2 = clf_dt2

clf3 = clf_rf

clf_eclf = VotingClassifier(estimators=[('knn', clf1), ('dt', clf2), ('rf', clf3)], voting='hard', n_jobs=-1)



#Call up the model to see the parameters you can tune (and their default setting)

print(clf_eclf)

#Fit clf to the training data

clf_eclf = clf_eclf.fit(features_train, target_train)

#Predict clf DT model again test data

target_predicted_eclf = clf_eclf.predict(features_test)
acc_eclf = accuracy_score(target_test, target_predicted_eclf)

prec_eclf = precision_score(target_test, target_predicted_eclf)

recall_eclf = recall_score(target_test, target_predicted_eclf)

f1_eclf = f1_score(target_test, target_predicted_eclf)

cm_eclf = confusion_matrix(target_test, target_predicted_eclf)

print("Stacking Accuracy Score", acc_eclf)

print(classification_report(target_test, target_predicted_eclf))

if DEBUG:

    print(confusion_matrix(target_test, target_predicted_eclf))
# Source: https://seaborn.pydata.org/examples/heatmap_annotation.html

plt.figure(figsize=(5.5,4))

sns.heatmap(cm_eclf, annot=True, fmt='d')

plt.title('Stacking Model 1 Confusion Matrix \nAccuracy:{0:.3f}'.format(acc_eclf))

plt.ylabel('True label')

plt.xlabel('Predicted label')

plt.show()
#verify RF with Cross Validation

scores_eclf = cross_val_score(clf_eclf, features_train, target_train, cv=10, n_jobs=-1)

print("Cross Validation Score for each K",scores_eclf)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores_eclf.mean(), scores_eclf.std() * 2))
for MV, label in zip([clf1, clf2, clf3, clf_eclf], ['KNN', 'Decision Tree', 'Random Forest', 'Ensemble Model 1']):

    scores2 = cross_val_score(MV, features_train, target_train, cv=10, scoring='recall', n_jobs=-1)

    print("Recall: %0.2f (+/- %0.2f) [%s]" % (scores2.mean(), scores2.std(), label))
# Random Forest train model. Call up my model and name it clf

clf1 = clf_bag

clf2 = clf_xdt2

clf3 = clf_gbc2

clf_eclf1 = VotingClassifier(estimators=[('bag', clf1), ('xdt', clf2), ('sgd', clf3)], voting='hard', n_jobs=-1)

#Call up the model to see the parameters you can tune (and their default setting)

print(clf_eclf1)

#Fit clf to the training data

clf_eclf1 = clf_eclf1.fit(features_train, target_train)

#Predict clf DT model again test data

target_predicted_eclf1 = clf_eclf1.predict(features_test)
acc_eclf1 = accuracy_score(target_test, target_predicted_eclf1)

prec_eclf1 = precision_score(target_test, target_predicted_eclf1)

recall_eclf1 = recall_score(target_test, target_predicted_eclf1)

f1_eclf1 = f1_score(target_test, target_predicted_eclf1)

cm_eclf1 = confusion_matrix(target_test, target_predicted_eclf1)

print("Stacking Accuracy Score", acc_eclf1)

print(classification_report(target_test, target_predicted_eclf1))

if DEBUG:

    print(confusion_matrix(target_test, target_predicted_eclf1))
# Source: https://seaborn.pydata.org/examples/heatmap_annotation.html

plt.figure(figsize=(5.5,4))

sns.heatmap(cm_eclf1, annot=True, fmt='d')

plt.title('Stacking Model 2 Confusion Matrix \nAccuracy:{0:.3f}'.format(acc_eclf1))

plt.ylabel('True label')

plt.xlabel('Predicted label')

plt.show()
#verify RF with Cross Validation

scores_eclf1 = cross_val_score(clf_eclf1, features_train, target_train, cv=10, n_jobs=-1)

print("Cross Validation Score for each K",scores_eclf1)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores_eclf1.mean(), scores_eclf1.std() * 2))
for MV, label in zip([clf1, clf2, clf3, clf_eclf1], ['Bagging', 'Extra Trees', 'Gradient Boost Classification', 'Ensemble Model 2']):

    scores2 = cross_val_score(MV, features_train, target_train, cv=10, scoring='recall', n_jobs=-1)

    print("Recall: %0.2f (+/- %0.2f) [%s]" % (scores2.mean(), scores2.std(), label))
# Random Forest train model. Call up my model and name it clf

clf1 = clf_sgd_log

clf2 = clf_lsvm

clf3 = clf_svc

clf4 = clf_NN2

clf_eclf2 = VotingClassifier(estimators=[('sgd', clf1), ('lsvm', clf2), ('svc', clf3), ('nn', clf4)], voting='hard', n_jobs=-1)

#Call up the model to see the parameters you can tune (and their default setting)

print(clf_eclf2)

#Fit clf to the training data

clf_eclf2 = clf_eclf2.fit(features_train, target_train)

#Predict clf DT model again test data

target_predicted_eclf2 = clf_eclf2.predict(features_test)
acc_eclf2 = accuracy_score(target_test, target_predicted_eclf2)

prec_eclf2  = precision_score(target_test, target_predicted_eclf2)

recall_eclf2 = recall_score(target_test, target_predicted_eclf2)

f1_eclf2 = f1_score(target_test, target_predicted_eclf2)

cm_eclf2 = confusion_matrix(target_test, target_predicted_eclf2)

print("Stacking Accuracy Score", acc_eclf2)

print(classification_report(target_test, target_predicted_eclf2))

if DEBUG:

    print(confusion_matrix(target_test, target_predicted_eclf2))
# Source: https://seaborn.pydata.org/examples/heatmap_annotation.html

plt.figure(figsize=(5.5,4))

sns.heatmap(cm_eclf2, annot=True, fmt='d')

plt.title('Stacking Model 3 Confusion Matrix \nAccuracy:{0:.3f}'.format(acc_eclf2))

plt.ylabel('True label')

plt.xlabel('Predicted label')

plt.show()
#verify RF with Cross Validation

scores_eclf2 = cross_val_score(clf_eclf2, features_train, target_train, cv=10, n_jobs=-1)

print("Cross Validation Score for each K",scores_eclf2)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores_eclf2.mean(), scores_eclf2.std() * 2))
for MV, label in zip([clf1, clf2, clf3, clf4, clf_eclf2], ['Stochastic Gradient Descent', 'Linear Support Vector Classification', 'Support Vector Model RBF', 'Neural Network', 'Ensemble Model 3']):

    scores2 = cross_val_score(MV, features_train, target_train, cv=10, scoring='recall', n_jobs=-1)

    print("Recall: %0.2f (+/- %0.2f) [%s]" % (scores2.mean(), scores2.std(), label))
# Random Forest train model. Call up my model and name it clf

clf1 = clf_dt2

clf2 = clf_rf

clf3 = clf_xdt2

clf4 = clf_gbc2

clf5 = clf_lsvm

clf6 = clf_NN2

clf_eclf3 = VotingClassifier(estimators=[('dt', clf1), ('rf', clf2), ('xdt', clf3), ('gbc', clf4), ('lsvm', clf5), ('nn', clf6)], voting='hard', n_jobs=-1)

#clf_eclf3 = VotingClassifier(estimators=[('knn', clf1), ('dt', clf2), ('rf', clf3), ('bag', clf4), ('xdt', clf5), ('gbc', clf6), ('sgd', clf7), ('lsvm', clf8), ('svc', clf9), ('nn', clf10)], voting='hard', n_jobs=-1)

#Call up the model to see the parameters you can tune (and their default setting)

print(clf_eclf3)

#Fit clf to the training data

clf_eclf3 = clf_eclf3.fit(features_train, target_train)

#Predict clf DT model again test data

target_predicted_eclf3 = clf_eclf3.predict(features_test)
acc_eclf3 = accuracy_score(target_test, target_predicted_eclf3)

prec_eclf3 = precision_score(target_test, target_predicted_eclf3)

recall_eclf3 = recall_score(target_test, target_predicted_eclf3)

f1_eclf3 = f1_score(target_test, target_predicted_eclf3)

cm_eclf3 = confusion_matrix(target_test, target_predicted_eclf3)

print("Stacking Accuracy Score", acc_eclf3)

print(classification_report(target_test, target_predicted_eclf3))

if DEBUG:

    print(confusion_matrix(target_test, target_predicted_eclf3))
# Source: https://seaborn.pydata.org/examples/heatmap_annotation.html

plt.figure(figsize=(5.5,4))

sns.heatmap(cm_eclf3, annot=True, fmt='d')

plt.title('Stacking Model 4 Confusion Matrix \nAccuracy:{0:.3f}'.format(acc_eclf3))

plt.ylabel('True label')

plt.xlabel('Predicted label')

plt.show()
#verify RF with Cross Validation

scores_eclf3 = cross_val_score(clf_eclf3, features_train, target_train, cv=10, n_jobs=-1)

print("Cross Validation Score for each K",scores_eclf3)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores_eclf3.mean(), scores_eclf3.std() * 2))
for MV, label in zip([clf1, clf2, clf3, clf4, clf5, clf6, clf_eclf3], ['Decision Tree','Random Forest','Extra Trees','Gradient Boost Classification','Linear Support Vector Classificaiton','Neural Network','Ensemble Model 4']):

    scores2 = cross_val_score(MV, features_train, target_train, cv=10, scoring='recall', n_jobs=-1)

    print("Recall: %0.2f (+/- %0.2f) [%s]" % (scores2.mean(), scores2.std(), label))
t = Table()

t[''] = ['Stacking Model 1','Stacking Model 2','Stacking Model 3','Stacking Model 4']

t['Cross Validation Score'] = [round(scores_eclf.mean(),4),round(scores_eclf1.mean(),4),round(scores_eclf2.mean(),4),round(scores_eclf3.mean(),4)]

t['Accuracy Score'] = [round(acc_eclf,4),round(acc_eclf1,4),round(acc_eclf2,4),round(acc_eclf3,4)]

t['Precision'] = [round(prec_eclf,4),round(prec_eclf1,4),round(prec_eclf2,4),round(prec_eclf3,4)]

t['Recall'] = [round(recall_eclf,4),round(recall_eclf1,4),round(recall_eclf2,4),round(recall_eclf3,4)]

t['F1 Score'] = [round(f1_eclf,4),round(f1_eclf1,4),round(f1_eclf2,4),round(f1_eclf3,4)]

t
acc_dt2 = accuracy_score(target_test, target_predicted_dt2)

prec_dt2 = precision_score(target_test, target_predicted_dt2)

recall_dt2 = recall_score(target_test, target_predicted_dt2)

f1_dt2 = f1_score(target_test, target_predicted_dt2)

cm_dt2 = confusion_matrix(target_test, target_predicted_dt2)

print("DT Accuracy Score", acc_dt2)

print(classification_report(target_test, target_predicted_dt2))

if DEBUG:

    print(confusion_matrix(target_test, target_predicted_dt2))
# Source: https://seaborn.pydata.org/examples/heatmap_annotation.html

plt.figure(figsize=(5.5,4))

sns.heatmap(cm_dt2, annot=True, fmt='d')

plt.title('Decision Tree Model 3 Confusion Matrix \nAccuracy:{0:.3f}'.format(acc_dt2))

plt.ylabel('True label')

plt.xlabel('Predicted label')

plt.show()
!pip install pydotplus

from sklearn.externals.six import StringIO  

from IPython.display import Image  

from sklearn.tree import export_graphviz

import pydotplus

import collections



data_feature_names = features[1:]

dot_data = tree.export_graphviz(clf_dt2, feature_names=data_feature_names, class_names=True, rounded = True, proportion = False, precision = 2, filled = True)

graph = pydotplus.graphviz.graph_from_dot_data(dot_data)

colors = ('turquoise', 'orange')

edges = collections.defaultdict(list)



for edge in graph.get_edge_list():

    edges[edge.get_source()].append(int(edge.get_destination()))



for edge in edges:

    edges[edge].sort()    

    for i in range(2):

        dest = graph.get_node(str(edges[edge][i]))[0]

        dest.set_fillcolor(colors[i])



Image(graph.create_png())