# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.compose import ColumnTransformer

from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedKFold,StratifiedShuffleSplit

from sklearn import metrics

from sklearn.model_selection import GridSearchCV

from imblearn.under_sampling import NearMiss

from imblearn.over_sampling import SMOTE

from imblearn.pipeline import make_pipeline

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report

from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score,auc,confusion_matrix



#Import learning models

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestClassifier

from sklearn.cluster import KMeans





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/creditcard.csv')
df.info()
df.head(5)
df.describe()
sns.distplot(df['Class'],kde=False,bins=30) 
print("Number of non Fraud samples:",df[df["Class"]==0].shape[0])

print("Number of Fraud samples:",df[df["Class"]==1].shape[0])
X = df.drop(["Class"],axis=1)

y = df["Class"]
std = StandardScaler()

X_new = pd.DataFrame(std.fit_transform(X))

X_new.columns = X.columns

X_new.head(5)
#Splitting dataset into training and test

X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state=101)
#Define a iterable for Grid Search Cross Validation

skf = StratifiedShuffleSplit(n_splits=3)
#Grid Search to determine best parameters for the learning models

log_reg_params = {"logisticregression__penalty": ['l1', 'l2'], 'logisticregression__C': [ 0.01, 0.1, 1, 10, 100]} 

#knears_params = {"n_neighbors": list(range(2,5,1)), 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}

#svc_params = {'C': [0.5, 0.7, 0.9, 1], 'kernel': ['rbf', 'poly', 'sigmoid', 'linear']}

#tree_params = {"criterion": ["gini", "entropy"],"max_depth": list(range(2,4,1)), "min_samples_leaf": list(range(5,7,1))}



log_reg = LogisticRegression(solver='liblinear')



#Created Pipeline to perform undersampling using Near Miss technique and then fit the Learning model

my_pipeline_undersample = make_pipeline(NearMiss(sampling_strategy='majority'),log_reg)

log_gridcv_model = GridSearchCV(my_pipeline_undersample,log_reg_params,scoring='f1',cv=skf)

log_gridcv_model.fit(X_train,y_train)

log_gridcv_model_best = log_gridcv_model.best_estimator_

log_gridcv_model_best
#Calculate training recall and precision

log_gridcv_model_best.fit(X_train,y_train)

pred_train = log_gridcv_model_best.predict(X_train)

pred_prob_train = log_gridcv_model_best.predict_proba(X_train)

print("Training Recall:",recall_score(y_train,pred_train))

print("Training Precision:",precision_score(y_train,pred_train))
#Function to Create Precision Recall Curve

def create_precision_recall_curve(y_train,prob):

    precision, recall, threshold = precision_recall_curve(y_train,prob)

    plt.step(recall, precision, color='b', alpha=0.2,where='post')

    plt.fill_between(recall, precision, alpha=0.2, color='b')

    plt.xlabel('Recall')

    plt.ylabel('Precision')

    plt.ylim([0.0, 1.05])

    plt.xlim([0.0, 1.0])
#Create precision recall curve for training result

create_precision_recall_curve(y_train,pred_prob_train[:,1])
#Function to Create ROC curve using the training result

def create_roc_curve(label,result):

    fpr, tpr, _ = roc_curve(label,result)

    plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % auc(fpr, tpr))

    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')

    plt.xlim([0.0, 1.0])

    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title('Receiver operating characteristic example')

    plt.legend(loc="lower right")

    plt.show()
#Create ROC curve for training result

create_roc_curve(y_train,pred_train)
#Calculate test recall and precision

pred_test = log_gridcv_model_best.predict(X_test)

pred_prob_test = log_gridcv_model_best.predict_proba(X_test)

print("Test Recall:",recall_score(y_test,pred_test))

print("Test Precision:",precision_score(y_test,pred_test))
#Create precision recall curve for test result

create_precision_recall_curve(y_test,pred_prob_test[:,1])
#Create ROC curve for test result

create_roc_curve(y_test,pred_test)
#Created Pipeline to perform oversampling using SMOTE technique and then fit the Learning model

my_pipeline_oversample = make_pipeline(SMOTE(sampling_strategy='minority'),log_reg)

log_gridcv_oversample_model = GridSearchCV(my_pipeline_oversample,log_reg_params,scoring='f1',cv=skf)

log_gridcv_oversample_model.fit(X_train,y_train)

log_gridcv_oversample_model_best = log_gridcv_oversample_model.best_estimator_

log_gridcv_oversample_model_best
#Calculate training recall and precision

log_gridcv_oversample_model_best.fit(X_train,y_train)

pred_train = log_gridcv_oversample_model_best.predict(X_train)

pred_prob_train = log_gridcv_oversample_model_best.predict_proba(X_train)

print("Training Recall:",recall_score(y_train,pred_train))

print("Training Precision:",precision_score(y_train,pred_train))
#Create precision recall curve for training result

create_precision_recall_curve(y_train,pred_prob_train[:,1])
threshold = [0.1,0.2,0.3,0.4,0.6,0.7,0.8,0.9]

f1_best = f1_score(y_train,pred_train)

threshold_best = 0.5

pred_best = pred_train

for threshold_value in threshold:

    pred_new_oversample = [1 if pred>threshold_value else 0 for pred in pred_prob_train[:,1]]

    f1 = f1_score(y_train,pred_new_oversample)

    if f1>f1_best:

        f1_best = f1

        threshold_best = threshold_value

        pred_best = pred_new_oversample

print("Best F1 score:",f1_best)

print("Best threshold:",threshold_best)

print("Best Recall:",recall_score(y_train,pred_best))

print("Best Precision:",precision_score(y_train,pred_best))
#Create ROC curve for training result

create_roc_curve(y_train,pred_best)
#Calculate test recall and precision using the threshold_best

pred_prob_test = log_gridcv_oversample_model_best.predict_proba(X_test)

pred_test = [1 if pred>threshold_best else 0 for pred in pred_prob_test[:,1]]

recall_oversample_test = recall_score(y_test,pred_test)

precision_oversample_test = precision_score(y_test,pred_test)

print("Test Recall:",recall_oversample_test)

print("Test Precision:",precision_oversample_test)
#Create precision recall curve for test result

create_precision_recall_curve(y_test,pred_prob_test[:,1])
#Create ROC curve for test result

create_roc_curve(y_test,pred_test)