# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df_fraud = pd.read_csv('../input/creditcard.csv')
def fraud_per(data):
    fraud = data[data == 1].count()/data.shape[0] 
    return fraud *100
fraud = fraud_per(df_fraud['Class'])
print('Percentage of fraud observations = ' + str(fraud)+'%')
df_fraud.describe()
df_fraud.Time.plot()
df_fraud['sin_time'] = np.sin(2*np.pi*df_fraud.Time/(24*60*60))
df_fraud['cos_time'] = np.cos(2*np.pi*df_fraud.Time/(24*60*60))
df_fraud.sample(100).plot.scatter('sin_time','cos_time').set_aspect('equal')
df_fraud.head()
X = df_fraud.drop(['Time', 'Class'], axis = 1)
y = df_fraud['Class']
from sklearn.preprocessing import MinMaxScaler
def scaling_data(X):
    x_col = X.columns
    scaler = MinMaxScaler()
    scaler.fit(X)
    #Xtr_norm = pd.DataFrame(scaler.transform(Xtr), columns = x_col)
    #Xte_norm = pd.DataFrame(scaler.transform(Xte), columns = x_col)
    X_norm = scaler.transform(X)
    return X_norm
X_scaled= scaling_data(X)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, random_state=0, test_size=0.2)
X_train1, X_valid, y_train1, y_valid = train_test_split(X_train, y_train, random_state=0, test_size=0.2)
print('Percentage of fraud observations on Train Data', fraud_per(y_train))
print('Percentage of fraud observations on Validation Data', fraud_per(y_valid))
print('Percentage of fraud observations on Test Data', fraud_per(y_test))
from sklearn.linear_model import LogisticRegression
def LogisticClassifier(Xtr, ytr, c):
    lr = LogisticRegression(C = c, solver = 'lbfgs', max_iter=2000)
    model = lr.fit(Xtr, ytr)
    return model
from sklearn.metrics import recall_score, precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
def recall_pre_accuracy(Xv, yv, model, target_value):
    yp = model.predict(Xv)
    accuracy = model.score(Xv, yv)
    recall= recall_score(yv, yp)
    precision = precision_score(yv, yp)
    report = classification_report(yv, yp, target_names = target_value)
    return accuracy, recall, precision, report
from sklearn.metrics import confusion_matrix
def conf_matrix(Xv, yv, model):
    # Negative class (0) is most frequent
    y_predicted = model.predict(Xv)
    confusion = confusion_matrix(yv, y_predicted)
    return confusion
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve

def precision_recall_curves(y_test, y_scores):
    # precision recall curve
    precisionc, recallc, thresholds = precision_recall_curve(y_test, y_scores)
    pr_rel_df = pd.DataFrame([precisionc, recallc], index = ['precision', 'recall']).T
    # find threshold closest to zero:
    close_zero = np.argmin(np.abs(thresholds))
    plt.plot(precisionc[close_zero], recallc[close_zero], 'o', markersize=10, label="threshold zero", 
             fillstyle="none", c='k', mew=2)
    plt.plot(precisionc, recallc, label="precision recall curve")
    plt.xlabel("precision")
    plt.ylabel("recall")
    return precisionc[close_zero], recallc[close_zero]
def auc_roc_curves(y_test, y_scores, title): 
    #roc_curve, auc
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)
    roc_df = pd.DataFrame([fpr, tpr], index = ['fpr', 'tpr']).T
    plt.plot(fpr, tpr, label="ROC Curve")
    plt.xlabel("FPR")
    plt.ylabel("TPR (recall)")
    plt.title(title)
    # find threshold closest to zero:
    close_zero = np.argmin(np.abs(thresholds))
    plt.plot(fpr[close_zero], tpr[close_zero], 'o', markersize=10, label="threshold zero", 
             fillstyle="none", c='k', mew=2)
    plt.legend(loc=4)
    return roc_auc
def best_C(Xt, yt, xv, yv, c_list):
    best_C = 0
    min_mae = 0
    for c in C_list:
        model = LogisticClassifier(X_train1, y_train1, c)
        # calculate mean absolute error
        #preds_val = model.predict(X_valid)
        mae = 1- model.score(X_valid, y_valid)
        if min_mae == 0 or mae < min_mae:
            min_mae = mae
            best_C = c
    return best_C
C_list = list(1/np.logspace(-4, 2, num=10))
C = best_C(X_train1, y_train1, X_valid, y_valid, C_list)
print('Best C:\n', C)
lr_modelC = LogisticClassifier(X_train1, y_train1, C)
y_scores_lrC = lr_modelC.decision_function(X_test)
target_value = [ 'not fraud', 'fraud']
accuracy1, recall1, precision1, report1 = recall_pre_accuracy(X_test, y_test, lr_modelC, target_value)
print('Logistic Regression on Original Data:')
print ('Accuracy Score = ', accuracy1)
print ('Recall Score = ', recall1)
print ('Precision Score1 = ', precision1)
print('Report Score:\n', report1)
print('Confusion Matrix: \n', conf_matrix(X_test, y_test, lr_modelC))
precision0, recall0  = precision_recall_curves(y_test, y_scores_lrC)
print(precision0, recall0)
roc_auc = auc_roc_curves(y_test, y_scores_lrC, 'ROC Curve on Original Data')
print('AUC = ', roc_auc)
from imblearn.over_sampling import SMOTE, ADASYN
X_resampled, y_resampled = SMOTE(kind='borderline2').fit_sample(X_train1, y_train1)
x_col = X.columns
df_fraud_over = pd.DataFrame(X_resampled, columns = x_col)
df_fraud_over['Class'] = y_resampled
df_fraud_over.shape
print('Percentage of fraud observations on Upsamled Data', fraud_per(df_fraud_over['Class']))
X_train_up, X_valid_up, y_train_up, y_valid_up = train_test_split(X_resampled, y_resampled,
                                                                random_state=0, test_size=0.2)
C_up = best_C(X_train_up, y_train_up, X_valid_up, y_valid_up, C_list)
print('Best C: ', C_up)
lr_model_up = LogisticClassifier(X_resampled, y_resampled, C_up)
y_scores_lr_up = lr_model_up.decision_function(X_test)
target_value = [ 'not fraud', 'fraud']
accuracyup, recallup, precisionup, reportup = recall_pre_accuracy(X_test, y_test, lr_model_up, target_value)
print('Logistic Regression on Over_Sampled Data Validation Set:')
print ('Accuracy Score = ', accuracyup)
print ('Recall Score = ', recallup)
print ('Precision Score1 = ', precisionup)
print('Report Score:\n', reportup)
print('Confusion Matrix: \n', conf_matrix(X_test, y_test, lr_model_up))
precision_up, recall_up = precision_recall_curves(y_test, y_scores_lr_up)
precision_up, recall_up
roc_auc_up = auc_roc_curves(y_test, y_scores_lr_up, 'ROC Curve on Over-sample Minority Class')
print('AUC on Over-sample = ', roc_auc_up)
from imblearn.under_sampling import NearMiss
nm1 = NearMiss(random_state=0, version=3)
X_resampledu, y_resampledu = nm1.fit_sample(X_train, y_train)
x_col = X.columns
df_fraud_Under = pd.DataFrame(X_resampledu, columns = x_col)
df_fraud_Under['Class'] = y_resampledu
df_fraud_Under.head()
df_fraud_Under.shape
print('Percentage of fraud observations on Under-samled Data', fraud_per(df_fraud_Under['Class']))
from sklearn.linear_model import LogisticRegressionCV
def lrCV(X, y):
    lr = LogisticRegressionCV(cv = 5, penalty= 'l2', scoring='recall',  max_iter=2000)
    model = lr.fit(X,y)
    return model 
lrCV_un = lrCV(X_resampledu, y_resampledu)
C_un = lrCV_un.C_[0]
print('Best C: ', lrCV_un.C_[0])
lr_model_under = LogisticClassifier(X_resampledu, y_resampledu, C_un)
y_scores_lr_under = lr_model_under.decision_function(X_test)
target_value = [ 'not fraud', 'fraud']
accuracy1, recall1, precision1, report1 = recall_pre_accuracy(X_test, y_test, lr_model_under, target_value)
print('Logistic Regression on Under_Sampled Data TEst Set:')
print ('Accuracy Score = ', accuracy1)
print ('Recall Score = ', recall1)
print ('Precision Score1 = ', precision1)
print('Report Score:\n', report1)
print('Confusion Matrix: \n', conf_matrix(X_test, y_test, lr_model_under))
recallund, precisionund = precision_recall_curves(y_test, y_scores_lr_under)
roc_auc_under = auc_roc_curves(y_test, y_scores_lr_under, 'ROC Curve on Under-sample Majority Class')
print('AUC on Under-sample = ', roc_auc_under)