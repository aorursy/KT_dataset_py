import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

plt.style.use('ggplot')

%matplotlib inline





from sklearn.utils import resample

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold

from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from xgboost import XGBClassifier

from lightgbm import LGBMClassifier

from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, precision_recall_curve, roc_curve, auc, average_precision_score

from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE, ADASYN

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

from sklearn.metrics import f1_score, confusion_matrix, precision_recall_curve, roc_curve

from sklearn.preprocessing import StandardScaler , Binarizer

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

import lightgbm as lgb

from lightgbm import LGBMClassifier

import time

import os, sys, gc, warnings, random, datetime

import math

import shap

import joblib

warnings.filterwarnings('ignore')

import gc
df = pd.read_pickle("../input/handling-imbalanced-data-eda-small-fe/df_for_use.pkl")

df_fe = pd.read_pickle("../input/handling-imbalanced-data-eda-small-fe/df_fe.pkl")
lgbm_clf = joblib.load('../input/handling-imbalanced-data-supervised-learning/lgbm_clf.pkl')

rf_clf = joblib.load('../input/handling-imbalanced-data-supervised-learning/rf_clf.pkl')

xgb_clf = joblib.load('../input/handling-imbalanced-data-supervised-learning/xgb_clf.pkl')

# ngb_clf = joblib.load('../input/handling-imbalanced-data-supervised-learning/ngb_clf.pkl')
X = df.drop('loan_condition_cat', axis=1)

y = df['loan_condition_cat']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2 , random_state = 2020, stratify = y)
# y_trainin = y_train.to_frame()

# for_sample_train_df = pd.concat([X_train, y_trainin], axis=1)

# y_testet = y_test.to_frame()

# for_sample_test_df = pd.concat([X_test, y_testet], axis=1)
# X_ = for_sample_train_df.drop('loan_condition_cat', axis=1)

# y_ = for_sample_train_df['loan_condition_cat']



# X_train, X_test_use, y_train, y_test_use = train_test_split(X_, y_, test_size = 0.8 , random_state = 2020, stratify = y_)

## function fork from https://www.kaggle.com/chirag19/fraud-detection-balancing-roc-pr-curves

def results(balancing_technique):

    print(balancing_technique)

    fig, (ax1, ax2) = plt.subplots(1,2,figsize = (12,6))

    model_name = [ 'RF',"XGB", "LGB"]

    RF = RandomForestClassifier(random_state = 2020)

    XGBC = XGBClassifier(random_state = 2020)

    LGBC = LGBMClassifier(random_state = 2020)



    for clf,i in zip([ RF, XGBC, LGBC], model_name):

        model = clf.fit(X_train, y_train,)

        y_pred = model.predict(X_test)

        y_pred_prob = model.predict_proba(X_test)[:,1]

        print("#"*25,i,"#"*25)

        print("Training Accuracy = {:.3f}".format(model.score(X_train, y_train)))

        print("Test Accuracy = {:.3f}".format(model.score(X_test, y_test)))

        print("ROC_AUC_score : %.6f" % (roc_auc_score(y_test, y_pred_prob)))

        #Confusion Matrix

        print(confusion_matrix(y_test, y_pred))

        print("-"*15,"CLASSIFICATION REPORT","-"*15)

        print(classification_report(y_test, y_pred))

        

        #precision-recall curve

        precision, recall, thresholds_pr = precision_recall_curve(y_test, y_pred_prob)

        avg_pre = average_precision_score(y_test, y_pred_prob)

        ax1.plot(precision, recall, label = i+ " average precision = {:0.2f}".format(avg_pre), lw = 3, alpha = 0.7)

        ax1.set_xlabel('Precision', fontsize = 14)

        ax1.set_ylabel('Recall', fontsize = 14)

        ax1.set_title('Precision-Recall Curve', fontsize = 18)

        ax1.legend(loc = 'best')

        #find default threshold

        close_default = np.argmin(np.abs(thresholds_pr - 0.5))

        ax1.plot(precision[close_default], recall[close_default], 'o', markersize = 8)



        #roc-curve

        fpr, tpr, thresholds_roc = roc_curve(y_test, y_pred_prob)

        roc_auc = auc(fpr,tpr)

        ax2.plot(fpr,tpr, label = i+ " area = {:0.2f}".format(roc_auc), lw = 3, alpha = 0.7)

        ax2.plot([0,1], [0,1], 'r', linestyle = "--", lw = 2)

        ax2.set_xlabel("False Positive Rate", fontsize = 14)

        ax2.set_ylabel("True Positive Rate", fontsize = 14)

        ax2.set_title("ROC Curve", fontsize = 18)

        ax2.legend(loc = 'best')

        #find default threshold

        close_default = np.argmin(np.abs(thresholds_roc - 0.5))

        ax2.plot(fpr[close_default], tpr[close_default], 'o', markersize = 8)

        plt.tight_layout()
results("Without Balancing")