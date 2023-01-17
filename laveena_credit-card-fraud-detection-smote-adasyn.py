import numpy as np # linear algebra

import pandas as pd # data processing



# for standardization

from sklearn.preprocessing import StandardScaler as ss



# for splitting into train and test datasets

from sklearn.model_selection import train_test_split 



# for modelling

from xgboost.sklearn import XGBClassifier

from sklearn.ensemble import RandomForestClassifier



# for balancing dataset by oversampling

from imblearn.over_sampling import SMOTE, ADASYN



# for performance metrics

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, precision_recall_curve

from sklearn.metrics import auc, roc_curve, precision_score, recall_score, f1_score

from sklearn.metrics import confusion_matrix, classification_report, average_precision_score



# for data visualization

import matplotlib.pyplot as plt



#Miscellaneous

import time

import random

import os





# set number of rows to be displayed

pd.options.display.max_columns = 300



# reading the dataset

data = pd.read_csv("../input/creditcard.csv")
print('Shape: ',data.shape)



print('\nColumns: ',data.columns.values)



print('\nData types:\n',data.dtypes.value_counts())
data.Class.value_counts()
# check if there are null values in the dataset

data.isnull().sum().sum()
# Time & Amount distributions



fig = plt.figure(figsize=(14,5))

ax = fig.add_subplot(1,2,1)

data.Time.plot(kind = "hist", bins = 40)

plt.xlabel('Time(in secs)', size='large')

ax = fig.add_subplot(1,2,2)

data.Amount.plot(kind = "hist", bins = 40)

plt.xlabel('Amount', size='large')

y = data.iloc[:,30]

X = data.iloc[:,0:30]



print(X.columns)

print(y.head(3))
X_train, X_test, y_train, y_test =   train_test_split(X, y, test_size = 0.3, stratify = y)



X_train.shape

y_train.value_counts()
# XGBoost - scale_pos_weight - Control the balance of positive and negative weights, useful for unbalanced classes.

# A typical value to consider: sum(negative instances) / sum(positive instances)



weight = 199020/344



# Using Random Forest and XGBoost

rf = RandomForestClassifier(n_estimators=100, class_weight={0:1,1:7})

xg = XGBClassifier(scale_pos_weight = weight, learning_rate = 0.7,

                   reg_alpha= 0.8,

                   reg_lambda= 1)



rf1 = rf.fit(X_train,y_train)

xg1 = xg.fit(X_train,y_train)
y_pred_rf = rf1.predict(X_test)

y_pred_xg= xg1.predict(X_test)



y_pred_rf_prob = rf1.predict_proba(X_test)

y_pred_xg_prob = xg1.predict_proba(X_test)



print("RF - Accuracy - ",accuracy_score(y_test,y_pred_rf))

print("XGBoost - Accuracy - ",accuracy_score(y_test,y_pred_xg))



print("RF:\n",confusion_matrix(y_test,y_pred_rf))

print("XGBoost:\n",confusion_matrix(y_test,y_pred_xg))



fpr_rf1, tpr_rf1, thresholds = roc_curve(y_test,

                                 y_pred_rf_prob[: , 1],

                                 pos_label= 1

                                 )



fpr_xg1, tpr_xg1, thresholds = roc_curve(y_test,

                                 y_pred_xg_prob[: , 1],

                                 pos_label= 1

                                 )



print("RF - AUC: ",auc(fpr_rf1,tpr_rf1))

print("XGBoost - AUC: ",auc(fpr_xg1,tpr_xg1))



p_rf,r_rf,f_rf,_ = precision_recall_fscore_support(y_test,y_pred_rf)

p_xg,r_xg,f_xg,_ = precision_recall_fscore_support(y_test,y_pred_xg)



print("Random Forest:\n Precision: ",p_rf, "Recall: ", r_rf)

print("XGBoost:\n Precision: ",p_xg, "Recall: ", r_xg)
# Oversampling and balancing using SMOTE

sm = SMOTE(random_state=42)

X_bal, y_bal = sm.fit_sample(X_train, y_train)



columns = X_train.columns

X_bal = pd.DataFrame(data = X_bal, columns = columns)



print(X_train.shape)

print(X_bal.shape)

print(np.unique(y_bal, return_counts=True))
# Initialising the models

rf_sm = RandomForestClassifier(n_estimators=100)

xg_sm = XGBClassifier(learning_rate=0.7,

                   reg_alpha= 0.8,

                   reg_lambda= 1

                   )



# training the models

rf_sm1 = rf_sm.fit(X_bal,y_bal)

xg_sm1 = xg_sm.fit(X_bal,y_bal)
print("With Smote: \n")



# Making predictions on the test data

y_pred_rf1 = rf_sm1.predict(X_test)

y_pred_xg1= xg_sm1.predict(X_test)



y_pred_rf_prob1 = rf_sm1.predict_proba(X_test)

y_pred_xg_prob1 = xg_sm1.predict_proba(X_test)



print("RF - Accuracy - ",accuracy_score(y_test,y_pred_rf1))

print("XGBoost - Accuracy - ",accuracy_score(y_test,y_pred_xg1))



print("RF:\n",confusion_matrix(y_test,y_pred_rf1))

print("XGBoost:\n",confusion_matrix(y_test,y_pred_xg1))



fpr_rf2, tpr_rf2, thresholds = roc_curve(y_test,

                                 y_pred_rf_prob1[: , 1],

                                 pos_label= 1

                                 )



fpr_xg2, tpr_xg2, thresholds = roc_curve(y_test,

                                 y_pred_xg_prob1[: , 1],

                                 pos_label= 1

                                 )



print("RF - AUC: ",auc(fpr_rf2,tpr_rf2))

print("XGBoost - AUC: ",auc(fpr_xg2,tpr_xg2))



p_rf,r_rf,f_rf,_ = precision_recall_fscore_support(y_test,y_pred_rf1)

p_xg,r_xg,f_xg,_ = precision_recall_fscore_support(y_test,y_pred_xg1)



print("Random Forest:\n Precision: ",p_rf, "Recall: ", r_rf)

print("XGBoost:\n Precision: ",p_xg, "Recall: ", r_xg)
# oversampling and balancing dataset with ADASYN

ad = ADASYN()

X_ad, y_ad = ad.fit_sample(X_train, y_train)

 

X_ad = pd.DataFrame(data = X_ad, columns = X_train.columns)



print(X_train.shape)

print(X_ad.shape)

print(np.unique(y_bal, return_counts=True))
# Initialising the models

rf_ad = RandomForestClassifier(n_estimators=100)

xg_ad = XGBClassifier(learning_rate=0.8,

                   reg_alpha= 0.8,

                   reg_lambda= 0.8)



# training the models

rf_ad1 = rf_ad.fit(X_ad,y_ad)

xg_ad1 = xg_ad.fit(X_ad,y_ad)
print("With ADASYN:\n")

# Making predictions on the test data

y_pred_rf2 = rf_ad1.predict(X_test)

y_pred_xg2 = xg_ad1.predict(X_test)



y_pred_rf_prob2 = rf_ad1.predict_proba(X_test)

y_pred_xg_prob2 = xg_ad1.predict_proba(X_test)



print("RF - Accuracy - ",accuracy_score(y_test,y_pred_rf2))

print("XGBoost - Accuracy - ",accuracy_score(y_test,y_pred_xg2))



print("RF:\n",confusion_matrix(y_test,y_pred_rf2))

print("XGBoost:\n",confusion_matrix(y_test,y_pred_xg2))



fpr_rf3, tpr_rf3, thresholds = roc_curve(y_test,

                                 y_pred_rf_prob2[: , 1],

                                 pos_label= 1

                                 )



fpr_xg3, tpr_xg3, thresholds = roc_curve(y_test,

                                 y_pred_xg_prob2[: , 1],

                                 pos_label= 1

                                 )



print("RF - AUC: ",auc(fpr_rf3,tpr_rf3))

print("XGBoost - AUC: ",auc(fpr_xg3,tpr_xg3))



p_rf,r_rf,f_rf,_ = precision_recall_fscore_support(y_test,y_pred_rf2)

p_xg,r_xg,f_xg,_ = precision_recall_fscore_support(y_test,y_pred_xg2)



print("Random Forest:\n Precision: ",p_rf, "Recall: ", r_rf)

print("XGBoost:\n Precision: ",p_xg, "Recall: ", r_xg)
fig = plt.figure(figsize=(14,16))   # Create window frame





roc = [["Imbalanced", fpr_rf1, tpr_rf1,'rf'],["Imbalanced", fpr_xg1, tpr_xg1,'xg'],["SMOTE", fpr_rf2, tpr_rf2,'rf'],["SMOTE", fpr_xg2, tpr_xg2,'xg'],

["ADASYN", fpr_rf3, tpr_rf3,'rf'], ["ADASYN", fpr_xg3, tpr_xg3,'xg']]



for i in range(6):

    #8.1 Connect diagonals

    ax = fig.add_subplot(3,2,i+1) 

    ax.plot([0, 1], [0, 1], ls="--")  # Dashed diagonal line



    #8.2 Labels 

    ax.set_xlabel('False Positive Rate')  # Final plot decorations

    ax.set_ylabel('True Positive Rate')

    ax.set_title(roc[i][0])



    #8.3 Set graph limits

    ax.set_xlim([0.0, 1.0])

    ax.set_ylim([0.0, 1.0])



    #8.4 Plot each graph now

    ax.plot(roc[i][1], roc[i][2], label = roc[i][3])



    #8.5 Set legend and show plot

    ax.legend(loc="lower right")

    
prc = [["Imbalanced", rf1,'rf'],["Imbalanced", xg1,'xg'],["SMOTE", rf_sm1,'rf'],["SMOTE", xg_sm1,'xg'],

["ADASYN", rf_ad1,'rf'], ["ADASYN", xg_ad1,'xg']]



fig = plt.figure(figsize=(14,16))



for i in range(6):

    ax = fig.add_subplot(3,2,i+1)

   

    precision, recall, _ = precision_recall_curve(y_test,prc[i][1].predict_proba(X_test)[:,-1])



    plt.step(recall, precision, color='b', alpha=0.2,

         where='post')

    plt.fill_between(recall, precision, step='post', alpha=0.2,

                 color='b')



    plt.xlabel('Recall')

    plt.ylabel('Precision')

    plt.ylim([0.0, 1.05])

    plt.xlim([0.0, 1.0])

    ttl = prc[i][0]+' '+prc[i][2]

    plt.title(ttl)

    

      

 