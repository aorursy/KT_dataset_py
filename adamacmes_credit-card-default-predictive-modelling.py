# Loading packages
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
# import csv
df = pd.read_csv('../input/UCI_Credit_Card.csv')
df.head()
df.shape
# Checking missing data
df.isnull().sum()
# Check the summary for each feature
df.describe().transpose()
df['default.payment.next.month'].value_counts()
plt.title('Default Payment Next Month - data imbalance check')
ax1 = sns.countplot(x= 'default.payment.next.month', data = df)
ax1.set_xticklabels(['No Default','Default'])
plt.show()
# Education Distribution
plt.title('Education Distribution')
ax2 = sns.countplot(x= 'EDUCATION', hue = 'default.payment.next.month', data = df)
ax2.set_xticklabels(['Unknown','graduate school','university','high school','others','unknown','unknown'],rotation = 90)
plt.show()
# SEX distribution
plt.title('Sex Distribution')
ax3 = sns.countplot(x= 'SEX', hue = 'default.payment.next.month', data = df)
ax3.set_xticklabels(['Male','Female'])
plt.show()
# Age Distribution
plt.title('Age Distribution \n Default(Red) vs. No Default(Grey)')
agedist0 = df[df['default.payment.next.month']==0]['AGE']
agedist1 = df[df['default.payment.next.month']==1]['AGE']
sns.distplot(agedist0, bins = 100, color = 'grey')
sns.distplot(agedist1, bins = 100, color = 'red')
plt.show()
# Credit Amount Distribution
plt.title('Credit Amount Distribution \n Default(Red) vs. No Default(Grey)')
cadist0 = df[df['default.payment.next.month']==0]['LIMIT_BAL']
cadist1 = df[df['default.payment.next.month']==1]['LIMIT_BAL']
sns.distplot(cadist0, bins = 100, color = 'grey')
sns.distplot(cadist1, bins = 100, color = 'red')
plt.xlabel('Credit Limit')
plt.show()
# Define predictor and target variables with X and Y
X = df.columns[:24]
Y = df.columns[-1]
# training and test dataset split, leaving 30% as test set
x_train, x_test, y_train, y_test = train_test_split(df[X],df[Y], 
                                                    test_size = .3, shuffle = True, random_state = 0)
# Check splitted data for train and test sets respectively
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
clfLR = LogisticRegression(solver = 'lbfgs',
                           max_iter = 500,
                          random_state = 0)

clfLR.fit(x_train,y_train)

predLR = clfLR.predict(x_test)
# Cross Validation
cross_val_score_LR = cross_val_score(clfLR, x_test, y_test, cv = 10)
print('cross_val_score: ',cross_val_score_LR.mean().round(2))

# Precision Score
print('precision score is ',precision_score(y_test, predLR).round(2))

# Recall Score
print('recall_score is ',recall_score(y_test, predLR).round(4))
# F1 Score
print('f1 score is ',f1_score(y_test, predLR).round(3))

# ROC_AUC
print('ROC AUC is ',roc_auc_score(y_test, predLR).round(2))
clfSVC = SVC(kernel = 'rbf',
             gamma = 'scale',
                random_state = 0)

clfSVC.fit(x_train,y_train)

predSVC = clfSVC.predict(x_test)
# Cross Validation
cross_val_score_SVC = cross_val_score(clfSVC, x_test, y_test, cv = 10)
print('cross_val_score: ',cross_val_score_SVC.mean().round(2))

# Precision Score
print('precision score is ',precision_score(y_test, predSVC).round(2))

# Recall Score
print('recall_score is ',recall_score(y_test, predSVC).round(4))
# F1 Score
print('f1 score is ',f1_score(y_test, predSVC).round(3))

# ROC_AUC
print('ROC AUC is ',roc_auc_score(y_test, predSVC).round(2))
clfKNN = KNeighborsClassifier(n_neighbors = 3)
clfKNN.fit(x_train,y_train)

predKNN = clfKNN.predict(x_test)
# Cross Validation
cross_val_score_KNN = cross_val_score(clfKNN, x_test, y_test, cv = 10)
print('cross_val_score: ',cross_val_score_KNN.mean().round(2))

# Precision Score
print('precision score is ',precision_score(y_test, predKNN).round(2))

# Recall Score
print('recall_score is ',recall_score(y_test, predKNN).round(4))
# F1 Score
print('f1 score is ',f1_score(y_test, predKNN).round(3))

# ROC_AUC
print('ROC AUC is ',roc_auc_score(y_test, predKNN).round(2))
clfRF = RandomForestClassifier(criterion = 'gini',
                              n_estimators = 100,
                              verbose = False,
                              random_state = 0)

clfRF.fit(x_train,y_train)

predRF = clfRF.predict(x_test)
# Cross Validation
cross_val_score_RF = cross_val_score(clfRF, x_test, y_test, cv = 10)
print('cross_val_score: ',cross_val_score_RF.mean().round(2))

# Precision Score
print('precision score is ',precision_score(y_test, predRF).round(2))

# Recall Score
print('recall_score is ',recall_score(y_test, predRF).round(4))
# F1 Score
print('f1 score is ',f1_score(y_test, predRF).round(3))

# ROC_AUC
print('ROC AUC is ',roc_auc_score(y_test, predRF).round(2))
clfXGB = xgb.XGBClassifier()
clfXGB.fit(x_train,y_train)
predXGB = clfXGB.predict(x_test)
# Cross Validation
cross_val_score_XGB = cross_val_score(clfXGB, x_test, y_test, cv = 10)
print('cross_val_score: ',cross_val_score_XGB.mean().round(2))

# Precision Score
print('precision score is ',precision_score(y_test, predXGB).round(2))

# Recall Score
print('recall_score is ',recall_score(y_test, predXGB).round(4))
# F1 Score
print('f1 score is ',f1_score(y_test, predXGB).round(3))

# ROC_AUC
print('ROC AUC is ',roc_auc_score(y_test, predXGB).round(2))
clfLGB = LGBMClassifier(n_estimators = 100,
                           learning_rate = .2,
                           random_state = 0)

clfLGB.fit(x_train,y_train)

predLGB = clfLGB.predict(x_test)
# Cross Validation
cross_val_score_LGB = cross_val_score(clfLGB, x_test, y_test, cv = 10)
print('cross_val_score: ',cross_val_score_LGB.mean().round(2))

# Precision Score
print('precision score is ',precision_score(y_test, predLGB).round(2))

# Recall Score
print('recall_score is ',recall_score(y_test, predLGB).round(4))
# F1 Score
print('f1 score is ',f1_score(y_test, predLGB).round(3))

# ROC_AUC
print('ROC AUC is ',roc_auc_score(y_test, predLGB).round(2))
clfCB = CatBoostClassifier(iterations = 100,
                           learning_rate = .2,
                           depth = 5,
                           eval_metric = 'AUC',
                           random_seed = 0)

clfCB.fit(x_train,y_train)

predCB = clfCB.predict(x_test)
# Cross Validation
cross_val_score_CB = cross_val_score(clfCB, x_test, y_test, cv = 10)
print('cross_val_score: ',cross_val_score_CB.mean().round(2))

# Precision Score
print('precision score is ',precision_score(y_test, predCB).round(2))

# Recall Score
print('recall_score is ',recall_score(y_test, predCB).round(4))
# F1 Score
print('f1 score is ',f1_score(y_test, predCB).round(3))

# ROC_AUC
print('ROC AUC is ',roc_auc_score(y_test, predCB).round(2))
# Confusion Matrix
cmLR = confusion_matrix(y_test, predLR)
cmSVC = confusion_matrix(y_test, predSVC)
cmKNN = confusion_matrix(y_test, predKNN)
cmRF = confusion_matrix(y_test, predRF)
cmXGB = confusion_matrix(y_test, predXGB)
cmLGB = confusion_matrix(y_test, predLGB)
cmCB = confusion_matrix(y_test, predCB)

# Confusion Matrix List
cmList = [cmLR, cmSVC,cmKNN, cmRF, cmXGB, cmLGB, cmCB]
cmTitle = ['Logistic Regression','Support Vector Machines','K Nearest Neighbors','Random Forest','XGB','LightGB','CatGBM',None]
i = 0
plt.figure()
fig, ax = plt.subplots(2,4, num = 6, figsize = (30,10))
for cm in cmList:
    i += 1
    plt.subplot(2,4,i)
    plt.title(cmTitle[i-1])
    sns.heatmap(cm, annot = True, cmap = 'YlGnBu')
plt.show();