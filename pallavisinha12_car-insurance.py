import pandas as pd

import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

import sklearn

import matplotlib.pyplot as plt

from sklearn import metrics

import scikitplot as skplt

from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, precision_recall_curve, auc, roc_curve, recall_score, classification_report 
data = pd.read_csv("../input/health-insurance-cross-sell-prediction/train.csv")

test = pd.read_csv("../input/health-insurance-cross-sell-prediction/test.csv")
print(test.isnull().sum())
print(data.isnull().sum())
print(data['Driving_License'].value_counts())
data = data.drop(columns = ['Driving_License'], axis = 1)

test = test.drop(columns = ['Driving_License'], axis = 1)
data['Previously_Insured'].value_counts()
pd.crosstab(data['Response'], data['Previously_Insured'])
le = LabelEncoder()

data['Gender'] = le.fit_transform(data["Gender"])

data['Vehicle_Damage'] = le.fit_transform(data["Vehicle_Damage"])

data.head()
num = ['Age', 'Vintage']

ss = StandardScaler()

data[num] = ss.fit_transform(data[num])

mm = MinMaxScaler()

data[['Annual_Premium']] = mm.fit_transform(data[['Annual_Premium']])

test[num] = ss.fit_transform(test[num])

test[['Annual_Premium']] = mm.fit_transform(test[['Annual_Premium']])
data.head()
ohe = pd.get_dummies(data['Vehicle_Age'], prefix='Vehicle_Age')

data = pd.concat([data, ohe], axis=1)
data = data.drop(columns = ['id', 'Vehicle_Age'], axis = 1)

data.head()
y = data['Response']

data = data.drop(columns = ['Response'], axis= 1)
ohe1 = pd.get_dummies(test['Vehicle_Age'], prefix='Vehicle_Age')

test = pd.concat([test, ohe1], axis=1)

id = test['id']

test = test.drop(columns = ['id', 'Vehicle_Age'], axis = 1)

test['Gender'] = le.fit_transform(test["Gender"])

test['Vehicle_Damage'] = le.fit_transform(test["Vehicle_Damage"])

test.head()
data.head()
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(data, y, train_size=0.8, random_state = 5)
def plot_ROC(fpr, tpr, m_name):

    roc_auc = sklearn.metrics.auc(fpr, tpr)

    plt.figure(figsize=(6, 6))

    lw = 2

    plt.plot(fpr, tpr, color='darkorange',

             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc, alpha=0.5)

    

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--', alpha=0.5)

    

    plt.xlim([0.0, 1.0])

    plt.ylim([0.0, 1.05])

    plt.xticks(fontsize=16)

    plt.yticks(fontsize=16)

    plt.grid(True)

    plt.xlabel('False Positive Rate', fontsize=16)

    plt.ylabel('True Positive Rate', fontsize=16)

    plt.title('Receiver operating characteristic for %s'%m_name, fontsize=20)

    plt.legend(loc="lower right", fontsize=16)

    plt.show()
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()

rf.fit(X = X_train, y = y_train)

acc = rf.score(X_valid, y_valid)

print("Accuracy of Random_Forest: ",acc)
rf_preds = rf.predict_proba(X_valid)

rf_score = roc_auc_score(y_valid, rf_preds[:,1], average = 'weighted')

(fpr, tpr, thresholds) = roc_curve(y_valid, rf_preds[:,1])

plot_ROC(fpr, tpr, 'rf')

rf_class = rf.predict(X_valid)

print('ROC AUC score for rf model: %.4f'%rf_score)

print('F1 score: %0.4f'%f1_score(y_valid, rf_class))

skplt.metrics.plot_confusion_matrix(y_valid, rf_class,

        figsize=(8,8))
from sklearn.ensemble import GradientBoostingClassifier

gb=GradientBoostingClassifier(n_estimators=134,learning_rate=0.2)

gb.fit(X_train, y_train)

accuracy2 = gb.score(X_valid, y_valid)

print("Accuracy of Gradient Boost", accuracy2)
gb_preds = gb.predict_proba(X_valid)

gb_score = roc_auc_score(y_valid, gb_preds[:,1], average = 'weighted')

print(gb_score)

(fpr, tpr, thresholds) = roc_curve(y_valid, gb_preds[:,1])

plot_ROC(fpr, tpr, 'gb')
gb_class = gb.predict(X_valid)

print('ROC AUC score for gb model: %.4f'%gb_score)

print('F1 score: %0.4f'%f1_score(y_valid, gb_class))

skplt.metrics.plot_confusion_matrix(y_valid, gb_class,

        figsize=(8,8))
from lightgbm import LGBMClassifier

LGB_model = LGBMClassifier(random_state = 5, max_depth = 8, n_estimators = 300, reg_lambda = 1.2, reg_alpha = 1.2, min_child_weight = 1, verbose  = 1,

                       learning_rate = 0.15, gamma = 0.3, colsample_bytree = 0.5, eval_metric = 'auc', is_higher_better = 1, plot = True)

LGB_model.fit(X_train, y_train)

accuracy3 = LGB_model.score(X_valid, y_valid)

print("Accuracy of lgb: ", accuracy3)
LGB_preds = LGB_model.predict_proba(X_valid)

LGB_class = LGB_model.predict(X_valid)

LGB_score = roc_auc_score(y_valid, LGB_preds[:,1], average = 'weighted')

(fpr, tpr, thresholds) = roc_curve(y_valid, LGB_preds[:,1])

plot_ROC(fpr, tpr, 'LGBM')
print('ROC AUC score for LGBM model: %.4f'%LGB_score)

print('F1 score: %0.4f'%f1_score(y_valid, LGB_class))

skplt.metrics.plot_confusion_matrix(y_valid, LGB_class,

        figsize=(8,8))
predictions = [pred[1] for pred in LGB_model.predict_proba(test)]

submission = pd.DataFrame(data = {'id': id, 'Response': predictions})

submission.to_csv('vehicle_insurance_lgb.csv', index = False)

submission.head()