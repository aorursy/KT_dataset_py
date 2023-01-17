# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.cluster import KMeans

sns.set()

%matplotlib inline



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.

#Algorithem Imports

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import StandardScaler 

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import GridSearchCV

import xgboost as xgb

from sklearn.metrics import confusion_matrix, roc_auc_score, precision_recall_curve, precision_score, accuracy_score

from sklearn.metrics import recall_score, classification_report, f1_score, roc_curve, auc
data = pd.read_csv('../input/creditcard.csv')

df = data.copy()
df.head()
df.info()
df.describe(include='all')
df.isnull().sum()
df.dtypes
X = df.drop('Class', axis=1)

Y = df['Class']
xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.3,random_state=0)

print(xtrain.shape, ytrain.shape)

print(xtest.shape, ytest.shape)
#Logisctic Regression

clf_lr = LogisticRegression().fit(xtrain, ytrain)

logistic_regression_pred = clf_lr.predict(xtest)

lr_pred_prb = clf_lr.predict_proba(xtest)[:,1]

#Accuracy

accuracy_logistic_regression = accuracy_score(ytest, logistic_regression_pred)

#Recall

recall_lr = recall_score(ytest,logistic_regression_pred)

#AUC

auc_lr = roc_auc_score(ytest,lr_pred_prb)

    

#KNN

scaler = StandardScaler()  

scaler.fit(xtrain)

X_train_ = scaler.transform(xtrain)

X_test_ = scaler.transform(xtest)

X_train = pd.DataFrame(data=X_train_, columns=xtrain.columns)

X_test = pd.DataFrame(data=X_test_, columns=xtest.columns)

clf_knn = KNeighborsClassifier(n_neighbors=6).fit(X_train,ytrain)

knn_pred = clf_knn.predict(X_test)

knn_pred_prb = clf_knn.predict_proba(xtest)[:,1]

#Accuracy

accuracy_knn = accuracy_score(ytest,knn_pred)

#Recall

recall_knn = recall_score(ytest,knn_pred)

#AUC

auc_knn = roc_auc_score(ytest,lr_pred_prb)



#Decision Tree Classifier

clf_dt = DecisionTreeClassifier(criterion='gini',max_depth=3).fit(xtrain, ytrain)

decision_tree_classifier_pred = clf_dt.predict(xtest)

dt_pred_prb = clf_dt.predict_proba(xtest)[:,1]

#Accuracy

accuracy_decision_tree_classifier = accuracy_score(ytest,decision_tree_classifier_pred)

#Recall

recall_dt = recall_score(ytest,decision_tree_classifier_pred)

#AUC

auc_dt = roc_auc_score(ytest,lr_pred_prb)

    

#Random Forest

clf_rf = RandomForestClassifier(max_depth=4).fit(xtrain, ytrain)

random_forest_pred = clf_rf.predict(xtest)

rf_pred_prb = clf_rf.predict_proba(xtest)[:,1]

#Accuracy

accuracy_random_forest = accuracy_score(ytest,random_forest_pred)

#Recalla

recall_rf = recall_score(ytest,random_forest_pred)

#AUC

auc_rf = roc_auc_score(ytest,rf_pred_prb)

    

#XGBoost

clf_xgb = xgb.XGBClassifier(seed=42,nthread=1).fit(xtrain, ytrain)

xgb_pred = clf_xgb.predict(xtest)

xgb_pred_prb = clf_xgb.predict_proba(xtest)[:,1]

#Accuracy

accuracy_xgboost = accuracy_score(ytest,xgb_pred)

#Recall

recall_xgb = recall_score(ytest,xgb_pred)

#AUC

auc_xgb = roc_auc_score(ytest,lr_pred_prb)
F_dict = {        

    "Algorithms":["Logistic Regression","KNN","Decision Tree Classifier","Random Forest","XGBoost"],

    "Accuracy":[accuracy_logistic_regression,accuracy_knn,accuracy_decision_tree_classifier,accuracy_random_forest,accuracy_xgboost],

    'AUC':[auc_lr,auc_knn,auc_dt,auc_rf,auc_xgb]

}

final_result = pd.DataFrame(F_dict)

final_result
def plot_roc_curve(fpr, tpr, label=None):

    plt.plot(fpr, tpr, linewidth=2, label=label)

    plt.plot([0, 1], [0, 1], 'k--')

    plt.axis([-0.005, 1, 0, 1.005])

    plt.xticks(np.arange(0,1, 0.05), rotation=90)

    plt.xlabel("False Positive Rate")

    plt.ylabel("True Positive Rate")

    plt.legend(loc='best')
plt.figure(figsize=(20, 38))

plt.subplot(321)

plt.title('ROC Curve Logistic Regression')

fpr_lr,tpr_lr,threshold_lr = roc_curve(ytest,lr_pred_prb)

plot_roc_curve(fpr_lr,tpr_lr,label='AUC = %0.3f'% auc_lr)



plt.subplot(322)

plt.title('ROC Curve KNN')

fpr_knn,tpr_knn,threshold_knn = roc_curve(ytest,knn_pred_prb)

plot_roc_curve(fpr_knn,tpr_knn,label='AUC = %0.3f'% auc_knn)



plt.subplot(323)

plt.title('ROC Curve Random Forest')

fpr_rf,tpr_rf,threshold_rf = roc_curve(ytest,rf_pred_prb)

plot_roc_curve(fpr_rf,tpr_rf,label='AUC = %0.3f'% auc_rf)



plt.subplot(324)

plt.title('ROC Curve XGBoost')

fpr_xgb,tpr_xgb,threshold_xbg = roc_curve(ytest,xgb_pred_prb)

plot_roc_curve(fpr_xgb,tpr_xgb,label='AUC = %0.3f'% auc_xgb)
features_tuple=list(zip(X.columns,clf_dt.feature_importances_))

feature_imp=pd.DataFrame(features_tuple,columns=["Feature Names","Importance"])

feature_imp=feature_imp.sort_values("Importance",ascending=False)

plt.figure(figsize=(12, 6))

sns.barplot(x="Importance",y="Feature Names", data=feature_imp, color='r')

sns.set_context('poster')

plt.xlabel("Dataset Features")

plt.ylabel("Importance")

plt.title("Decision Classifier - Features Importance")