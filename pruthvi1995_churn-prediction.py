# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import sklearn
import matplotlib as mlp
import matplotlib.pyplot as plt

from sklearn import cross_validation
from sklearn import tree
from sklearn import svm
from sklearn import ensemble
from sklearn import neighbors
from sklearn import linear_model
from sklearn import metrics
from sklearn import preprocessing
from sklearn.cross_validation import KFold
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/bigml_59c28831336c6604c800002a.csv')

data.head(5)
data.describe()
print("data has {} rows".format(data.shape[0]))
print("data has {} columns".format(data.shape[1]))
y = data["churn"].value_counts()
sns.barplot(y.index, y.values)
print("Churn Percentage is {}".format(data["churn"].sum()*100/data["churn"].shape[0]))
data.groupby(["state", "churn"]).size().unstack().plot(kind='bar', stacked=True, figsize=(40,10))
# converting the catagorical columns into integer values
label_encoder = preprocessing.LabelEncoder()

data['state'] = label_encoder.fit_transform(data['state'])
data['international plan'] = label_encoder.fit_transform(data['international plan'])
data['voice mail plan'] = label_encoder.fit_transform(data['voice mail plan'])
data['churn'] = label_encoder.fit_transform(data['churn'])
# one hot encoding of catagorical variables
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()

catagories = ['state', 'area code']
catagorical_data = data.loc[:,catagories]

enc.fit(catagorical_data)
one_hot = pd.DataFrame(enc.transform(catagorical_data).toarray())
one_hot.iloc[-3:]
# combining it with the data

data_f = pd.concat([data, one_hot], axis = 1)
data_f.drop(["phone number","state","area code","churn"], axis = 1, inplace = True)
# features
features = data_f.columns
# transforming the dataframe into matrix
X = data_f.as_matrix().astype(np.float)
X.shape
Y = data['churn']
print(X.shape)
Y.shape
# to ignore warnings
import warnings
warnings.filterwarnings("ignore")

def model_runner(X, Y, model, n_fold, **kwargs):
    # we shall use stratified sampling because the data is not balanced
    k_fold_cv = cross_validation.StratifiedKFold(Y, n_folds = n_fold,shuffle = True)
    Y_predict = Y.copy()
    
    for i, j in k_fold_cv:
        X_train, X_test = X[i], X[j]
        Y_train = Y[i]
        mdl = model(**kwargs)
        mdl.fit(X_train, Y_train)
        Y_predict[j] = mdl.predict(X_test)
    return Y_predict

def accuracy_printer (name, Y, Y_predict):
    print(str(name) + ' :' + '{}'.format(metrics.accuracy_score(Y, Y_predict)))
    
def confusion_matrix (name, Y, Y_predict):
    con_mtx = metrics.confusion_matrix(Y, Y_predict)
    sns.heatmap(con_mtx, annot = True, fmt = '')
    title = name
    plt.title(title)
    return con_mtx
# Logistic Regression
 
Y_lr = model_runner(X, Y, linear_model.LogisticRegression, 10)
accuracy_printer('Logistic Regression', Y, Y_lr)
cm_lr = confusion_matrix('Logistic Regression', Y, Y_lr)
# K Nearest Neighbor
 
Y_kn = model_runner(X, Y, neighbors.KNeighborsClassifier, 10)
accuracy_printer('K Nearest Neighbor', Y, Y_kn)
cm_kn = confusion_matrix('K Nearest Neighbor', Y, Y_kn)
# Decision Tree
 
Y_dt = model_runner(X, Y, tree.DecisionTreeClassifier, 10)
accuracy_printer('Decision Tree', Y, Y_dt)
cm_dt = confusion_matrix('Decision Tree', Y, Y_dt)
# Support vector machine
 
Y_svm = model_runner(X, Y, svm.SVC, 10)
accuracy_printer('Support vector machine', Y, Y_svm)
cm_svm = confusion_matrix('Support vector machine', Y, Y_svm)
# Random Forest
 
Y_rf = model_runner(X, Y, ensemble.RandomForestClassifier, 10)
accuracy_printer('Random Forest', Y, Y_rf)
cm_rf = confusion_matrix('Random Forest', Y, Y_rf)
# Gradient Boosting
 
Y_gb = model_runner(X, Y, ensemble.GradientBoostingClassifier, 10)
accuracy_printer('Gradient Boosting', Y, Y_gb)
cm_gb = confusion_matrix('Gradient Boosting', Y, Y_gb)
# Bagging
 
Y_bg = model_runner(X, Y, ensemble.BaggingClassifier, 10)
accuracy_printer('Bagging', Y, Y_bg)
cm_bg = confusion_matrix('Bagging', Y, Y_bg)
# Each subscription cost's $100 and for a future churned customer we are allocating $20 off to prevent churn
# if a churned customer is correctly predicted we will save 100(subscription) - 20 (offer) = $80
# if we predict non churn customer as a churned customer we would loose = $20
# if a non churn customer is correctly predicted we save nothing 
# if we predict churn customer as a non churn customer we would loose = $100

def cost_cal (con_mtx):
    savings = con_mtx[0,1]*(-20) + con_mtx[1,0]*(-100) + con_mtx[1,1]*(80)
    with_out_ml = (con_mtx[1,0] + con_mtx[1,1])*(-100)
    total_savings = savings - with_out_ml
    return total_savings
print('Total profit by using Logistic Regression :'+ str(cost_cal(con_mtx = cm_lr)))
print('Total profit by using Gradient Boosting :'+ str(cost_cal(con_mtx = cm_gb)))
print('Total profit by using Support vector machine :'+ str(cost_cal(con_mtx = cm_svm)))
print('Total profit by using Random Forest :'+ str(cost_cal(con_mtx = cm_rf)))
print('Total profit by using K Nearest Neighbor :'+ str(cost_cal(con_mtx = cm_kn)))
print('Total profit by using Decision Tree :'+ str(cost_cal(con_mtx = cm_dt)))
print('Total profit by using Bagging :'+ str(cost_cal(con_mtx = cm_bg)))
# Our main aim is to predict all the churn customers and to maximize the profits. 
# even tho we are saving 63760 using Gradient Boosting but, we are misclassifying 125 customers as non churn customers
# This can be taken care by adjusting the threshold for classification.


def model_runner_thr(X, Y, model, n_fold, name, **kwargs):
    # we shall use stratified sampling because the data is not balanced
    k_fold_cv = cross_validation.StratifiedKFold(Y, n_folds = n_fold,shuffle = True)
    Y_predict = Y.copy()
    dic1 = {}
    dic2 = {}
    for k in np.arange(0,1,0.1):
        for i, j in k_fold_cv:
            X_train, X_test = X[i], X[j]
            Y_train = Y[i]
            mdl = model(**kwargs)
            mdl.fit(X_train, Y_train)
            Y_predict[j] = (mdl.predict_proba(X_test)[:,1] >= k ).astype(bool)
        con_mtx = metrics.confusion_matrix(Y,Y_predict)
        dic1[k] = con_mtx
        net_profit = cost_cal(con_mtx = con_mtx)
        dic2[k] = net_profit
        opt_thr = max(dic2, key=dic2.get)
    return dic1[opt_thr], dic2[opt_thr]
# Logistic Regression

cm_lr_thr, sav_lr_thr = model_runner_thr(X, Y, linear_model.LogisticRegression, 10, 'Logistic Regression')
print(cm_lr_thr, sav_lr_thr)
# K Nearest Neighbor

cm_kn_thr, sav_kn_thr = model_runner_thr(X, Y, neighbors.KNeighborsClassifier, 10, 'K Nearest Neighbor')
print(cm_kn_thr, sav_kn_thr)
# Decision Tree

cm_dt_thr, sav_dt_thr = model_runner_thr(X, Y, tree.DecisionTreeClassifier, 10, 'Decision Tree')
print(cm_dt_thr, sav_dt_thr)
# Random Forest

cm_rf_thr, sav_rf_thr = model_runner_thr(X, Y, ensemble.RandomForestClassifier, 10, 'Random Forest')
print(cm_rf_thr, sav_rf_thr)
# Gradient Boosting

cm_gb_thr, sav_gb_thr = model_runner_thr(X, Y, ensemble.GradientBoostingClassifier, 10, 'Gradient Boosting')
print(cm_gb_thr, sav_gb_thr)
# Bagging

cm_bg_thr, sav_bg_thr = model_runner_thr(X, Y, ensemble.BaggingClassifier, 10, 'Bagging')
print(cm_bg_thr, sav_bg_thr)