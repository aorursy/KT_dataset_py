

import numpy as np

import pandas as pd 

from matplotlib import pyplot as plt

import seaborn as sns

import os

import xgboost as xgb

from xgboost import XGBClassifier



def warn(*args, **kwargs):

    pass

import warnings

warnings.warn = warn



from sklearn import preprocessing

from sklearn import model_selection

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import  f1_score

from sklearn import preprocessing ,decomposition, model_selection,metrics,pipeline

from sklearn.model_selection import GridSearchCV



from sklearn import tree

from sklearn.metrics import confusion_matrix

from sklearn.ensemble import AdaBoostClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, plot_confusion_matrix

from sklearn.svm import LinearSVC, SVC, NuSVC

from sklearn.linear_model import LogisticRegression, SGDClassifier

from sklearn.naive_bayes import MultinomialNB, BernoulliNB

import random
df = pd.read_csv('/kaggle/input/wine-quality-binary-classification/wine.csv')

df_data = df.replace(['good', 'bad'], [0,1])

target = df_data.quality

features = df.columns

df_data.drop(columns = ['quality'], inplace=True)

df_data = preprocessing.scale(df_data)

df_data = pd.DataFrame(data=df_data, columns=features[0:11])
X_train, X_test, y_train, y_test = train_test_split(df_data, target, random_state = 1, shuffle = True)
def show_cm(classifier, X_test, y_test):

    plt.style.use('default')

    class_names = ['Bad', 'Good']

    titles_options = [("Confusion matrix, without normalization", None),

                  ("Normalized confusion matrix", 'true')]

    for title, normalize in titles_options:

        disp = plot_confusion_matrix(classifier, X_test, y_test,

                                 display_labels=class_names,

                                 cmap=plt.cm.Blues,

                                 normalize=normalize,

                                 xticks_rotation = 30)

        plt.title(title)

        plt.show()
# Logistic Regression

lr = LogisticRegression()

lr.fit(X_train ,y_train)

lr_pred = lr.predict(X_test)

lr_score = accuracy_score(y_test, lr_pred)

print('LogisticRegression Score: ', lr_score)

show_cm(lr, df_data, target)



# XGBoost

xgb_classifier=xgb.XGBClassifier(objective='binary:logistic',learning_rate = 0.1,gamma=0.01,max_depth = 10,booster="gbtree")

xgb_classifier.fit(X_train ,y_train)

xgb_pred = xgb_classifier.predict(X_test)

xgb_score = accuracy_score(y_test,xgb_pred)

print('XGBoost Score: ',xgb_score)

show_cm(xgb_classifier, df_data, target)



# Decision Tree

dt_clf = tree.DecisionTreeClassifier()

dt_clf.fit(X_train ,y_train)

dt_pred = dt_clf.predict(X_test)

dt_score = accuracy_score(y_test,dt_pred)

print('Decision Tree Score: ',dt_score)

show_cm(dt_clf, df_data, target)



# AdaBoost

from sklearn.ensemble import AdaBoostClassifier

ada_clf = AdaBoostClassifier(n_estimators=1500,learning_rate=1,algorithm='SAMME.R')

ada_clf.fit(X_train ,y_train)

ada_pred = ada_clf.predict(X_test)

ada_score = accuracy_score(y_test,ada_pred)

print('AdaBoost Decision Tree Score: ',ada_score)

show_cm(ada_clf, df_data, target)



# Random Forest

from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier(max_depth=10, random_state=0)

rf_clf.fit(X_train ,y_train)

rf_pred = rf_clf.predict(X_test)

rf_score = accuracy_score(y_test,rf_pred)

print('Random Forest Score: ',rf_score)

show_cm(rf_clf, df_data, target)



# Linear Support Vector Classification

lsvc = LinearSVC()

lsvc.fit(X_train ,y_train)

lsvc_pred = lsvc.predict(X_test)

lsvc_score = accuracy_score(y_test,lsvc_pred)

print('LinearSVC Score: ', lsvc_score)

show_cm(lsvc, df_data, target)



# Support Vector Classification

svc = SVC()

svc.fit(X_train ,y_train)

svc_pred = svc.predict(X_test)

svc_score = accuracy_score(y_test,svc_pred)

print('SVC Score: ', svc_score)

show_cm(svc, df_data, target)



# Nu-Support Vector Classification

nusvc = NuSVC()

nusvc.fit(X_train ,y_train)

nusvc_pred = nusvc.predict(X_test)

nusvc_score = accuracy_score(y_test, nusvc_pred)

print('NuSVC Score: ', nusvc_score)

show_cm(nusvc, df_data, target)



# Stochastic Gradient Descent

sgd = SGDClassifier()

sgd.fit(X_train ,y_train)

sgd_pred = sgd.predict(X_test)

sgd_score = accuracy_score(y_test,sgd_pred)

print('SGD Score: ', sgd_score)

show_cm(sgd, df_data, target)



models = pd.DataFrame({

    'Model': ['Linear Support Vector Classification', 'Support vector Classification', 'Nu-Support Vector Classification', 

              'Stochastic Gradient Decent', 'Logistic Regression','XGBoost','AdaBoost', 'Decision Tree', 'Random Forest'],

    'Score': [ 

              lsvc_score, svc_score, nusvc_score, 

              sgd_score, lr_score, xgb_score, ada_score, dt_score, rf_score]})

models = models.sort_values(by="Score",ascending=False)

print(models.to_string())



features = df.columns.values

importances = xgb_classifier.get_booster().get_score(importance_type="gain")

importances = {k: v for k, v in sorted(importances.items(), key=lambda item: item[1])}



plt.style.use('fivethirtyeight')

plt.title('Feature Importances')

plt.barh(list(importances.keys()), list(importances.values()), color='b', align='center')

plt.xlabel('Relative Importance')

plt.show()



from xgboost import plot_tree



plot_tree(xgb_classifier, num_trees=0, rankdir='LR')

fig = plt.gcf()

fig.set_size_inches(150, 200)

plt.show()