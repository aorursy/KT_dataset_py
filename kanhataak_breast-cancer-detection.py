import pandas as pd 

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sn
from sklearn.datasets import load_breast_cancer

ds= load_breast_cancer()

ds
type(ds)
ds.keys()
ds['data']
# malignant or benign value

ds['target']
ds['target_names']
ds['DESCR']
ds['feature_names']  # name of features
ds['filename'] # Location/ path of data file
df=pd.DataFrame(np.c_[ds['data'],ds['target']],columns=

               np.append(ds['feature_names'],['target']))
# dataframe to csv

df.to_csv('breast_cancer_dataset.csv')
df.head()
df.info()
df.describe()
# pairplot of cancer dataframe

# sn.pairplot(df,hue ='target')
# pairplot of sample feature

sn.pairplot(df,hue="target",vars=['mean radius','mean texture','mean perimeter',

                                  'mean area','mean smoothness','mean compactness'])
# count target class

sn.countplot(df["target"])
# counter plot of feature mean radius

plt.figure(figsize=(25,10))

sn.countplot(df['mean radius'])
plt.figure(figsize=(25,10))

sn.heatmap(df)
df.corr()
# heatmap of correlation matrix of breast canver dataframe

plt.figure(figsize=(25,25))

sn.heatmap(df.corr(),annot=True,linewidths=3)
# creat second dataframe by droping target

df2=df.drop(['target'],axis = 1)
df2.shape
plt.figure(figsize = (20,10))

ax = sn.barplot(df2.corrwith(df.target).index, df2.corrwith(df.target))

ax.tick_params(labelrotation = 90)
# input variable

X = df.drop(['target'], axis = 1)

X.head(6)
# output variable

y = df['target']

y.head(6)
# split dataset into train and test

from sklearn.model_selection import train_test_split

X_train, X_test, y_train,y_test = train_test_split(X,y, test_size=0.2,random_state=5)
# Feature scaling

# we need converting different values into one unit

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_sc =scaler.fit_transform(X_train)

X_test_sc = scaler.transform(X_test)
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score
# Support vector classifier

from sklearn.svm import SVC

svc_classifier = SVC()

svc_classifier.fit(X_train, y_train)

y_pred_scv = svc_classifier.predict(X_test)

accuracy_score(y_test, y_pred_scv)
# Train with Standard scaled Data

svc_classifier2 = SVC()

svc_classifier2.fit(X_train_sc, y_train)

y_pred_svc_sc = svc_classifier2.predict(X_test_sc)

accuracy_score(y_test, y_pred_svc_sc)
# Logistic Regression

from sklearn.linear_model import LogisticRegression

lr_classifier = LogisticRegression(random_state = 51)

lr_classifier.fit(X_train, y_train)

y_pred_lr = lr_classifier.predict(X_test)

accuracy_score(y_test, y_pred_lr)
# K – Nearest Neighbor Classifier

from sklearn.neighbors import KNeighborsClassifier

knn_classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)

knn_classifier.fit(X_train, y_train)

y_pred_knn = knn_classifier.predict(X_test)

accuracy_score(y_test, y_pred_knn)
# Train with Standard scaled Data

knn_classifier2 = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)

knn_classifier2.fit(X_train_sc, y_train)

y_pred_knn_sc = knn_classifier.predict(X_test_sc)

accuracy_score(y_test, y_pred_knn_sc)
# Naive Bayes Classifier

from sklearn.naive_bayes import GaussianNB

nb_classifier = GaussianNB()

nb_classifier.fit(X_train, y_train)

y_pred_nb = nb_classifier.predict(X_test)

accuracy_score(y_test, y_pred_nb)
# Train with Standard scaled Data

nb_classifier2 = GaussianNB()

nb_classifier2.fit(X_train_sc, y_train)

y_pred_nb_sc = nb_classifier2.predict(X_test_sc)

accuracy_score(y_test, y_pred_nb_sc)
# Decision Tree Classifier

from sklearn.tree import DecisionTreeClassifier

dt_classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 51)

dt_classifier.fit(X_train, y_train)

y_pred_dt = dt_classifier.predict(X_test)

accuracy_score(y_test, y_pred_dt)
# Train with Standard scaled Data

dt_classifier2 = DecisionTreeClassifier(criterion = 'entropy', random_state = 51)

dt_classifier2.fit(X_train_sc, y_train)

y_pred_dt_sc = dt_classifier.predict(X_test_sc)

accuracy_score(y_test, y_pred_dt_sc)
# Random Forest Classifier

from sklearn.ensemble import RandomForestClassifier

rf_classifier = RandomForestClassifier(n_estimators = 20, criterion = 'entropy', random_state = 51)

rf_classifier.fit(X_train, y_train)

y_pred_rf = rf_classifier.predict(X_test)

accuracy_score(y_test, y_pred_rf)
# Train with Standard scaled Data

rf_classifier2 = RandomForestClassifier(n_estimators = 20, criterion = 'entropy', random_state = 51)

rf_classifier2.fit(X_train_sc, y_train)

y_pred_rf_sc = rf_classifier.predict(X_test_sc)

accuracy_score(y_test, y_pred_rf_sc)
# Adaboost Classifier

from sklearn.ensemble import AdaBoostClassifier

adb_classifier = AdaBoostClassifier(DecisionTreeClassifier(criterion = 'entropy', random_state = 200),

                                    n_estimators=2000,

                                    learning_rate=0.1,

                                    algorithm='SAMME.R',

                                    random_state=1,)

adb_classifier.fit(X_train, y_train)

y_pred_adb = adb_classifier.predict(X_test)

accuracy_score(y_test, y_pred_adb)
# Train with Standard scaled Data

adb_classifier2 = AdaBoostClassifier(DecisionTreeClassifier(criterion = 'entropy', random_state = 200),

                                    n_estimators=2000,

                                    learning_rate=0.1,

                                    algorithm='SAMME.R',

                                    random_state=1,)

adb_classifier2.fit(X_train_sc, y_train)

y_pred_adb_sc = adb_classifier2.predict(X_test_sc)

accuracy_score(y_test, y_pred_adb_sc)
# XGBoost Classifier

from xgboost import XGBClassifier

xgb_classifier = XGBClassifier()

xgb_classifier.fit(X_train, y_train)

y_pred_xgb = xgb_classifier.predict(X_test)

accuracy_score(y_test, y_pred_xgb)
# !pip install xgboost
# Train with Standard scaled Data

xgb_classifier2 = XGBClassifier()

xgb_classifier2.fit(X_train_sc, y_train)

y_pred_xgb_sc = xgb_classifier2.predict(X_test_sc)

accuracy_score(y_test, y_pred_xgb_sc)
# XGBoost classifier most required parameters

params={

 "learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,

 "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],

 "min_child_weight" : [ 1, 3, 5, 7 ],

 "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],

 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ] 

}

1

2

3

4

# Randomized Search

from sklearn.model_selection import RandomizedSearchCV

random_search = RandomizedSearchCV(xgb_classifier, param_distributions=params, scoring= 'roc_auc', n_jobs= -1, verbose= 3)

random_search.fit(X_train, y_train)
random_search.best_params_
XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,

       colsample_bynode=1, colsample_bytree=0.3, gamma=0.4,

       learning_rate=0.3, max_delta_step=0, max_depth=3,

       min_child_weight=1, missing=None, n_estimators=100, n_jobs=1,

       nthread=None, objective='binary:logistic', random_state=0,

       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,

       silent=None, subsample=1, verbosity=1)

1

2

3

4

5

6

7

8

9

10

11

# training XGBoost classifier with best parameters

xgb_classifier_pt = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,

       colsample_bynode=1, colsample_bytree=0.4, gamma=0.2,

       learning_rate=0.1, max_delta_step=0, max_depth=15,

       min_child_weight=1, missing=None, n_estimators=100, n_jobs=1,

       nthread=None, objective='binary:logistic', random_state=0,

       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,

       silent=None, subsample=1, verbosity=1)

 

xgb_classifier_pt.fit(X_train, y_train)

y_pred_xgb_pt = xgb_classifier_pt.predict(X_test)
# confusion Matrix

cm = confusion_matrix(y_test, y_pred_xgb_pt)

plt.title('Heatmap of Confusion Matrix', fontsize = 15)

sn.heatmap(cm, annot = True)

plt.show()
print(classification_report(y_test, y_pred_xgb_pt))
# Cross validation

from sklearn.model_selection import cross_val_score

cross_validation = cross_val_score(estimator = xgb_classifier2, X = X_train_sc, y = y_train, cv = 10)

print("Cross validation of XGBoost model = ",cross_validation)

print("Cross validation of XGBoost model (in mean) = ",cross_validation.mean())

from sklearn.model_selection import cross_val_score

cross_validation = cross_val_score(estimator = xgb_classifier_pt, X = X_train_sc,y = y_train, cv = 10)

print("Cross validation accuracy of XGBoost model = ", cross_validation)

print("\nCross validation mean accuracy of XGBoost model = ", cross_validation.mean())
