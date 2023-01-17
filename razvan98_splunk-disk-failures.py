import numpy as np 

import pandas as pd

import seaborn as sns

import scipy

import matplotlib.pyplot as plt

import graphviz 

from sklearn.preprocessing import LabelEncoder

from sklearn import model_selection

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.datasets import make_regression

from sklearn.cluster import KMeans

from sklearn.datasets import make_blobs

from sklearn.metrics import completeness_score

from sklearn.linear_model import Lasso

from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble

from xgboost import XGBClassifier

from sklearn.model_selection import cross_val_predict

from sklearn.metrics import confusion_matrix

from sklearn.metrics import f1_score

from sklearn.metrics import precision_score, recall_score, accuracy_score

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import ExtraTreesClassifier, AdaBoostClassifier



from sklearn.metrics import precision_recall_curve

import os

import sys



import warnings

warnings.filterwarnings('ignore')
train_data_disk = pd.read_csv('/kaggle/input/splunk/disk_failures_0.csv')

test_data_disk = pd.read_csv('/kaggle/input/splunk/disk_failures_1.csv')



train_df_disk = train_data_disk.copy(deep = True)

test_df_disk = test_data_disk.copy(deep = True)

combine = [train_df_disk, test_df_disk]



train_df_disk.head()

test_df_disk.head()



train_df_disk.info()

test_df_disk.info()

print(train_df_disk.head())

print("------------------------------------------------------------")

print("------------------------------------------------------------")

print("------------------------------------------------------------")

print(test_df_disk.head())
sns.barplot(x="CapacityBytes",y="DiskFailure", data=train_df_disk) ;
sns.barplot(x="SMART_1_Raw",y="DiskFailure", data=train_df_disk) ;
sns.barplot(x="SMART_2_Raw",y="DiskFailure", data=train_df_disk) ;
sns.barplot(x="SMART_3_Raw",y="DiskFailure", data=train_df_disk) ;
sns.barplot(x="SMART_4_Raw",y="DiskFailure", data=train_df_disk) ;
sns.barplot(x="SMART_5_Raw",y="DiskFailure", data=train_df_disk) ;
Target=['DiskFailure']

columns_to_check = ['SMART_1_Raw','SMART_2_Raw','SMART_3_Raw','SMART_4_Raw', 'SMART_5_Raw']



data_x_bin = ['SMART_1_Raw','SMART_2_Raw','SMART_3_Raw','SMART_4_Raw', 'SMART_5_Raw']

data_xy_bin = Target + data_x_bin



data = train_df_disk[data_xy_bin]

X = train_df_disk[data_x_bin]

y = train_df_disk[Target]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

X_pred = test_df_disk[data_x_bin]



scaler = StandardScaler() 

scaler.fit(X_train)

X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test) 
for dataset in combine:    

    dataset['SMART_1_Raw'].fillna(((dataset['SMART_1_Raw']-dataset['SMART_1_Raw'].min())/(dataset['SMART_1_Raw'].max()-dataset['SMART_1_Raw'].min())) , inplace = True)

    dataset['SMART_2_Raw'].fillna(((dataset['SMART_2_Raw']-dataset['SMART_2_Raw'].min())/(dataset['SMART_2_Raw'].max()-dataset['SMART_2_Raw'].min())) , inplace = True)

    dataset['SMART_3_Raw'].fillna(((dataset['SMART_3_Raw']-dataset['SMART_3_Raw'].min())/(dataset['SMART_3_Raw'].max()-dataset['SMART_3_Raw'].min())) , inplace = True)

    dataset['SMART_4_Raw'].fillna(((dataset['SMART_4_Raw']-dataset['SMART_4_Raw'].min())/(dataset['SMART_4_Raw'].max()-dataset['SMART_4_Raw'].min())) , inplace = True)

    dataset['SMART_5_Raw'].fillna(((dataset['SMART_5_Raw']-dataset['SMART_5_Raw'].min())/(dataset['SMART_5_Raw'].max()-dataset['SMART_5_Raw'].min())) , inplace = True)



label = LabelEncoder()

for dataset in combine:    

    dataset['SMART_1_Transformed'] = label.fit_transform(dataset['SMART_1_Raw'])

    dataset['SMART_2_Transformed'] = label.fit_transform(dataset['SMART_2_Raw'])

    dataset['SMART_3_Transformed'] = label.fit_transform(dataset['SMART_3_Raw'])

    dataset['SMART_4_Transformed'] = label.fit_transform(dataset['SMART_4_Raw'])

    dataset['SMART_5_Transformed'] = label.fit_transform(dataset['SMART_5_Raw'])

    

    dataset = dataset.reindex(columns=['SMART_1_Raw', 'SMART_2_Raw', 'SMART_3_Raw', 'SMART_4_Raw', 'SMART_5_Raw'])

               
Target=['DiskFailure']

columns_to_check = ['SMART_1_Raw','SMART_2_Raw','SMART_3_Raw' ,'SMART_4_Raw' ,'SMART_5_Raw']



data_x_bin = ['SMART_1_Transformed','SMART_2_Transformed','SMART_3_Transformed' ,'SMART_4_Transformed' ,'SMART_5_Transformed']

data_xy_bin = Target + data_x_bin



data = train_df_disk[data_xy_bin]

X = train_df_disk[data_x_bin]

y = train_df_disk[Target]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

X_pred = test_df_disk[data_x_bin]



scaler = StandardScaler() 

scaler.fit(X_train)

X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test) 
# Support Vector Machines



np.any(np.isnan(X_train))

np.all(np.isfinite(X_train))





svc = SVC()

svc.fit(X_train, y_train)

acc_train_svc = round(svc.score(X_train, y_train) * 100, 2)

acc_test_svc = round(svc.score(X_test, y_test) * 100, 2)



predictions = cross_val_predict(svc, X_train, y_train, cv=3)

confusion_matrix(y_train, predictions)

print("Support Vector Machine Precision SVC:", precision_score(y_train, predictions, pos_label='No'))

print("Support Vector Machine Recall SVC:",recall_score(y_train, predictions,pos_label='No'))

print('Support Vector Machine test accurary: ',acc_test_svc)

print("Support Vector Machine F1 SVC:", f1_score(y_train, predictions,pos_label='No'))
# Decision Tree



decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train, y_train)

acc_train_decision_tree = round(decision_tree.score(X_train, y_train) * 100, 2)

acc_test_decision_tree = round(decision_tree.score(X_test, y_test) * 100, 2)

predictions1 = cross_val_predict(decision_tree, X_train, y_train, cv=3)



confusion_matrix(y_train, predictions1)





print("Precision Decision Tree:", precision_score(y_train, predictions1,pos_label="No"))

print("Recall  Decision Tree:",recall_score(y_train, predictions1,pos_label="No"))

print('Decision Tree test accurary: ',acc_test_decision_tree)

print("F1  Decision Tree:", f1_score(y_train, predictions1,pos_label="No"))
# Random Forest



random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, y_train)

acc_train_random_forest = round(random_forest.score(X_train, y_train) * 100, 2)

acc_test_random_forest = round(random_forest.score(X_test, y_test) * 100, 2)



predictions2 = cross_val_predict(random_forest, X_train, y_train, cv=3)

confusion_matrix(y_train, predictions2)



print("Precision Random Forest:", precision_score(y_train, predictions2, pos_label="No"))

print("Recall Random Forest:",recall_score(y_train, predictions2, pos_label="No"))

print('Random Forest test accurary: ',acc_test_random_forest)

print("F1 Random Forest:", f1_score(y_train, predictions2, pos_label="No"))
#Logistic Regression



log_reg = LogisticRegression()

log_reg.fit(X_train, y_train)

y_pred_logReg = log_reg.predict(X_train)



predictions4 = cross_val_predict(log_reg, X_train, y_train, cv=3)

confusion_matrix(y_train, predictions4)



print("Precision Logistic:", precision_score(y_train, predictions4, pos_label="No"))

print("Recall Logistic:",recall_score(y_train, predictions4, pos_label="No"))

print('Logistic Regresion accurasy: ',accuracy_score(y_train, y_pred_logReg))

print("F1 Logistic:", f1_score(y_train, predictions4, pos_label="No"))
# K-Nearest Neighbors



knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train, y_train)

acc_train_knn = round(knn.score(X_train, y_train) * 100, 2)

acc_test_knn = round(knn.score(X_test, y_test) * 100, 2)



predictions3 = cross_val_predict(knn, X_train, y_train, cv=3)

confusion_matrix(y_train, predictions3)





print("Precision KNN:", precision_score(y_train, predictions3, pos_label = "No"))

print("Recall KNN:",recall_score(y_train, predictions3,pos_label = "No"))

print('K-Nearest Neighbors test accurary: ',acc_test_knn)

print("F1 KNN:", f1_score(y_train, predictions3,pos_label = "No"))
# Ada Boost



ada_params = {'base_estimator__criterion': ['gini', 'entropy'],

              'base_estimator__splitter': ['best', 'random'],

              'algorithm': ['SAMME', 'SAMME.R'],

              'n_estimators': [1, 2],

              'learning_rate': [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 1.5]}



ada_gridsearch = GridSearchCV(AdaBoostClassifier(DecisionTreeClassifier(random_state=42)),

                              param_grid = ada_params,

                              cv=StratifiedKFold(n_splits=10),

                              scoring='accuracy',

                              n_jobs= -1)



ada_gridsearch.fit(X_train, y_train)



ada_gridsearch.best_score_

y_pred_ada = ada_gridsearch.best_estimator_.predict(X_train)



predictions6 = cross_val_predict(ada_gridsearch, X_train, y_train, cv=3)

confusion_matrix(y_train, predictions6)



print('Ada Boost accuracy: ',accuracy_score(y_train, y_pred_ada))

print("Precision Ada:", precision_score(y_train, predictions6,pos_label = "No"))

print("Recall Ada:",recall_score(y_train, predictions6,pos_label = "No"))

print("F1 Ada:", f1_score(y_train, predictions6,pos_label="No"))
accuracy_score_df = pd.DataFrame({'Models': ['Logistic Regression',

                                             'AdaBoost',],

                                  'Accuracy score': [accuracy_score(y_train, y_pred_logReg),

                                                     accuracy_score(y_train, y_pred_ada)]},

                                 columns=['Models', 'Accuracy score'])



plt.figure(figsize=(10, 5))

plt.barh(np.arange(len(accuracy_score_df['Models'])),

         accuracy_score_df['Accuracy score'],

         align='center',

         height=0.5)



plt.yticks(np.arange(len(accuracy_score_df['Models'])), accuracy_score_df['Models'])

plt.tick_params(labelsize=12)

plt.xlabel('Accuracy score', fontdict={'fontsize': 13})

plt.ylabel('Models', fontdict={'fontsize': 13})



plt.show()
cv_split = model_selection.ShuffleSplit(n_splits = 10, test_size = .3, train_size = .6, random_state = 0 )



param_grid = {'criterion': ['gini', 'entropy'],

              'max_depth': [2,4,6,8,10,None],

              'random_state': [0] 

             }



tune_model = model_selection.GridSearchCV(tree.DecisionTreeClassifier(), param_grid=param_grid, scoring = 'roc_auc', cv = cv_split)

tune_model.fit(train_df_disk[data_x_bin], train_df_disk[Target])



dtree = tree.DecisionTreeClassifier(criterion='gini', max_depth=4, random_state = 0)

dtree.fit(train_df_disk[data_x_bin], train_df_disk[Target])

dot_data = tree.export_graphviz(dtree, out_file=None, 

                                feature_names = data_x_bin, class_names = True,

                                filled = True, rounded = True)

graph = graphviz.Source(dot_data) 

graph
