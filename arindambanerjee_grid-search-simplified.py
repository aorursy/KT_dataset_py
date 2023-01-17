# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc

from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt

import seaborn as sns

from prettytable import PrettyTable

%matplotlib inline

import warnings

warnings.filterwarnings("ignore")



# Any results you write to the current directory are saved as output.
X, y = load_breast_cancer(return_X_y=True)

print(X.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 123)
ss = StandardScaler()

X_train_ss = ss.fit_transform(X_train)

X_test_ss = ss.transform(X_test)
clf = RandomForestClassifier()

clf.fit(X_train_ss, y_train)

y_pred = clf.predict(X_test_ss)
def plot_conf_matrix (conf_matrix, dtype):

    class_names = [0,1]

    fontsize=14

    df_conf_matrix = pd.DataFrame(

            conf_matrix, index=class_names, columns=class_names, 

        )

    fig = plt.figure()

    heatmap = sns.heatmap(df_conf_matrix, annot=True, fmt="d")

    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)

    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    plt.title('Confusion Matrix for {0}'.format(dtype))
acc_rf = accuracy_score(y_test, y_pred)

print(acc_rf)
plot_conf_matrix(confusion_matrix(y_test, y_pred), "Test data")


estimators = [10, 50, 100, 200, 500] 

max_depths = [3, 6, 10, 15, 20] 



grid_values = {'n_estimators': estimators, 'max_depth':max_depths}



clf = GridSearchCV(RandomForestClassifier(), grid_values, scoring='roc_auc', n_jobs=-1, verbose=10, cv=3)

clf.fit(X_train_ss, y_train)

best_n_estimators_value = clf.best_params_['n_estimators']

best_max_depth_value = clf.best_params_['max_depth']

best_score = clf.best_score_
max_depth_list = list(clf.cv_results_['param_max_depth'].data)

estimators_list = list(clf.cv_results_['param_n_estimators'].data)



sns.set_style("whitegrid")

plt.figure(figsize=(16,6))

plt.subplot(1,2,1)

data = pd.DataFrame(data={'Estimators':estimators_list, 'Max Depth':max_depth_list, 'AUC':clf.cv_results_['mean_train_score']})

data = data.pivot(index='Estimators', columns='Max Depth', values='AUC')

sns.heatmap(data, annot=True, cmap="YlGnBu").set_title('AUC for Training data')

plt.subplot(1,2,2)

data = pd.DataFrame(data={'Estimators':estimators_list, 'Max Depth':max_depth_list, 'AUC':clf.cv_results_['mean_test_score']})

data = data.pivot(index='Estimators', columns='Max Depth', values='AUC')

sns.heatmap(data, annot=True, cmap="YlGnBu").set_title('AUC for Test data')

plt.show()
def plot_roc_curve(roc_auc_train, roc_auc_test):

    plt.title('Receiver Operating Characteristic')

    plt.plot(fpr_tr, tpr_tr, 'g', label = 'Training AUC = %0.2f' % roc_auc_train)

    plt.plot(fpr_ts, tpr_ts, 'b', label = 'Testing AUC = %0.2f' % roc_auc_test)

    plt.legend(loc = 'lower right')

    plt.plot([0, 1], [0, 1],'r--')

    plt.xlim([0, 1])

    plt.ylim([0, 1])

    plt.ylabel('True Positive Rate')

    plt.xlabel('False Positive Rate')

    plt.show()
#Best hyper parameter 

clf = RandomForestClassifier(n_estimators=best_n_estimators_value, max_depth=best_max_depth_value)

clf.fit(X_train_ss, y_train)



y_pred_train = clf.predict_proba(X_train_ss)[:,1]

y_pred_test = clf.predict_proba(X_test_ss)[:,1]



   

#train data ROC

fpr_tr, tpr_tr, threshold = roc_curve(y_train, y_pred_train)

roc_auc_train = auc(fpr_tr, tpr_tr)



#test data ROC

fpr_ts, tpr_ts, threshold = roc_curve(y_test, y_pred_test)

roc_auc_test = auc(fpr_ts, tpr_ts)



#Plot ROC curve

plot_roc_curve(roc_auc_train, roc_auc_test)
acc_rf_grid = accuracy_score(y_test, clf.predict(X_test_ss))



print(acc_rf_grid)
plot_conf_matrix(confusion_matrix(y_train, clf.predict(X_train_ss)), "Training data")
plot_conf_matrix(confusion_matrix(y_test, clf.predict(X_test_ss)), "Test data")
# Compare both the models using Prettytable library    

x = PrettyTable()



x.field_names = ["Model", "n_estimators", "max_depth","Accuracy"]



x.add_row(["Random Forest w/o GridSearch", "default 10", "None", acc_rf])

x.add_row(["Random Forest with GridSearch", best_n_estimators_value, best_max_depth_value, acc_rf_grid])



print(x)