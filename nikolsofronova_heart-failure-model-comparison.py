import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
dataset = pd.read_csv('../input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')
dataset.head()
dataset.isnull().sum()
plt.figure(figsize=(12,8))

sns.heatmap(dataset.drop(columns=['DEATH_EVENT']).corr(),vmin=-1, cmap='coolwarm', annot=True)

plt.show()
dataset.drop(columns=['DEATH_EVENT']).corrwith(dataset['DEATH_EVENT'])
X = dataset.iloc[:, 0:11].values # all columns except for the last two (time and death variables)

y = dataset.iloc[:, -1].values



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
from sklearn.linear_model import LogisticRegression

classifier_lr = LogisticRegression(random_state = 0)

classifier_lr.fit(X_train, y_train)
y_pred_lr = classifier_lr.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score

cm_lr = confusion_matrix(y_test, y_pred_lr)

print(cm_lr)

accuracy_score(y_test, y_pred_lr)
from sklearn.naive_bayes import GaussianNB

classifier_NB = GaussianNB()

classifier_NB.fit(X_train, y_train)
y_pred_NB = classifier_NB.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score

cm_NB = confusion_matrix(y_test, y_pred_NB)

print(cm_NB)

accuracy_score(y_test, y_pred_NB)
from sklearn.svm import SVC

classifier_SVM = SVC(kernel = 'linear', random_state = 0)

classifier_SVM.fit(X_train, y_train)
y_pred_SVM = classifier_SVM.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score

cm_SVM = confusion_matrix(y_test, y_pred_SVM)

print(cm_SVM)

accuracy_score(y_test, y_pred_SVM)
from sklearn.svm import SVC

classifier_KSVM = SVC(kernel = 'rbf', random_state = 0)

classifier_KSVM.fit(X_train, y_train)
y_pred_KSVM = classifier_KSVM.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score

cm_KSVM = confusion_matrix(y_test, y_pred_KSVM)

print(cm_KSVM)

accuracy_score(y_test, y_pred_KSVM)
from sklearn.ensemble import RandomForestClassifier

classifier_RF = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)

classifier_RF.fit(X_train, y_train)
y_pred_RF = classifier_RF.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score

cm_RF = confusion_matrix(y_test, y_pred_RF)

print(cm_RF)

accuracy_score(y_test, y_pred_RF)
from sklearn.tree import DecisionTreeClassifier

classifier_DT = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)

classifier_DT.fit(X_train, y_train)
y_pred_DT = classifier_DT.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score

cm_DT = confusion_matrix(y_test, y_pred_DT)

print(cm_DT)

accuracy_score(y_test, y_pred_DT)
from sklearn.neighbors import KNeighborsClassifier

classifier_KNN = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)

classifier_KNN.fit(X_train, y_train)
y_pred_KNN = classifier_KNN.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score

cm_KNN = confusion_matrix(y_test, y_pred_KNN)

print(cm_KNN)

accuracy_score(y_test, y_pred_KNN)
from xgboost import XGBClassifier

classifier_XGBoost = XGBClassifier()

classifier_XGBoost.fit(X_train, y_train)
y_pred_XGBoost = classifier_XGBoost.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score

cm_XGBoost = confusion_matrix(y_test, y_pred_XGBoost)

print(cm_XGBoost)

accuracy_score(y_test, y_pred_XGBoost)
comparison = []

comparison_accuracy = []

comparison.append('Logistic Regression')

comparison_accuracy.append(accuracy_score(y_test, y_pred_lr))

comparison.append('Naive Bayes')

comparison_accuracy.append(accuracy_score(y_test, y_pred_NB))

comparison.append('Linear SVM')

comparison_accuracy.append(accuracy_score(y_test, y_pred_SVM))

comparison.append('Kernel SVM')

comparison_accuracy.append(accuracy_score(y_test, y_pred_KSVM))

comparison.append('Random Forest')

comparison_accuracy.append(accuracy_score(y_test, y_pred_RF))

comparison.append('Decision Tree')

comparison_accuracy.append(accuracy_score(y_test, y_pred_DT))

comparison.append('K-Nearest Neighbors (K-NN)')

comparison_accuracy.append(accuracy_score(y_test, y_pred_KNN))

comparison.append('XG Boost')

comparison_accuracy.append(accuracy_score(y_test, y_pred_XGBoost))



accuracy_comparison = pd.DataFrame({'Classification Model': comparison, 'Accuracy': comparison_accuracy})

print(accuracy_comparison)
from sklearn.model_selection import cross_val_score

acc_lr = cross_val_score(estimator = classifier_lr, X = X_train, y = y_train, cv = 10)

acc_NB = cross_val_score(estimator = classifier_NB, X = X_train, y = y_train, cv = 10)

acc_SVM = cross_val_score(estimator = classifier_SVM, X = X_train, y = y_train, cv = 10)

acc_KSVM = cross_val_score(estimator = classifier_KSVM, X = X_train, y = y_train, cv = 10)

acc_RF = cross_val_score(estimator = classifier_RF, X = X_train, y = y_train, cv = 10)

acc_DT = cross_val_score(estimator = classifier_DT, X = X_train, y = y_train, cv = 10)

acc_KNN = cross_val_score(estimator = classifier_KNN, X = X_train, y = y_train, cv = 10)

acc_XGBoost = cross_val_score(estimator = classifier_XGBoost, X = X_train, y = y_train, cv = 10)



kfold_acc_mean = [np.mean(acc_lr), np.mean(acc_NB), np.mean(acc_SVM), np.mean(acc_KSVM), np.mean(acc_RF), 

                  np.mean(acc_DT), np.mean(acc_KNN), np.mean(acc_XGBoost)]

kfold_acc_std = [np.std(acc_lr), np.std(acc_NB), np.std(acc_SVM), np.std(acc_KSVM), np.std(acc_RF), 

                 np.std(acc_DT), np.std(acc_KNN), np.std(acc_XGBoost)]



K_Fold_cross_val = pd.DataFrame({'Classification Model': comparison, 'KFold accuracy mean': kfold_acc_mean, 

                                 'KFold accuracy std': kfold_acc_std})

print(K_Fold_cross_val)