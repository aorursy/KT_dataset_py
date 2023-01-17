# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# linear algebra

import numpy as np  



# data processing

import pandas as pd  



# data visualising

import matplotlib.pyplot as plt 

import seaborn as sns



# data preprocessing

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler



# model building

from sklearn.linear_model import LogisticRegression # Logistic Regression

from sklearn.naive_bayes import GaussianNB # Naive Bayes

from sklearn.svm import SVC # Support Vector Machine

from sklearn.ensemble import RandomForestClassifier # Random Forest

from sklearn.tree import DecisionTreeClassifier # Decision Tree



# model evaluation

from sklearn.metrics import confusion_matrix, accuracy_score



# k-fold cross validation

from sklearn.model_selection import cross_val_score
dataset = pd.read_csv('/kaggle/input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')
dataset.head()
dataset.tail()
dataset.columns
dataset.info()
plt.figure(figsize=(20,12))

plt.suptitle('Continuous Features', fontsize=20)

for i in range(0, dataset[['age', 'creatinine_phosphokinase', 'ejection_fraction', 'platelets', 'serum_creatinine', 'serum_sodium', 'time']].shape[1]):

    plt.subplot(3, 3, i+1)



    sns.distplot(dataset[['age', 'creatinine_phosphokinase', 'ejection_fraction', 'platelets', 'serum_creatinine', 'serum_sodium', 'time']].iloc[:, i])
plt.figure(figsize=(20,12))

plt.suptitle('Categorical Features', fontsize=20)

for i in range(0, dataset[['anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking']].shape[1]):

    plt.subplot(3, 3, i+1)



    sns.countplot(dataset[['anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking']].iloc[:, i], hue=dataset['DEATH_EVENT'])
plt.figure(figsize=(12,8))

sns.heatmap(dataset.drop(columns=['DEATH_EVENT']).corr(), annot=True)

plt.show()
dataset.drop(columns=['DEATH_EVENT']).corrwith(dataset['DEATH_EVENT'])
plt.figure(figsize=(12,8))

dataset.drop(columns=['DEATH_EVENT']).corrwith(dataset['DEATH_EVENT']).plot.bar(title = 'Correlations with the response variable', rot = 45, grid = True, color = 'orange')

plt.show()
# Finding the features having correlation more than 0.1 or less than -0.1

columns = dataset.drop(columns=['DEATH_EVENT']).columns[np.array(abs(dataset.drop(columns=['DEATH_EVENT']).corrwith(dataset['DEATH_EVENT']).array) > 0.1)]

print(columns)
# Rejecting the features with weak correlations

X = dataset[columns]

y = dataset['DEATH_EVENT']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
sc = StandardScaler()

X_train = pd.DataFrame(sc.fit_transform(X_train))

X_test = pd.DataFrame(sc.transform(X_test))
classifier_lr = LogisticRegression(random_state=0)

classifier_lr.fit(X_train, y_train)

y_pred_lr = classifier_lr.predict(X_test)
sns.heatmap(confusion_matrix(y_test, y_pred_lr), annot=True)

plt.show()
accuracy_score(y_test, y_pred_lr)
model = []

model_accuracy = []
model.append('Logistic Regression')

model_accuracy.append(accuracy_score(y_test, y_pred_lr))
classifier_nb = GaussianNB()

classifier_nb.fit(X_train, y_train)

y_pred_nb = classifier_nb.predict(X_test)
sns.heatmap(confusion_matrix(y_test, y_pred_nb), annot=True)

plt.show()
accuracy_score(y_test, y_pred_nb)
model.append('Naive Bayes')

model_accuracy.append(accuracy_score(y_test, y_pred_nb))
classifier_svm_l = SVC(random_state = 0, kernel='linear')

classifier_svm_l.fit(X_train, y_train)

y_pred_svm_l = classifier_svm_l.predict(X_test)
sns.heatmap(confusion_matrix(y_test, y_pred_svm_l), annot=True)

plt.show()
accuracy_score(y_test, y_pred_svm_l)
model.append('SVM Linear')

model_accuracy.append(accuracy_score(y_test, y_pred_svm_l))
classifier_svm_rbf = SVC(random_state = 0, kernel='rbf')

classifier_svm_rbf.fit(X_train, y_train)

y_pred_svm_rbf = classifier_svm_rbf.predict(X_test)
sns.heatmap(confusion_matrix(y_test, y_pred_svm_rbf), annot=True)

plt.show()
accuracy_score(y_test, y_pred_svm_rbf)
model.append('SVM rbf')

model_accuracy.append(accuracy_score(y_test, y_pred_svm_rbf))
classifier_rf = RandomForestClassifier(criterion='entropy', n_jobs=10, random_state=10)

classifier_rf.fit(X_train, y_train)

y_pred_rf = classifier_rf.predict(X_test)
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True)

plt.show()
accuracy_score(y_test, y_pred_rf)
model.append('Random Forest')

model_accuracy.append(accuracy_score(y_test, y_pred_rf))
classifier_dt = DecisionTreeClassifier(criterion = 'entropy', random_state=0)

classifier_dt.fit(X_train, y_train)

y_pred_dt = classifier_dt.predict(X_test)
sns.heatmap(confusion_matrix(y_test, y_pred_dt), annot=True)

plt.show()
accuracy_score(y_test, y_pred_dt)
model.append('Decision Tree')

model_accuracy.append(accuracy_score(y_test, y_pred_dt))
# Printing the models alongside their accuracies

print(np.concatenate((np.array(model).reshape(len(model),1), np.array(model_accuracy).reshape(len(model),1)), axis=1))
plt.figure(figsize=(9,6))

sns.barplot(model, model_accuracy)

plt.show()
# Calculating K Fold Cross Validation scores for the models

accuracies_lr = cross_val_score(estimator = classifier_lr, X = X_train, y = y_train, cv = 10)

accuracies_nb = cross_val_score(estimator = classifier_nb, X = X_train, y = y_train, cv = 10)

accuracies_svm_l = cross_val_score(estimator = classifier_svm_l, X = X_train, y = y_train, cv = 10)

accuracies_svm_rbf = cross_val_score(estimator = classifier_svm_rbf, X = X_train, y = y_train, cv = 10)

accuracies_rf = cross_val_score(estimator = classifier_rf, X = X_train, y = y_train, cv = 10)

accuracies_dt = cross_val_score(estimator = classifier_dt, X = X_train, y = y_train, cv = 10)
kfold_acc_mean = [np.mean(accuracies_lr), np.mean(accuracies_nb), np.mean(accuracies_svm_l), np.mean(accuracies_svm_rbf), np.mean(accuracies_rf), np.mean(accuracies_dt)]
kfold_acc_std = [np.std(accuracies_lr), np.std(accuracies_nb), np.std(accuracies_svm_l), np.std(accuracies_svm_rbf), np.std(accuracies_rf), np.std(accuracies_dt)]
KFold = pd.DataFrame({'Model': model, 'KFold accuracies mean': kfold_acc_mean, 'KFold accuracies std': kfold_acc_std})

print(KFold)
plt.figure(figsize=(9,6))

sns.barplot(model, kfold_acc_mean)

plt.show()