# Basic

import numpy as np 

import pandas as pd



# Plotting

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



# Splitting

from sklearn.model_selection import train_test_split



# Import models

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier



# Evaluation metrics

from sklearn.metrics import jaccard_score, f1_score, log_loss, accuracy_score, confusion_matrix, classification_report, roc_auc_score



# Cross validation

from sklearn.model_selection import cross_val_score
records = pd.read_csv('/kaggle/input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv', index_col=False)

print(records.shape)

records.head(3)
records.info()
records.describe()
records.isna().sum()
records_corr = records.corr()

plt.figure(figsize=(14,12))

sns.heatmap(records_corr, annot=True)

plt.title('Correlation between features')

plt.show()
records_corr['DEATH_EVENT'].sort_values(ascending = False)
X = records[['serum_creatinine', 'age', 'time', 'ejection_fraction', 'serum_sodium']].values

y = records.iloc[:, -1].values



print('Shape of X ', X.shape)

print('Shape of y ', y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)



print('Shape of training set ', X_train.shape)

print('Shape of test set ', X_test.shape)
classifiers = [XGBClassifier(), LogisticRegression(max_iter=1000), KNeighborsClassifier(n_neighbors = 10, metric='minkowski', p=2), SVC(kernel = 'linear'), SVC(kernel = 'rbf'), DecisionTreeClassifier(criterion='entropy'), RandomForestClassifier(n_estimators = 10, criterion = 'entropy')]



for classifier in classifiers:

    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    

    # print classifier name

    print(str(type(classifier)).split('.')[-1][:-2])

    

    # Accuracy Score

    print('Accuracy Score: {}'.format(accuracy_score(y_test, y_pred)))

    

    # jaccard Score

    print('\nJaccard Score: {}'.format(jaccard_score(y_test, y_pred)))

    

    # F1 score

    print('\nF1 Score: {}'.format(f1_score(y_test, y_pred)))

    

    # Log Loss

    print('\nLog Loss: {}'.format(log_loss(y_test, y_pred)))

    

    print('CROSS VALIDATION')

    accuracy = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv=10)

    print('Accuracies after CV: ', accuracy)

    print('Mean Accuracy of the model: ', accuracy.mean()*100)

    

    # confusion matrix

    cm = confusion_matrix(y_test, y_pred)

    sns.heatmap(cm, annot=True, lw = 2, cbar=False)

    plt.xlabel('Predicted')

    plt.ylabel('True')

    plt.title('Confusion Matrix: {}'.format(str(type(classifier)).split('.')[-1][:-2]))

    plt.show()
classifier = KNeighborsClassifier(n_neighbors = 10, metric='minkowski', p=2)

classifier.fit(X_train, y_train)



y_pred = classifier.predict(X_test)



print(classification_report(y_test, y_pred))



print('ROC-AUC Score: ',roc_auc_score(y_test, y_pred))
# confusion matrix

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, lw = 2, cbar=False)

plt.xlabel('Predicted')

plt.ylabel('True')

plt.title('Confusion Matrix: {}'.format(str(type(classifier)).split('.')[-1][:-2]))

plt.show()
acc=[]

for i in range(1, 20):

    y_p = KNeighborsClassifier(n_neighbors=i, metric='minkowski', p=2).fit(X_train, y_train).predict(X_test)

    

    acc.append(accuracy_score(y_test, y_p))
plt.figure(figsize=(10,8))

plt.scatter(np.arange(1,20, step=1), acc)

plt.xticks(np.arange(1,20, step=1))

plt.grid(b=True, which='major', axis='both', color='#999999', linestyle='-', alpha=0.1)
classifier = KNeighborsClassifier(n_neighbors = 6, metric='minkowski', p=2)

classifier.fit(X_train, y_train)



y_pred = classifier.predict(X_test)



print(classification_report(y_test, y_pred))



print('ROC-AUC Score: ',roc_auc_score(y_test, y_pred))

print('Accuracy: ', accuracy_score(y_test, y_pred)*100)
# confusion matrix

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, lw = 2, cbar=False)

plt.xlabel('Predicted')

plt.ylabel('True')

plt.title('Confusion Matrix: {}'.format(str(type(classifier)).split('.')[-1][:-2]))

plt.show()