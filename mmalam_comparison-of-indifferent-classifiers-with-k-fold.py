import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

from sklearn.model_selection import cross_val_score

from sklearn.metrics import confusion_matrix, accuracy_score
df = pd.read_csv('../input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')

df = df.drop('anaemia', 1).drop('creatinine_phosphokinase', 1).drop('diabetes', 1).drop('high_blood_pressure', 1).drop('platelets', 1).drop('serum_sodium', 1).drop('sex', 1).drop('smoking', 1)

X = df.iloc[:, :-2].values#excluding time

y = df.iloc[:, -1].values
df.head()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
result_comparison = pd.DataFrame(columns=['Model', 'Cross Validation Mean Accuracy', 'Cross Validation Standard Deviation', 'Test Data Accuracy', 'Test Data Precision', 'Test Data Recall', 'Test Data Specificity' ])
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state = 0)

classifier.fit(X_train, y_train)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)

print("Accuracy: {:.2f} %".format(accuracies.mean()*100))

print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))
y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

print(cm)

accuracy_score(y_test, y_pred)
precision = cm[0,0]/(cm[0,0]+cm[1,0]) * 100

print (precision)

recall = cm[0,0]/(cm[0,0]+cm[0,1]) * 100

print (recall)

specificity = cm[1,1]/(cm[1,1]+cm[1,0]) * 100

print (specificity)
result_comparison = result_comparison.append({'Model':'Logistic Regression', 'Cross Validation Mean Accuracy': accuracies.mean()*100, 'Cross Validation Standard Deviation': accuracies.std()*100, 'Test Data Accuracy': accuracy_score(y_test, y_pred)*100, 'Test Data Precision':precision, 'Test Data Recall':recall, 'Test Data Specificity':specificity}, ignore_index=True)
from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)

classifier.fit(X_train, y_train)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)

print("Accuracy: {:.2f} %".format(accuracies.mean()*100))

print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))
y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

print(cm)

accuracy_score(y_test, y_pred)*100
precision = cm[0,0]/(cm[0,0]+cm[1,0]) * 100

print (precision)

recall = cm[0,0]/(cm[0,0]+cm[0,1]) * 100

print (recall)

specificity = cm[1,1]/(cm[1,1]+cm[1,0]) * 100

print (specificity)
result_comparison = result_comparison.append({'Model':'Decision Tree', 'Cross Validation Mean Accuracy': accuracies.mean()*100, 'Cross Validation Standard Deviation': accuracies.std()*100, 'Test Data Accuracy': accuracy_score(y_test, y_pred)*100, 'Test Data Precision':precision, 'Test Data Recall':recall, 'Test Data Specificity':specificity}, ignore_index=True)
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)

classifier.fit(X_train, y_train)



accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)

print("Accuracy: {:.2f} %".format(accuracies.mean()*100))

print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))



y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix")

print(cm)

print ("Accuracy on Test Set: {:.2f} %".format(accuracy_score(y_test, y_pred)*100))



precision = cm[0,0]/(cm[0,0]+cm[1,0]) * 100

print (precision)

recall = cm[0,0]/(cm[0,0]+cm[0,1]) * 100

print (recall)

specificity = cm[1,1]/(cm[1,1]+cm[1,0]) * 100

print (specificity)

result_comparison = result_comparison.append({'Model':'Random Forest', 'Cross Validation Mean Accuracy': accuracies.mean()*100, 'Cross Validation Standard Deviation': accuracies.std()*100, 'Test Data Accuracy': accuracy_score(y_test, y_pred)*100, 'Test Data Precision':precision, 'Test Data Recall':recall, 'Test Data Specificity':specificity}, ignore_index=True)
from sklearn.svm import SVC

classifier = SVC(kernel = 'linear')

classifier.fit(X_train, y_train)



accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)

print("Accuracy: {:.2f} %".format(accuracies.mean()*100))

print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))



y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix")

print(cm)

print ("Accuracy on Test Set: {:.2f} %".format(accuracy_score(y_test, y_pred)*100))



precision = cm[0,0]/(cm[0,0]+cm[1,0]) * 100

print (precision)

recall = cm[0,0]/(cm[0,0]+cm[0,1]) * 100

print (recall)

specificity = cm[1,1]/(cm[1,1]+cm[1,0]) * 100

print (specificity)
result_comparison = result_comparison.append({'Model':'SVM', 'Cross Validation Mean Accuracy': accuracies.mean()*100, 'Cross Validation Standard Deviation': accuracies.std()*100, 'Test Data Accuracy': accuracy_score(y_test, y_pred)*100, 'Test Data Precision':precision, 'Test Data Recall':recall, 'Test Data Specificity':specificity}, ignore_index=True)
classifier = SVC(kernel = 'rbf')

classifier.fit(X_train, y_train)



accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)

print("Accuracy: {:.2f} %".format(accuracies.mean()*100))

print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))



y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix")

print(cm)

print ("Accuracy on Test Set: {:.2f} %".format(accuracy_score(y_test, y_pred)*100))



precision = cm[0,0]/(cm[0,0]+cm[1,0]) * 100

print (precision)

recall = cm[0,0]/(cm[0,0]+cm[0,1]) * 100

print (recall)

specificity = cm[1,1]/(cm[1,1]+cm[1,0]) * 100

print (specificity)
result_comparison = result_comparison.append({'Model':'Kernel SVM', 'Cross Validation Mean Accuracy': accuracies.mean()*100, 'Cross Validation Standard Deviation': accuracies.std()*100, 'Test Data Accuracy': accuracy_score(y_test, y_pred)*100, 'Test Data Precision':precision, 'Test Data Recall':recall, 'Test Data Specificity':specificity}, ignore_index=True)
from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()

classifier.fit(X_train, y_train)



accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)

print("Accuracy: {:.2f} %".format(accuracies.mean()*100))

print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))



y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix")

print(cm)

print ("Accuracy on Test Set: {:.2f} %".format(accuracy_score(y_test, y_pred)*100))



precision = cm[0,0]/(cm[0,0]+cm[1,0]) * 100

print (precision)

recall = cm[0,0]/(cm[0,0]+cm[0,1]) * 100

print (recall)

specificity = cm[1,1]/(cm[1,1]+cm[1,0]) * 100

print (specificity)
result_comparison = result_comparison.append({'Model':'Naive Bayes', 'Cross Validation Mean Accuracy': accuracies.mean()*100, 'Cross Validation Standard Deviation': accuracies.std()*100, 'Test Data Accuracy': accuracy_score(y_test, y_pred)*100, 'Test Data Precision':precision, 'Test Data Recall':recall, 'Test Data Specificity':specificity}, ignore_index=True)
from xgboost import XGBClassifier

classifier = XGBClassifier()

classifier.fit(X_train, y_train)



accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)

print("Accuracy: {:.2f} %".format(accuracies.mean()*100))

print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))



y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix")

print(cm)

print ("Accuracy on Test Set: {:.2f} %".format(accuracy_score(y_test, y_pred)*100))



precision = cm[0,0]/(cm[0,0]+cm[1,0]) * 100

print (precision)

recall = cm[0,0]/(cm[0,0]+cm[0,1]) * 100

print (recall)

specificity = cm[1,1]/(cm[1,1]+cm[1,0]) * 100

print (specificity)
result_comparison = result_comparison.append({'Model':'XG Boost', 'Cross Validation Mean Accuracy': accuracies.mean()*100, 'Cross Validation Standard Deviation': accuracies.std()*100, 'Test Data Accuracy': accuracy_score(y_test, y_pred)*100, 'Test Data Precision':precision, 'Test Data Recall':recall, 'Test Data Specificity':specificity}, ignore_index=True)
# Params Copied from https://www.kaggle.com/para24/comparing-the-performance-of-12-classifiers



params = {'learning_rate': 0.014724527414939945,

          'num_boost_round': 3451,

          'gamma': 0.4074467665676125,

          'reg_lambda': 31.082862686792716,

          'reg_alpha': 0.008543705214252668,

          'max_depth': 7,

          'min_child_weight': 3.2435633342899867e-06,

          'subsample': 0.15432895096353877,

          'colsample_bytree': 0.7665394913603492}

classifier = XGBClassifier(**params,

                        random_state=0, n_jobs=-1)

classifier.fit(X_train, y_train)



accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)

print("Accuracy: {:.2f} %".format(accuracies.mean()*100))

print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))



y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix")

print(cm)

print ("Accuracy on Test Set: {:.2f} %".format(accuracy_score(y_test, y_pred)*100))



precision = cm[0,0]/(cm[0,0]+cm[1,0]) * 100

print (precision)

recall = cm[0,0]/(cm[0,0]+cm[0,1]) * 100

print (recall)

specificity = cm[1,1]/(cm[1,1]+cm[1,0]) * 100

print (specificity)
result_comparison = result_comparison.append({'Model':'XG Boost with Params', 'Cross Validation Mean Accuracy': accuracies.mean()*100, 'Cross Validation Standard Deviation': accuracies.std()*100, 'Test Data Accuracy': accuracy_score(y_test, y_pred)*100, 'Test Data Precision':precision, 'Test Data Recall':recall, 'Test Data Specificity':specificity}, ignore_index=True)
from catboost import CatBoostClassifier

classifier = CatBoostClassifier()



classifier.fit(X_train, y_train, 

                 eval_set=(X_train, y_train),

                 verbose=False)



accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)

print("Accuracy: {:.2f} %".format(accuracies.mean()*100))

print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))



y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix")

print(cm)

print ("Accuracy on Test Set: {:.2f} %".format(accuracy_score(y_test, y_pred)*100))



precision = cm[0,0]/(cm[0,0]+cm[1,0]) * 100

print (precision)

recall = cm[0,0]/(cm[0,0]+cm[0,1]) * 100

print (recall)

specificity = cm[1,1]/(cm[1,1]+cm[1,0]) * 100

print (specificity)
result_comparison = result_comparison.append({'Model':'CatBoost', 'Cross Validation Mean Accuracy': accuracies.mean()*100, 'Cross Validation Standard Deviation': accuracies.std()*100, 'Test Data Accuracy': accuracy_score(y_test, y_pred)*100, 'Test Data Precision':precision, 'Test Data Recall':recall, 'Test Data Specificity':specificity}, ignore_index=True)
from lightgbm import LGBMClassifier

classifier = LGBMClassifier(n_jobs=-1)

classifier.fit(X_train, y_train);



accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)

print("Accuracy: {:.2f} %".format(accuracies.mean()*100))

print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))



y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix")

print(cm)

print ("Accuracy on Test Set: {:.2f} %".format(accuracy_score(y_test, y_pred)*100))



precision = cm[0,0]/(cm[0,0]+cm[1,0]) * 100

print (precision)

recall = cm[0,0]/(cm[0,0]+cm[0,1]) * 100

print (recall)

specificity = cm[1,1]/(cm[1,1]+cm[1,0]) * 100

print (specificity)
result_comparison = result_comparison.append({'Model':'Light GBM', 'Cross Validation Mean Accuracy': accuracies.mean()*100, 'Cross Validation Standard Deviation': accuracies.std()*100, 'Test Data Accuracy': accuracy_score(y_test, y_pred)*100, 'Test Data Precision':precision, 'Test Data Recall':recall, 'Test Data Specificity':specificity}, ignore_index=True)
result_comparison