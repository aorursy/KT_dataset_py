# Importing required libraries to get started

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style("whitegrid")



%matplotlib inline
data = pd.read_csv('../input/diabetes.csv')

data.head()
data.describe()
sns.set(style='ticks')

plt.figure(figsize=(20,20))

sns.pairplot(data, hue='Outcome')
print(data[data.BloodPressure == 0].shape[0])

print(data[data.BloodPressure == 0].index.tolist())

print(data[data.BloodPressure == 0].groupby('Outcome')['Age'].count())
print(data[data.Glucose == 0].shape[0])

print(data[data.Glucose == 0].index.tolist())

print(data[data.Glucose == 0].groupby('Outcome')['Age'].count())
print(data[data.SkinThickness == 0].shape[0])

print(data[data.SkinThickness == 0].index.tolist())

print(data[data.SkinThickness == 0].groupby('Outcome')['Age'].count())
print(data[data.BMI == 0].shape[0])

print(data[data.BMI == 0].index.tolist())

print(data[data.BMI == 0].groupby('Outcome')['Age'].count())
print(data[data.Insulin == 0].shape[0])

print(data[data.Insulin == 0].index.tolist())

print(data[data.Insulin == 0].groupby('Outcome')['Age'].count())
plt.figure(figsize=(14,3))

bp_pivot = data.groupby('BloodPressure').Outcome.mean().reset_index()

sns.barplot(bp_pivot.BloodPressure, bp_pivot.Outcome)

plt.title('% chance of being diagnosed with diabetes by blood pressure reading')

plt.show()



plt.figure(figsize=(14,3))

bp_pivot = data.groupby('BloodPressure').Outcome.count().reset_index()

sns.distplot(data[data.Outcome == 0]['BloodPressure'], color='turquoise', kde=False, label='0 Class')

sns.distplot(data[data.Outcome == 1]['BloodPressure'], color='coral', kde=False, label='1 Class')

plt.legend()

plt.title('count # of people with blood pressure values')

plt.show()
plt.figure(figsize=(20,5))

glucose_pivot = data.groupby('Glucose').Outcome.mean().reset_index()

sns.barplot(glucose_pivot.Glucose, glucose_pivot.Outcome)

plt.title('% chance of being diagnosed with diabetes by Glucose reading')

plt.show()



plt.figure(figsize=(14,3))

glucose_pivot = data.groupby('Glucose').Outcome.count().reset_index()

sns.distplot(data[data.Outcome == 0]['Glucose'], color='turquoise', kde=False, label='0 Class')

sns.distplot(data[data.Outcome == 1]['Glucose'], color='coral', kde=False, label='1 class')

plt.legend()

plt.title('count # of people with Glucose values')

plt.show()
plt.figure(figsize=(20,5))

BMI_pivot = data.groupby('BMI').Outcome.mean().reset_index()

sns.barplot(BMI_pivot.BMI, BMI_pivot.Outcome)

plt.title('% chance of being diagnosed with diabetes by BMI reading')

plt.show()



plt.figure(figsize=(14,3))

BMI_pivot = data.groupby('BMI').Outcome.count().reset_index()

sns.distplot(data[data.Outcome == 0]['BMI'], color='turquoise', kde=False, label='Class 0')

sns.distplot(data[data.Outcome == 1]['BMI'], color='coral', kde=False, label='Class 1')

plt.legend()

plt.title('count # of people with BMI values')

plt.show()
plt.figure(figsize=(14,3))

Insulin_pivot = data.groupby('Insulin').Outcome.mean().reset_index()

sns.barplot(Insulin_pivot.Insulin, Insulin_pivot.Outcome)

plt.title('% chance of being diagnosed with diabetes by Insulin reading')

plt.show()



plt.figure(figsize=(14,3))

Insulin_pivot = data.groupby('Insulin').Outcome.count().reset_index()

sns.distplot(data[data.Outcome == 0]['Insulin'], color='turquoise', kde=False, label='Class 0')

sns.distplot(data[data.Outcome == 1]['Insulin'], color='coral', kde=False, label='Class 1')

plt.legend()

plt.title('count # of people with Insulin values')

plt.show()
plt.figure(figsize=(14,3))

SkinThickness_pivot = data.groupby('SkinThickness').Outcome.mean().reset_index()

sns.barplot(SkinThickness_pivot.SkinThickness, SkinThickness_pivot.Outcome)

plt.title('% chance of being diagnosed with diabetes by skin thickness reading')

plt.show()



plt.figure(figsize=(14,3))

SkinThickness_pivot = data.groupby('SkinThickness').Outcome.count().reset_index()

sns.distplot(data[data.Outcome == 0]['SkinThickness'], color='turquoise', kde=False, label='Class 0')

sns.distplot(data[data.Outcome == 1]['SkinThickness'], color='coral', kde=False, label='Class 1')

plt.legend()

plt.title('count # of people with Skin thickness values')

plt.show()
from sklearn import linear_model as lm

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier as dtc

from sklearn.neighbors import KNeighborsClassifier as knnc

from sklearn.naive_bayes import GaussianNB as gnb

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split 

from sklearn.metrics import accuracy_score, confusion_matrix
data_mod = data[(data.BloodPressure != 0) & (data.BMI != 0) & (data.Glucose != 0)]

train, test = train_test_split(data_mod, test_size=0.2)

print(data_mod.shape)

print(train.shape)

print(test.shape)
features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness'\

            , 'BMI', 'Age', 'Insulin', 'DiabetesPedigreeFunction']

target = 'Outcome'

classifiers = [

    knnc(),

    dtc(),

    SVC(),

    SVC(kernel='linear'),

    gnb()

]

classifier_names = [

    'K nearest neighbors',

    'Decision Tree Classifier',

    'SVM classifier with RBF kernel',

    'SVM classifier with linear kernel',

    'Gaussian Naive Bayes'

]
for clf, clf_name in zip(classifiers, classifier_names):

    cv_scores = cross_val_score(clf, train[features], train[target], cv=5)

    

    print(clf_name, ' mean accuracy: ', round(cv_scores.mean()*100, 3), '% std: ', round(cv_scores.var()*100, 3),'%')
final_model_smv_lin = SVC(kernel='linear').fit(train[features], train[target])

final_model_gnb = gnb().fit(train[features], train[target])
y_hat_svm = final_model_smv_lin.predict(test[features])

y_hat_gnb = final_model_gnb.predict(test[features])



print('test accuracy for SVM classifier with a linear kernel:'\

      , round(accuracy_score(test[target], y_hat_svm)*100, 2), '%')

plt.title('Confusion matrix for SVM classifier with a linear kernel')

sns.heatmap(confusion_matrix(test[target], y_hat_svm), annot=True, cmap="YlGn")

plt.xlabel('Predicted classes')

plt.ylabel('True Classes')

plt.show()



print('test accuracy for Gaussian naive bayes classifier:', \

      round(accuracy_score(test[target], y_hat_gnb)*100, 2),'%')

plt.title('confusion matrix for Gaussian naive bayes classifier')

sns.heatmap(confusion_matrix(test[target], y_hat_gnb), annot=True, cmap="YlGn")

plt.show()