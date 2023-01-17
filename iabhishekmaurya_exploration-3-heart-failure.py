# Importing necessary libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()



from imblearn.over_sampling import RandomOverSampler

from collections import Counter



from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import StratifiedKFold



from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score, classification_report, accuracy_score



from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.svm import SVC



from xgboost import XGBClassifier
# Reading the dataset

heart_data = pd.read_csv('../input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')

heart_data.head()
print("Shape of data: ",heart_data.shape)
heart_data.info()
heart_data['time'].describe()
heart_data['DEATH_EVENT'].value_counts()
heart_data.columns
var = 'age'

fig, ax = plt.subplots()

fig.set_size_inches(20, 8)

plt.xticks(rotation=90);

sns.countplot(x = var,palette="ch:.4", data = heart_data)

ax.set_xlabel('AGE', fontsize=15)

ax.set_ylabel('Count', fontsize=15)

ax.set_title('AGE Count Distribution', fontsize=15)

sns.despine()
var = 'DEATH_EVENT'

fig, ax = plt.subplots()

fig.set_size_inches(11.7, 8.27)

sns.swarmplot(x = var, y ='age', data = heart_data)
heart_data['ejection_fraction'].describe()
var = 'age'

data = pd.concat([heart_data['ejection_fraction'], heart_data[var]], axis=1)

f, ax = plt.subplots(figsize=(20, 10))

fig = sns.boxplot(x=var, y="ejection_fraction", data=data)

fig.axis(ymin=0, ymax=90);

plt.xticks(rotation=90);
var = 'anaemia'

fig, ax = plt.subplots()

fig.set_size_inches(5,5)

sns.countplot(x = var, data = heart_data)

ax.set_xlabel('anaemia', fontsize=15)

ax.set_ylabel('Count', fontsize=15)

ax.set_title('anaemia Count Distribution', fontsize=15)

ax.tick_params(labelsize=15)

sns.despine()
fig, ax = plt.subplots()

fig.set_size_inches(5,5)

sns.countplot(x='DEATH_EVENT',hue='diabetes', data=heart_data)

ax.set_xlabel('DEATH_EVENT', fontsize=15)

ax.set_ylabel('Count', fontsize=15)

ax.set_title('DEATH_EVENT count basis of diabetes', fontsize=10)

ax.tick_params(labelsize=15)

plt.show() 
fig, ax = plt.subplots()

fig.set_size_inches(20, 8)

sns.countplot(x='ejection_fraction',hue='diabetes', data=heart_data)

ax.set_xlabel('ejection_fraction', fontsize=15)

ax.set_ylabel('Count', fontsize=15)

ax.set_title('ejection_fraction count basis of diabetes', fontsize=10)

ax.tick_params(labelsize=15)

plt.show() 
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows = 2, ncols = 2, figsize = (10,10))



sns.countplot(x='DEATH_EVENT',hue='diabetes', data=heart_data, ax = ax1)

ax1.set_xlabel('DEATH_EVENT')

ax1.set_ylabel('Count')

ax1.set_title('DEATH_EVENT count basis of diabetes', fontsize=10)

ax1.tick_params(labelsize=15)



sns.countplot(x='high_blood_pressure',hue='diabetes', data=heart_data, ax = ax2)

ax2.set_xlabel('high_blood_pressure')

ax2.set_ylabel('Count')

ax2.set_title('DEATH_EVENT count basis of diabetes', fontsize=10)

ax2.tick_params(labelsize=15)



sns.countplot(x='sex',hue='diabetes', data=heart_data, ax = ax3)

ax3.set_xlabel('sex')

ax3.set_ylabel('Count')

ax3.set_title('sex count basis of diabetes', fontsize=10)

ax3.tick_params(labelsize=15)



sns.countplot(x='smoking',hue='diabetes', data=heart_data, ax = ax4)

ax4.set_xlabel('smoking')

ax4.set_ylabel('Count')

ax4.set_title('smoking count basis of diabetes', fontsize=10)

ax4.tick_params(labelsize=15)



plt.subplots_adjust(wspace=0.5)

plt.tight_layout() 
heart_data['platelets'].describe()
f, ax = plt.subplots(figsize=(15,8))

sns.distplot(heart_data['platelets'])

plt.xlim([0,900000])
plt.figure(figsize=(18,18))

sns.heatmap(heart_data.corr(),annot=True,cmap='RdYlGn')



plt.show()
X = heart_data.drop(['DEATH_EVENT'], axis = 1)

y = heart_data['DEATH_EVENT']
X_train,X_test,y_train,y_test = train_test_split(X , y, test_size=0.2, random_state=25)
os=RandomOverSampler(1)

X_train_ns,y_train_ns=os.fit_sample(X_train,y_train)

print("The number of classes before fit {}".format(Counter(y_train)))

print("The number of classes after fit {}".format(Counter(y_train_ns)))
models = []

models.append(('LR', LogisticRegression(solver='liblinear')))

models.append(('LDA', LinearDiscriminantAnalysis()))

models.append(('KNN', KNeighborsClassifier()))

models.append(('DTC', DecisionTreeClassifier()))

models.append(('SVM', SVC(gamma='auto')))

models.append(('RFC', RandomForestClassifier()))

models.append(('ABC', AdaBoostClassifier()))

models.append(('XGB', XGBClassifier()))

# evaluate each model in turn

results = []

names = []

for name, model in models:

    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)

    cv_results = cross_val_score(model, X_train_ns, y_train_ns, cv=kfold, scoring='accuracy')

    results.append(cv_results)

    names.append(name)

    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))