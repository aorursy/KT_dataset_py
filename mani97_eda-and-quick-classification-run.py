# Basic 

import numpy as np

import pandas as pd



# Plotting

import matplotlib.pyplot as plt

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



from sklearn.ensemble import VotingClassifier



# Evaluation metrics

from sklearn.metrics import jaccard_score, f1_score, log_loss, accuracy_score, confusion_matrix, classification_report, roc_auc_score



# Cross validation

from sklearn.model_selection import cross_val_score
path = '/kaggle/input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv'



records = pd.read_csv(path, index_col=False)

records.head(3)
records.describe()
records.info()
records.isna().sum()
records_corr = records.corr()



plt.figure(figsize=(12,12))

sns.heatmap(records_corr, annot=True, cmap="YlGnBu")

plt.title('Correlation')

plt.show()
# Age and death_event



ax = sns.violinplot(x='DEATH_EVENT', y='age', data = records)





medians = records.groupby(['DEATH_EVENT'])['age'].median().values

nobs = records['DEATH_EVENT'].value_counts().values

nobs = [str(x) for x in nobs.tolist()]

nobs = ['n: ' + i for i in nobs]



pos = range(len(nobs))

for tick, label in zip(pos, ax.get_xticklabels()):

    ax.text(pos[tick], medians[tick]+0.10, nobs[tick], horizontalalignment='center', size='x-small', color='w', weight='semibold')



plt.title('1. Age vs Death Event')

plt.show()
# Age and serum creatinine



plt.figure(figsize=(8,6))

sns.regplot(x='age', y='serum_creatinine', data = records)

plt.minorticks_on()

plt.grid(b=True, which='both', axis='both', alpha=0.1)

plt.title('2. Age and Serum_creatinine')

plt.show()
# Age and diabetes



ax = sns.violinplot(x='diabetes', y='age', data = records)





medians = records.groupby(['diabetes'])['age'].median().values

nobs = records['diabetes'].value_counts().values

nobs = [str(x) for x in nobs.tolist()]

nobs = ['n: ' + i for i in nobs]



pos = range(len(nobs))

for tick, label in zip(pos, ax.get_xticklabels()):

    ax.text(pos[tick], medians[tick]+0.10, nobs[tick], horizontalalignment='center', size='x-small', color='w', weight='semibold')



plt.title('3. Age and Diabetes')

plt.show()
# Age, diabetes and death_event



ax = sns.violinplot(x='diabetes', y='age', data = records, hue = 'DEATH_EVENT')

plt.title('4. Age, Diabetes vs Death Event')

plt.show()
# anaemia and death-event



sns.countplot(x = 'DEATH_EVENT', data = records, hue = 'anaemia')

plt.title('1. Anaemia and death_event')

plt.xlabel('Death-event')

plt.ylabel('Number of people')

plt.show()

# anaemia and creatinine phosphokinase



ax = sns.violinplot(x='anaemia', y='creatinine_phosphokinase', data = records)





medians = records.groupby(['anaemia'])['creatinine_phosphokinase'].median().values

nobs = records['anaemia'].value_counts().values

nobs = [str(x) for x in nobs.tolist()]

nobs = ['n: ' + i for i in nobs]



pos = range(len(nobs))

for tick, label in zip(pos, ax.get_xticklabels()):

    ax.text(pos[tick], medians[tick]+0.10, nobs[tick], horizontalalignment='center', size='x-small', color='w', weight='semibold')



plt.title('2. Anaemia and Creatinine phosphokinase')

plt.show()
# Anaemia and smoking

sns.countplot(x = 'smoking', data = records, hue = 'anaemia')

plt.title('3. Anaemia and smoking')

plt.xlabel('Smoking')

plt.ylabel('Number of people')

plt.show()

# creatinine phosphokinaseand death_event



ax = sns.violinplot(x='DEATH_EVENT', y='creatinine_phosphokinase', data = records)





medians = records.groupby(['DEATH_EVENT'])['creatinine_phosphokinase'].median().values

nobs = records['DEATH_EVENT'].value_counts().values

nobs = [str(x) for x in nobs.tolist()]

nobs = ['n: ' + i for i in nobs]



pos = range(len(nobs))

for tick, label in zip(pos, ax.get_xticklabels()):

    ax.text(pos[tick], medians[tick]+0.10, nobs[tick], horizontalalignment='center', size='x-small', color='w', weight='semibold')



plt.title('Creatinine phosphokinase and death event')

plt.show()
# 'diabetes' and 'DEATH_EVENT'

sns.countplot(x = 'diabetes', data = records, hue = 'DEATH_EVENT')

plt.title('1. Diabetes and death event')

plt.xlabel('diabetes')

plt.ylabel('Number of people')

plt.show()

# diabetes and sex

sns.countplot(x = 'sex', data = records, hue = 'diabetes')

plt.title('2. diabetes and sex')

plt.xlabel('sex')

plt.ylabel('Number of people')

plt.show()
# diabetes and smoking

sns.countplot(x = 'diabetes', data = records, hue = 'smoking')

plt.title('3. diabetes and smoking')

plt.xlabel('diabetes')

plt.ylabel('Number of people')

plt.show()
print(records['ejection_fraction'].median())
# ejection fraction and death event



ax = sns.violinplot(x='DEATH_EVENT', y='ejection_fraction', data = records)





medians = records.groupby(['DEATH_EVENT'])['ejection_fraction'].median().values

nobs = records['DEATH_EVENT'].value_counts().values

nobs = [str(x) for x in nobs.tolist()]

nobs = ['n: ' + i for i in nobs]



pos = range(len(nobs))

for tick, label in zip(pos, ax.get_xticklabels()):

    ax.text(pos[tick], medians[tick]+0.10, nobs[tick], horizontalalignment='center', size='x-small', color='w', weight='semibold')



plt.title('1. Ejection fraction vs Death Event')

plt.show()
# ejection fraction and serum sodium



plt.figure(figsize=(8,6))

sns.regplot(x='serum_sodium', y='ejection_fraction', data = records)

plt.minorticks_on()

plt.grid(b=True, which='both', axis='both', alpha=0.1)

plt.title('2. Ejection fraction and Serum sodium')

plt.show()
# ejection fraction and sex



# Age and death_event



ax = sns.violinplot(x='sex', y='ejection_fraction', data = records)





medians = records.groupby(['sex'])['ejection_fraction'].median().values

nobs = records['sex'].value_counts().values

nobs = [str(x) for x in nobs.tolist()]

nobs = ['n: ' + i for i in nobs]



pos = range(len(nobs))

for tick, label in zip(pos, ax.get_xticklabels()):

    ax.text(pos[tick], medians[tick]+0.10, nobs[tick], horizontalalignment='center', size='x-small', color='w', weight='semibold')



plt.title('3. Ejection fraction vs Sex')

plt.show()
# High BP and death-event

sns.countplot(x='DEATH_EVENT', data = records, hue='high_blood_pressure')

plt.title('1. High BP and death-event')

plt.xlabel('Death Event')

plt.ylabel('Number of people')

plt.show()
# high BP and sex

sns.countplot(x='sex', data = records, hue='high_blood_pressure')

plt.title('2. High BP and sex')

plt.xlabel('sex')

plt.ylabel('Number of people')

plt.show()
# high BP and time



ax = sns.violinplot(x='high_blood_pressure', y='time', data = records)





medians = records.groupby(['high_blood_pressure'])['time'].median().values

nobs = records['high_blood_pressure'].value_counts().values

nobs = [str(x) for x in nobs.tolist()]

nobs = ['n: ' + i for i in nobs]



pos = range(len(nobs))

for tick, label in zip(pos, ax.get_xticklabels()):

    ax.text(pos[tick], medians[tick]+0.10, nobs[tick], horizontalalignment='center', size='x-small', color='w', weight='semibold')



plt.title('3. High BP vs time')

plt.show()
# platelets and death event



ax = sns.violinplot(x='DEATH_EVENT', y='platelets', data = records)





medians = records.groupby(['DEATH_EVENT'])['platelets'].median().values

nobs = records['DEATH_EVENT'].value_counts().values

nobs = [str(x) for x in nobs.tolist()]

nobs = ['n: ' + i for i in nobs]



pos = range(len(nobs))

for tick, label in zip(pos, ax.get_xticklabels()):

    ax.text(pos[tick], medians[tick]+0.10, nobs[tick], horizontalalignment='center', size='x-small', color='w', weight='semibold')



plt.title('1. Platelet count vs death event')

plt.show()
# platelets and sex



ax = sns.violinplot(x='sex', y='platelets', data = records)





medians = records.groupby(['sex'])['platelets'].median().values

nobs = records['sex'].value_counts().values

nobs = [str(x) for x in nobs.tolist()]

nobs = ['n: ' + i for i in nobs]



pos = range(len(nobs))

for tick, label in zip(pos, ax.get_xticklabels()):

    ax.text(pos[tick], medians[tick]+0.10, nobs[tick], horizontalalignment='center', size='x-small', color='w', weight='semibold')



plt.title('1. Platelet count vs sex')

plt.show()
# serum creatinine and death event



ax = sns.violinplot(x='DEATH_EVENT', y='serum_creatinine', data = records)





medians = records.groupby(['DEATH_EVENT'])['serum_creatinine'].median().values

nobs = records['DEATH_EVENT'].value_counts().values

nobs = [str(x) for x in nobs.tolist()]

nobs = ['n: ' + i for i in nobs]



pos = range(len(nobs))

for tick, label in zip(pos, ax.get_xticklabels()):

    ax.text(pos[tick], medians[tick]+0.10, nobs[tick], horizontalalignment='center', size='x-small', color='w', weight='semibold')



plt.title('1. Serum creatinine vs death event')

plt.show()
# serum creatinine and age

sns.regplot(x = 'age', y = 'serum_creatinine', data = records)

plt.title('2. Serum creatinine and age')

plt.xlabel('age')

plt.ylabel('Serum creatinine')

plt.show()
# serum creatinine and serum sodium 

sns.regplot(x = 'serum_sodium', y = 'serum_creatinine', data = records)

plt.title('3. Serum creatinine and Serum sodium')

plt.xlabel('Serum sodium')

plt.ylabel('Serum creatinine')

plt.show()
# serum creatinine and time 

sns.regplot(x = 'time', y = 'serum_creatinine', data = records)

plt.title('4. Serum creatinine and Follow up time')

plt.xlabel('time')

plt.ylabel('Serum creatinine')

plt.show()
# serum sodium and death event



ax = sns.violinplot(x='DEATH_EVENT', y='serum_sodium', data = records)





medians = records.groupby(['DEATH_EVENT'])['serum_sodium'].median().values

nobs = records['DEATH_EVENT'].value_counts().values

nobs = [str(x) for x in nobs.tolist()]

nobs = ['n: ' + i for i in nobs]



pos = range(len(nobs))

for tick, label in zip(pos, ax.get_xticklabels()):

    ax.text(pos[tick], medians[tick]+0.10, nobs[tick], horizontalalignment='center', size='x-small', color='w', weight='semibold')



plt.title('1. Serum sodium vs death event')

plt.show()
# sex and death event

sns.countplot(x='sex', data = records, hue = 'DEATH_EVENT')

plt.title('1. Sex and death-event')

plt.xlabel('Sex')

plt.ylabel('Number of people')

plt.show()
# sex and smoking

sns.countplot(x = 'sex', data = records, hue = 'smoking')

plt.title('2. Sex and smoking')

plt.xlabel('Sex')

plt.ylabel('Number of people')

plt.show()
# smoking and death event

sns.countplot(x = 'DEATH_EVENT', data = records, hue = 'smoking')

plt.title('1. Smoking and death event')

plt.xlabel('Death event')

plt.ylabel('Number of people')

plt.show()
ls = records.groupby(['DEATH_EVENT', 'smoking']).count().values[:, 0]

print('Percentge smoking but no heart failure: ', (ls[1]/(ls[0]+ls[1]))*100)

print('Percentge smoking and heart failure: ', (ls[3]/(ls[2]+ls[3]))*100)
# time and death_event



ax = sns.violinplot(x='DEATH_EVENT', y='time', data = records)





medians = records.groupby(['DEATH_EVENT'])['time'].median().values

nobs = records['DEATH_EVENT'].value_counts().values

nobs = [str(x) for x in nobs.tolist()]

nobs = ['n: ' + i for i in nobs]



pos = range(len(nobs))

for tick, label in zip(pos, ax.get_xticklabels()):

    ax.text(pos[tick], medians[tick]+0.10, nobs[tick], horizontalalignment='center', size='x-small', color='w', weight='semibold')



plt.title('1. Follow up period vs death event')

plt.show()
records_corr['DEATH_EVENT'].sort_values(ascending=False)
X = records[['serum_creatinine', 'age', 'time', 'ejection_fraction', 'serum_sodium']].values

y = records.iloc[:, -1].values



print('Shape of X ', X.shape)

print('Shape of y ', y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)



print('Shape of training set ', X_train.shape)

print('Shape of test set ', X_test.shape)
classifier = KNeighborsClassifier(n_neighbors = 6, metric='minkowski', p=2)
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

print('\nMean Accuracy of the model: ', accuracy.mean()*100)

    

# confusion matrix

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, lw = 2, cbar=False)

plt.xlabel('Predicted')

plt.ylabel('True')

plt.title('Confusion Matrix: {}'.format(str(type(classifier)).split('.')[-1][:-2]))

plt.show()