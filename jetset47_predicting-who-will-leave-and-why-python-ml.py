import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
hr_data = pd.read_csv('../input/HR_comma_sep.csv')

hr_data.head()
hr_data.describe()
hr_data.info()
print('Departments: ', ', '.join(hr_data['sales'].unique()))

print('Salary levels: ', ', '.join(hr_data['salary'].unique()))
hr_data.rename(columns={'sales':'department'}, inplace=True)

hr_data_new = pd.get_dummies(hr_data, ['department', 'salary'] ,drop_first = True)
hr_data_new.head()
# Correlation matrix

sns.heatmap(hr_data.corr(), annot=True)
hr_data_new.columns
dept_table = pd.crosstab(hr_data['department'], hr_data['left'])

dept_table.index.names = ['Department']

dept_table
dept_table_percentages = dept_table.apply(lambda row: (row/row.sum())*100, axis = 1)

dept_table_percentages
sns.countplot(x='department', hue='left', data=hr_data)
sns.boxplot(x='department', y='satisfaction_level', data=hr_data)
sns.countplot(x='salary', hue='left', data=hr_data)
sns.boxplot(x='salary', y='satisfaction_level', data=hr_data)
sns.factorplot(x='number_project', y='last_evaluation', hue='department', data=hr_data)
sns.boxplot(x='number_project', y='satisfaction_level', data=hr_data_new)
timeplot = sns.factorplot(x='time_spend_company', hue='left', y='department', row='salary', data=hr_data, aspect=2)
accidentplot = plt.figure(figsize=(10,6))

accidentplotax = accidentplot.add_axes([0,0,1,1])

accidentplotax = sns.violinplot(x='department', y='average_montly_hours', hue='Work_accident', split=True, data = hr_data, jitter = 0.47)
satisaccident = plt.figure(figsize=(10,6))

satisaccidentax = satisaccident.add_axes([0,0,1,1])

satisaccidentax = sns.violinplot(x='left', hue='Work_accident', y='satisfaction_level', split=True, data=hr_data)
# We now use model_selection instead of cross_validation

from sklearn.model_selection import train_test_split



X = hr_data_new.drop('left', axis=1)

y = hr_data_new['left']



X_train, X_test, y_train, y_test, = train_test_split(X, y, test_size = 0.3, random_state = 47)
from sklearn.tree import DecisionTreeClassifier



dt = DecisionTreeClassifier()

dt.fit(X_train, y_train)
from sklearn.model_selection import cross_val_score



# Score first on our training data

print('Score: ', dt.score(X_train, y_train))

print('Cross validation score, 10-fold cv: \n', cross_val_score(dt, X_train, y_train, cv=10))

print('Mean cross validation score: ', cross_val_score(dt,X_train,y_train,cv=10).mean())
predictions = dt.predict(X_test)



print('Score: ', dt.score(X_test, y_test))

print('Cross validation score, 10-fold cv: \n', cross_val_score(dt, X, y, cv=10))

print('Mean cross validation score: ', cross_val_score(dt,X,y,cv=10).mean())
from sklearn.metrics import confusion_matrix, classification_report



print('Confusion matrix: \n', confusion_matrix(y_test, predictions), '\n')

print('Classification report: \n', classification_report(y_test, predictions))
from sklearn.metrics import roc_curve, roc_auc_score

probabilities = dt.predict_proba(X_test)

fpr, tpr, thresholds = roc_curve(y_test, probabilities[:,1])



rates = pd.DataFrame({'False Positive Rate': fpr, 'True Positive Rate': tpr})



roc = plt.figure(figsize = (10,6))

rocax = roc.add_axes([0,0,1,1])

rocax.plot(fpr, tpr, color='g', label='Decision Tree')

rocax.plot([0,1],[0,1], color='gray', ls='--', label='Baseline (Random Guessing)')

rocax.set_xlabel('False Positive Rate')

rocax.set_ylabel('True Positive Rate')

rocax.set_title('ROC Curve')

rocax.legend()



print('Area Under the Curve:', roc_auc_score(y_test, probabilities[:,1]))
importances = dt.feature_importances_

print("Feature importances: \n")

for f in range(len(X.columns)):

    print('•', X.columns[f], ":", importances[f])
featureswithimportances = list(zip(X.columns, importances))

featureswithimportances.sort(key = lambda f: f[1], reverse=True)



print('Ordered feature importances: \n', '(From most important to least important)\n')



for f in range(len(featureswithimportances)):

    print(f+1,". ", featureswithimportances[f][0], ": ", featureswithimportances[f][1])
sorted_features, sorted_importances = zip(*featureswithimportances)

plt.figure(figsize=(12,6))

sns.barplot(sorted_features, sorted_importances)

plt.title('Feature Importances (Gini Importance)')

plt.ylabel('Decrease in Node Impurity')

plt.xlabel('Feature')

plt.xticks(rotation=90);