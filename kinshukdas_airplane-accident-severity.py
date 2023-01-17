import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline
train_df = pd.read_csv('/kaggle/input/airplane-accidents-severity-dataset/train.csv')

test_df = pd.read_csv('/kaggle/input/airplane-accidents-severity-dataset/test.csv')
train_df.head()
train_df.describe().T
train_df.isnull().sum()
train_df.columns
target = train_df[['Severity']]

features = train_df.drop(['Severity', 'Accident_ID'], axis=1)
features.head()
target.head()
train_df['Severity'].value_counts()
def find_distribution(col):

    plt.figure(figsize=(13,8))

    sns.distplot(features[col], kde=False)

    plt.show()

    

def create_boxplot(col):

    plt.figure(figsize=(13,8))

    sns.set(style="whitegrid")

    sns.boxplot(x=features[col])

    plt.show()
features.columns
for cols in features.columns:

    find_distribution(cols)

    create_boxplot(cols)
plt.figure(figsize=(13,8))

ax = sns.barplot(x=target['Severity'].unique(),y=target['Severity'].value_counts(), data=target)

ax.set(xlabel='Accident Severity', ylabel='# of records', title='Meter type vs. # of records')

ax.set_xticklabels(ax.get_xticklabels(), rotation=50, ha="right")

plt.show()
class_map = {

    'Minor_Damage_And_Injuries': 0,

    'Significant_Damage_And_Fatalities': 1,

    'Significant_Damage_And_Serious_Injuries': 2,

    'Highly_Fatal_And_Damaging': 3

}

inverse_class_map = {

    0: 'Minor_Damage_And_Injuries',

    1: 'Significant_Damage_And_Fatalities',

    2: 'Significant_Damage_And_Serious_Injuries',

    3: 'Highly_Fatal_And_Damaging'

}
target['Severity'] = target['Severity'].map(class_map).astype(np.uint8)
target.head()
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.multiclass import OneVsOneClassifier

from sklearn import metrics
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
X_train.shape
y_train.shape
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
X_train[:5]
X_test[:5]
y_train[:5]
y_train.shape
clf = OneVsOneClassifier(LogisticRegression())

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(metrics.f1_score(y_test, y_pred, average='weighted'))
clf.get_params()
from sklearn.model_selection import GridSearchCV



param_grid = {

    'estimator' : [LogisticRegression()],

     'estimator__penalty' : ['l1', 'l2'],

    'estimator__C' : np.logspace(-4, 4, 20),

    'estimator__solver' : ['liblinear']

}



gridsearchCV = GridSearchCV(estimator=clf, param_grid=param_grid, scoring='f1_weighted', cv=10)

gridsearchCV.fit(X_train, y_train)

gridsearchCV.best_score_

gridsearchCV.best_estimator_

gridsearchCV.best_params_
clf = OneVsOneClassifier(LogisticRegression(C=4.281332398719396, class_weight=None, dual=False,

                    fit_intercept=True, intercept_scaling=1, l1_ratio=None,

                    max_iter=100, multi_class='warn', n_jobs=None, penalty='l2',

                    random_state=None, solver='liblinear', tol=0.0001, verbose=0,

                    warm_start=False))



clf.fit(X_train, y_train)



y_pred = clf.predict(X_test)

print(metrics.f1_score(y_test, y_pred, average='weighted'))
clf_rf = OneVsOneClassifier(RandomForestClassifier(n_estimators=100))

clf_rf.fit(X_train, y_train)



y_pred = clf_rf.predict(X_test)

print(metrics.f1_score(y_test, y_pred, average='weighted'))
clf_rf.estimator.fit(X_train, y_train)

clf_rf.estimator.feature_importances_
test_df.head()
test = test_df.drop('Accident_ID', axis=1)

test = scaler.transform(test)
preds = clf_rf.predict(test)
submission = pd.DataFrame([test_df['Accident_ID'],preds], index=['Accident_ID', 'Severity']).T
submission['Severity'] = submission['Severity'].map(inverse_class_map)

submission.head()
from IPython.display import FileLink

submission.to_csv('submission.csv', index=False)

FileLink('submission.csv')