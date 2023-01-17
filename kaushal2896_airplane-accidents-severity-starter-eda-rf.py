# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('/kaggle/input/airplane-accidents-severity-dataset/train.csv')

test_df = pd.read_csv('/kaggle/input/airplane-accidents-severity-dataset/test.csv')

sample_sub_df = pd.read_csv('/kaggle/input/airplane-accidents-severity-dataset/sample_submission.csv')
train_df.head()
test_df.head()
sample_sub_df.head()
print(f'Shape of training data: {train_df.shape}')

print(f'Shape of testing data: {test_df.shape}')
train_df.isna().sum()
test_df.isna().sum()
X_train = train_df.drop(['Severity', 'Accident_ID'], axis=1)

Y_train = train_df['Severity']
Y_train.unique()
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
Y_train = Y_train.map(class_map).astype(np.uint8)
plt.figure(figsize=(13,8))

ax = sns.barplot(np.vectorize(inverse_class_map.get)(pd.unique(Y_train)), Y_train.value_counts().sort_index())

ax.set(xlabel='Accident Severity', ylabel='# of records', title='Meter type vs. # of records')

ax.set_xticklabels(ax.get_xticklabels(), rotation=50, ha="right")

plt.show()
plt.figure(figsize=(13,8))

sns.distplot(X_train['Safety_Score'], kde=False)

plt.show()
plt.figure(figsize=(13,8))

sns.distplot(X_train['Days_Since_Inspection'], kde=False)

plt.show()
plt.figure(figsize=(13,8))

sns.distplot(X_train['Total_Safety_Complaints'], kde=False)

plt.show()
plt.figure(figsize=(13,8))

sns.distplot(X_train['Control_Metric'], kde=False)

plt.show()
plt.figure(figsize=(13,8))

sns.distplot(X_train['Turbulence_In_gforces'], kde=False)

plt.show()
plt.figure(figsize=(13,8))

sns.distplot(X_train['Cabin_Temperature'], kde=False)

plt.show()
plt.figure(figsize=(13,8))

sns.distplot(X_train['Max_Elevation'], kde=False)

plt.show()
plt.figure(figsize=(13,8))

sns.distplot(X_train['Violations'], kde=False)

plt.show()
plt.figure(figsize=(13,8))

sns.distplot(X_train['Adverse_Weather_Metric'], kde=False)

plt.show()
X_train['Total_Safety_Complaints'] = np.power(2, X_train['Total_Safety_Complaints'])

X_train['Days_Since_Inspection'] = np.power(2, X_train['Days_Since_Inspection'])

X_train['Safety_Score'] = np.power(2, X_train['Safety_Score'])
rf = RandomForestClassifier(n_estimators=1250, random_state=666, oob_score=True)



# 0.8589427

param_grid = { 

    'n_estimators': [1000],

    'max_features': [None],

    'min_samples_split': [3],

    'max_depth': [50]

    

}



CV_rf = GridSearchCV(estimator=rf, param_grid=param_grid, cv=6, verbose=100, n_jobs=-1)

CV_rf.fit(X_train, Y_train)

print (f'Best Parameters: {CV_rf.best_params_}')
test_df['Total_Safety_Complaints'] = np.power(2, test_df['Total_Safety_Complaints'])

test_df['Days_Since_Inspection'] = np.power(2, test_df['Days_Since_Inspection'])

test_df['Safety_Score'] = np.power(2, test_df['Safety_Score'])
preds = CV_rf.predict(test_df.drop(['Accident_ID'], axis=1))
submission = pd.DataFrame([test_df['Accident_ID'], np.vectorize(inverse_class_map.get)(preds)], index=['Accident_ID', 'Severity']).T

submission.to_csv('submission.csv', index=False)

submission.head()
from IPython.display import FileLink, FileLinks



FileLink('submission.csv')