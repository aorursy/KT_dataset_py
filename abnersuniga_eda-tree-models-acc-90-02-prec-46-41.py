import numpy as np

import pandas as pd

from sklearn import tree

from sklearn.metrics import accuracy_score

from sklearn.metrics import precision_score

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_validate

import seaborn as sns

import shap

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier

%matplotlib inline  
dataset = pd.read_excel('../input/covid19/dataset.xlsx')

print(f"Num rows: {len(dataset)}")

print(f"Num columns: {len(dataset.columns)}")
print(f"Num positive cases: {len(dataset[dataset['SARS-Cov-2 exam result'] == 'positive'])}")

print(f"Num negative cases: {len(dataset[dataset['SARS-Cov-2 exam result'] == 'negative'])}")
# Drop columns with all NaNs

dataset.dropna(axis=1, how='all', inplace=True)
# Predict confirmed COVID-19 cases

X = dataset.iloc[:,[1] + list(range(6,106))]

Y = dataset.iloc[:,2].apply(lambda result: 1 if result == 'positive' else 0)



# Transform object columns in categories

transform_columns = X.select_dtypes(['object']).columns

for col in transform_columns:

    X[col] = X[col].astype('category')

    X[col] = X[col].cat.codes

    

# Replacing NaNs with mean or 0.0

X = X.fillna(X.median())

# X = X.fillna(0.0)
# Model DecisionTreeClassifier

clf = tree.DecisionTreeClassifier()

clf_rf = RandomForestClassifier(random_state=42)
scores = cross_validate(clf, X.values, Y.values, cv=5, scoring=('accuracy','precision'))

print("Test accuracy: %.2f%%" % (scores['test_accuracy'].mean() * 100.0))

print("Test precision: %.2f%%" % (scores['test_precision'].mean() * 100.0))
scores = cross_validate(clf_rf, X.values, Y.values, cv=5, scoring=('accuracy','precision'))

print("Test accuracy: %.2f%%" % (scores['test_accuracy'].mean() * 100.0))

print("Test precision: %.2f%%" % (scores['test_precision'].mean() * 100.0))
# Feature importance

clf_rf.fit(X.values, Y.values)

feat_importances = pd.Series(clf_rf.feature_importances_, index=X.columns)

feat_importances.nlargest(20).plot(kind='barh', figsize=(6, 8), alpha=0.5)
TOP_N_FEATURES = 5

top_features = feat_importances.sort_values(ascending=False)[:TOP_N_FEATURES].index



sns.set(style="white", palette="muted", color_codes=True)

f, axes = plt.subplots(TOP_N_FEATURES, 2, figsize=(10,TOP_N_FEATURES*2))

sns.despine(left=True)



for i, feature in enumerate(top_features):

    sns.distplot(dataset[dataset['SARS-Cov-2 exam result'] == 'negative'][feature], kde=False, ax=axes[i, 0], color='r');

    axes[i][0].set_title(f'{feature} for negative exams')



    sns.distplot(dataset[dataset['SARS-Cov-2 exam result'] == 'positive'][feature], kde=False, ax=axes[i, 1]);

    axes[i][1].set_title(f'{feature} for positive exams')



plt.setp(axes, yticks=[])

plt.tight_layout()
# Same vision in the same plot

for i, feature in enumerate(top_features):

    plt.figure(figsize=(6,2))    

    plt.title(f'{feature} for SARS-Cov-2 exam result')

    plt.hist([

        dataset[dataset['SARS-Cov-2 exam result'] == 'negative'][feature], 

        dataset[dataset['SARS-Cov-2 exam result'] == 'positive'][feature]

    ], color=['r','b'], alpha=0.5)
