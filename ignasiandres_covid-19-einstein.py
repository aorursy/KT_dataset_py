# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        df = pd.read_excel(os.path.join(dirname, filename), encoding = "utf-8")

# Any results you write to the current directory are saved as output.
# THis is another approach for the 2 tasks based on the previous work on my colleagues as published in:
# https://www.kaggle.com/aiopstivit/covid-19
# First: load and configure plots
# Matplotlib
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [20, 10]
# General shape of the dataset: 111 columns
df.shape
# Look first columns
df.head(100)
# Too much NaN values. Need to clean.
# Encode categories for the SARS-Cov-2 exam result
df_test = df['SARS-Cov-2 exam result'].astype("category").cat.codes
df_test.head()
df_test.value_counts()
# Plot SARS-Cov-2 exam result
df_test.hist()
# Plot Patient addmited to regular ward, semi-intensive unit or intensive care unit (1=yes, 0=no) by result on the exam
df_result = df[['Patient addmited to regular ward (1=yes, 0=no)', 'Patient addmited to semi-intensive unit (1=yes, 0=no)', 'Patient addmited to intensive care unit (1=yes, 0=no)']].groupby(df['SARS-Cov-2 exam result'].astype("category").cat.codes)
num_groups = df_result.ngroups
# df[['Patient addmited to regular ward (1=yes, 0=no)', 'Patient addmited to semi-intensive unit (1=yes, 0=no)', 'Patient addmited to intensive care unit (1=yes, 0=no)']].hist(by=df['SARS-Cov-2 exam result'].astype("category").cat.codes)
groups = ['Teste negativo SARS-Cov-2', 'Teste positivo SARS-Cov-2']
labels= ["Regular ward","Semi-intensive unit", "Intensive care unit"]
axes = df_result.plot(kind='hist', stacked=True, alpha=0.5)
for i, (groupname, group) in enumerate(df_result):
    axes[i].set_title(groups[groupname])
plt.show()
# It can be seen that even when the results of the test are positive, the most part of patients are not admited
# Show empty rows per columns
df.isna().sum().plot.bar()
# Ordering the above bar plot
df_nan_values = df.isna().sum().values
df_nan_dict = dict(zip(df.columns, df_nan_values))
order = sorted(df_nan_dict.keys(), key=df_nan_dict.get)
df[order].isna().sum().plot.bar()
# Lots of columns have a lot of NaN values so we have to clean records with low information
df_percent = (1 - (df.isna().sum().values / df.shape[0]))*100
df_percent_dict = dict(zip(df.columns, df_percent))
order = sorted(df_percent_dict.keys(), key=df_percent_dict.get, reverse=True)
# We keep columns with at least 1% of values
order_filtered = [x for x in order if df_percent_dict[x] > 1]
df_final = df[order_filtered]
df_final.shape
# And we drop rows with NaN
# We count the number of NaN per column, and drop those that have more than 90 fields
empty_records = df_final.isna().sum(axis=1) > 80
drop_index = empty_records.index[empty_records.values]
df_final = df_final.drop(drop_index, axis=0)
df_final.shape
df_final.head()
# We encode the string columns:
for col in df_final.columns:
    df_final[col] = df_final[col].astype("category").cat.codes
    print(df_final[col].value_counts())

df_final.head()
# We obtain the correlation matrix:
corr_matrix = df_final.corr()
# We can represent relations better with a heatmap
import seaborn as sns; sns.set()
sns.heatmap(corr_matrix, linewidths=.1, cmap="YlGnBu")
# correlation = df_final.corr()
# correlation.style.background_gradient(cmap='coolwarm').set_precision(2)
import numpy as np

indices = np.where(corr_matrix > 0.5)
indices = [(corr_matrix.index[x], corr_matrix.columns[y]) for x, y in zip(*indices)
                                        if x != y and x < y]
len(indices)

# For task 1: prediction of Covid positive
# Grid search: iterate different configurations and algorithms
# First separate the variable we want to predict: SARS-Cov-2 Exam result
y = df_final['SARS-Cov-2 exam result']
X = df_final.drop(['SARS-Cov-2 exam result'], axis = 1)

# Import sklearn libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
# We want now to create the different train and test groups
# X_train, X_test, y_train, y_test = train_test_split(X, y)
X_train = X
X_test = X
y_train = y
y_test = y
X_train, X_test, y_train, y_test = train_test_split(X, y)
# We should scale the data for some algorithms
feature_scaler = StandardScaler()
X_train = feature_scaler.fit_transform(X_train)
X_test = feature_scaler.transform(X_test)
# Start with RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=300, random_state=0)
# Cross validation with 5 folds
all_accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=5)
print(all_accuracies)
grid_param = {
    'n_estimators': [10, 30, 40, 50, 100, 300, 500, 800, 1000],
    'criterion': ['gini', 'entropy'],
    'bootstrap': [True, False]
}
gd_sr = GridSearchCV(estimator=classifier,
                     param_grid=grid_param,
                     scoring='accuracy',
                     cv=5,
                     n_jobs=-1)
gd_sr.fit(X_train, y_train)

gd_sr.fit(X_test, y_test)
results = gd_sr.cv_results_