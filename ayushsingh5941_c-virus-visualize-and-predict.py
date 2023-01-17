import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import random

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.preprocessing import RobustScaler

import category_encoders as ce

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report, confusion_matrix

from sklearn.svm import SVC
%matplotlib inline
seed = 42

random.seed(seed)
import matplotlib.style as style

style.available
style.use('tableau-colorblind10')
data = pd.read_csv('../input/coronavirusdataset/patient.csv')
data.drop('id', axis=1, inplace=True)
data.head()
data.info()
data.describe()
# missing data

total = data.isnull().sum().sort_values(ascending=False)

percent = (data.isnull().sum() / data.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(20)
fig = plt.figure(figsize=(20, 5))

missing_plt = sns.barplot(x=missing_data.index, y=missing_data.Percent)

missing_plt.set_xticklabels(missing_plt.get_xticklabels(), rotation=45)

print()
print([pd.value_counts(data[cols]) for cols in data.columns])
print('No. of NA values in Sex:', data['sex'].isna().sum())
data_cleaned = data.dropna(axis=0, subset=['sex'])
data_cleaned['birth_year'].fillna(1973.0, inplace=True)
data_cleaned = data_cleaned.drop('region', axis=1)
data_cleaned['disease'].fillna(0.0, inplace=True)
data_cleaned = data_cleaned.drop('group', axis=1)
data_cleaned['infection_reason'].fillna('Unknown', inplace=True)
data_cleaned = data_cleaned.drop(['infected_by','infection_order', 'contact_number', 'released_date', 'deceased_date'], axis=1)
data_cleaned.head()
data_cleaned.info()
gender_count_plt = sns.countplot(data_cleaned['sex'], )

gender_count_plt.set_title('Gender Count')

plt.show()
gender_vs_state_plt = sns.countplot(x='sex', hue='state', data=data_cleaned)

gender_count_plt.set_title('Gender VS State')

plt.show()
data_cleaned['age'] = 2019 - data_cleaned['birth_year']

data_cleaned['age_bin'] = (data_cleaned['age'] // 10) * 10

data_cleaned = data_cleaned.drop('birth_year', axis=1)
sns.distplot(data_cleaned['age_bin'])
plt.figure(figsize=(25, 10))

age_sex_plt = sns.countplot(x='age', hue='sex', data=data_cleaned)

age_sex_plt.set_xticklabels(age_sex_plt.get_xticklabels(), rotation=-45)

age_sex_plt.legend(loc='upper right')
plt.figure(figsize=(25, 6))

age_sex_plt1 = sns.countplot(x='age_bin', hue='sex', data=data_cleaned)

age_sex_plt1.set_xticklabels(age_sex_plt1.get_xticklabels(), rotation=-45)

age_sex_plt1.legend(loc='upper right')

plt.figure(figsize=(25, 6))

age_state_plt = sns.countplot(x='age_bin', hue='state', data=data_cleaned)

age_state_plt.set_xticklabels(age_state_plt.get_xticklabels(), rotation=-45)

age_state_plt.legend(loc='upper right')

age_state_plt.set_title('Age VS State')
sns.countplot(data_cleaned['state'])
plt.figure(figsize=(20, 5))

infection_reason_plt = sns.countplot(data['infection_reason'])

infection_reason_plt.set_xticklabels(infection_reason_plt.get_xticklabels(), rotation=-45)

plt.show()
plt.figure(figsize=(20, 5))

infection_reason_plt1 = sns.countplot(data_cleaned['infection_reason'])

infection_reason_plt1.set_xticklabels(infection_reason_plt1.get_xticklabels(), rotation=-45)

plt.show()
plt.figure(figsize=(20, 5))

reason_age_plt = sns.countplot(x='age_bin', hue='infection_reason', data=data_cleaned)

reason_age_plt.set_xticklabels(reason_age_plt.get_xticklabels(), rotation=-45)

reason_age_plt.legend(loc='upper right')

print()
plt.figure(figsize=(20, 5))

reason_age_plt = sns.countplot(x=data_cleaned['age_bin'], hue=data['infection_reason'])

reason_age_plt.set_xticklabels(reason_age_plt.get_xticklabels(), rotation=-45)

reason_age_plt.legend(loc='upper right')

print()
plt.figure(figsize=(20, 5))

reason_age_plt = sns.countplot(x=data_cleaned['infection_reason'], hue=data_cleaned['state'])

reason_age_plt.set_xticklabels(reason_age_plt.get_xticklabels(), rotation=-45)

reason_age_plt.legend(loc='upper right')

print()
disease_vs_state_plt = sns.countplot(x='disease', hue='state', data=data_cleaned)
disease_vs_age_plt = sns.countplot(x='age_bin', hue='disease', data=data_cleaned)
# Removing redundant confirmed date and age columns

data_cleaned.drop(['confirmed_date', 'age'], axis=1, inplace=True)
data_cleaned.head()
target = data_cleaned['state']
print(target.head(),'\n',  target.shape)
features = data_cleaned.drop('state', axis=1)
print(features.head(), '\n',  features.shape)
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.20, stratify=target, random_state=42)
x_train.shape
x_test.shape
ohe = ce.OneHotEncoder()
sc = RobustScaler()
pipe = Pipeline(steps=[('ohe', ohe)])
x_train = pipe.fit_transform(x_train)
x_test = pipe.transform(x_test)
rf = LogisticRegression()
grid_param = {'C': [1.0, 1.2, 1.3],

             'fit_intercept': [True, False], 

             'solver': ['newton-cg', 'liblinear', 'lbfgs'],

             'tol': [1e-3, 1e-4],

             'max_iter': [500, 1000,  2000]}
grid = GridSearchCV(rf, grid_param, scoring='accuracy', n_jobs=-1, cv=5)
grid.fit(x_train, y_train)
grid.best_score_
grid.best_params_
model = grid.best_estimator_
y_pred = model.predict(x_test)
report = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True))
report
confusion_mat = confusion_matrix(y_test, y_pred)
confusion_mat
conf_mat = sns.heatmap(confusion_mat, square=True, vmax= 15, vmin=4, annot=True, cmap='YlGnBu')