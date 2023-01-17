import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import imblearn



%matplotlib inline

pd.options.display.max_columns = 150



from sklearn import metrics

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
df = pd.read_csv('./data/xAPI-Edu-Data.csv')
df.head()
df['Class'].value_counts()
df.head()
grade_map = {'L': 0, 'M': 1, 'H': 2}

df = df.replace({'Class': grade_map})

df.head()
df.columns
# One-hot encode string columns

columns_to_one_hot_encode = ['gender', 'NationalITy', 'PlaceofBirth', 'StageID', 'GradeID', 'SectionID', 'Topic', 'Semester', 'Relation', 'ParentAnsweringSurvey', 'ParentschoolSatisfaction',

'StudentAbsenceDays']



df_one_hot_encoded = pd.get_dummies(df, columns = columns_to_one_hot_encode)

df_one_hot_encoded.head()
df_one_hot_encoded.describe()
df_one_hot_encoded.columns
X = df_one_hot_encoded.ix[:, df_one_hot_encoded.columns != 'Class']

y = df_one_hot_encoded.ix[:, df_one_hot_encoded.columns == 'Class'].values.ravel()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
from sklearn.ensemble import RandomForestClassifier
model_1 = RandomForestClassifier()

model_1.fit(X_train, y_train)
expected = y

predicted = model_1.predict(X)



print(metrics.confusion_matrix(expected, predicted))

print(metrics.classification_report(expected, predicted))
df_X_vars_only = df_one_hot_encoded

del df_X_vars_only['Class']
plt.figure(figsize=(16,8))

plt.plot(model_1.feature_importances_, 'o')

plt.xticks(range(len(df_X_vars_only.columns)), df_X_vars_only.columns.values, rotation=90);
for index, f in enumerate(model_1.feature_importances_):

    if f > 0.05:

        print(df_one_hot_encoded.columns[index], f)
from sklearn.model_selection import GridSearchCV
random_forest_classifier_model = RandomForestClassifier(random_state=0)



param_grid = {'max_features': [None, 'auto', 'sqrt', 'log2'],

              'n_estimators': [1, 2, 4, 8, 10, 20, 30, 50],

              'min_samples_leaf': [1,5,10,50]}



model_2 = GridSearchCV(estimator=random_forest_classifier_model, 

                       param_grid=param_grid, cv=5)

model_2.fit(X_train, y_train)
expected_2 = y

predicted_2 = model_2.predict(X)



print(metrics.confusion_matrix(expected_2, predicted_2))

print(metrics.classification_report(expected_2, predicted_2))
plt.figure(figsize=(16,8))

plt.plot(model_2.best_estimator_.feature_importances_, 'o')

plt.xticks(range(len(df_X_vars_only.columns)), df_X_vars_only.columns.values, rotation=90);
for index, f in enumerate(model_2.best_estimator_.feature_importances_):

    if f > 0.05:

        print(df_one_hot_encoded.columns[index], f)