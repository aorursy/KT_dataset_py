# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df_ = pd.read_csv('/kaggle/input/sections.csv')

df_.head()
df_ = pd.read_csv('/kaggle/input/teachings.csv')

df_.head()
df_grades = pd.read_csv('/kaggle/input/grade_distributions.csv')

df_courses = pd.read_csv('/kaggle/input/course_offerings.csv')
df_grades = df_grades[df_grades.columns[:9]]

# df_grades.drop(['section_number'], axis=1, inplace=True)

df_grades.head()
df_courses = df_courses[['uuid', 'name']]

df_courses.head()
df_merge = df_grades.merge(df_courses, left_on='course_offering_uuid', right_on='uuid', how='inner')

df_merge.drop(['uuid', 'course_offering_uuid'], axis=1, inplace=True)

df_merge.head()
df_merge = df_merge.groupby('name').sum()
df_merge['a_count'].sort_values(ascending=False)[:20].plot(kind='bar', title='Top 20 A Programs')
df_merge['a_count'].value_counts()[:100].plot(kind='bar', title='A Counts', figsize=(20,5))
corr = df_merge.drop(['section_number'], axis=1).corr()

corr.style.background_gradient(cmap='coolwarm')
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.model_selection import GridSearchCV



np.random.seed(0)
df_ = df_merge.drop(['section_number'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(df_[df_.columns[1:]], df_[df_.columns[0]], test_size=0.20, random_state=42)
model = RandomForestClassifier()

grid = GridSearchCV(estimator=model, param_grid=dict(n_estimators=[25], 

                                                     n_jobs=[2], 

                                                     random_state=[0], 

                                                     max_depth=list(range(1,6)), 

                                                     max_features=list(range(1, 6))))

grid.fit(X_train, y_train)



y_predict = grid.predict(X_test)

score = accuracy_score(y_test.values, y_predict)



print('Best features:\n- max depth of tree: {}\n- number of features: {}\n'.format(grid.best_estimator_.max_depth, grid.best_estimator_.max_features))



print('Accuracy score: {:0.2f}%'.format(score * 100))