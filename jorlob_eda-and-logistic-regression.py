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



# Any results you write to the current directory are saved as output.
import seaborn as sns

import matplotlib.pyplot as plt

import math

%matplotlib inline

sns.set_context('notebook')

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
df = pd.read_csv('/kaggle/input/real-or-fake-fake-jobposting-prediction/fake_job_postings.csv')
df.head()
df.info()
df.describe()
plt.figure(figsize=(24, 6))

sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
df['employment_type'].value_counts()
sns.countplot(df['employment_type'], hue = df['fraudulent'])
df['salary_range'].head(15)
sns.countplot(df['has_company_logo'], hue = df['fraudulent'])
sns.countplot(df['telecommuting'], hue = df['fraudulent'])
type_job = pd.get_dummies(df['employment_type'], drop_first = True)
df['salary_min'] = df['salary_range'][df['salary_range'].notnull()].apply(lambda x :x.split('-')[0])

df['salary_max'] = df['salary_range'][df['salary_range'].notnull()].apply(lambda x :x.split('-')[-1])

df['salary_min'] = pd.to_numeric(df['salary_min'], errors='coerce').fillna("0")

df['salary_max'] = pd.to_numeric(df['salary_max'], errors='coerce').fillna("0")
df = pd.concat([df, type_job], axis = 1)
X= df[['telecommuting', 'has_company_logo', 'has_questions', 'Full-time', 'Other',

       'Part-time', 'Temporary', 'salary_min', 'salary_max']]

y = df['fraudulent']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state = 101)
logmodel = LogisticRegression()

logmodel.fit(X_train, y_train)
predictions = logmodel.predict(X_test)
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test,predictions))
print(accuracy_score(y_test,predictions))