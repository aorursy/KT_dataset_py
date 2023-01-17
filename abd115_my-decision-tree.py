import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

df = pd.read_csv('/kaggle/input/simple-decision-tree/salaries.csv')

df.head()
input = df.drop('salary_more_then_100k', axis = 'columns')

input.head()
target = df['salary_more_then_100k']

target.head()
from sklearn.preprocessing import LabelEncoder

le_company = LabelEncoder()

le_job = LabelEncoder()

le_degree = LabelEncoder()
input['company_n'] = le_company.fit_transform(input['company'])

input['job_n'] = le_job.fit_transform(input['job'])

input['degree_n'] = le_degree.fit_transform(input['degree'])



input
input.drop(['company','job','degree'], axis = 'columns', inplace = True)

input
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()
model.fit(input, target)
model.score(input,target)
model.predict([[2,1,0]])
model.predict([[2,1,1]])