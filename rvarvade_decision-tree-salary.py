import pandas as pd

df = pd.read_csv("../input/salaries.csv")

df.head()
inputs = df.drop('salary_more_then_100k',axis = 'columns')

target = df['salary_more_then_100k']
from sklearn.preprocessing import LabelEncoder



le_company = LabelEncoder()

le_job = LabelEncoder()

le_degree = LabelEncoder()

inputs['company_n'] = le_company.fit_transform(inputs['company'])

inputs['job_n'] = le_company.fit_transform(inputs['job'])

inputs['degree_n'] = le_company.fit_transform(inputs['degree'])

inputs.head()

inputs_n = inputs.drop(['company','job','degree'],axis='columns')

inputs_n
from sklearn import tree



model = tree.DecisionTreeClassifier()

model.fit(inputs_n,target)
model.score(inputs_n,target)
model.predict([[2,1,0]])
model.predict([[2,1,1]])