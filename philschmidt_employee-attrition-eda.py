import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# Import statements required for Plotly 

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls



from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, log_loss
df = pd.read_csv('../input/WA_Fn-UseC_-HR-Employee-Attrition.csv')

df.head()
df.info()
df.groupby('Attrition')['Attrition'].agg('count').plot.bar()
# convert to numeric for mean

df.Attrition_numeric = df.Attrition

df.loc[df.Attrition == 'Yes','Attrition_numeric'] = 1

df.loc[df.Attrition == 'No','Attrition_numeric'] = 0



plt.figure(figsize=(12,8))

sns.barplot(x = 'Gender', y = 'Attrition_numeric', data=df)
plt.figure(figsize=(12,8))

sns.barplot(x = 'OverTime', y = 'Attrition_numeric', data=df)
plt.figure(figsize=(12,8))

sns.barplot(y = 'JobRole', x = 'Attrition_numeric', data=df)
plt.figure(figsize=(12,8))

sns.barplot(x = 'MaritalStatus', y = 'Attrition_numeric', data=df)
plt.figure(figsize=(12,8))

sns.barplot(x = 'TotalWorkingYears', y = 'Attrition_numeric', data=df)
plt.figure(figsize=(12,8))

sns.barplot(x = 'PerformanceRating', y = 'Attrition_numeric', data=df)
plt.figure(figsize=(12,8))

sns.barplot(x = 'Department', y = 'Attrition_numeric', data=df)
plt.figure(figsize=(12,8))

sns.barplot(x = 'Education', y = 'Attrition_numeric', data=df)
plt.figure(figsize=(12,8))

sns.barplot(x = 'BusinessTravel', y = 'Attrition_numeric', data=df)
plt.figure(figsize=(12,8))

sns.barplot(x = 'EducationField', y = 'Attrition_numeric', data=df)
plt.figure(figsize=(12,8))

sns.barplot(x = 'YearsAtCompany', y = 'Attrition_numeric', data=df)
plt.figure(figsize=(12,8))

sns.barplot(x = 'WorkLifeBalance', y = 'Attrition_numeric', data=df)
plt.figure(figsize=(12,8))

sns.barplot(x = 'RelationshipSatisfaction', y = 'Attrition_numeric', data=df)
plt.figure(figsize=(12,8))

sns.barplot(x = 'NumCompaniesWorked', y = 'Attrition_numeric', data=df)
plt.figure(figsize=(12,8))

sns.barplot(x = 'Age', y = 'Attrition_numeric', data=df)
from sklearn.preprocessing import LabelBinarizer



LabelBinarizer().fit_transform(df['Department'])