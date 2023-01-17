import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import LabelEncoder
train = pd.read_csv('/kaggle/input/av-healthcare-analytics-ii/healthcare/train_data.csv')
test = pd.read_csv('/kaggle/input/av-healthcare-analytics-ii/healthcare/test_data.csv')
train.head()
train.drop('case_id', axis=1, inplace=True)
test.drop('case_id', axis=1, inplace=True)
print(train.info())
print(test.info())
fig = go.Figure() 
fig.add_trace(go.Box(x=train['Admission_Deposit'],
                     marker_color="blue",
                     name="Train"))
fig.add_trace(go.Box(x=test['Admission_Deposit'],
                     marker_color="red",
                     name="Test"))
fig.update_layout(title="Distributions of Admission Deposit")
fig.show()

fig = go.Figure() 
fig.add_trace(go.Box(x=train['Age'],
                     marker_color="blue",
                     name="Train"))
fig.add_trace(go.Box(x=test['Age'],
                     marker_color="red",
                     name="Test"))
fig.update_layout(title="Distributions of Age")
fig.show()

train.corr()['Stay']
fig = px.scatter(train, x=train['Visitors with Patient'], y=train['Stay'])
fig.update_layout(title='Number of Visitors vs. Duration of Stay',xaxis_title="Visitors",yaxis_title="Duration")
fig.show()
le = LabelEncoder()
for column in train.columns:
    if train[column].dtype == 'object': 
        train[column] = le.fit_transform(train[column])
for column in test.columns:
    if test[column].dtype == 'object': 
        test[column] = le.fit_transform(test[column])

sns.heatmap(train.corr())
