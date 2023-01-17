import numpy as np

import pandas as pd



import os



import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.ticker as ticker



import timeit



%matplotlib inline
data = pd.read_csv(r"../input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv")

data.head()
data.describe(include='all')
data.info()
#checking for any missing values in the dataset

data.isnull().sum().sort_values(ascending=False)
count_churn = data['Churn'].value_counts()
import plotly_express as px

c_churn = pd.DataFrame(data['Churn'].value_counts().reset_index().values,

                        columns=['Churn', 'count_churn'])



c_churn = c_churn.sort_values('count_churn', ascending=False)

group_by = c_churn.groupby('Churn')['count_churn'].sum().reset_index()

fig = px.bar(group_by.sort_values('Churn', ascending = False)[:20][::-1], x = 'Churn', y = 'count_churn',

            title = 'Total value counts for people that left and people that did not leave', text = 'count_churn', height = 500, orientation = 'v' )

fig.show()
from plotly.offline import init_notebook_mode,iplot

import plotly.graph_objects as go

from plotly.subplots import make_subplots



dataset = data.copy()
yes_churn = dataset[dataset.Churn=='Yes']

no_churn = dataset[dataset.Churn=='No']
trace1 = go.Histogram(

    x=yes_churn.gender  ,

    opacity=0.75,

    name='churned')



trace2 = go.Histogram(

    x=no_churn.gender  ,

    opacity=0.75,

    name='did not churn')



data = [trace1, trace2]

layout = go.Layout(barmode='stack',

                   title='Churn according to Gender',

                   xaxis=dict(title='Gender'),

                   yaxis=dict( title='Count'),

                   paper_bgcolor='beige',

                   plot_bgcolor='beige'

)

fig = go.Figure(data=data, layout=layout)

iplot(fig)
trace1 = go.Histogram(

    x=yes_churn.SeniorCitizen  ,

    opacity=0.75,

    name='churned')



trace2 = go.Histogram(

    x=no_churn.SeniorCitizen  ,

    opacity=0.75,

    name='did not churn')



data = [trace1, trace2]

layout = go.Layout(barmode='stack',

                   title='Churn according to senior citizenship',

                   xaxis=dict(title='Senior Citizen'),

                   yaxis=dict( title='Count'),

                   paper_bgcolor='beige',

                   plot_bgcolor='beige'

)

fig = go.Figure(data=data, layout=layout)

iplot(fig)
trace1 = go.Histogram(

    x=yes_churn.tenure,

    opacity=0.75,

    name='churned')



trace2 = go.Histogram(

    x=no_churn.tenure,

    opacity=0.75,

    name='did not churn')



data = [trace1, trace2]

layout = go.Layout(barmode='stack',

                   title='Churn according to tenure',

                   xaxis=dict(title='tenure'),

                   yaxis=dict( title='Count'),

                   paper_bgcolor='beige',

                   plot_bgcolor='beige'

)

fig = go.Figure(data=data, layout=layout)

iplot(fig)
trace1 = go.Histogram(

    x=yes_churn.Partner,

    opacity=0.75,

    name='churned')



trace2 = go.Histogram(

    x=no_churn.Partner,

    opacity=0.75,

    name='did not churn')



data = [trace1, trace2]

layout = go.Layout(barmode='stack',

                   title='Churn according to partner',

                   xaxis=dict(title='Partner'),

                   yaxis=dict( title='Count'),

                   paper_bgcolor='beige',

                   plot_bgcolor='beige'

)

fig = go.Figure(data=data, layout=layout)

iplot(fig)
trace1 = go.Histogram(

    x=yes_churn.MonthlyCharges,

    opacity=0.75,

    name='churned')



trace2 = go.Histogram(

    x=no_churn.MonthlyCharges, 

    opacity=0.75,

    name='did not churn')



data = [trace1, trace2]

layout = go.Layout(barmode='stack',

                   title='Churn according to Monthly Charges',

                   xaxis=dict(title='Monthly Charges'),

                   yaxis=dict( title='Count'),

                   paper_bgcolor='beige',

                   plot_bgcolor='beige'

)

fig = go.Figure(data=data, layout=layout)

iplot(fig)
val_monthcharges = dataset['MonthlyCharges'].values
sns.distplot(val_monthcharges, color='g')
dataset.hist(figsize=(12,4))

plt.show()
correlation_matrix = dataset.corr()

fig = plt.figure(figsize=(12,10))

sns.heatmap(correlation_matrix, vmax=0.8, square=True)

plt.show()
y = dataset['Churn']

X = dataset.drop(['Churn'], axis = 1)
X = pd.get_dummies(X, drop_first=True)
y = y.map({'Yes':1, 'No':0})
from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
from sklearn.model_selection import cross_val_score
classifier = LogisticRegression()

classifier.fit(X_train, y_train)
training_score = cross_val_score(classifier, X_train, y_train, cv=5)
training_score
# Use GridSearchCV to find the best parameters.

from sklearn.model_selection import GridSearchCV





# Logistic Regression 

log_reg_params = {"penalty": ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}







grid_log_reg = GridSearchCV(LogisticRegression(), log_reg_params)

grid_log_reg.fit(X_train, y_train)

# We automatically get the logistic regression with the best parameters.

log_reg = grid_log_reg.best_estimator_
log_reg
log_reg_score = cross_val_score(log_reg, X_train, y_train, cv=5)
log_reg_score
log_reg_score.mean()
y_pred = log_reg.predict(X_test)
y_pred
y_test = y_test.reset_index()
y_test = y_test.drop(['index'], axis=1)
y_test.head(5)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_pred, y_test)
cm
from sklearn.metrics import classification_report
print('Logistic Regression:')

print(classification_report(y_test, y_pred))
#If you like this kernel please kindly UPVOTE. THANKS 