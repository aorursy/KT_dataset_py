# Data Libraries

import pandas as pd

import numpy as np



# Visualization Libraries

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import plotly.graph_objects as go

import plotly.express as px

import plotly.io as pio

pio.templates.default = "plotly_dark"

sns.set(style="darkgrid")



# Ignore Warnings

import warnings

warnings.filterwarnings('ignore')
# Read in the data

f = pd.read_csv('../input/graduate-admissions/Admission_Predict_Ver1.1.csv')
# Basic info of Data set

f.info()
# List of Columns

f.columns

f.rename({'Chance of Admit ': 'Chance of Admit', 'LOR ':'LOR'}, axis=1, inplace=True)

f.columns
# Basic Statistics on numeric columns

f.describe()
# Read first 5 values using head()

f.head()
# Correlation Heatmap

f = f.drop('Serial No.',axis = 1)

sns.heatmap(f.corr())
# Finding which parameter has the strong correlation with 'Chance of Admit'

f.corr()['Chance of Admit'].sort_values()
# Distribution of CGPA

fig = go.Figure()



fig.add_trace(go.Histogram(x = f['CGPA'], marker_color='white'))



fig.update_layout(title='Distribution of CGPA',bargap=0.05,

                  xaxis = dict(title = 'CGPA'),

                  yaxis = dict(title = 'Number of Students')

                 )

fig.show()
# Distribution of GRE Scores

fig = go.Figure()



fig.add_trace(go.Histogram(x = f['GRE Score'], marker_color='white'))



fig.update_layout(title='Distribution of GRE Score',bargap=0.05,

                  xaxis = dict(title = 'GRE Score'),

                  yaxis = dict(title = 'Number of Students')

                 )

fig.show()
# Distribution of TOEFL Scores

fig = go.Figure()



fig.add_trace(go.Histogram(x = f['TOEFL Score'], marker_color='white'))



fig.update_layout(title='Distribution of TOEFL Score',bargap=0.05,

                  xaxis = dict(title = 'TOEFL Score'),

                  yaxis = dict(title = 'Number of Students')

                 )

fig.show()
# CGPA Analysis



bins = [6, 7.01, 8.01, 9.01]

names = ['6 to 7', '7.01 to 8', '8.01 to 9','9+']

d = dict(enumerate(names, 1))

f['CGPA_Category'] = np.vectorize(d.get)(np.digitize(f['CGPA'], bins))



fig = go.Figure()



fig.add_trace(go.Histogram(x = f['CGPA_Category'], marker_color='#5852a8'))



fig.update_layout(title='Number of Students Categorised by CGPA',

                  xaxis = dict(title = 'CGPA_Category'),

                  yaxis = dict(title = 'Number of Students'),

                  bargap=0.05

                 )

fig.show()
# Admit chances as per CGPA Category



fig = px.box(f,x = f['CGPA_Category'], y = f['Chance of Admit'],

             color= 'Research', title="Box plot of Chance of Admit for Each CGPA Category (Categorised by Research & No-Research Students)"

             )

fig.show()
# GRE Score Analysis



bins = [290, 301, 311, 321, 331]

names = ['<=300', '301 - 310', '311 - 320','321 - 330','330+']

d = dict(enumerate(names, 1))

f['GRE_Category'] = np.vectorize(d.get)(np.digitize(f['GRE Score'], bins))



fig = go.Figure()



fig.add_trace(go.Histogram(x = f['GRE_Category'], marker_color='#5852a8'))



fig.update_layout(title='Number of Students Categorised by GRE',

                  xaxis = dict(title = 'GRE_Category'),

                  yaxis = dict(title = 'Number of Students'),

                  bargap=0.05

                 )

fig.show()
# Admit chances as per GRE Score Category



fig = px.box(f,x = f['GRE_Category'], y = f['Chance of Admit'],

             color= 'Research', title="Box plot of Chance of Admit for Each GRE Score Category (Categorised by Research & No-Research Students)"

             )

fig.show()
# TOEFL Score Analysis



bins = [91, 101, 111]

names = ['91 - 100','101 - 110','110+']

d = dict(enumerate(names, 1))

f['TOEFL_Category'] = np.vectorize(d.get)(np.digitize(f['TOEFL Score'], bins))



fig = go.Figure()



fig.add_trace(go.Histogram(x = f['TOEFL_Category'], marker_color='#5852a8'))



fig.update_layout(title='Number of Students Categorised by TOEFL',

                  xaxis = dict(title = 'TOEFL_Category'),

                  yaxis = dict(title = 'Number of Students'),

                  bargap=0.05

                 )

fig.show()
# Admit chances as per TOEFL Score Category



fig = px.box(f,x = f['TOEFL_Category'], y = f['Chance of Admit'],

             color= 'Research', title="Box plot of Chance of Admit for Each TOEFL Score Category (Categorised by Research & No-Research Students)"

             )

fig.show()
# Admit chances as per GRE Score and TOEFL Score Category



fig = px.box(f,x = f['GRE_Category'], y = f['Chance of Admit'],

             color= 'TOEFL_Category', title="Box plot of Chance of Admit for Each GRE Score Category (Categorised by TOEFL Score)"

             )

fig.show()
# Admit chances as per GRE Score and SOP Rating Category



fig = px.box(f,x = f['SOP'], y = f['Chance of Admit'],

             color= 'GRE_Category', title="Box plot of Chance of Admit for Each SOP Rating Category (Categorised by GRE Score)"

             )

fig.show()
# Admit chances as per GRE Score and LOR Rating Category



fig = px.box(f,x = f['LOR'], y = f['Chance of Admit'],

             color= 'GRE_Category', title="Box plot of Chance of Admit for Each LOR Rating Category (Categorised by GRE Score)"

             )

fig.show()
# Admit chances as per SOP and LOR Rating Category



fig = px.box(f,x = f['LOR'], y = f['Chance of Admit'],

             color= 'SOP', title="Box plot of Chance of Admit for Each LOR Rating Category (Categorised by SOP Rating)"

             )

fig.show()
# Admit chances as per GRE_Category and CGPA_Category



fig = px.box(f,x = f['GRE_Category'], y = f['Chance of Admit'],

             color= 'CGPA_Category', title="Box plot of Chance of Admit for Each GRE Score Category (Categorised by CGPA Category)"

             )

fig.show()
# Admit chances as per GRE_Category and University Rating



fig = px.box(f,x = f['GRE_Category'], y = f['Chance of Admit'],

             color= 'University Rating', title="Box plot of Chance of Admit for Each GRE Score Category (Categorised by University Rating)"

             )

fig.show()
# Admit chances as per CGPA_Category and University Rating



fig = px.box(f,x = f['CGPA_Category'], y = f['Chance of Admit'],

             color= 'University Rating', title="Box plot of Chance of Admit for Each CGPA Category (Categorised by University Rating)"

             )

fig.show()
from sklearn.model_selection import train_test_split



X = f.drop(['Chance of Admit','CGPA_Category','GRE_Category','TOEFL_Category'], axis=1)

y = f['Chance of Admit']



X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.1, random_state=0)
from sklearn.linear_model import LinearRegression



lr = LinearRegression()



plr = lr.fit(X_train, y_train)

predictions = lr.predict(X_test)
# Let's grab predictions off our test set and see how well it did!



fig = go.Figure()

fig.add_trace(go.Scatter(x=y_test,

                y=predictions,

                marker_color='white',

                mode='markers'

                ))

fig.update_layout(title='Predicted Chance of Admit vs Actual Chance of Admit',

                  xaxis = dict(title = 'Actual Chance of Admit'),

                  yaxis = dict(title = 'Predicted Chance of Admit')

                  )

fig.show()
from sklearn import metrics



print("Results...")

print('MAE:', metrics.mean_absolute_error(y_test, predictions))

print('MSE:', metrics.mean_squared_error(y_test, predictions))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
# Feature Importance using random forest regressor

from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor()

rfr.fit(X,y)

feature_names = X.columns

imp_f = pd.DataFrame()

imp_f['Parameters'] = X.columns

imp_f['Parameters_Importance'] = rfr.feature_importances_

imp_f = imp_f.sort_values(by=['Parameters_Importance'], ascending=0)

imp_f['Parameters_Importance'] = round(imp_f['Parameters_Importance'],3)

fig = go.Figure()



fig.add_trace(go.Bar(x = imp_f['Parameters'],

                     y = imp_f['Parameters_Importance'],

                     textposition = "outside",

                     text = imp_f['Parameters_Importance']))



fig.update_layout(title='Parameters Importance for High Chance of Admit',

                  xaxis = dict(title = 'Parameters'),

                  yaxis = dict(title = 'Parameters_Importance')

                  )



fig.show()