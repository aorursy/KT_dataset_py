import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
from plotly import tools
import plotly.figure_factory as ff
import seaborn as sns
import matplotlib.pyplot as plt
df=pd.read_csv('../input/StudentsPerformance.csv')
df.head()
df.isna().sum()
gender = df['gender'].value_counts()
label = gender.index
size = gender.values

trace = go.Pie(labels=label, 
               values=size, 
              )

data = [trace]
layout = go.Layout(title='Gebder Distribution')

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
race = df['race/ethnicity'].value_counts()
label = race.index
size = race.values

trace = go.Pie(labels=label, 
               values=size, 
              )

data = [trace]
layout = go.Layout(title='Race/ethhicity Distribution')

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
race = df['parental level of education'].value_counts()
label = race.index
size = race.values

trace = go.Pie(labels=label, 
               values=size, 
              )

data = [trace]
layout = go.Layout(title='parental level of education')

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
group_a = df[df['race/ethnicity']=='group A']
race_a=group_a['parental level of education'].value_counts()
label_a = race_a.index
size_a = race_a.values

group_b = df[df['race/ethnicity']=='group B']
race_b=group_b['parental level of education'].value_counts()
label_b = race_b.index
size_b = race_b.values

group_c = df[df['race/ethnicity']=='group C']
race_c=group_c['parental level of education'].value_counts()
label_c = race_c.index
size_c = race_c.values

group_d = df[df['race/ethnicity']=='group D']
race_d=group_d['parental level of education'].value_counts()
label_d = race_d.index
size_d = race_d.values

group_e = df[df['race/ethnicity']=='group E']
race_e=group_e['parental level of education'].value_counts()
label_e = race_e.index
size_e = race_e.values

fig = {
    'data': [
        {
            'labels': label_a,
            'values': size_a,
            'marker': {'colors': ['rgb(146, 123, 21)',
                                  'rgb(177, 180, 34)',
                                  'rgb(206, 206, 40)',
                                  'rgb(175, 51, 21)',
                                  'rgb(35, 36, 21)']},
            'type': 'pie',
            'name': 'group A',
            'domain': {'x': [0, .48],
                       'y': [0, .49]},
            'hoverinfo':'label+percent+name',
            'textinfo':'none'
        },
        {
            'labels': label_b,
            'values': size_b,
            'marker': {'colors': ['rgb(146, 123, 21)',
                                  'rgb(177, 180, 34)',
                                  'rgb(206, 206, 40)',
                                  'rgb(175, 51, 21)',
                                  'rgb(35, 36, 21)']},
            'type': 'pie',
            'name': 'group B',
            'domain': {'x': [.52, 1],
                       'y': [0, .49]},
            'hoverinfo':'label+percent+name',
            'textinfo':'none'

        },
        {
            'labels': label_c,
            'values': size_c,
            'marker': {'colors': ['rgb(146, 123, 21)',
                                  'rgb(177, 180, 34)',
                                  'rgb(206, 206, 40)',
                                  'rgb(175, 51, 21)',
                                  'rgb(35, 36, 21)']},
            'type': 'pie',
            'name': 'group C',
            'domain': {'x': [0, .48],
                       'y': [.51, 1]},
            'hoverinfo':'label+percent+name',
            'textinfo':'none'
        },
        {
            'labels': label_d,
            'values': size_d,
            'marker': {'colors': ['rgb(146, 123, 21)',
                                  'rgb(177, 180, 34)',
                                  'rgb(206, 206, 40)',
                                  'rgb(175, 51, 21)',
                                  'rgb(35, 36, 21)']},
            'type': 'pie',
            'name':'group D',
            'domain': {'x': [.52, 1],
                       'y': [.51, 1]},
            'hoverinfo':'label+percent+name',
            'textinfo':'none'
        }
    ],
    'layout': {'title': 'Distribution on education by etnic groups',
               'showlegend': False}
}

py.iplot(fig)
female = df[df['gender']=='female']
female=female['parental level of education'].value_counts()
label_f = female.index
size_f = female.values

male = df[df['gender']=='male']
male=male['parental level of education'].value_counts()
label_m = male.index
size_m = male.values



fig = {
    'data':[
        {
            'labels': label_f,
            'values': size_f,
            'marker': {'colors': ['rgb(146, 123, 21)',
                                  'rgb(177, 180, 34)',
                                  'rgb(206, 206, 40)',
                                  'rgb(175, 51, 21)',
                                  'rgb(35, 36, 21)']},
            'type': 'pie',
            'name': 'female',
            'domain': {'x': [0, .48],
                       'y': [.51, 1]},
            'hoverinfo':'label+percent+name',
            'textinfo':'none'
        },
        {
            'labels': label_m,
            'values': size_m,
            'marker': {'colors': ['rgb(146, 123, 21)',
                                  'rgb(177, 180, 34)',
                                  'rgb(206, 206, 40)',
                                  'rgb(175, 51, 21)',
                                  'rgb(35, 36, 21)']},
            'type': 'pie',
            'name':'male',
            'domain': {'x': [.52, 1],
                       'y': [.51, 1]},
            'hoverinfo':'label+percent+name',
            'textinfo':'none'
        }
    ],
    'layout': {'title': 'Distribution on education gender',
               'showlegend': False}
}

py.iplot(fig)
fig = plt.subplots(figsize=(15, 15))
female=df[df['gender']=='female']
male=df[df['gender']=='male']
sns.distplot(female["math score"],label="female")
sns.distplot(male["math score"],label="male")
plt.legend()
plt.show()
fig = plt.subplots(figsize=(15, 15))
female=df[df['gender']=='female']
male=df[df['gender']=='male']
sns.distplot(female["reading score"],label="female")
sns.distplot(male["reading score"],label="male")
plt.legend()
plt.show()
fig = plt.subplots(figsize=(15, 15))
female=df[df['gender']=='female']
male=df[df['gender']=='male']
sns.distplot(female["writing score"],label="female")
sns.distplot(male["writing score"],label="male")
plt.legend()
plt.show()
fig = plt.subplots(figsize=(25, 25))
sns.distplot(group_a['math score'],label="race_a")
sns.distplot(group_b['math score'],label="race_b")
sns.distplot(group_c['math score'],label="race_c")
sns.distplot(group_d['math score'],label="race_d")
sns.distplot(group_e['math score'],label="race_e")
plt.legend()
plt.show()
fig = plt.subplots(figsize=(25, 25))
sns.distplot(group_a['reading score'],label="race_a")
sns.distplot(group_b['reading score'],label="race_b")
sns.distplot(group_c['reading score'],label="race_c")
sns.distplot(group_d['reading score'],label="race_d")
sns.distplot(group_e['reading score'],label="race_e")
plt.legend()
plt.show()
fig = plt.subplots(figsize=(25, 25))
sns.distplot(group_a['writing score'],label="race_a")
sns.distplot(group_b['writing score'],label="race_b")
sns.distplot(group_c['writing score'],label="race_c")
sns.distplot(group_d['writing score'],label="race_d")
sns.distplot(group_e['writing score'],label="race_e")
plt.legend()
plt.show()
df1=pd.get_dummies(df)
fig = plt.subplots(figsize=(25, 25))
sns.heatmap(df1.corr(), annot=True, fmt=".1f")
df1.head()
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

df_scaled = pd.DataFrame(scaler.fit_transform(df1), columns=df1.columns)
X=df_scaled.iloc[:,3:].values
Y_m=df_scaled.iloc[:,0].values
Y_r=df_scaled.iloc[:,1].values
Y_w=df_scaled.iloc[:,2].values
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

rfc=RandomForestRegressor(random_state=42)
param_grid = { 
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['mse','mae']
}
CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
CV_rfc.fit(X,Y_r)
CV_rfc.best_params_
CV_rfc.score(X,Y_r)
from sklearn.linear_model import Lasso


lasso=Lasso()
param_grid = { 
    'alpha': [0.01,0,1, 1,10],
    'max_iter':[100,1000,10000]
}
CV_lasso = GridSearchCV(estimator=lasso, param_grid=param_grid, cv= 5)
CV_lasso.fit(X,Y_r)
CV_lasso.score(X,Y_r)