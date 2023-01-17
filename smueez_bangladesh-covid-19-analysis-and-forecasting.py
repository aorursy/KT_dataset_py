import os

from pathlib import Path

import random

import sys



from tqdm.notebook import tqdm

import numpy as np

import pandas as pd

import scipy as sp





import matplotlib.pyplot as plt

import seaborn as sns



from IPython.core.display import display, HTML



# --- plotly ---

from plotly import tools, subplots

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.express as px

import plotly.figure_factory as ff

import plotly.io as pio

pio.templates.default = "simple_white"



# --- models ---

from sklearn import preprocessing

from sklearn.model_selection import KFold

import lightgbm as lgb

import xgboost as xgb

import catboost as cb



# --- setup ---

pd.set_option('max_columns', 50)
bd_df = pd.read_csv('../input/covid19-in-bangladesh/COVID-19_in_bd.csv')

bd_df.head()


bd_df = bd_df

bd_df['prev_confirmed'] = bd_df['Confirmed'].shift(1)

bd_df['new_case'] = bd_df['Confirmed'] - bd_df['prev_confirmed']

bd_df.loc[bd_df['new_case'] < 0, 'new_case'] = 0.
fig = px.line(bd_df,

              x='Date', y='new_case',

              title=f'DAILY NEW Confirmed cases in Bangladesh')

fig.show()
bd_df = bd_df

bd_df['prev_confirmed'] = bd_df['Deaths'].shift(1)

bd_df['new_case'] = bd_df['Deaths'] - bd_df['prev_confirmed']

bd_df.loc[bd_df['new_case'] < 0, 'new_case'] = 0.
fig = px.line(bd_df,

              x='Date', y='new_case',

              title=f'DAILY NEW Deaths in Bangladesh')

fig.show()
bd_df = bd_df

bd_df['prev_confirmed'] = bd_df['Recovered'].shift(1)

bd_df['new_case'] = bd_df['Recovered'] - bd_df['prev_confirmed']

bd_df.loc[bd_df['new_case'] < 0, 'new_case'] = 0.
fig = px.line(bd_df,

              x='Date', y='new_case',

              title=f'DAILY Recovering rate in Bangladesh')

fig.show()
df1 = pd.read_csv('../input/covid19-bangladesh-dataset/COVID-19-Bangladesh.csv')

df1.head()
# Grouping cases by date 

df = pd.read_csv('../input/covid19-bangladesh-dataset/COVID-19-Bangladesh.csv')

temp = df.groupby('Date')['Confirmed', 'Recovered', 'Deaths','Quarantine'].sum().reset_index() 

# Unpivoting 

temp = temp.melt(id_vars='Date',value_vars = ['Confirmed', 'Recovered', 'Deaths'], var_name='Case', value_name='Count') 

# Visualization

fig = px.line(temp, x='Date', y='Count', color='Case', color_discrete_sequence=['#ff0000', '#FFFF00', '#0000FF' , '#0020FF'], template='presentation') 

fig.update_layout(title="COVID-19 Cases Over Time in Bangladesh")

fig.show()
# Grouping cases by date 

df = pd.read_csv('../input/covid19-bangladesh-dataset/COVID-19-Bangladesh.csv')

temp = df.groupby('Date')['Quarantine','Released From Quarantine'].sum().reset_index() 

# Unpivoting 

temp = temp.melt(id_vars='Date',value_vars = ['Quarantine','Released From Quarantine'], var_name='Case', value_name='Count') 

# Visualization

fig = px.line(temp, x='Date', y='Count', color='Case', color_discrete_sequence=[ '#ff0000', '#00FF00'], template='ggplot2') 

fig.update_layout(title="COVID-19 Cases Over Time in Bangladesh")

fig.show()
# Grouping cases by date 

df = bd_df

temp = df.groupby('Date')['Confirmed', 'Recovered', 'Deaths'].sum().reset_index() 

# Unpivoting 

temp = temp.melt(id_vars='Date',value_vars = ['Confirmed', 'Recovered', 'Deaths'], var_name='Case', value_name='Count') 



# Visualization

fig = px.line(temp, x='Date', y='Count', color='Case', color_discrete_sequence=['#ff0000', '#FFFF00', '#0000FF'], template='plotly_dark') 

fig.update_layout(title="COVID-19 Cases in Bangladesh")

fig.show()
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pandas as pd 

import numpy as np 

import matplotlib.pyplot as plt 

from sklearn.linear_model import LinearRegression, Ridge, Lasso 

from sklearn.model_selection import train_test_split, cross_val_score 

from statistics import mean 



import datetime as dt

df = df.dropna(how='any',axis=0)



df['Date'] = pd.to_datetime(df['Date'])

df['Date']= df['Date'].map(dt.datetime.toordinal)



X = df.iloc[:, 2:].values



y = df.iloc[:, 1].values



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=9)



#Linear regression

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor_fit = regressor.fit(X_train, y_train)



y_pred = regressor.predict(X_test)



df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})



print(regressor_fit.score(X_test, y_test)*100)



from sklearn import datasets, linear_model, metrics





reg = linear_model.LogisticRegression() 

   

# train the model using the training sets 

reg.fit(X_train, y_train) 

  

# making predictions on the testing set 

y_pred = reg.predict(X_test) 



print("Logistic Regression model accuracy(in %):",  

metrics.accuracy_score(y_test, y_pred)*100)