# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 

import seaborn as sns

%matplotlib inline

import plotly.express as px

import plotly.graph_objects as go

import plotly.offline as py

import plotly.express as px



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/psagot2020explorationlecturedata/winter.csv', encoding='ISO-8859-2')

df.head()
fig = px.bar(df, 

             x='Sport', y='Gender', color_discrete_sequence=['darkgreen'],

             title='Winter Sports', text='City')

fig.show()
fig = px.bar(df, 

             x='Athlete', y='Medal', color_discrete_sequence=['darkgreen'],

             title='Winter Sports', text='Event')

fig.show()
fig = px.bar(df, 

             x='Year', y='City', color_discrete_sequence=['crimson'],

             title='Winter Sports', text='Event')

fig.show()
fig = px.bar(df, 

             x='Medal', y='Discipline', color_discrete_sequence=['crimson'],

             title='Winter Sports', text='Gender')

fig.show()
fig = px.bar(df, 

             x='City', y='Sport', color_discrete_sequence=['moccasin'],

             title='Winter Sports', text='Discipline')

fig.show()
fig = px.bar(df, 

             x='Discipline', y='Country', color_discrete_sequence=['magenta'],

             title='Winter Sports', text='Medal')

fig.show()
fig = px.line(df, x="Sport", y="City", color_discrete_sequence=['darkseagreen'], 

              title="Winter Sports")

fig.show()
fig = px.line(df, x="Event", y="City", color_discrete_sequence=['navy'], 

              title="Winter Sports")

fig.show()
from category_encoders import OneHotEncoder

from sklearn.linear_model import LinearRegression, Ridge

from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler



cols_selected = ['Medal']

ohe = OneHotEncoder(cols=cols_selected, use_cat_names=True)

df_t = ohe.fit_transform(df[cols_selected+['Year']])



#scaler = MaxAbsScaler()

X = df_t.iloc[:,:-1]

y = df_t.iloc[:, -1].fillna(df_t.iloc[:, -1].mean()) / df_t.iloc[:, -1].max()



mdl = Ridge(alpha=0.1)

mdl.fit(X,y)



pd.Series(mdl.coef_, index=X.columns).sort_values().head(10).plot.barh()