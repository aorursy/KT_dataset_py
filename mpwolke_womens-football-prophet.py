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
df = pd.read_csv('../input/womens-international-football-results/results.csv', encoding='ISO-8859-2')

df.head()
df.isna().sum()
# Distribution of different type of amount

fig , ax = plt.subplots(1,2,figsize = (12,5))



home_score = df.home_score.values

away_score = df.away_score.values





sns.distplot(home_score , ax = ax[0] , color = 'blue').set_title('Womens Football Home Score' , fontsize = 14)

sns.distplot(away_score , ax = ax[1] , color = 'pink').set_title('Womens Football Away Score' , fontsize = 14)







plt.show()
import matplotlib.pyplot as plt

import seaborn as sns

sns.countplot(x="neutral",data=df,palette="GnBu_d", edgecolor="black")

plt.title('Womens Football')

plt.xticks(rotation=45)

plt.yticks(rotation=45)

# changing the font size

sns.set(font_scale=1)
px.histogram(df, x='date', color='home_team', title='Womens Football Home team')
df["home_score"] = df["away_score"]/100

fig = px.scatter(df, x="home_score", y="away_score", color="tournament", error_x="home_score", error_y="home_score")

fig.show()
from fbprophet import Prophet

df1=df.rename(columns={"date": "ds", "home_score": "y"})

df1

m = Prophet()

m.fit(df1)
future = m.make_future_dataframe(periods=365)

future.tail()
forecast = m.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
fig1 = m.plot(forecast)
fig2 = m.plot_components(forecast)
from fbprophet.plot import plot_plotly

import plotly.offline as py

py.init_notebook_mode()



fig = plot_plotly(m, forecast)  # This returns a plotly Figure

py.iplot(fig)
_ = pd.pivot_table(df, values='home_score', index='date').plot(style='-o', title="Women's Football")
# 3D Scatter Plot

fig = px.scatter_3d(df, x='home_score', y='tournament', z='date')

fig.show()