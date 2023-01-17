# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.offline as py

import plotly.figure_factory as ff

py.init_notebook_mode(connected=True)

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
df = pd.read_csv('../input/buildingdatagenomeproject2/steam_cleaned.csv')

df.head()
df.isna().sum()
from fbprophet import Prophet

df1=df.rename(columns={"timestamp": "ds", "Peacock_lodging_Jamaal": "y"})

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
_ = pd.pivot_table(df, values='Peacock_lodging_Jamaal', index='timestamp').plot(style='-o', title="Peacock Lodging Jamaal")

plt.xticks(rotation=45)
# 3D Scatter Plot

fig = px.scatter_3d(df, x='Peacock_lodging_Jamaal', y='Cockatoo_office_Ada', z='Cockatoo_public_Shad')

fig.show()