# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/ghana-covid19-dataset/Ghana_Covid19_DailyActive.csv')



df.info()
df.head()
df.describe()
df['confirmed'].sum()
df['recovered'].sum()
df['death'].sum()
import plotly.graph_objects as go

import plotly.express as px



fig = px.line(df, x='date', y='confirmed', title='Positive Confirmed Cases by Day')





fig.update_xaxes(

    rangeslider_visible=True

)

fig.show()
import plotly.graph_objects as go

import plotly.express as px



fig = px.line(df, x='date', y='recovered', title='Number of Recoveries by Day')





fig.update_xaxes(

    rangeslider_visible=True

)

fig.show()
import plotly.graph_objects as go

import plotly.express as px



fig = px.line(df, x='date', y='death', title='Number of Deaths by Day')





fig.update_xaxes(

    rangeslider_visible=False

)

fig.show()
import plotly.graph_objects as go

import plotly.express as px



fig = px.line(df, x='date', y='active_cases', title='Daily Active Cases')





fig.update_xaxes(

    rangeslider_visible=False

)

fig.show()