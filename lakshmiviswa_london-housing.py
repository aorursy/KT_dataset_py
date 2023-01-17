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
#pd.set_option("display.max_rows", None, "display.max_columns", None)

lon_m=pd.read_csv('/kaggle/input/housing-in-london/housing_in_london_monthly_variables.csv')

lon_m.head()
import plotly.express as px



fig = px.line(lon_m, x="date", y="average_price", color='area')

fig.show()
fig_crime = px.line(lon_m, x="date", y="no_of_crimes", color='area')

fig_crime.show()
lon_m['no_of_crimes'].isnull().sum()
lon_m_pivot=lon_m.pivot_table(index=['date'], values=['average_price','no_of_crimes'], aggfunc='sum')

lon_m_pivot.style.background_gradient(cmap='summer')
lon_m_pivot.plot(kind='line', figsize=(15,10))
flattened = pd.DataFrame(lon_m_pivot.to_records())

#df1=pd.DataFrame({'count' : df.groupby(['cs_host']).size()}).reset_index()
flattened.head()
import plotly.graph_objects as go



fig = go.Figure()

fig.add_trace(go.Scatter(x=flattened['date'], y=flattened['average_price'], name='Average Price',

                         line=dict(color='firebrick', width=2)))

fig.add_trace(go.Scatter(x=flattened['date'], y=flattened['no_of_crimes'],name = 'Crimes',

                         line=dict(color='royalblue', width=2, dash='dashdot')))

fig.show()
lon_m['borough_flag'].value_counts()
flag_1=lon_m.loc[lon_m['borough_flag'] == 1]

flag_1.head()
fig = px.line(flag_1, x="date", y="average_price", color='area')

fig.show()
fig_crime = px.line(flag_1, x="date", y="no_of_crimes", color='area')

fig_crime.show()
flag_1_pivot=flag_1.pivot_table(index=['date'], values=['average_price','no_of_crimes'], aggfunc='sum')
lon_m_pivot.style.background_gradient(cmap='summer')
flattened_1 = pd.DataFrame(flag_1_pivot.to_records())

flattened_1.head()
import plotly.graph_objects as go



fig = go.Figure()

fig.add_trace(go.Scatter(x=flattened_1['date'], y=flattened_1['average_price'], name='Average Price',

                         line=dict(color='firebrick', width=2)))

fig.add_trace(go.Scatter(x=flattened_1['date'], y=flattened_1['no_of_crimes'],name = 'Crimes',

                         line=dict(color='royalblue', width=2, dash='dashdot')))

fig.show()