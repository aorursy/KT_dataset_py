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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import seaborn as sns
import plotly.graph_objs as go


df = pd.read_csv('../input/our-world-in-data/owid-covid-data.csv')

df.columns
df_pt = pd.read_csv('../input/oxford-policy-tracker/OxCGRT_latest.csv', parse_dates=['Date'])
a =  list(df_pt.columns)
a
Indonesia = df_pt[df_pt['CountryName']=='Indonesia']
df_pivot = pd.melt(Indonesia, id_vars =['Date'], 
        value_vars =['C1_School closing',
 'C2_Workplace closing',
 'C3_Cancel public events',
 'C4_Restrictions on gatherings',
 'C5_Close public transport',
 'C6_Stay at home requirements',
 'C7_Restrictions on internal movement',
 'C8_International travel controls',
 'E1_Income support',
 'E2_Debt/contract relief',
 'H1_Public information campaigns',
 'H2_Testing policy',
 'H3_Contact tracing',
 'H4_Emergency investment in healthcare',
 'H5_Investment in vaccines',
 'M1_Wildcard',], var_name = "Kebijakan", value_name="Nilai")
df_pivot[df_pivot['Kebijakan'] == 'H1_Public information campaigns']
fig = go.Figure(data=go.Heatmap(
        z=df_pivot['Nilai'],
        x=df_pivot['Date'],
        y=df_pivot['Kebijakan'],
        colorscale='GnBu',
        showlegend=False,
        text=df_pivot['Kebijakan']))
fig.show()
df_mobility = pd.read_csv('../input/global-mobility-report-21-may/Global_Mobility_Report.csv')
df_mobility_indo = df_mobility[df_mobility['country_region'] == 'Indonesia']
df_mobility_indo
df_pivot = pd.melt(df_mobility_indo, id_vars =['date'], 
        value_vars =['retail_and_recreation_percent_change_from_baseline',
 'grocery_and_pharmacy_percent_change_from_baseline',
 'parks_percent_change_from_baseline',
 'transit_stations_percent_change_from_baseline',
 'workplaces_percent_change_from_baseline',
 'residential_percent_change_from_baseline',],
var_name = "Mobilitas", value_name="Nilai")
#nilai yang diambil heatmap adalah rata-rata
fig = go.Figure(data=go.Heatmap(
        z=df_pivot['Nilai'],
        x=df_pivot['date'],
        y=df_pivot['Mobilitas'],
        colorscale='GnBu',
        showlegend=False,
        text=df_pivot['Mobilitas']))
fig.show()
df_pivot_money = pd.melt(df_pt, id_vars =['CountryName','Date'], 
        value_vars =[
  'E3_Fiscal measures'], var_name = "Kebijakan", value_name="Nilai")
nilai =df_pivot_money[df_pivot_money['CountryName']=='Indonesia']
nilai[nilai['Nilai']==24650000000]
df_pivot_money
temp = df_pivot_money[df_pivot_money['Nilai']>0].sort_values('CountryName', ascending=False)
fig = px.scatter(temp, x='Date', y='CountryName', size='Nilai', color='Nilai', height=2000, 
           color_continuous_scale=px.colors.sequential.Viridis)
fig.update_layout(yaxis = dict(dtick = 1))
fig.update(layout_coloraxis_showscale=True)
fig.show()
