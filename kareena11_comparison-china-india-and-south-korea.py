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
datafile = "../input/world-development-indicators/Indicators.csv"



df = pd.read_csv(datafile)
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

plt.style.use('seaborn-whitegrid')

import plotly.express as px

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
df.head()
df_subset = df.pivot_table(index=['CountryName', 'CountryCode', 'Year'], columns='IndicatorName', values='Value').reset_index()



countries = ['Korea, Rep.', 'India', 'China']



df_subset1 = df_subset[df_subset['CountryName'].isin(countries)]
fig = px.line(df_subset1, x="Year", y="GDP per capita (constant 2005 US$)", color='CountryName')

fig.update_layout(title='GDP per capita comparison' , showlegend=True)

fig.show()
fig1 = px.line(df_subset1, x="Year", y="Gross capital formation (% of GDP)", color='CountryName')

fig1.update_layout(title='Gross Capital Formation Comparison', showlegend=True)

fig1.show()
fig2 = px.line(df_subset1, x="Year", y="Population growth (annual %)", color='CountryName')

fig2.update_layout(title='Population Comparison', showlegend=True)

fig2.show()
fig3 = px.line(df_subset1, x="Year", y="Age dependency ratio, young (% of working-age population)", color='CountryName')

fig3.update_layout(title='Working population comparison' , showlegend=True)

fig3.show()
fig4 = px.line(df_subset1, x="Year", y="Ratio of female to male labor force participation rate (%) (modeled ILO estimate)", color='CountryName')

fig4.update_layout(title='Labor force by gender comparison', showlegend=True)

fig4.show()
fig5 = px.line(df_subset1, x="Year", y="Research and development expenditure (% of GDP)", color='CountryName')

fig5.update_layout(title='R&D Comparison' , showlegend=True)

fig5.show()
fig6 = px.line(df_subset1, x="Year", y="Mobile cellular subscriptions (per 100 people)", color='CountryName')

fig6.update_layout(title='Mobile subscriptions comparison' , showlegend=True)

fig6.show()
fig7 = px.line(df_subset1, x="Year", y="Exports of goods and services (% of GDP)", color='CountryName')

fig7.update_layout(title='Exports Comparison' , showlegend=True)

fig7.show()
fig8 = px.line(df_subset1, x="Year", y="Time required to start a business (days)", color='CountryName')

fig8.update_layout(title='Business-friendly environment comparison' , showlegend=True)

fig8.show()
fig9 = px.line(df_subset1, x="Year", y="Life expectancy at birth, total (years)", color='CountryName')

fig9.update_layout(title='Life expectancy Comparison', showlegend=True)

fig9.show()
fig10 = px.line(df_subset1, x="Year", y="Fertility rate, total (births per woman)", color='CountryName')

fig10.update_layout(title='Fertility rate Comparison' , showlegend=True)

fig10.show()