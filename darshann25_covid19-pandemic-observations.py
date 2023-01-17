# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
covid = pd.read_csv('/kaggle/input/worldometer-stats/wworldometer.csv')
covid.head()
covid = covid.drop(['newcases','newdeath','criticalcases','totaltestsOver1M'], axis=1)

columns = covid.columns
columns = columns.drop('name')

for col in columns:
    covid[col] = covid[col].str.replace(',','')
    covid[col] = pd.to_numeric(covid[col], errors='coerce')
    covid[col] = covid[col].fillna(0)
    
columns = columns.sort_values()
columns = columns.insert(0,'name')
covid = covid[columns]
rename_cols = ['country','active_cases','total_cases','total_deaths','total_recovered','total_tests']
covid.columns = rename_cols
# Feature Engineering - Mortality Rate, Cases by Tests, Deaths by Tests
covid['mortality_rate'] = (covid['total_deaths'] / covid['total_cases']) * 100
covid['cases_rate'] = (covid['total_cases'] / covid['total_tests']) * 100
covid['deaths_rate'] = (covid['total_deaths'] / covid['total_tests']) * 100

covid = covid.replace([np.inf, -np.inf], np.nan)
covid['cases_rate'] = covid['cases_rate'].fillna(0)
covid['deaths_rate'] = covid['deaths_rate'].fillna(0)
covid = covid.dropna(subset=['country'])
import matplotlib.pyplot as plt
from matplotlib import colors

covid.sort_values('total_cases', ascending=False).style.background_gradient(cmap='Blues',subset=['total_cases'])\
            .background_gradient(cmap='Reds', subset=['total_deaths'])\
            .background_gradient(cmap='Greens', subset=['total_recovered'])\
            .background_gradient(cmap='Purples', subset=['active_cases'])\
            .background_gradient(cmap='Greys', subset=['total_tests'])\
            .background_gradient(cmap='Reds', subset=['mortality_rate'])\
            .background_gradient(cmap='Blues', subset=['cases_rate'])\
            .background_gradient(cmap='Reds', subset=['deaths_rate'])
import plotly.express as px
fig = px.scatter_geo(covid, locations="country", color="country",
                     hover_name="country", size="total_deaths",
                     projection="natural earth")
fig.show()
