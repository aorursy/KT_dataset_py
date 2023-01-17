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
# Death and Recoverd of a single country

df = pd.read_csv('/kaggle/input/corona-virus-report/covid_19_clean_complete.csv')
bd_df = df[df['Country/Region'] == 'Bangladesh']
bd_df[['Deaths', 'Recovered']].tail(1).reset_index().iloc[0]
# bd_df.head()
# Top death recovered ratio countries

df_dr = pd.DataFrame()
dr_series = pd.Series({})
death_series = pd.Series({})
recovered_series = pd.Series({})
confirmed_series = pd.Series({})
for group, frame in df.groupby('Country/Region'):
    tmp_df = df[df['Country/Region'] == group]
    tmp_df = tmp_df[['Country/Region', 'Deaths', 'Recovered', 'Confirmed']].tail(1)
    tmp_series = tmp_df.reset_index().set_index('Country/Region').iloc[0]
    
    death_series[group] = tmp_series.loc['Deaths']
    recovered_series[group] = tmp_series.loc['Recovered']
    confirmed_series[group] = tmp_series.loc['Confirmed']
    
    if(tmp_series.loc['Recovered'] > 0.0):
        dr_ratio = tmp_series.loc['Deaths'] / tmp_series.loc['Recovered']
        dr_series[group] = dr_ratio

dr_series = dr_series.sort_values(ascending=False)
df_dr.insert(0, "D/R Ratio", dr_series)
df_dr.insert(1, "Deaths", death_series)
df_dr.insert(2, "Recovered", recovered_series)
df_dr.insert(3, "Confirmed", confirmed_series)
df_dr.head(20)
# top countries with confirmed case more than 1000 [Considering them as community transmission]

df_dr_5 = df_dr[df_dr['Confirmed'] > 1000]
df_dr_5.head(10)
# bar chart

tmp_df = df_dr_5['D/R Ratio']
tmp_df.head(10).plot.bar(stacked=True)


asian_countries = {
    "south": ["India", "Pakistan", "Afghanistan", "Bangladesh", "Nepal", "Sri Lanka", "Bhutan", "Maldives"]
}
asian_countries["south"]
# South Asian countries

df_south_asia = df[df['Country/Region'].isin(asian_countries["south"])]
df_south_asia