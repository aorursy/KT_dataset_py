# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
print ("Datasets:")
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
       
india_covid19_df = pd.read_csv("/kaggle/input/covid19-in-india/covid_19_india.csv") 
india_covid19_df.head()
statewise_covid19_df = india_covid19_df.groupby(by='State/UnionTerritory', as_index=True).agg({"Cured" : max, "Deaths": max, "Confirmed":max} )
#statewise_covid19_df_pvt = statewise_covid19_df.pivot(index='Sno', columns="State/UnionTerritory", values=['Cured', 'Deaths', 'Confirmed'])

statewise_covid19_df["Mortality Rate %"] = round(statewise_covid19_df["Deaths"] / statewise_covid19_df["Confirmed"], 3) * 100
statewise_covid19_df = statewise_covid19_df.sort_values(by="Confirmed", ascending=False)
statewise_covid19_df.style.background_gradient(cmap='Reds')
#display(statewise_covid19_df)

statewise_covid19_df.plot.bar(figsize=(20,10))
statelist = india_covid19_df['State/UnionTerritory'].unique()
statelist.sort()
for s_ut in statelist:
    if (statewise_covid19_df.loc[s_ut]["Confirmed"] > 100):
        state_covid19_df = india_covid19_df[india_covid19_df['State/UnionTerritory'] == s_ut]
        state_covid19_df.plot(x='Date', y=['Confirmed', 'Cured', 'Deaths'], figsize=[10, 5], title="State: {}".format(s_ut))

