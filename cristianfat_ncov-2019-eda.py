# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import networkx as nx

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
conf = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')

dth = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv')

recov = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv')
conf[:5]
dth[:5]
recov[:5]
conf_time_cols = conf.columns.tolist()[4:]

dth_time_cols = dth.columns.tolist()[4:]

recov_time_cols = recov.columns.tolist()[4:]
conf_graph = nx.Graph()

dth_graph = nx.Graph()

recov_graph = nx.Graph()
for i in range(conf.__len__()):

    conf_graph.add_edge(conf.iloc[i]['Country/Region'],conf.iloc[i]['Province/State'],relation='state-level')

    conf_graph.add_edge(conf.iloc[i]['Province/State'],conf.iloc[i][conf_time_cols[-1]],relation='number-level')
plt.figure(figsize=(50,50))

nx.draw_networkx(conf_graph)
for i in range(dth.__len__()):

    dth_graph.add_edge(dth.iloc[i]['Country/Region'],dth.iloc[i]['Province/State'],relation='state-level')

    dth_graph.add_edge(conf.iloc[i]['Province/State'],conf.iloc[i][dth_time_cols[-1]],relation='number-level')
plt.figure(figsize=(50,50))

nx.draw_networkx(dth_graph)
for i in range(recov.__len__()):

    recov_graph.add_edge(dth.iloc[i]['Country/Region'],dth.iloc[i]['Province/State'],relation='state-level')

    recov_graph.add_edge(conf.iloc[i]['Province/State'],conf.iloc[i][recov_time_cols[-1]],relation='number-level')
plt.figure(figsize=(50,50))

nx.draw_networkx(recov_graph)
conf[conf_time_cols].describe().boxplot(figsize=(55,20))
dth[dth_time_cols].describe().boxplot(figsize=(55,20))
recov[recov_time_cols].describe().boxplot(figsize=(55,20))
conf.loc[:,['Country/Region','Province/State',conf_time_cols[-1]]].groupby(['Country/Region','Province/State']).agg({

    conf_time_cols[-1]: 'sum'

})
dth.loc[:,['Country/Region','Province/State',dth_time_cols[-1]]].groupby(['Country/Region','Province/State']).agg({

    dth_time_cols[-1]: 'sum'

})
recov.loc[:,['Country/Region','Province/State',recov_time_cols[-1]]].groupby(['Country/Region','Province/State']).agg({

    recov_time_cols[-1]: 'sum'

})
total_cases = conf[conf_time_cols[-1]].sum()

total_deaths = dth[dth_time_cols[-1]].sum()

total_recovs = recov[recov_time_cols[-1]].sum()
total_cases, total_deaths, total_recovs
death_rate = (total_deaths / total_cases) * 100.0

recov_rate = (total_recovs / total_cases) * 100.0
plt.figure(figsize=(10,10))

objects = ('Death Ratio', 'Recover Ratio')

y_pos = np.arange(len(objects))

performance = [death_rate,recov_rate]



plt.bar(y_pos, performance, align='center', alpha=0.5)

plt.xticks(y_pos, objects)

plt.ylabel('%')

plt.title('Death vs Recover Ratio')



plt.show()
plt.figure(figsize=(10,10))

objects = ('Total Confirmed','Total Deaths', 'Total Recovered')

y_pos = np.arange(len(objects))

performance = [total_cases, total_deaths, total_recovs]



plt.bar(y_pos, performance, align='center', alpha=0.5)

plt.xticks(y_pos, objects)

plt.title("Numbers of " + " - ".join(objects))



plt.show()
import plotly.express as px
conf['cases'] = conf[conf_time_cols[-1]]

dth['cases'] = dth[dth_time_cols[-1]]

recov['cases'] = recov[recov_time_cols[-1]]
hov_data = ["Province/State", "Country/Region","cases"]
plt.figure(figsize=(25,25))

fig = px.scatter_mapbox(conf, lat="Lat", lon="Long", hover_name="Province/State", hover_data=hov_data,

                        color_discrete_sequence=["fuchsia"],zoom=3)

fig.update_layout(mapbox_style="carto-darkmatter")

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

fig.show()
plt.figure(figsize=(50,50))

fig = px.scatter_mapbox(dth, lat="Lat", lon="Long", hover_name="Province/State", hover_data=hov_data,

                        color_discrete_sequence=["fuchsia"],zoom=3)

fig.update_layout(mapbox_style="carto-darkmatter")

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

fig.show()
plt.figure(figsize=(25,25))

fig = px.scatter_mapbox(recov, lat="Lat", lon="Long", hover_name="Province/State", hover_data=hov_data,

                        color_discrete_sequence=["fuchsia"],zoom=3)

fig.update_layout(mapbox_style="carto-darkmatter")

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

fig.show()