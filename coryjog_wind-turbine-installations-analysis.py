import pandas as pd

# Import plotting
%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.style.use('ggplot')
wind_turbines = pd.read_csv("../input/uswtdb/uswtdb_v1_0_20180419.csv")
wind_turbines = wind_turbines.loc[wind_turbines.t_manu!='missing',:]
wind_turbines = wind_turbines.loc[wind_turbines.p_year>0,:]
print(wind_turbines.columns.tolist())
turbine_count_by_manufacturer = wind_turbines.groupby(['t_manu']).case_id.count()
turbine_capacity_by_manufacturer = wind_turbines.groupby(['t_manu']).t_cap.sum()/1000000.0

fig = plt.figure(figsize=(15,6))
ax = fig.add_subplot(111)
turbine_count_by_manufacturer[turbine_count_by_manufacturer>100].sort_values(ascending=False).plot(kind='bar', ax=ax)
ax.set_title('How many turbines installed by manufacturer?')
ax.set_xlabel('Turbine manufacturer')
ax.set_ylabel('Turbine count')
plt.show()

fig = plt.figure(figsize=(15,6))
ax = fig.add_subplot(111)
turbine_capacity_by_manufacturer[turbine_capacity_by_manufacturer>0.25].sort_values(ascending=False).plot(kind='bar', ax=ax)
ax.set_title('How many GW installed by manufacturer?')
ax.set_xlabel('Turbine manufacturer')
ax.set_ylabel('Total turbine capacity [GW]')
plt.show()
top_six_manufacturers = ['GE Wind', 'Vestas', 'Siemens', 'Mitsubishi', 'Gamesa', 'Suzlon']
turbines_by_manufacturer_and_year = wind_turbines.groupby(['p_year','t_manu']).t_cap.sum().reset_index()

fig = plt.figure(figsize=(15,6))
ax = fig.add_subplot(111)
for top_manufacturer in top_six_manufacturers:
    turbines_by_manufacturer_and_year.loc[turbines_by_manufacturer_and_year.t_manu==top_manufacturer,:].plot(kind='line',
                                                                                                             x='p_year',
                                                                                                             y='t_cap',
                                                                                                             ax=ax,
                                                                                                             label=top_manufacturer)
ax.set_title('How many turbines per year did year manufacturer install?')
ax.set_xlabel('Turbine manufacturer')
ax.set_ylabel('Turbine count')
plt.show()
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.offline as offline
offline.init_notebook_mode(connected=True)
wind_turbines_by_state = wind_turbines.groupby('t_state').case_id.count()

plot_data = [dict(type='choropleth',
                autocolorscale = True,
                locations = wind_turbines_by_state.index,
                z = wind_turbines_by_state,
                locationmode = 'USA-states',
                colorbar = dict(title = "Turbine count"))]

layout = dict(title = 'How many turbines installed per state?',
              autosize=False,
              width=1000,
              geo = dict(scope='usa',
                         projection=dict( type='albers usa' ),
                         showlakes = True,
                         lakecolor = 'rgb(255, 255, 255)'))

fig = go.Figure(data=plot_data, layout=layout)
offline.iplot(fig)
wind_capacity_by_state = wind_turbines.groupby('t_state').t_cap.sum()/1000000.0

plot_data = [dict(type='choropleth',
                autocolorscale = True,
                locations = wind_capacity_by_state.index,
                z = wind_capacity_by_state,
                locationmode = 'USA-states',
                colorbar = dict(title = "Capacity [GW]"))]

layout = dict(title = 'How much capacity installed per state?',
              autosize=False,
              width=1000,
              geo = dict(scope='usa',
                         projection=dict( type='albers usa' ),
                         showlakes = True,
                         lakecolor = 'rgb(255, 255, 255)'))

fig = go.Figure(data=plot_data, layout=layout)
offline.iplot(fig)
