import numpy as np

import pandas as pd

import plotly.express as px

from IPython.display import display

from ashrae_utils import reduce_mem_usage
data_path = '../input/ashrae-energy-prediction/'
X_train = pd.read_csv(data_path + 'train.csv', engine='python')

X_train, na_list = reduce_mem_usage(X_train)

X_train['timestamp'] = pd.to_datetime(X_train['timestamp'], format='%Y-%m-%d %H:%M:%S')

X_train['meter'] = pd.Categorical(X_train['meter']).rename_categories({0: 'electricity', 1: 'chilledwater', 2: 'steam', 3: 'hotwater'})
X_train.head()
building_metadata = pd.read_csv(data_path + 'building_metadata.csv', engine='python')

building_metadata, na_list = reduce_mem_usage(building_metadata)
building_metadata.head()
daily_train = X_train

daily_train['date'] = daily_train['timestamp'].dt.date

daily_train = daily_train.groupby(['date', 'building_id', 'meter']).sum()

daily_train
daily_train_agg = daily_train.groupby(['date', 'meter']).agg(['sum', 'mean', 'idxmax', 'max'])

daily_train_agg = daily_train_agg.reset_index()

level_0 = daily_train_agg.columns.droplevel(0)

level_1 = daily_train_agg.columns.droplevel(1)

level_0 = ['' if x == '' else '-' + x for x in level_0]

daily_train_agg.columns = level_1 + level_0

daily_train_agg.rename_axis(None, axis=1)

daily_train_agg.head()
fig_total = px.line(daily_train_agg, x='date', y='meter_reading-sum', color='meter', render_mode='svg')

fig_total.update_layout(title='Total kWh per energy aspect')

fig_total.show()
fig_maximum = px.line(daily_train_agg, x='date', y='meter_reading-max', color='meter', render_mode='svg')

fig_maximum.update_layout(title='Maximum kWh value per energy aspect')

fig_maximum.show()
daily_train_agg['building_id_max'] = [x[1] for x in daily_train_agg['meter_reading-idxmax']]

daily_train_agg.head()
def show_building(building, energy_aspects=None):

    fig = px.line(daily_train.loc[(slice(None), building, slice(None)), :].reset_index(),

                  x='date',

                  y='meter_reading',

                  color='meter',

                  render_mode='svg')

    if energy_aspects:

        if 'electricity' not in energy_aspects:

            fig['data'][0].visible = 'legendonly'

        if 'chilledwater' not in energy_aspects:

            fig['data'][1].visible = 'legendonly'

        if 'steam' not in energy_aspects:

            fig['data'][2].visible = 'legendonly'

        if 'hotwater' not in energy_aspects:

            fig['data'][3].visible = 'legendonly'

    fig.update_layout(title='Building ID: {}'.format(building))        

    fig.show()

    display(building_metadata[building_metadata['building_id']==building])
print('Number of days that a building has the maximum electricity consumption of all the buildings:\n')

print(daily_train_agg[daily_train_agg['meter'] == 'electricity']['building_id_max'].value_counts())
daily_train_electricity = daily_train_agg[daily_train_agg['meter']=='electricity'].copy()

daily_train_electricity['building_id_max'] = pd.Categorical(daily_train_electricity['building_id_max'])

fig_daily_electricity = px.scatter(daily_train_electricity,

                                   x='date',

                                   y='meter_reading-max',

                                   color='building_id_max',

                                   render_mode='svg')

fig_daily_electricity.update_layout(title='Maximum consumption values for the day and energy aspect')

fig_daily_electricity.show()
show_building(803, ['electricity'])
show_building(801, ['electricity'])
show_building(799, ['electricity'])
show_building(1088, ['electricity'])
show_building(993, ['electricity'])
show_building(794, ['electricity'])
print('Number of days that a building has the maximum chilledwater consumption of all the buildings:\n')

print(daily_train_agg[daily_train_agg['meter'] == 'chilledwater']['building_id_max'].value_counts())
daily_train_chilledwater = daily_train_agg[daily_train_agg['meter']=='chilledwater'].copy()

daily_train_chilledwater['building_id_max'] = pd.Categorical(daily_train_chilledwater['building_id_max'])

fig_daily_chilledwater = px.scatter(daily_train_chilledwater,

                                    x='date',

                                    y='meter_reading-max',  

                                    color='building_id_max', 

                                    render_mode='svg')

fig_daily_chilledwater.update_layout(title='Maximum consumption values for the day and energy aspect')

fig_daily_chilledwater.show()
show_building(778, ['chilledwater'])
show_building(1088, ['chilledwater'])
print('Number of days that a building has the maximum steam consumption of all the buildings:\n')

print(daily_train_agg[daily_train_agg['meter'] == 'steam']['building_id_max'].value_counts())
daily_train_steam = daily_train_agg[daily_train_agg['meter']=='steam'].copy()

daily_train_steam['building_id_max'] = pd.Categorical(daily_train_steam['building_id_max'])

fig_daily_steam = px.scatter(daily_train_steam,

                             x='date',

                             y='meter_reading-max',

                             color='building_id_max',

                             render_mode='svg')

fig_daily_steam.update_layout(title='Maximum consumption values for the day and energy aspect')

fig_daily_steam.show()
show_building(1099, ['steam'])
show_building(1168, ['steam'])
show_building(1197, ['steam'])
show_building(1148, ['steam'])
print('Number of days that a building has the maximum hotwater consumption of all the buildings:\n')

print(daily_train_agg[daily_train_agg['meter'] == 'hotwater']['building_id_max'].value_counts())
daily_train_hotwater = daily_train_agg[daily_train_agg['meter']=='hotwater'].copy()

daily_train_hotwater['building_id_max'] = pd.Categorical(daily_train_hotwater['building_id_max'])

fig_daily_hotwater = px.scatter(daily_train_hotwater,

                                x='date',

                                y='meter_reading-max',

                                color='building_id_max',

                                render_mode='svg')

fig_daily_hotwater.update_layout(title='Maximum consumption values for the day and energy aspect')

fig_daily_hotwater.show()
show_building(1021, ['hotwater'])
show_building(1331, ['hotwater'])