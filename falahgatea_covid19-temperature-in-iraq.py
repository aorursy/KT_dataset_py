import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/all-cases-weather/merge_all_casses_with_weather.csv')
import plotly.graph_objects as go

grouped_multiple = data.groupby(['date']).agg({'confirmed': ['sum']})

grouped_multiple.columns = ['confirmed']

grouped_multiple = grouped_multiple.reset_index()

fig = go.Figure()

fig.update_layout(template='plotly_dark')

fig.add_trace(go.Scatter(x=grouped_multiple['date'], 

                         y=grouped_multiple['confirmed'],

                         mode='lines+markers',

                         name='deaths',

                         line=dict(color='orange', width=2)))

fig.show()
#cases

cases = ['confirmed', 'deaths', 'recovered','tempC']

data['country'] = data['country'].replace('iraq', 'Iraq')
# iraq and the row

iraq_df = data[data['country']=='Iraq']

iraq_df
iraq_df.to_csv('iraq_all_casses_temp.csv')
iraq_df['deaths']
iraq_df['confirmed']
iraq_df['recovered']
iraq_df['active']

iraq_df['tempC']
Iraq_conf_cases = iraq_df.groupby('date').sum().apply(list).reset_index()

Iraq_conf_cases
i=1

iraq_df['Days'] = 1

for ind in iraq_df.index: 

    iraq_df['Days'][ind] = i

    i=i+1   
iraq_df
x=iraq_df['tempC']

y=iraq_df['active']

raw_data = {'TempC': x,'Active':y}

#x=np.array(1,2,3,4,5,6,7,8,9,10,11,12,13)

df_Iraq = pd.DataFrame(raw_data, columns = ['TempC', 'Active'])

df_Iraq
import numpy as np

import matplotlib.pyplot as plt

np.random.seed(0)

plt.scatter(x,y, s=10)

plt.show()

import plotly.graph_objects as go

grouped_multiple = iraq_df.groupby(['tempC']).agg({'active': ['sum']})

grouped_multiple.columns = ['active']

grouped_multiple = grouped_multiple.reset_index()

fig = go.Figure()

fig.update_layout(template='plotly_dark',title="COVID19 Iraq Active Count With Temperature  ",xaxis_title="Temperature C",

    yaxis_title="Active Count",)

fig.add_trace(go.Scatter(x=grouped_multiple['tempC'], 

                         y=grouped_multiple['active'],

                         mode='lines+markers',

                         name='active',

                         line=dict(color='orange', width=2)))

fig.show()
import plotly.graph_objects as go

grouped_multiple = iraq_df.groupby(['tempC']).agg({'deaths': ['sum']})

grouped_multiple.columns = ['deaths']

grouped_multiple = grouped_multiple.reset_index()

fig = go.Figure()

fig.update_layout(template='plotly_dark',title="COVID19 Iraq Deaths Count With Temperature  ",xaxis_title="Temperature C",

    yaxis_title="Deaths Count",)

fig.add_trace(go.Scatter(x=grouped_multiple['tempC'], 

                         y=grouped_multiple['deaths'],

                         mode='lines+markers',

                         name='deaths',

                         line=dict(color='orange', width=2)))

fig.show()
import plotly.graph_objects as go

grouped_multiple = iraq_df.groupby(['tempC']).agg({'confirmed': ['sum']})

grouped_multiple.columns = ['confirmed']

grouped_multiple = grouped_multiple.reset_index()

fig = go.Figure()

fig.update_layout(template='plotly_dark',title="COVID19 Iraq Confirmed Count With Temperature  ",xaxis_title="Temperature C",

    yaxis_title="Confirmed Count",)

fig.add_trace(go.Scatter(x=grouped_multiple['tempC'], 

                         y=grouped_multiple['confirmed'],

                         mode='lines+markers',

                         name='confirmed',

                         line=dict(color='orange', width=2)))

fig.show()
import plotly.graph_objs as go

import plotly.express as px

fig = px.scatter_geo(iraq_df.fillna(0), locations="country", locationmode='country names', 

                     color="tempC", size='confirmed', hover_name="country", 

                     range_color= [-20, 45], 

                     projection="natural earth", animation_frame="date", 

                     title='COVID-19: Confirmed VS Temperature by Iraq Country', color_continuous_scale="portland")

# fig.update(layout_coloraxis_showscale=False)

fig.show()