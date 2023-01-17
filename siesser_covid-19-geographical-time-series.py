import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly.graph_objects as go
from ipywidgets import widgets, Layout, interact, fixed
from datetime import date
import pickle

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
print("UPDATED: {}".format(date.today()))
# read in the data
data = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv", parse_dates=['ObservationDate'])
max_date = data['ObservationDate'].max()


# pull out countries and states
countries = sorted(data['Country/Region'].unique())
states = sorted([s for s in data['Province/State'].unique() if isinstance(s, str)])
# create graph dropdowns for geography

country_dropdown = widgets.Dropdown(
    options=countries,
    value='US',
    description='Country/Region',
    style={'description_width': 'initial'}
)

state_dropdown = widgets.Dropdown(
    options=sorted(
        [s for s in data[data['Country/Region']=='US']['Province/State'].unique() if isinstance(s,str)]
    ),
    value='Georgia',
    description='Province/State',
    style={'description_width': 'initial'}
)

cum_dropdown = widgets.Dropdown(
    options=['Cumulative', 'New'],
    value='New',
    description='Cumulative or New',
    style={'description_width': 'initial'}
)

type_dropdown = widgets.Dropdown(
    options=['Confirmed', 'Deaths', 'Recovered'],
    value='Confirmed',
    description='Series Type',
    style={'description_width': 'initial'}
)

rolling_avg_window = widgets.IntSlider(
    value=7,
    min=1,
    max=14,
    step=1,
    description='Rolling Avg # Days):',
    style={'description_width': 'initial'}
)


# update the local dropdown based regional selection
def update_state_dropdown(*args):
    country = country_dropdown.value
    state_dropdown.options = sorted(
        [s for s in data[data['Country/Region']==country]['Province/State'].unique() if isinstance(s,str)]
    )
    
country_dropdown.observe(update_state_dropdown, 'value')

# plot the results
@interact
def create_geo_timeseries(country=country_dropdown, state=state_dropdown, data=fixed(data), cum=cum_dropdown, t=type_dropdown,
                         r=rolling_avg_window):
    """ Creates a timeseries plot of COVID cases for a specific geography
    
    :param country: country for the time series
    :param state: state for the time series
    :return: plotly figure
    """
    
    data = data.groupby(['Country/Region', 'Province/State', 'ObservationDate']).sum().reset_index()
    
    # these are the three traces we will plot for each geography
    types=['Confirmed', 'Deaths', 'Recovered']
    colors={
        'Confirmed': 'orange',
        'Deaths': 'red',
        'Recovered': 'green'
    }
    
    return go.FigureWidget(
        data=[
            go.Scatter(
            #go.Bar(
                name=t,
                x=data[data['Province/State'] == state]['ObservationDate'],
                y=data[data['Province/State'] == state][t].diff().rolling(r).mean() \
                    if cum == 'New' else data[data['Province/State'] == state][t],
                marker_color=colors[t]
            )
            #for t in types
        ],
        layout=go.Layout(
            title='{}, {} - {} COVID-19 {} {}'.format(
                state, 
                country, 
                cum if cum == 'Cumulative' else 'Daily', 
                'Confirmed Cases' if t == 'Confirmed' else t, 
                '' if cum == 'Cumulative' else '<br> (Rolling {} Day Average)'.format(r)
            ),
            template='presentation',
            #yaxis_type='log'
        )
    )

us_states=["Alabama","Alaska","Arizona","Arkansas","California","Colorado",
  "Connecticut","Delaware","Florida","Georgia","Hawaii","Idaho","Illinois",
  "Indiana","Iowa","Kansas","Kentucky","Louisiana","Maine","Maryland",
  "Massachusetts","Michigan","Minnesota","Mississippi","Missouri","Montana",
  "Nebraska","Nevada","New Hampshire","New Jersey","New Mexico","New York",
  "North Carolina","North Dakota","Ohio","Oklahoma","Oregon","Pennsylvania",
  "Rhode Island","South Carolina","South Dakota","Tennessee","Texas","Utah",
  "Vermont","Virginia","Washington","West Virginia","Wisconsin","Wyoming"]

# select only the US states (so we don't double count by including the cities and counties)
state_data = data[data['Province/State'].isin(us_states)].groupby(['Province/State', 'ObservationDate']).sum().reset_index()

# look at the data only after cases started picking up in the US
state_data = state_data[state_data['ObservationDate'] >= pd.to_datetime('03/10/2020')]

# rank the states by number of confirmed cases
states_ranked = state_data[state_data['ObservationDate']==max_date].groupby(['Province/State']).sum().reset_index().sort_values(
    by='Confirmed', ascending=False)['Province/State'].to_list()


y_axis_scale = widgets.ToggleButtons(
    options=[('Linear','-'), ('Logarithmic','log')],
    description='Y-Axis Scale:',
    
    #button_style='info', # 'success', 'info', 'warning', 'danger' or ''
)

data_category = widgets.ToggleButtons(
    options=['Confirmed','Deaths'],
    description='Select Data:',
)


@interact 
def all_states(y_axis_scale=y_axis_scale, data_category=data_category):
        
    return go.FigureWidget(
            data=[
                # stacked bars for the states
                go.Bar(
                    name=state,
                    x=state_data[state_data['Province/State'] == state]['ObservationDate'],
                    y=state_data[state_data['Province/State'] == state][data_category]
                )
                for state in reversed(states_ranked)
            ] + [

                # line with sum
                go.Scatter(
                    name='US Total',
                    x=state_data.groupby('ObservationDate').sum().index,
                    y=state_data.groupby('ObservationDate').sum()[data_category],
                    #text=state_data.groupby('ObservationDate').sum()[data_category],
                    #textposition='top center',
                    mode='lines+markers'#+text'
                )
            ],
            layout=go.Layout(
                title='US {} COVID-19 Cases - Ranked by State'.format(data_category),
                yaxis_title='{} Cases'.format(data_category),
                yaxis_type=y_axis_scale,
                template='presentation',
                barmode='stack'
            )
        )
us_states=["Alabama","Alaska","Arizona","Arkansas","California","Colorado",
  "Connecticut","Delaware","Florida","Georgia","Hawaii","Idaho","Illinois",
  "Indiana","Iowa","Kansas","Kentucky","Louisiana","Maine","Maryland",
  "Massachusetts","Michigan","Minnesota","Mississippi","Missouri","Montana",
  "Nebraska","Nevada","New Hampshire","New Jersey","New Mexico","New York",
  "North Carolina","North Dakota","Ohio","Oklahoma","Oregon","Pennsylvania",
  "Rhode Island","South Carolina","South Dakota","Tennessee","Texas","Utah",
  "Vermont","Virginia","Washington","West Virginia","Wisconsin","Wyoming"]


y_axis_scale3 = widgets.ToggleButtons(
    options=[('Linear','-'), ('Logarithmic','log')],
    description='Y-Axis Scale:',
)

data_category3 = widgets.ToggleButtons(
    options=['Confirmed','Deaths'],
    description='Select Data:',
)


@interact 
def us_new_cases(y_axis_scale=y_axis_scale3, data_category=data_category3):
        
    return go.FigureWidget(
            data=[
                # line with sum
                go.Scatter(
                    name='US Total',
                    x=state_data.groupby('ObservationDate').sum().index,
                    y=state_data.groupby('ObservationDate').sum()[data_category] \
                        .diff() \
                        .rolling(7) \
                        .mean()#,
#                     mode='lines+markers'#+text'
                )
            ],
            layout=go.Layout(
                title='US Daily Daily COVID {}'.format(
                    'Confirmed Cases' if data_category == 'Confirmed' else data_category),
                yaxis_title='{} Cases'.format(data_category),
                yaxis_type=y_axis_scale,
                template='presentation',
                barmode='stack'
            )
        )
# first rank countries by cases 
top_countries = data[data['ObservationDate']==max_date].groupby(['Country/Region']).sum().reset_index().sort_values(
    by='Confirmed', ascending=False)['Country/Region'].to_list()

by_country = data.groupby(['Country/Region', 'ObservationDate']).sum().reset_index()


y_axis_scale2 = widgets.ToggleButtons(
    options=[('Linear','-'), ('Logarithmic','log')],
    description='Y-Axis Scale:',
)

data_category2 = widgets.ToggleButtons(
    options=['Confirmed','Deaths'],
    description='Select Data:',
)

cum_dropdown2 = widgets.Dropdown(
    options=['Cumulative', 'New'],
    value='New',
    description='Cumulative or New',
    style={'description_width': 'initial'}
)

rolling_avg_window2 = widgets.IntSlider(
    value=7,
    min=1,
    max=14,
    step=1,
    description='Rolling Avg # Days):',
    style={'description_width': 'initial'}
)


@interact
def all_countries(
    y_axis_scale=y_axis_scale2, 
    data_category=data_category2): #, 
#     cum=cum_dropdown2, 
#     r=rolling_avg_window2):
    
    return go.FigureWidget(
            data=[
                go.Bar(
                    name=c,
                    x=by_country[by_country['Country/Region'] == c]['ObservationDate'],
                    y=by_country[by_country['Country/Region'] == c][data_category]
                )
                for (i,c) in enumerate(reversed(top_countries)) if i >= (len(top_countries) - 50)
                #for c in reversed(top_countries)
            ] + [
                go.Scatter(
                    name='Global Total',
                    x=by_country.groupby('ObservationDate').sum().index,
                    y=by_country.groupby('ObservationDate').sum()[data_category],

                    #text=by_country.groupby('ObservationDate').sum()[data_category],
                    #textposition='top center',
                    mode='lines+markers'#+text'
                )
            ],
            layout=go.Layout(
                title='Global COVID-19 {} Cases - Ranked by Country'.format(data_category),
                template='presentation',
                barmode='stack',
                yaxis_type=y_axis_scale
            )
        )

