import numpy as np
import pandas as pd
from google.cloud import bigquery #For BigQuery
from bq_helper import BigQueryHelper #For BigQuery
us_traffic = BigQueryHelper("bigquery-public-data", "nhtsa_traffic_fatalities")
us_traffic.head("accident_2015")
accidents_query = """SELECT longitude, latitude, number_of_fatalities, timestamp_of_crash
                     FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
                     WHERE number_of_drunk_drivers > 0
                     AND longitude < 0
                     AND longitude > -140
                     AND month_of_crash = 12 """ 
accidents_latlong = us_traffic.query_to_pandas(accidents_query)
import datetime
accidents_latlong['timestamp_of_crash'] = accidents_latlong['timestamp_of_crash'].apply(lambda x: x.strftime("%Y-%m-%d %H:%M"))
#Ref: https://plot.ly/python/scatter-plots-on-maps/
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode()

data = [ dict(
        type = 'scattergeo',
        locationmode = 'USA-states',
        lon = accidents_latlong['longitude'],
        lat = accidents_latlong['latitude'],
        text = accidents_latlong['timestamp_of_crash'],
        mode = "markers",
        marker = dict(
            size = accidents_latlong['number_of_fatalities']*10,
            opacity = 0.8,
        ))]

layout = dict(
        title = 'US Fatalities by Location (December 2016, Drunk Drivers)',
        colorbar = True,
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showland = True,
            landcolor = "rgb(250, 250, 250)",
            subunitcolor = "rgb(217, 217, 217)",
            countrycolor = "rgb(217, 217, 217)",
            countrywidth = 0.5,
            subunitwidth = 0.5
        ),
    )

fig = dict(data=data, layout=layout)
iplot(fig, validate=False, filename='fatalties')
us_traffic_crashes_by_month = us_traffic.query_to_pandas_safe("""
    SELECT month_of_crash, count(month_of_crash) AS months_totals
    FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
    GROUP BY month_of_crash
    ORDER BY months_totals DESC
""")
us_traffic_crashes_by_month
us_traffic_fatalities = us_traffic.query_to_pandas_safe("""
    SELECT number_of_fatalities
    FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
""")
x = us_traffic_fatalities['number_of_fatalities']
data = [go.Histogram(x=x)]

layout = go.Layout(
    title='Number of Fatalities Per Incident',
    yaxis=dict(
        title='Count'
    ),
    xaxis=dict(
        title='Number of Fatalities'
    ),
    bargap=0.2,
    bargroupgap=0.1
)
fig = go.Figure(data=data, layout=layout)
iplot(fig, validate=False, filename='fatalties')
us_traffic_fatality_by_month = us_traffic.query_to_pandas_safe("""
    SELECT month_of_crash, sum(number_of_fatalities) AS months_fatalities_totals
    FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
    GROUP BY month_of_crash
    ORDER BY months_fatalities_totals DESC
""")
us_traffic_fatality_by_month
us_traffic_fatality_by_state = us_traffic.query_to_pandas_safe("""
    SELECT state_name, 
    sum(number_of_fatalities) AS fatality_total
    FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
    GROUP BY state_name
    ORDER BY fatality_total DESC
""")
us_traffic_fatality_by_state
us_traffic.head("cevent_2016")
us_traffic_events = us_traffic.query_to_pandas_safe("""
    SELECT sequence_of_events_name, 
    count(sequence_of_events_name) AS events_counts
    FROM `bigquery-public-data.nhtsa_traffic_fatalities.cevent_2016`
    GROUP BY sequence_of_events_name
    ORDER BY events_counts DESC
    LIMIT 10
""")
us_traffic_events
us_traffic_factors = us_traffic.query_to_pandas_safe("""
    SELECT contributing_circumstances_motor_vehicle_name
    FROM `bigquery-public-data.nhtsa_traffic_fatalities.factor_2016`
""")
us_traffic_factors_counts = us_traffic_factors.groupby(['contributing_circumstances_motor_vehicle_name']).agg('contributing_circumstances_motor_vehicle_name').count().sort_values(ascending = False)
us_traffic_factors_counts
us_traffic_vision = us_traffic.query_to_pandas_safe("""
    SELECT drivers_vision_obscured_by_name
    FROM `bigquery-public-data.nhtsa_traffic_fatalities.vision_2016`
""")
us_traffic_vision.groupby(['drivers_vision_obscured_by_name']).agg('drivers_vision_obscured_by_name').count().sort_values(ascending = False)
us_traffic_vision_by_month = us_traffic.query_to_pandas_safe("""
    SELECT a.drivers_vision_obscured_by_name, a.state_number AS vision_state, b.state_number, b.state_name, b.month_of_crash
    FROM `bigquery-public-data.nhtsa_traffic_fatalities.vision_2016` a
    JOIN `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016` b
    ON a.state_number = b.state_number
    WHERE b.month_of_crash = 12
    AND number_of_drunk_drivers > 0
    AND a.drivers_vision_obscured_by_name != 'No Obstruction Noted'
    AND a.drivers_vision_obscured_by_name != 'Unknown'
""")
us_traffic_vision_by_month.groupby(['state_name', 'drivers_vision_obscured_by_name']).agg('state_name').count().sort_values(ascending = False).head(15)
state_chronic = pd.read_csv('../input/chronic-disease/U.S._Chronic_Disease_Indicators.csv')
state_chronic.head(5)
state_chronic_topics = state_chronic.groupby(['Topic']).agg('Topic').count().sort_values(ascending = False)
state_chronic_topics.head(20)
state_chronic_alcohol = state_chronic[state_chronic['Topic'] == 'Alcohol'].groupby(['Question']).agg('Question').count().sort_values(ascending = False)
state_chronic_alcohol.head(20)
state_chronic_2015 = state_chronic[(state_chronic['YearStart'] == 2015) & (state_chronic['Question'] == 'Alcohol use among youth')]
state_chronic_2015.head(5)
state_chronic_2015_value = state_chronic_2015.groupby(['LocationDesc','DataValue']).mean()
state_chronic_2015_value.sortlevel('DataValue', ascending = False)
merged = pd.merge(us_traffic_fatality_by_state, state_chronic_2015, left_on='state_name', right_on='LocationDesc')
merged.head()
fatalities_and_alcohol = merged.groupby(['LocationDesc','DataValue']).mean()
fatalities_and_alcohol = fatalities_and_alcohol.reset_index() #Tidy up the column headers
fatalities_and_alcohol.head()
from scipy import stats

slope, intercept, r_value, p_value, std_err = stats.linregress(fatalities_and_alcohol['DataValue'],fatalities_and_alcohol['fatality_total'])
line = slope*fatalities_and_alcohol['DataValue']+intercept

trace1 = go.Scatter(
    x = fatalities_and_alcohol['DataValue'],
    y = fatalities_and_alcohol['fatality_total'],
    text = fatalities_and_alcohol['LocationDesc'],
    mode = "markers")

trace2 = go.Scatter(
    x=fatalities_and_alcohol['DataValue'],
    y=line,
    mode='lines',
    hoverinfo='none',
    marker=go.Marker(color='red'),
    name='Fit'
    )

layout = go.Layout(
    title = 'Alcohol Use Among Youth vs Number of Fatalities by State',
    xaxis=go.XAxis(title = 'Alcohol Use Among Youth (%)'),
    yaxis=go.XAxis(title = 'Number of Fatalities'),
    showlegend=False
)

data = [trace1, trace2]
fig = go.Figure(data=data, layout=layout)
iplot(fig, validate=False, filename='alcohol')
election = pd.read_csv('../input/2016-us-election/primary_results.csv')
election.head()
results_by_state_candidate = election.groupby(['state', 'candidate']).agg('votes').sum()
results_by_state_candidate = results_by_state_candidate.to_frame()
results_by_state_candidate = results_by_state_candidate.reset_index()
results_by_state_candidate.head(10)
results_by_state_candidate_top = results_by_state_candidate.groupby(['state']).agg('votes').idxmax()
results_by_state_candidate_top = results_by_state_candidate_top.to_frame()
votes = results_by_state_candidate_top.reset_index()
results_by_state_candidate_top = results_by_state_candidate.iloc[results_by_state_candidate_top['votes'].tolist(),:]
results_by_state_candidate_top.head()
fatalities_election = pd.merge(us_traffic_fatality_by_state, results_by_state_candidate_top, left_on='state_name', right_on='state')
fatalities_election.head(10)
y0 = fatalities_election['fatality_total'][fatalities_election['candidate'] == 'Hillary Clinton']
y1 = fatalities_election['fatality_total'][fatalities_election['candidate'] == 'Donald Trump']
y2 = fatalities_election['fatality_total'][fatalities_election['candidate'] == 'Bernie Sanders']
y3 = fatalities_election['fatality_total'][fatalities_election['candidate'] == 'Ted Cruz']
y4 = fatalities_election['fatality_total'][fatalities_election['candidate'] == 'John Kasich']

trace0 = go.Box(
    name = 'Hillary Clinton',
    y=y0
)
trace1 = go.Box(
    name = 'Donald Trump',
    y=y1
)
trace2 = go.Box(
    name = 'Bernie Sanders',
    y=y2
)
trace3 = go.Box(
    name = 'Ted Cruz',
    y=y3
)
trace4 = go.Box(
    name = 'John Kasich',
    y=y4
)

layout = go.Layout(
    title = 'Votes vs Fatalities',
    xaxis=go.XAxis(title = 'Votes'),
    yaxis=go.XAxis(title = 'Number of Fatalities'),
    showlegend=False
)

data = [trace0, trace1, trace2, trace3, trace4]
fig = go.Figure(data=data, layout=layout)
iplot(fig, validate=False, filename='votes')
vege = pd.read_csv('../input/vegetarian-vegan-restaurants/vegetarian_restaurants_US_datafiniti.csv')
vege = vege[vege.cuisines.notnull()]
vegan = vege[vege['cuisines'].str.contains("Vegan")]
vegan.head()
vegan_count_by_state = vegan.groupby(['province']).agg('address').count()
vegan_count_by_state = vegan_count_by_state.to_frame()
vegan_count_by_state = vegan_count_by_state.reset_index()
vegan_count_by_state.columns = ['province', 'count']
vegan_count_by_state.head()
#Ref: https://gist.github.com/rogerallen/1583593

us_state_abbrev = {
    'Alabama': 'AL',
    'Alaska': 'AK',
    'Arizona': 'AZ',
    'Arkansas': 'AR',
    'California': 'CA',
    'Colorado': 'CO',
    'Connecticut': 'CT',
    'Delaware': 'DE',
    'Florida': 'FL',
    'Georgia': 'GA',
    'Hawaii': 'HI',
    'Idaho': 'ID',
    'Illinois': 'IL',
    'Indiana': 'IN',
    'Iowa': 'IA',
    'Kansas': 'KS',
    'Kentucky': 'KY',
    'Louisiana': 'LA',
    'Maine': 'ME',
    'Maryland': 'MD',
    'Massachusetts': 'MA',
    'Michigan': 'MI',
    'Minnesota': 'MN',
    'Mississippi': 'MS',
    'Missouri': 'MO',
    'Montana': 'MT',
    'Nebraska': 'NE',
    'Nevada': 'NV',
    'New Hampshire': 'NH',
    'New Jersey': 'NJ',
    'New Mexico': 'NM',
    'New York': 'NY',
    'North Carolina': 'NC',
    'North Dakota': 'ND',
    'Ohio': 'OH',
    'Oklahoma': 'OK',
    'Oregon': 'OR',
    'Pennsylvania': 'PA',
    'Rhode Island': 'RI',
    'South Carolina': 'SC',
    'South Dakota': 'SD',
    'Tennessee': 'TN',
    'Texas': 'TX',
    'Utah': 'UT',
    'Vermont': 'VT',
    'Virginia': 'VA',
    'Washington': 'WA',
    'West Virginia': 'WV',
    'Wisconsin': 'WI',
    'Wyoming': 'WY',
}
us_state_abbrev_df = pd.DataFrame(list(us_state_abbrev.items()), columns=['state', 'abb'])
vegan_with_states = pd.merge(vegan_count_by_state, us_state_abbrev_df, left_on='province', right_on='abb')
fatalities_vegan = pd.merge(us_traffic_fatality_by_state, vegan_with_states, left_on='state_name', right_on='state')
fatalities_vegan.head()
slope, intercept, r_value, p_value, std_err = stats.linregress(fatalities_vegan['count'],fatalities_vegan['fatality_total'])
line = slope*fatalities_vegan['count']+intercept

trace1 = go.Scatter(
    x = fatalities_vegan['count'],
    y = fatalities_vegan['fatality_total'],
    text = fatalities_vegan['state_name'],
    mode = "markers")

trace2 = go.Scatter(
    x=fatalities_vegan['count'],
    y=line,
    mode='lines',
    hoverinfo='none',
    marker=go.Marker(color='red'),
    name='Fit'
    )

layout = go.Layout(
    title = 'Number of Vegan Food Outlets vs Number of Fatalities by State',
    xaxis=go.XAxis(title = 'Number of Vegan Food Outlets'),
    yaxis=go.XAxis(title = 'Number of Fatalities'),
    showlegend=False
)

data = [trace1, trace2]
fig = go.Figure(data=data, layout=layout)
iplot(fig, validate=False, filename='vegan')
pop = pd.read_csv('../input/2016-us-election/county_facts.csv')
pop_state = pop.groupby(['state_abbreviation']).agg('PST045214').sum()
pop_state = pop_state.to_frame()
pop_state = pop_state.reset_index()
pop_state['PST045214'].sum()
pop_state_name = pd.merge(us_state_abbrev_df, pop_state, left_on='abb', right_on='state_abbreviation')
fatalities_with_pop = pd.merge(us_traffic_fatality_by_state, pop_state_name, left_on='state_name', right_on='state')
fatalities_with_pop = fatalities_with_pop.drop(['abb', 'state', 'state_abbreviation'], axis=1)
fatalities_with_pop['Fatalities Per Capita'] = fatalities_with_pop['fatality_total'] / fatalities_with_pop['PST045214']
fatalities_with_pop['Fatalities Per 100k'] = fatalities_with_pop['Fatalities Per Capita'] * 100000
fatalities_with_pop.sort_values('Fatalities Per 100k', ascending=False)