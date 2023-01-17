# import libraries 
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import re
from datetime import datetime
from fbprophet import Prophet
# import cleaned summary data - found clean data set that updates 24 h 
summary = pd.read_csv("../input/corona-virus-report/covid_19_clean_complete.csv", parse_dates = ['Date']).fillna(0)
# get recent stats
recent_stats = summary.groupby('Date', as_index = False)['Confirmed', 'Deaths', 'Recovered'].sum()
sorted_stats = recent_stats.sort_values(by = 'Date', ascending = False)

# our latest data 
sorted_stats.head(5)
def calculate_percentage_increase(original_number, new_number):
    ''' Calculates the percentage increase given two positive numbers
    '''
    increase = new_number - original_number 
    return "{0:.0f}%".format((increase / original_number) * 100) 
# total confirmed, deaths, and recovered so far 
first_date = sorted_stats.iloc[-1]
most_recent_date = sorted_stats.iloc[0]

confirmed_increase = calculate_percentage_increase(first_date['Confirmed'], most_recent_date['Confirmed'])
deaths_increase = calculate_percentage_increase(first_date['Deaths'], most_recent_date['Deaths'])
recovered_increase = calculate_percentage_increase(first_date['Recovered'], most_recent_date['Recovered'])

increase_summary = pd.DataFrame([[confirmed_increase,deaths_increase,recovered_increase]],columns=['Confirmed','Deaths','Recovered'], index = ['% Increase'])
increase_summary
def calculate_case_fatality(confirmed, death):
    ''' Calculates the case fatality rates given the total number of confirmed and death cases
    '''
    return "{0:.2f}%".format((death / confirmed) * 100)
print("The death-to-case ratio is {}".format(calculate_case_fatality(most_recent_date['Confirmed'], most_recent_date['Deaths'])))
fig = go.Figure() 
fig.add_trace(go.Scatter(
                x= recent_stats['Date'],
                y= recent_stats['Confirmed'],
                name = "Confirmed",
                line_color= "deepskyblue",
                opacity= 0.8))
fig.add_trace(go.Scatter(
                x= recent_stats['Date'],
                y= recent_stats['Deaths'],
                name= "Deaths",
                line_color= "gray",
                opacity= 0.8))
fig.add_trace(go.Scatter(
                x= recent_stats['Date'],
                y= recent_stats['Recovered'],
                name= "Recovered",
                line_color= "deeppink",
                opacity= 0.8))

fig.update_layout(title_text= "Overview of reported confirmed, dead, and recovered cases across countries")

fig.show()
# see most recent totals in terms of country & province
summary['Country/Region'].replace({"Mainland China":"China", "US":"United States"}, inplace = True)
summary_by_country = summary.groupby(["Country/Region", "Province/State"], as_index = False)['Confirmed', 'Deaths', 'Recovered', 'Lat', 'Long']
recent_geo_summary = summary_by_country.last().groupby(['Country/Region']).sum()
# create iso mapping for easy use with plotly
country_codes = pd.read_csv("../input/iso-country-codes-global/wikipedia-iso-country-codes.csv", usecols = ["English short name lower case", "Alpha-3 code"])
country_codes.columns = ['Name', 'Code']
recent_geo_summary = recent_geo_summary.merge(country_codes, left_on = "Country/Region", right_on = "Name")
recent_geo_summary.sort_values("Confirmed", ascending = False).head(10)
map_fig = px.choropleth(recent_geo_summary, locations="Code",
                    color="Confirmed", 
                    hover_name="Name", 
                    color_continuous_scale=px.colors.sequential.Plasma)
map_fig.update_layout(
        title = 'Most affected areas by Geography')
map_fig.show()
china_summary = summary[summary['Country/Region'] == 'China']
china_prov_summary = china_summary.groupby('Province/State', as_index = False).last()
china_fig = go.Figure(data=go.Scattergeo(
        lon = china_prov_summary['Long'],
        lat = china_prov_summary['Lat'],
        text = china_prov_summary['Province/State'],
        mode = 'markers',
        marker_color = china_prov_summary['Confirmed'],
        marker = dict(
            size = 6,
            reversescale = True,
            autocolorscale = False,
            colorscale = 'Bluered_r',
            cmin = 0,
            color = china_prov_summary['Confirmed'],
            cmax = china_prov_summary['Confirmed'].max(),
            colorbar_title="Confirmed cases"
        )))

china_fig.update_layout(
        title = 'Most affected Chinese provinces',
        geo_scope='asia',
    )
china_fig.show()
china = recent_geo_summary["Name"] == 'China'
china_cases = recent_geo_summary[china]['Confirmed'].iloc[0]
world_cases = recent_geo_summary[-china]['Confirmed'].sum()
bar_fig = go.Figure([go.Bar(x=['China', 'Other'], y=[china_cases, world_cases])])
bar_fig.update_traces(marker_color='rgb(255,69,0)', marker_line_color='rgb(255,0,0)',
                  marker_line_width=1.5, opacity=0.6)
bar_fig.update_layout(title_text='China vs. World Confirmed Cases')
bar_fig.show()
iran = summary['Country/Region'] == 'Iran'
italy = summary['Country/Region'] == 'Italy'
united_states = summary['Country/Region'] == 'United States'
iran_summary = summary[iran]
italy_summary = summary[italy]
us_summary = summary[united_states]
i_fig = go.Figure() 
i_fig.add_trace(go.Scatter(
                x= iran_summary['Date'],
                y= iran_summary['Confirmed'],
                name = "Iran",
                line_color= "deepskyblue",
                opacity= 0.8))
i_fig.add_trace(go.Scatter(
                x= italy_summary['Date'],
                y= italy_summary['Confirmed'],
                name= "Italy",
                line_color= "deeppink",
                opacity= 0.8))
i_fig.add_trace(go.Scatter(
                x= us_summary['Date'],
                y= us_summary['Confirmed'],
                name= "United States",
                line_color= "green",
                opacity= 0.8))
i_fig.update_layout(title_text= "Overview of the case growth in Italy, Iran, and United States")

i_fig.show()
patient_info = pd.read_csv("../input/covid19-patientlevel-data/DXY.cn patient level data - Line-list.csv").fillna("NA")
patient_info.columns
gender_fig = px.histogram(patient_info, x="gender")
gender_fig.show()
symptoms = pd.DataFrame(data = patient_info['symptom'].value_counts().head(17)[1:])
words = symptoms.index
weights = symptoms.symptom
word_cloud_data = go.Scatter(x=[4,2,2,3, 1.5, 5, 4, 4,0],
                 y=[2,2,3,3,1, 5,1,3,0],
                 mode='text',
                 text=words,
                 marker={'opacity': 0.5},
                 textfont={'size': weights, 'color':["red", "green", "blue", "purple", "black", "orange", "blue", "black"]})
layout = go.Layout({'xaxis': {'showgrid': False, 'showticklabels': False, 'zeroline': False},
                    'yaxis': {'showgrid': False, 'showticklabels': False, 'zeroline': False}})
word_cloud = go.Figure(data=[word_cloud_data], layout=layout)
word_cloud.update_layout(title_text='Word cloud of most common symptoms by frequency')
word_cloud.show()
# it seems that some recovered and death entries are written as the date of recovery/death instead of 1 indicating 'true'

# create a new cleaned feature for recovered/death data in order to plot
def is_date(value):
    '''
    Returns a boolean indicating whether a given value is a date.
    '''
    regex = re.compile(r'\d{1,2}/\d{1,2}/\d{4}')
    return bool(regex.match(value))
patient_info['clean_recovered'] = patient_info['recovered'].apply(lambda x: '1' if is_date(x) else x)
patient_info['clean_recovered'] = patient_info['clean_recovered'].astype('category')
patient_info['clean_death'] = patient_info['death'].apply(lambda x: '1' if is_date(x) else x)
patient_info['clean_death'] = patient_info['clean_death'].astype('category')
rec_age_fig = make_subplots(rows=1, cols=2, subplot_titles=("Age vs. Recovered", "Age vs. Death"))

rec_age_fig.add_trace(go.Box(x=patient_info['clean_recovered'], y=patient_info['age'], name="Recovered"),
              row=1, col=1)
rec_age_fig.add_trace(go.Box(x=patient_info['clean_death'], y=patient_info['age'], name = "Death"), 
              row=1, col=2)
rec_age_fig.update_traces(boxpoints='all')
rec_age_fig.update_layout(title_text="Subplots of age in relation to recovery and death")
rec_age_fig.show()
total_instances = len(patient_info)
visiting_or_from_wuhan = patient_info['visiting Wuhan'].value_counts()[1] + patient_info['from Wuhan'].value_counts()[1]
not_visiting_or_from_wuhan = total_instances - visiting_or_from_wuhan 
wuhan_summary = pd.DataFrame([visiting_or_from_wuhan, not_visiting_or_from_wuhan],columns = ['Total'], index=['Visiting/From Wuhan', 'Not Visiting/From Wuhan'])
pie_fig = go.Figure(data=[go.Pie(labels=wuhan_summary.index, values=wuhan_summary['Total'], opacity = 0.8)])
pie_fig.show()
# prep data 
time_series_data = summary[['Date', 'Confirmed']].groupby('Date', as_index = False).sum()
time_series_data.columns = ['ds', 'y']
time_series_data.ds = pd.to_datetime(time_series_data.ds)
train_range = np.random.rand(len(time_series_data)) < 0.8
train_ts = time_series_data[train_range]
test_ts = time_series_data[~train_range]
test_ts = test_ts.set_index('ds')
prophet_model = Prophet()
prophet_model.fit(train_ts)
future = pd.DataFrame(test_ts.index)
predict = prophet_model.predict(future)
forecast = predict[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
forecast = forecast.set_index('ds')
test_fig = go.Figure() 
test_fig.add_trace(go.Scatter(
                x= test_ts.index,
                y= test_ts.y,
                name = "Actual Cases",
                line_color= "deepskyblue",
                mode = 'lines',
                opacity= 0.8))
test_fig.add_trace(go.Scatter(
                x= forecast.index,
                y= forecast.yhat,
                name= "Prediction",
                mode = 'lines',
                line_color = 'red',
                opacity= 0.8))
test_fig.add_trace(go.Scatter(
                x= forecast.index,
                y= forecast.yhat_lower,
                name= "Prediction Lower Bound",
                mode = 'lines',
                line = dict(color='gray', width=2, dash='dash'),
                opacity= 0.8))
test_fig.add_trace(go.Scatter(
                x= forecast.index,
                y= forecast.yhat_upper,
                name= "Prediction Upper Bound",
                mode = 'lines',
                line = dict(color='royalblue', width=2, dash='dash'),
                opacity = 0.8
                ))

test_fig.update_layout(title_text= "Prophet Model's Test Prediction",
                       xaxis_title="Date", yaxis_title="Cases",)

test_fig.show()
def calculate_mse(actual, predicted):
    '''Calculates the Mean Squared Error given estimated and actual values.
    '''
    errors = 0
    n = len(actual)
    for i in range(n):
        errors += (actual[i] - predicted[i]) **2
    return errors / n
print("The MSE for the Prophet time series model is {}".format(calculate_mse(test_ts.y, forecast.yhat)))
prophet_model_full = Prophet()
prophet_model_full.fit(time_series_data)
future_full = prophet_model_full.make_future_dataframe(periods=150)
forecast_full = prophet_model_full.predict(future_full)
forecast_full = forecast_full.set_index('ds')
prediction_fig = go.Figure() 
prediction_fig.add_trace(go.Scatter(
                x= time_series_data.ds,
                y= time_series_data.y,
                name = "Actual",
                line_color= "red",
                opacity= 0.8))
prediction_fig.add_trace(go.Scatter(
                x= forecast_full.index,
                y= forecast_full.yhat,
                name = "Prediction",
                line_color= "deepskyblue",
                opacity= 0.8))
prediction_fig.update_layout(title_text= "Prophet Model Forecasting", 
                             xaxis_title="Date", yaxis_title="Cases",)

prediction_fig.show()