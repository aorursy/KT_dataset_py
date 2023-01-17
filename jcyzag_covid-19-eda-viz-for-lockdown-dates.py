import plotly.express as px
from pandas import Timestamp
import numpy as np
import pandas as pd

dfLockdownDates = pd.read_csv('/kaggle/input/covid19-lockdown-dates-by-country/countryLockdowndatesJHUMatch.csv', parse_dates=['Date'])
dfLockdownDates.head()
dfConfirmedCases = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv', parse_dates=['ObservationDate'])
dfConfirmedCases = dfConfirmedCases.groupby(['ObservationDate', 'Country/Region'])['Confirmed'].sum().reset_index()
# dfFullGB = dfFullGB.sort_values('Confirmed',ascending=False, axis=0)#.reset_index()
dfConfirmedCases = dfConfirmedCases.rename(columns={'index': 'Country/Region'})
# get a sorted list of countries based of number of cases per country
def getListofSortedCountries(dfRaw, thresholdCases=1):
    dfSorted = dfRaw.groupby(['ObservationDate','Country/Region'])['Confirmed'].sum().reset_index()
    dfSorted = dfSorted.groupby(['Country/Region'])['Confirmed'].max().reset_index()
    dfSorted= dfSorted.sort_values('Confirmed',ascending=False, axis=0).reset_index()
    dfSorted = dfSorted[dfSorted['Confirmed']>=thresholdCases]
    countries = dfSorted['Country/Region'].unique()
    return countries

countries = getListofSortedCountries(dfConfirmedCases)
# pivot john hopkins data, each column will be a country total countrs, country new cases
def getPivotedCountryData(dfRaw, columns):
    dfFullPivot = dfRaw.pivot_table('Confirmed',['ObservationDate'], 'Country/Region').reset_index()
    for column in columns:
        dfFullPivot[column+'NewCases'] = dfFullPivot[column].diff()

    dfFullPivot = dfFullPivot.reindex(sorted(dfFullPivot.columns), axis=1)
    return dfFullPivot

dfFullPivot = getPivotedCountryData(dfConfirmedCases, countries)
dfFullPivot.head()
lockdownDate= dfLockdownDates[dfLockdownDates['Country/Region']=='Italy']['Date'].values[0]
dfFullPivot[dfFullPivot['ObservationDate']==lockdownDate]['Italy'].values[0]
def showCountry(country, countryToShow):
    if country not in countryToShow:
        return 'legendonly'
    else:
        return True
fig = px.line(title="Number of cases - lockdown highlighted with a marker - click legend to add/remove country")
colors = [
    '#1f77b4',  # muted blue
    '#ff7f0e',  # safety orange
    '#2ca02c',  # cooked asparagus green
    '#d62728',  # brick red
    '#9467bd',  # muted purple
    '#8c564b',  # chestnut brown
    '#e377c2',  # raspberry yogurt pink
    '#7f7f7f',  # middle gray
    '#bcbd22',  # curry yellow-green
    '#17becf'   # blue-teal
]

i = 0
confirmedCases = 0
for country in countries:
    color = colors[i]
    i+=1
    if i == 10:
        i=0

    if country != 'Others':
        fig.add_scatter(x=dfFullPivot['ObservationDate'], y=dfFullPivot[country], 
                        mode='lines', 
                        name= country, 
                        legendgroup=country,
                        visible = showCountry(country,'UK,US,Spain,Italy,Germany,France,Turkey,Iran'),
                        line_color=color)

        try:
            lockdownDate = Timestamp(dfLockdownDates[dfLockdownDates['Country/Region']==country]['Date'].values[0])
            confirmedCases = dfFullPivot[dfFullPivot['ObservationDate']==lockdownDate][country].values[0]
            fig.add_scatter(x=[lockdownDate], y=[confirmedCases]
                            ,mode='markers',  
                            name=country,
                            legendgroup=country,
                            visible = showCountry(country,'UK,US,Spain,Italy,Germany,France,Turkey,Iran'),
                            line_color=color)
        except Exception as e:
#             print(e)
            confirmedCases = confirmedCases
    
fig.update_layout( yaxis_type="log")
# fig.update_layout( xaxis_type="log")
fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Total confirmed Cases",
)
fig.show()
fig = px.line(title="New Cases - Lockdown highlighted with a marker - click legend to add/remove country")
colors = [
    '#1f77b4',  # muted blue
    '#ff7f0e',  # safety orange
    '#2ca02c',  # cooked asparagus green
    '#d62728',  # brick red
    '#9467bd',  # muted purple
    '#8c564b',  # chestnut brown
    '#e377c2',  # raspberry yogurt pink
    '#7f7f7f',  # middle gray
    '#bcbd22',  # curry yellow-green
    '#17becf'   # blue-teal
]

i = 0
for country in countries:
    color = colors[i]
    i+=1
    if i == 10:
        i=0

    if country == 'Others':
        break
    
    fig.add_scatter(x=dfFullPivot['ObservationDate'], y=dfFullPivot[country+'NewCases'], 
                    mode='lines', 
                    name= country, 
                    visible = showCountry(country,'UK,US,Spain,Italy,Germany,France,Turkey,Iran'),
                    legendgroup=country,
                    line_color=color)
    
    try:
        lockdownDate = Timestamp(dfLockdownDates[dfLockdownDates['Country/Region']==country]['Date'].values[0])
        confirmedCases = dfFullPivot[dfFullPivot['ObservationDate']==lockdownDate][country+'NewCases'].values[0]
        fig.add_scatter(x=[lockdownDate], y=[confirmedCases]
                        ,mode='markers',  
                        name=country,
                        visible = showCountry(country,'UK,US,Spain,Italy,Germany,France,Turkey,Iran'),
                        legendgroup=country,
                        line_color=color)
    except Exception as e:
#         print('no lock down', country)
        confirmedCases = confirmedCases
    
# fig.update_layout( yaxis_type="log")
# fig.update_layout( xaxis_type="log")
fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Daily New Cases",
)
fig.show()

