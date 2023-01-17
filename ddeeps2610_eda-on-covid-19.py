# Imports

# Visualization

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

import plotly.graph_objs as go

from plotly.subplots import make_subplots





# hide warnings

import warnings

warnings.filterwarnings('ignore')



# Load the dataset

outbreak = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv',parse_dates=['ObservationDate'])



# Data cleaning

# Rename the columns for easy use

outbreak = outbreak.rename(columns={'Country/Region':'Country','ObservationDate':'Date','Province/State':'State'})



# Combine the counts from China.

outbreak.Country = outbreak.Country.str.replace('Mainland China','China')



outbreakOverall = outbreak.groupby('Date')['Confirmed','Deaths','Recovered'].sum()

outbreakOverall.reset_index(inplace=True)

outbreakOverall['Active'] = outbreakOverall.Confirmed - outbreakOverall.Deaths - outbreakOverall.Recovered



latest = outbreakOverall[outbreakOverall.Date == max(outbreakOverall.Date)]

latest.style.background_gradient(cmap='Reds',axis=1)
# Compute the first degree of differences

outbreakOverall['New_Confirmed'] = outbreakOverall.Confirmed.diff()

outbreakOverall['New_Recovered'] = outbreakOverall.Recovered.diff()

outbreakOverall['New_Deaths']    = outbreakOverall.Deaths.diff()



# Let's get the axes

fig = plt.figure(figsize=(10,10))

ax1 = fig.add_subplot(211)

ax2 = fig.add_subplot(212)



# Plot the charts

# ax1 : Confirmed cases, deaths and recovered cases by Date

outbreakOverall.plot(x='Date',y=['Confirmed','Deaths','Recovered'],kind='line', ax=ax1)



# ax2 : New Confirmations per day

outbreakOverall.plot(x='Date',y=['New_Confirmed'],kind='bar', ax=ax2,

                     title='Upshot in new cases on 02/13 is in China which is due to stricter reporting requirements')



plt.subplots_adjust(hspace = 0.5)
countriesOverTime = outbreak.groupby('Date').Country.nunique()

countriesOverTime.plot()
outbreakByCountry = outbreak.groupby(['Date', 'Country'])['Confirmed', 'Deaths', 'Recovered'].sum()

outbreakByCountry.reset_index(inplace=True)



outbreakByCountry['DateStr'] = outbreakByCountry['Date'].dt.strftime('%m/%d/%Y')

outbreakByCountry['size'] = outbreakByCountry.Confirmed.pow(0.3)



fig = px.scatter_geo(outbreakByCountry, locations="Country", locationmode='country names', 

                     color="Confirmed", size='size', hover_name="Country",hover_data=['Confirmed','Deaths'],

                     range_color= [0, max(outbreakByCountry['Confirmed'])], 

                     projection="natural earth", animation_frame="DateStr", 

                     title='Spread over time')

fig.update(layout_coloraxis_showscale=False)

fig.show()
outbreakByCountry['DeathsSize'] = outbreakByCountry.Deaths.pow(0.5)

fig = px.scatter_geo(outbreakByCountry, locations="Country", locationmode='country names', 

                     size='DeathsSize', hover_name="Country", hover_data=['Deaths'],

                     projection="natural earth", animation_frame="DateStr",

                     color_continuous_scale='red',

                     # color_discrete_map='red',

                     title='Deaths over time')

fig.update(layout_coloraxis_showscale=False)

fig.show()
latestByCountries = outbreakByCountry[outbreakByCountry.Date == max(outbreakByCountry.Date)]

latestByCountries.sort_values('Confirmed',ascending=False)[['Date','Country','Confirmed','Deaths','Recovered']].head(20).style.background_gradient(cmap='Reds')
fig = px.bar(latestByCountries.sort_values('Confirmed',ascending=False).head(20).sort_values('Confirmed',ascending=True)

            ,x='Confirmed',y='Country', title='Top 20 countries by confirmed cases',text='Confirmed'

            ,orientation='h')

fig.update_traces(marker_color='#084177', opacity=0.8, textposition='outside')

fig.show()
fig = px.bar(latestByCountries.sort_values('Deaths',ascending=False).head(20).sort_values('Deaths',ascending=True)

            ,x='Deaths',y='Country', title='Top 20 countries by deaths', text='Deaths'

            ,orientation='h')

fig.update_traces(marker_color='Red', opacity=0.8, textposition='outside')

fig.show()
topCountries = latestByCountries.sort_values('Confirmed',ascending=False).Country.head(10).tolist()

outBreakForTopCountries=outbreakByCountry[outbreakByCountry.Country.isin(topCountries)]

fig = px.line(outBreakForTopCountries, x="Date", y="Confirmed", color='Country', title='COVID-19 outbreak in top 10 countries', height=400)

fig.update_layout(xaxis_rangeslider_visible=True)

fig.show()



topCountries = latestByCountries.sort_values('Confirmed',ascending=False).Country.iloc[11:20].tolist()

outBreakForTopCountries=outbreakByCountry[outbreakByCountry.Country.isin(topCountries)]

fig = px.line(outBreakForTopCountries, x="Date", y="Confirmed", color='Country', title='COVID-19 outbreak in the next 10 countries', height=400)

fig.update_layout(xaxis_rangeslider_visible=True)

fig.show()
topCountries = latestByCountries.sort_values('Confirmed',ascending=False).Country.head(10).tolist()

outBreakForTopCountries=outbreakByCountry[outbreakByCountry.Country.isin(topCountries)]

outBreakForTopCountries = outBreakForTopCountries.sort_values('Confirmed',ascending=False)

fig = px.line(outBreakForTopCountries, x="Date", y="Confirmed", title='Confirmed cases by Country', height=1000

              ,facet_col='Country', facet_col_wrap=4

              )

fig.update_layout(xaxis_rangeslider_visible=False#,yaxis_type="log"

                 )

fig.show()
outbreakByCountryPost100 = outbreakByCountry.copy()

outbreakByCountryPost100 = outbreakByCountryPost100[outbreakByCountryPost100.Confirmed >= 100]



countries=['China','US','Italy','Spain','France','South Korea','Japan','Singapore','Hong Kong','Taiwan']

fig = px.line(outbreakByCountryPost100[outbreakByCountryPost100.Country.isin(countries)]

              , y='Confirmed',color='Country',height=800)



annotations = []

annotations.append(dict(xref='paper', yref='paper', x=0.58, y=0.6,

                        xanchor='left', yanchor='bottom',

                        text='Early large scale testing',

                        showarrow=False))



annotations.append(dict(xref='paper', yref='paper', x=0.55, y=0.4,

                        xanchor='left', yanchor='bottom',

                        text='Civil obedience & Mask wearing',

                        showarrow=False))



annotations.append(dict(xref='paper', yref='paper', x=0.4, y=0.25,

                        xanchor='left', yanchor='bottom',

                        text='Quarantine & community response',

                        showarrow=False))



annotations.append(dict(xref='paper', yref='paper', x=0.43, y=0.3,

                        xanchor='left', yanchor='bottom',

                        text='Strict quarantine & contact tracing',

                        showarrow=False))





fig.update_layout(annotations=annotations)

fig.update_layout(xaxis_rangeslider_visible=False, yaxis_type='log')

fig.show()
Italy = outbreakByCountry[outbreakByCountry.Country=='Italy']

Italy['Stage'] = Italy.Date.apply(lambda x : 'Pre-lockdown' if x <= pd.to_datetime('2020-03-08') else 'Post-lockdown')



Spain = outbreakByCountry[outbreakByCountry.Country=='Spain']

Spain['Stage'] = Spain.Date.apply(lambda x : 'Pre-lockdown' if x <= pd.to_datetime('2020-03-14') else 'Post-lockdown')



France = outbreakByCountry[outbreakByCountry.Country=='France']

France['Stage'] = France.Date.apply(lambda x : 'Pre-lockdown' if x <= pd.to_datetime('2020-03-17') else 'Post-lockdown')



lockdown = Italy.append(Spain).append(France)

fig = px.line(Italy,x='Date',y='Confirmed', color='Stage', title='Outbreak in Italy before and after lockdown')

fig.show()



fig = px.line(Spain,x='Date',y='Confirmed', color='Stage', title='Outbreak in Spain before and after lockdown')

fig.show()



fig = px.line(France,x='Date',y='Confirmed', color='Stage', title='Outbreak in France before and after lockdown')

fig.show()
China=outbreakByCountry[outbreakByCountry.Country=='China']

China['NewConfirmed'] = China.Confirmed.diff()

China['GrowthFactor'] = China.NewConfirmed.pct_change() + 1 

China['MovingAvg_GF'] = China.GrowthFactor.rolling(window=5, min_periods=2).mean()





fig = make_subplots(rows=1, cols=2,subplot_titles=("Outbreak in China", "Growth Factor"))



fig.add_trace(

    go.Scatter(x=China.Date, y=China.Confirmed),

    row=1, col=1

)



fig.add_trace(

    go.Scatter(x=China.Date, y=China.MovingAvg_GF),

    row=1, col=2

)

fig.update_layout(showlegend=False, title_text="Subplots")

fig.show()
SKorea=outbreakByCountry[outbreakByCountry.Country=='South Korea']

SKorea['NewConfirmed'] = SKorea.Confirmed.diff()

SKorea['GrowthFactor'] = SKorea.NewConfirmed.pct_change() + 1 

SKorea['MovingAvg_GF'] = SKorea.GrowthFactor.rolling(window=5, min_periods=2).mean()





fig = make_subplots(rows=1, cols=2,subplot_titles=("Outbreak in South Korea", "Growth Factor"))



fig.add_trace(

    go.Scatter(x=SKorea.Date, y=SKorea.Confirmed),

    row=1, col=1

)



fig.add_trace(

    go.Scatter(x=SKorea.Date, y=SKorea.MovingAvg_GF),

    row=1, col=2

)

fig.update_layout(showlegend=False, title_text="Subplots")

fig.show()
US=outbreakByCountry[outbreakByCountry.Country=='US']

US['NewConfirmed'] = US.Confirmed.diff()

US['GrowthFactor'] = US.NewConfirmed.pct_change() + 1 

US['MovingAvg_GF'] = US.GrowthFactor.rolling(window=5, min_periods=2).mean()



fig = make_subplots(rows=1, cols=2,subplot_titles=("Outbreak in US", "Growth Factor"))



fig.add_trace(

    go.Scatter(x=US.Date, y=US.Confirmed),

    row=1, col=1

)



fig.add_trace(

    go.Scatter(x=US.Date, y=US.MovingAvg_GF),

    row=1, col=2

)

fig.update_layout(showlegend=False, title_text="Subplots")

fig.show()
outbreakOverall['ActivePerc'] = outbreakOverall.Active * 100.0 / outbreakOverall.Confirmed

outbreakOverall['RecoveryPerc'] = outbreakOverall.Recovered * 100.0 / outbreakOverall.Confirmed

outbreakOverall['DeathPerc'] = outbreakOverall.Deaths * 100.0 / outbreakOverall.Confirmed



outbreakOverall.plot(x='Date',y=['RecoveryPerc', 'DeathPerc'],kind='line')
# Compute the speed of spread, recovery and death

outbreakOverall['Speed_Confirmed'] = outbreakOverall.New_Confirmed * 100.0 / outbreakOverall.Confirmed

outbreakOverall['Speed_Recovered'] = outbreakOverall.New_Recovered * 100.0 / outbreakOverall.Recovered

outbreakOverall['Speed_Deaths'] = outbreakOverall.New_Deaths * 100.0 / outbreakOverall.Deaths



outbreakOverall.plot(x='Date',y=['Speed_Confirmed','Speed_Recovered'],kind='line')
outbreakByCountry = outbreak.groupby(['Date','Country'])['Confirmed','Deaths','Recovered'].sum()

outbreakByCountry.reset_index(inplace=True)



latestByCountry = outbreakByCountry[outbreakByCountry.Date == max(outbreakByCountry.Date)].sort_values('Confirmed',ascending=False)



latestByCountry['ConfirmedPerc'] = latestByCountry.Confirmed * 100/ latestByCountry.Confirmed.sum()

latestByCountry.head(10)
states = {

        'AK': 'Alaska',

        'AL': 'Alabama',

        'AR': 'Arkansas',

        'AS': 'American Samoa',

        'AZ': 'Arizona',

        'CA': 'California',

        'CO': 'Colorado',

        'CT': 'Connecticut',

        'DC': 'District of Columbia',

        'DE': 'Delaware',

        'FL': 'Florida',

        'GA': 'Georgia',

        'GU': 'Guam',

        'HI': 'Hawaii',

        'IA': 'Iowa',

        'ID': 'Idaho',

        'IL': 'Illinois',

        'IN': 'Indiana',

        'KS': 'Kansas',

        'KY': 'Kentucky',

        'LA': 'Louisiana',

        'MA': 'Massachusetts',

        'MD': 'Maryland',

        'ME': 'Maine',

        'MI': 'Michigan',

        'MN': 'Minnesota',

        'MO': 'Missouri',

        'MP': 'Northern Mariana Islands',

        'MS': 'Mississippi',

        'MT': 'Montana',

        'NA': 'National',

        'NC': 'North Carolina',

        'ND': 'North Dakota',

        'NE': 'Nebraska',

        'NH': 'New Hampshire',

        'NJ': 'New Jersey',

        'NM': 'New Mexico',

        'NV': 'Nevada',

        'NY': 'New York',

        'OH': 'Ohio',

        'OK': 'Oklahoma',

        'OR': 'Oregon',

        'PA': 'Pennsylvania',

        'PR': 'Puerto Rico',

        'RI': 'Rhode Island',

        'SC': 'South Carolina',

        'SD': 'South Dakota',

        'TN': 'Tennessee',

        'TX': 'Texas',

        'UT': 'Utah',

        'VA': 'Virginia',

        'VI': 'Virgin Islands',

        'VT': 'Vermont',

        'WA': 'Washington',

        'WI': 'Wisconsin',

        'WV': 'West Virginia',

        'WY': 'Wyoming'

}



statesRev = dict(zip(states.values(), states.keys()))



US = outbreak[outbreak.Country == 'US']

US.State = US.State.str.split(',').str[-1].str.strip()

US = US[~US['State'].isin(['Diamond Princess', 'Grand Princess'])]

US['StateCode'] = US.State.apply(lambda x: statesRev[x] if x in statesRev.keys() else x)

US['State'] = US.StateCode.apply(lambda x: states[x] if x in states.keys() else x)

US = US.groupby(['Date','State','StateCode'])['Confirmed','Deaths','Recovered'].sum()

US.reset_index(inplace=True)

latestUS = US[US.Date == max(US.Date)]

latestUS.sort_values('Confirmed',ascending=False)[['Date','StateCode','Confirmed','Deaths']].head(10).style.background_gradient(cmap='Reds')
fig = px.choropleth(latestUS, locations='StateCode', locationmode="USA-states", color='Confirmed', scope="usa", color_continuous_scale="Sunsetdark", 

                   hover_data = ['State', 'Confirmed', 'Deaths', 'Recovered'], title='COVID-19 Outbreak across US states')

fig.show()
US['DateStr'] = US['Date'].dt.strftime('%m/%d/%Y')

US['size'] = US.Confirmed.pow(0.3)



fig = px.scatter_geo(US, locations='StateCode', locationmode="USA-states", color='Confirmed', scope="usa",

                   size='size', hover_data = ['State', 'Confirmed', 'Deaths', 'Recovered'], title='COVID-19 Outbreak across US states over time'

                   #,projection="natural earth"

                    ,range_color= [0, max(US['Confirmed'])]

                    , animation_frame="DateStr"

                   )

fig.show()
fig = px.line(US, x="Date", y="Confirmed", color='StateCode', title='Confirmed cases in US by state', height=600)

fig.update_layout(xaxis_rangeslider_visible=True)

fig.show()
TopStates = [#'New York',

 'Washington',

 'California',

 'New Jersey',

 'Michigan',

 'Illinois',

 'Florida',

 'Louisiana',

 'Texas',

 'Massachusetts']



USTopStates = US[US.State.isin(TopStates)].sort_values('Confirmed',ascending=False)

fig = px.line(USTopStates, x="Date", y="Confirmed", title='Confirmed cases in US by state', height=600

             ,facet_col='State', facet_col_wrap=4)

fig.update_layout(xaxis_rangeslider_visible=False)

fig.show()
USByCountryPost100 = US.copy()

USByCountryPost100 = USByCountryPost100[USByCountryPost100.Confirmed >= 100]



topStates = USByCountryPost100[USByCountryPost100.Date == max(USByCountryPost100.Date)].sort_values('Confirmed',ascending=False).State.head(5).tolist()

topStates.append('Washington')

#topStates = ['New York','Washington','New Jersey','California']

fig = px.line(USByCountryPost100[USByCountryPost100.State.isin(topStates)], y='Confirmed',color='State',height=800,title='Outbreak in top 5 states compared with Washington')

fig.update_layout(xaxis_rangeslider_visible=False, yaxis_type='log')

fig.show()