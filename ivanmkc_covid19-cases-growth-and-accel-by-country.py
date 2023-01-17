# MODIFY THIS TO GET YOUR TARGET COUNTRY



# COUNTRIES INCLUDE



# ['Afghanistan' 'Albania' 'Algeria' 'Andorra' 'Antigua and Barbuda'

#  'Argentina' 'Armenia' 'Aruba' 'Australia' 'Austria' 'Azerbaijan'

#  'Bahrain' 'Bangladesh' 'Barbados' 'Belarus' 'Belgium' 'Benin' 'Bhutan'

#  'Bolivia' 'Bosnia and Herzegovina' 'Brazil' 'Brunei' 'Bulgaria'

#  'Burkina Faso' 'Cambodia' 'Cameroon' 'Canada' 'Central African Republic'

#  'Chile' 'China' 'Colombia' 'Congo (Brazzaville)' 'Congo (Kinshasa)'

#  'Costa Rica' "Cote d'Ivoire" 'Croatia' 'Cruise Ship' 'Cuba' 'Cyprus'

#  'Czechia' 'Denmark' 'Dominican Republic' 'Ecuador' 'Egypt'

#  'Equatorial Guinea' 'Estonia' 'Eswatini' 'Ethiopia' 'Finland' 'France'

#  'Gabon' 'Georgia' 'Germany' 'Ghana' 'Greece' 'Greenland' 'Guatemala'

#  'Guernsey' 'Guinea' 'Guyana' 'Holy See' 'Honduras' 'Hungary' 'Iceland'

#  'India' 'Indonesia' 'Iran' 'Iraq' 'Ireland' 'Israel' 'Italy' 'Jamaica'

#  'Japan' 'Jersey' 'Jordan' 'Kazakhstan' 'Kenya' 'Korea, South' 'Kosovo'

#  'Kuwait' 'Latvia' 'Lebanon' 'Liberia' 'Liechtenstein' 'Lithuania'

#  'Luxembourg' 'Malaysia' 'Maldives' 'Malta' 'Martinique' 'Mauritania'

#  'Mexico' 'Moldova' 'Monaco' 'Mongolia' 'Montenegro' 'Morocco' 'Namibia'

#  'Nepal' 'Netherlands' 'New Zealand' 'Nigeria' 'North Macedonia' 'Norway'

#  'Oman' 'Pakistan' 'Panama' 'Paraguay' 'Peru' 'Philippines' 'Poland'

#  'Portugal' 'Qatar' 'Republic of the Congo' 'Romania' 'Russia' 'Rwanda'

#  'Saint Lucia' 'Saint Vincent and the Grenadines' 'San Marino'

#  'Saudi Arabia' 'Senegal' 'Serbia' 'Seychelles' 'Singapore' 'Slovakia'

#  'Slovenia' 'Somalia' 'South Africa' 'Spain' 'Sri Lanka' 'Sudan'

#  'Suriname' 'Sweden' 'Switzerland' 'Taiwan*' 'Tanzania' 'Thailand'

#  'The Bahamas' 'The Gambia' 'Togo' 'Trinidad and Tobago' 'Tunisia'

#  'Turkey' 'US' 'Ukraine' 'United Arab Emirates' 'United Kingdom' 'Uruguay'

#  'Uzbekistan' 'Venezuela' 'Vietnam']



TARGET_COUNTRY = "Canada"
import pandas as pd

df_confirmed = pd.read_csv("../input/covid19-coronavirus/2019_nCoV_data.csv")



# Rename Province/State and Country/Region

df_confirmed = df_confirmed.rename(columns={'Province/State': 'provincestate', 

                        'Date': 'date', 

                        'Country': 'countryregion',

                        'Confirmed': 'confirmed',

                        'Deaths': 'deaths',

                        'Recovered': 'recovered',

                       })



df_confirmed['date'] = pd.to_datetime(df_confirmed['date'], errors='coerce', format='%m/%d/%Y %H:%M')



# # Remove unneeded columns

df_confirmed = df_confirmed.drop(['Last Update', 'Sno', 'deaths', 'recovered'], axis=1)



df_confirmed.head()



# print(df_confirmed.dtypes)
from itertools import product



# Restrict to target country

df_confirmed_grouped = df_confirmed.query('countryregion == "{}"'.format(TARGET_COUNTRY)).copy()

df_confirmed_grouped = df_confirmed_grouped.drop(['countryregion'], axis=1)

df_confirmed_grouped['provincestate'] = df_confirmed_grouped['provincestate'].fillna('Unknown')



## Dates/Provinces with no confirmed are not included in the dataset, so we have to fill these missing rows

# Create the full combinations of dates and provinces

dates_unique = df_confirmed_grouped['date'].unique()

provinces_unique = df_confirmed_grouped['provincestate'].unique()

df_full = pd.DataFrame(list(product(dates_unique, provinces_unique)), columns=['date', 'provincestate'])

# Merge the full df with the one with actual data

df_full = pd.merge(df_full, df_confirmed_grouped, how='outer', on=['date','provincestate']).fillna(0)



df_confirmed_grouped = df_full.sort_values(['date', 'provincestate'])



df_confirmed_grouped.head(500)
# Calculate change

window = 5

growthCalculation = lambda x: (x.max())



from sklearn.linear_model import LinearRegression

linear_regressor = LinearRegression()



x = pd.Series(range(0,window)).values.reshape(-1,1)

linearCoefficient = lambda values: (linear_regressor.fit(x, values.reshape(-1,1)).coef_)



df_confirmed_grouped['growth_rate'] = df_confirmed_grouped.groupby(['provincestate'])['confirmed'].rolling(window).apply(linearCoefficient, raw=True).fillna(0).reset_index(level=0, drop=True)

df_confirmed_grouped['acceleration_rate'] = df_confirmed_grouped['growth_rate'].rolling(window).apply(linearCoefficient, raw=True).fillna(0)



print(df_confirmed_grouped.shape)



df_confirmed_grouped.head(500)
# Show fig

import plotly.offline as py

py.init_notebook_mode(connected=True)



import plotly.express as px

import plotly.graph_objects as go



colorSequence = ["red", "royalblue"]



layout = go.Layout(title="COVID-19 confirmed cases by province/state", margin={"l": 100, "r": 100},

                   colorway=["#287D95", "#EF533B"], legend={"x": 0.7, "y": 1, 'orientation': "h"},

                   yaxis={'title': 'Confirmed cases'},                   

                   xaxis={"title": "Date",

                         'domain': [0, 0.92]},

                   height=1000)



chinaColor = colorSequence[0]



# Only plot 'confirmed', since we can't control the dashed lines and second-axes using Plotly Express

trace_confirmed = px.line(df_confirmed_grouped, 

                           x='date', 

                           y='confirmed', 

                           color='provincestate',).data



# Add growth rate

trace_growth = px.line(df_confirmed_grouped, 

                           x='date', 

                           y='growth_rate', 

                           color='provincestate',).data



# Add acceleration rate

trace_accel = px.line(df_confirmed_grouped, 

                           x='date', 

                           y='acceleration_rate', 

                           color='provincestate',).data
layout = go.Layout(title="COVID-19 confirmed cases by province/state",

                   yaxis={'title': 'Confirmed cases'},

                   height=1000)



fig = go.Figure(data=trace_confirmed, layout=layout)

fig.show()
layout = go.Layout(title="COVID-19 growth rate by province/state",

                   yaxis={'title': 'Confirmed cases'},

                   height=1000)



fig = go.Figure(data=trace_confirmed, layout=layout)

fig.show()
layout = go.Layout(title="COVID-19 acceleration by province/state",

                   yaxis={'title': 'Confirmed cases'},

                   height=1000)



fig = go.Figure(data=trace_accel, layout=layout)

fig.show()