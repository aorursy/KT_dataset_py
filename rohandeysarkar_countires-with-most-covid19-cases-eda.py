import os

import numpy as np

import pandas as pd

pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', None)



import matplotlib.style as style

style.use('fivethirtyeight')

import matplotlib.pyplot as plt

%matplotlib inline

plt.rcParams['figure.dpi'] = 150 #set figure size



from plotly.offline import iplot, init_notebook_mode

import plotly.express as px

#import plotly.plotly as py

import plotly.graph_objs as go

import cufflinks

cufflinks.go_offline(connected=True)

init_notebook_mode(connected=True)



import folium
os.listdir("../input")
os.listdir("../input/covid19-us-county-jhu-data-demographics")
os.listdir("../input/countries-of-the-world-iso-codes-and-population")
os.listdir("../input/usa-states")
os.listdir("../input/novel-corona-virus-2019-dataset")

df = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv", parse_dates=['Last Update'])

df.rename(columns={'Country/Region':'Country'}, inplace=True)

df = df.drop(columns = ['SNo', "Last Update"]) #only confuses



df_conf = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv")

df_conf.rename(columns={'Country/Region':'Country'}, inplace=True)



df_death = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv")

df_death.rename(columns={'Country/Region':'Country'}, inplace=True)



countries = pd.read_csv("../input/countries-of-the-world-iso-codes-and-population/countries_by_population_2019.csv")

countries_iso = pd.read_csv("../input/countries-of-the-world-iso-codes-and-population/country_codes_2020.csv")



us_covid = pd.read_csv('../input/covid19-us-county-jhu-data-demographics/covid_us_county.csv')

us_county = pd.read_csv('../input/covid19-us-county-jhu-data-demographics/us_county.csv')
df.head()
df['Country'] = df['Country'].str.strip()



df.loc[df['Province/State'].isnull(), 'Province/State'] = df.loc[df['Province/State'].isnull(), 'Country']



#keep most recent line per Province/State and Country

df.sort_values(['Country', 'Province/State', 'ObservationDate'], ascending = [True,True,False], inplace = True)

df = df.drop_duplicates(['Country', 'Province/State'], keep = "first")



df_state = df.copy()



df = df.drop(columns = "ObservationDate")



#groupby Country

df_country = df.groupby(['Country'], as_index=False)['Confirmed', 'Deaths'].sum()

df_country.head()
countries.head()
countries_iso.head()
cols_to_drop = ['Rank', 'pop2018','GrowthRate', 'area', 'Density']

countries = countries.drop(columns = cols_to_drop)



# merge countries_iso on countries

countries = countries.merge(countries_iso[['name', 'cca3']], on = ['name'], how = "left")



cols_to_rename = {'name': 'Country', 'pop2019': 'Population', 'cca3': 'ISO'}

countries = countries.rename(columns = cols_to_rename)

countries.head()
countries_to_rename = {'US': 'United States',

                       'Mainland China': 'China',

                       'UK': 'United Kingdom',

                       'Congo (Kinshasa)': 'DR Congo',

                       'North Macedonia': 'Macedonia',

                       'Republic of Ireland': 'Ireland',

                       'Congo (Brazzaville)': 'Republic of the Congo'}



# merge countries on df_country

df_country['Country'] = df_country['Country'].replace(countries_to_rename)



df_country = df_country.merge(countries, on = "Country", how = "left")





#dropping not matching countries, only small islands left

df_country = df_country.dropna()



#rounding population to millions

df_country['Population'] = round((df_country['Population']/1000),2)

df_country = df_country.rename(columns = {'Population': 'Population (million)'})

df_country['Cases per Million'] = round((df_country['Confirmed']/df_country['Population (million)']),2)

df_country['Deaths per Million'] = round((df_country['Deaths']/df_country['Population (million)']),2)



df_country = df_country[(df_country['Population (million)'] > 1)]



df_country.head()
df_country = df_country.sort_values(['Cases per Million'], ascending=False).reset_index(drop=True)

# give subset or else colors all relative cols

df_country.drop(columns = ['ISO', 'Deaths', 'Deaths per Million']).head(10).style.background_gradient(cmap='Reds', subset = ['Cases per Million'])
fig = px.choropleth(df_country, 

                    locations = "ISO",

                    color = "Cases per Million",

                    hover_name = "Country",

                    color_continuous_scale = px.colors.sequential.YlOrRd)



layout = go.Layout(

    title = go.layout.Title(

        text = "Corona confirmed cases per million inhabitants",

        x=0.5),

    font = dict(size=14),

    width = 750,

    height = 350,

    margin=dict(l=0,r=0,b=0,t=30)

)



fig.update_layout(layout)

fig.show()
df_country = df_country.sort_values(['Deaths per Million'], ascending = False).reset_index(drop=True)



countries = df_country.copy()



df_country.drop(columns = ['ISO', 'Confirmed', 'Cases per Million']).head(10).style.background_gradient(cmap='Reds', subset = ['Deaths per Million'])
fig = px.choropleth(df_country, 

                    locations = "ISO",

                    color = "Deaths per Million",

                    hover_name = "Country",

                    color_continuous_scale = px.colors.sequential.YlOrRd)



layout = go.Layout(

    title = go.layout.Title(

        text = "Corona deaths per million inhabitants",

        x=0.5),

    font = dict(size=14),

    width = 750,

    height = 350,

    margin=dict(l=0,r=0,b=0,t=30)

)



fig.update_layout(layout)

fig.show()
df_conf.head()
#only keep last date available

cols_to_keep = list(df_conf.columns[0:4]) + list(df_conf.columns[-1:])

df_conf_last = df_conf[cols_to_keep]

cols_to_rename = {'5/1/20' : 'Confirmed Cases'}

df_conf_last = df_conf_last.rename(columns = cols_to_rename)

df_conf_last['Confirmed Cases'] = df_conf_last['Confirmed Cases'].astype(float)



df_conf_last.head()
map1 = folium.Map(location = [30.6, 114], zoom_start=3) # US=[39,-98] Europe =[45, 5]



# Now to add circles in our map

for i in range(0,len(df_conf_last)):

   folium.Circle(

      location = [df_conf_last.iloc[i]['Lat'], df_conf_last.iloc[i]['Long']],

      tooltip = "Country: "+ df_conf_last.iloc[i]['Country']+

       "<br>Province/State: "+ str(df_conf_last.iloc[i]['Province/State'])+

       "<br>Confirmed cases: "+ str(df_conf_last.iloc[i]['Confirmed Cases'].astype(int)),

      radius = df_conf_last.iloc[i]['Confirmed Cases'] * 5,

      color = 'crimson',

      fill = True,

      fill_color = 'crimson'

   ).add_to(map1)

    

map1
df_death.head()
cols_to_keep = list(df_death.columns[0:4]) + list(df_death.columns[-1:])

df_death_last = df_death[cols_to_keep]

df_death_last.columns.values[-1] = "Deaths"

df_death_last["Deaths"] = df_death_last["Deaths"].astype(float)



df_death_last.head()
map2 = folium.Map(location = [30.6, 114], zoom_start=3)



for i in range(0,len(df_death_last)):

   folium.Circle(

      location = [df_death_last.iloc[i]['Lat'], df_death_last.iloc[i]['Long']],

      tooltip = "Country: "+ df_death_last.iloc[i]['Country']+

       "<br>Province/State: "+ str(df_death_last.iloc[i]['Province/State'])+

       "<br>Confirmed cases: "+ str(df_death_last.iloc[i]['Deaths'].astype(int)),

      radius = df_death_last.iloc[i]['Deaths'] * 50,

      color = 'crimson',

      fill = True,

      fill_color = 'crimson'

   ).add_to(map2)

    

map2
df_conf.head()
ts_country = df_conf.drop(columns = ['Lat', 'Long', 'Province/State'])

ts_country = ts_country.groupby(['Country']).sum()

# sort by most confirmed cases(last col has total conf. cases so sorting by last col)

ts_country = ts_country.sort_values(by = ts_country.columns[-1], ascending = False).head(10)

ts_country
ts_country.transpose().iplot(title = 'Time series of confirmed cases of countries with most confirmed cases')
df_death.head()
ts_country = df_death.drop(columns = ['Lat', 'Long', 'Province/State'])

ts_country = ts_country.groupby(['Country']).sum()



ts_country = ts_country.sort_values(by = ts_country.columns[-1], ascending = False).head(10)

ts_country
ts_country.transpose().iplot(title = 'Time series of deaths of countries with most victims')
ts_country = ts_country.transpose()

ts_country.head()
df1 = ts_country.iloc[:,0].to_frame()

df1.head()
# df1 = df1[df1.iloc[:, 0] !=0].reset_index(drop=True)

df1 = df1.loc[df1['US'] != 0, :].reset_index(drop=True)

df1.head()
for i in range(1, ts_country.shape[1]):

    df = ts_country.iloc[:, i].to_frame()

    df = df[df.iloc[:,0] != 0].reset_index(drop=True)

    df1 = pd.concat([df1, df], join = 'outer', axis=1)

    

df1.head()
df1.iplot(title = 'Time series of deaths since first victim', xTitle = 'Days since first reported Death', yTitle = 'Number of Deaths')