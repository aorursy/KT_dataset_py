# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.graph_objs as go

from plotly.offline import iplot

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv("../input/covid19-data/owid-covid-data.csv")
df.head()
df.info()
df[df["location"]=="World"].index
# drop world and International column because the world and International  contains total data, which disrupts our visualization

df.drop(df.loc[df["location"]=="World"].index,inplace=True)

df.drop(df.loc[df["location"]=="International"].index,inplace=True)
# Object date convert to datetime64 

df["date"]=pd.to_datetime(df["date"]) # tarihe çeviriyor
df["date"]
from sklearn import preprocessing

x = df[['new_cases']].values.astype(float)

min_max_scaler = preprocessing.MinMaxScaler()

x_scaled = min_max_scaler.fit_transform(x)

df["new_cases"]=x_scaled

x = df[['new_deaths']].values.astype(float)

min_max_scaler = preprocessing.MinMaxScaler()

x_scaled = min_max_scaler.fit_transform(x)

df["new_deaths"]=x_scaled
df.head()
df_turkey=df[df["location"]=="Turkey"]

df_spain=df[df["location"]=="Spain"]

df_italy=df[df["location"]=="Italy"]

df_usa=df[df["location"]=="United States"]

df_uk=df[df["location"]=="United Kingdom"]

df_german=df[df["location"]=="Germany"]
fig=go.Figure()

fig.add_trace(go.Scatter(x=df_turkey["date"], y=df_turkey["total_deaths"],

                    mode='lines',

                    name='Turkey'))

fig.add_trace(go.Scatter(x=df_spain["date"], y=df_spain["total_deaths"],

                    mode='lines',

                    name='spain'))

fig.add_trace(go.Scatter(x=df_italy["date"], y=df_italy["total_deaths"],

                    mode='lines',

                    name='italy'))

fig.add_trace(go.Scatter(x=df_usa["date"], y=df_usa["total_deaths"],

                    mode='lines',

                    name='USA'))

fig.add_trace(go.Scatter(x=df_uk["date"], y=df_uk["total_deaths"],

                    mode='lines',

                    name='UK'))

fig.add_trace(go.Scatter(x=df_uk["date"], y=df_german["total_deaths"],

                    mode='lines',

                    name='Germany'))
import plotly.express as px

fig = px.scatter(df_turkey, x=df_turkey["date"], y=df_turkey["total_deaths"], color="location")

fig.show()
# Total Cases in deaths in Turkey

sumcase_turkey=df_turkey["new_cases"].sum()

sumdeaths_turkey=df_turkey["new_deaths"].sum()
# Total Cases in deaths in Spain

sumcase_spain=df_spain["new_cases"].sum()

sumdeaths_spain=df_spain["new_deaths"].sum()
fig = go.Figure(data=[

    go.Bar(name='Turkey', y=[sumcase_turkey,sumdeaths_turkey]),

    go.Bar(name='Spain', y=[sumcase_spain,sumdeaths_spain])

])

# Change the bar mode

fig.update_layout(barmode='group')

fig.show()
# Sort location by total deaths per million for the top 10 data 

data_sorted=df.groupby("location").sum().sort_values(by="total_deaths_per_million",ascending=False)[:10]
data_sorted
fig = px.bar(data_sorted, x=data_sorted.index, y='total_deaths_per_million')

iplot(fig)
fig = px.bar(data_sorted, x=data_sorted.index, y='total_deaths_per_million',

             hover_data=['total_deaths', 'total_cases'], color='total_deaths_per_million',

             labels={'pop':'population of Canada'}, height=400)

fig.show()
fig = px.pie(data_sorted, values='total_deaths_per_million', names=data_sorted.index)

fig.show()
df2=df.groupby(['location', 'iso_code']).sum().sort_values(by="new_deaths",ascending=False)
df2.head()
iso_code=df2.index.get_level_values('iso_code')

df2_location=df2.index.get_level_values('location')
fig = px.scatter_geo(df2, locations=iso_code, color=df2_location,

                     hover_name=df2_location, size="new_cases",

                     projection="natural earth")

fig.show()
fig = px.choropleth(df2, locations=iso_code,

                    color="new_cases", # lifeExp is a column of gapminder

                    hover_name=df2_location, # column to add to hover information

                    color_continuous_scale=px.colors.sequential.Plasma)

fig.show()
df_date_sorted=df.sort_values(by="date")

fig = px.choropleth(df_date_sorted, locations="location", locationmode='country names', color=np.log(df_date_sorted["total_cases"]), 

                    hover_name="location", animation_frame=df_date_sorted["date"].dt.strftime('%Y-%m-%d'),

                    title='Cases over time', color_continuous_scale=px.colors.sequential.Magenta)

fig.update(layout_coloraxis_showscale=False)

fig.show()
# Listing cities of continents

eu_list=[

    "Austria",

"Albania",

"Andorra",

"Belarus",

"Bosnia and Herzegovina",

"Croatia",

"European Union",

"Faroe Islands",

"Gibraltar",

"Guerney and Alderney",

"Iceland",

"Jersey",

"Kosovo",

"Liechtenstein",

"Man, Island of",

"Moldova",

"Monaco",

"Montenegro",

"North Macedonia",

"Norway",

"Russia",

"San Marino",

"Serbia",

"Svalbard and Jan Mayen Islands",

"Switzerland",

"Turkey",

"Ukraine",

"United Kingdom",

"Vatican City State (Holy See)",

"Albania",

"Belgium",

"Bulgaria",

"Croatia",

"Cyprus",

"Czech Republic",

"Denmark",

"Estonia",

"Finland",

"France",

"Germany",

"Greece",

"Hungary",

"Ireland",

"Italy",

"Latvia",

"Lithuania",

"Luxembourg",

"Malta",

"Netherlands",

"Poland",

"Portugal",

"Romania",

"Slovakia",

"Slovenia",

"Spain",

"Sweden"]

asia_list=[

"Afghanistan",

"Armenia",

"Azerbaijan",

"Brunei",

"Bangladesh",

"Bhutan",

"Brunei Darussalam",

"Cambodia",

"China",

"Georgia",

"Hong Kong",

"India",

"Indonesia",

"Japan",

"Kazakhstan",

"South Korea",

"Kyrgyzstan",

"Laos",

"Macao",

"Malaysia",

"Maldives",

"Mongolia",

"Myanmar",

"Nepal",

"Pakistan",

"Phillipines",

"Singapore",

"Sri Lanka",

"Taiwan",

"Tajikistan",

"Thailand"

"Timor Leste (West)",

"Turkmenistan",

"Uzbekistan",

"Vietnam"

]

mid_east_list=[

"Bahrain",

"Iraq",

"Iran",

"Israel",

"Jordan",

"Kuwait",

"Lebanon",

"Oman",

"Palestine",

"Qatar",

"Saudi Arabia",

"Syria",

"United Arab Emirates",

"Yemen"

]

ocencia_list=[

"Australia",

"Fiji",

"French Polynesia",

"Guam",

"Kiribati",

"Marshall Islands",

"Micronesia",

"New Caledonia",

"New Zealand",

"Papua New Guinea",

"Samoa",

"Samoa, American",

"Solomon, Islands",

"Tonga",

"Vanuatu"

]

caribbean_list=[

"Anguilla",

"Antigua and Barbuda",

"Aruba",

"Bahamas",

"Barbados",

"Bonaire Sint Eustatius and Saba",

"British Virgin Islands",

"Cayman Islands",

"Cuba",

"Curaçao",

"Dominica",

"Dominican Republic",

"Grenada",

"Guadeloupe",

"Haiti",

"Jamaica",

"Martinique",

"Monserrat",

"Puerto Rico",

"Saint-Barthélemy",

"Saint Kitts and Nevis",

"Saint Lucia",

"Saint Martin",

"Saint Vincent and the Grenadines",

"Sint Maarten",

"Trinidad and Tobago",

"Turks and Caicos Islands",

"United States Virgin Islands"

]



central_america_list=[

"Belize",

"Costa Rica",

"El Salvador",

"Guatemala",

"Honduras",

"Mexico",

"Nicaragua",

"Panama"

]



south_america_list=[

"Argentina",

"Bolivia",

"Brazil",

"Chile",

"Colombia",

"Ecuador",

"Falkland Islands",

"French Guiana",

"Guyana",

"Paraguay",

"Peru",

"Suriname",

"Uruguay",

"Venezuela"

]

north_america_list=[

"Bermuda",

"Canada",

"Greenland",

"Saint Pierre and Miquelon",

"United States"

]

africa_list=[

"Burundi",

"Comoros",

"Djibouti",

"Eritrea",

"Ethiopia",

"Kenya",

"Madagascar",

"Malawi",

"Mauritius",

"Mayotte"

"Mozambique",

"Reunion",

"Rwanda",

"Seychelles",

"Somalia",

"Tanzania",

"Uganda",

"Zambia",

"Zimbabwe",

"Democratic Republic of Congo",

    "Congo",

"Angola",

"Cameroon",

"Central African Republic",

"Chad",

"Equatorial Guinea",

"Gabon",

"Sao Tome and Principe",

"Algeria",

"Egypt",

"Libyan Arab Jamahiriya",

"Morroco",

"South Sudan",

"Sudan",

"Tunisia",

"Western Sahara",

"Botswana",

"Eswatini (ex-Swaziland)",

"Lesotho",

"Namibia",

"South Africa",

"Benin",

"Burkina Faso",

"Cape Verde",

"Cote d'Ivoire",

"Gambia",

"Ghana",

"Guinea",

"Guinea-Bissau",

"Liberia",

"Mali",

"Mauritania",

"Niger",

"Nigeria",

"Saint Helena",

"Senegal",

"Sierra Leone",

"Togo"

]

continent_list=[eu_list,asia_list,mid_east_list,ocencia_list,caribbean_list,central_america_list,south_america_list,north_america_list,africa_list]

continent_str_list=["europe","asia","mid_east","ocencia","carribean","central_america","south_america","north_america","africa"]
indexlocation=[]

indexlocation2=[]

index_list=[0]

for i in range(len(continent_list)):

    for j in continent_list[i]:

        indexlocation.append(df[df["location"]=="{}".format(j)].index)     

    indexlocation2 = np.concatenate(indexlocation)

    print("number of cities in ",continent_str_list[i],":",len(indexlocation2))

    index_list.append(len(indexlocation2))

    

for l in range(len(index_list)):

    if(l+1>9):

        break

    print("loading cities of",continent_str_list[l],"......")

    for m in range(index_list[l],index_list[l+1]):

        df.loc[indexlocation2[m],"continent"]=continent_str_list[l]
#cities without continent

df["continent"].isnull().sum()
#transform cities without continents into other

df["continent"].fillna("other",inplace=True)
# convert the date to string format for chart

df["stringdate"]=[ i.strftime('%Y-%m-%d') for i in df["date"]]
#assigning to "which_day" which day of the epidemic

df["which_day"]=[float(i.split("-")[2])-30 if(i.split("-")[1]=="12") else False for i in df["stringdate"]]

df.loc[df["which_day"]==False,"which_day"]=[float(i.split("-")[2]) if(i.split("-")[1]=="01") else False for i in df[df["which_day"]==False]["stringdate"]]

df.loc[df["which_day"]==False,"which_day"]=[float(i.split("-")[2])+30 if(i.split("-")[1]=="02") else False for i in df[df["which_day"]==False]["stringdate"]]

df.loc[df["which_day"]==False,"which_day"]=[float(i.split("-")[2])+60 if(i.split("-")[1]=="03") else False  for i in df[df["which_day"]==False]["stringdate"]]

df.loc[df["which_day"]==False,"which_day"]=[float(i.split("-")[2])+90 if(i.split("-")[1]=="04") else False for i in df[df["which_day"]==False]["stringdate"]]

df.loc[df["which_day"]==False,"which_day"]=[float(i.split("-")[2])+120 if(i.split("-")[1]=="05") else False for i in df[df["which_day"]==False]["stringdate"]]
#sort the epidemic by day

df_graph=df.sort_values(by="which_day")
import plotly.express as px

fig = px.bar(df_graph[:], x="continent", y="total_cases", color="continent",

  animation_frame="which_day", animation_group="location", range_y=[100,1000000])

fig.show()

from IPython.core.display import HTML

HTML('''<div class="flourish-embed flourish-bar-chart-race" data-src="visualisation/1571387"><script src="https://public.flourish.studio/resources/embed.js"></script></div>''')
