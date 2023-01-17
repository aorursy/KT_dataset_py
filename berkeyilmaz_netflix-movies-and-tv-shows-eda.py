# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # visualization library

import seaborn as sns

from plotly.offline import init_notebook_mode, plot, iplot

import plotly as py

init_notebook_mode(connected=True) 

from pandas.plotting import parallel_coordinates

import plotly.graph_objs as go # plotly graphical object

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import warnings            

warnings.filterwarnings("ignore") 

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/netflix-shows/netflix_titles.csv")



## add new features in the dataset

df["date_added"] = pd.to_datetime(df['date_added'])

df['year_added'] = df['date_added'].dt.year

df['month_added'] = df['date_added'].dt.month



df['season_count'] = df.apply(lambda x : x['duration'].split(" ")[0] if "Season" in x['duration'] else "", axis = 1)

df['duration'] = df.apply(lambda x : x['duration'].split(" ")[0] if "Season" not in x['duration'] else "", axis = 1)

df.head()
df.columns
dataset = df.loc[:,["type","title","director","cast","country","date_added","release_year","duration", "listed_in", "year_added", "month_added" ]]
dataset.head()
col = "month_added"

grouped = dataset[col].value_counts().reset_index()

grouped = grouped.rename(columns = {col : "count", "index" : col})



## plot

trace = go.Pie(labels=grouped[col], values=grouped['count'], pull=[0.05, 0], marker=dict(colors=["#6ad49b", "#a678de"]))

layout = go.Layout(title="", height=400, legend=dict(x=0.1, y=1.1))

fig = go.Figure(data = [trace], layout = layout)

iplot(fig)
col = "type"

grouped = dataset[col].value_counts().reset_index()

grouped = grouped.rename(columns = {col : "count", "index" : col})



## plot

trace = go.Pie(labels=grouped[col], values=grouped['count'], pull=[0.05, 0], marker=dict(colors=["#6ad49b", "#a678de"]))

layout = go.Layout(title="", height=400, legend=dict(x=0.1, y=1.1))

fig = go.Figure(data = [trace], layout = layout)

iplot(fig)
d1 = df[df["type"] == "TV Show"]

d2 = df[df["type"] == "Movie"]



col = "year_added"



vc1 = d1[col].value_counts().reset_index()

vc1 = vc1.rename(columns = {col : "count", "index" : col})

vc1['percent'] = vc1['count'].apply(lambda x : 100*x/sum(vc1['count']))

vc1 = vc1.sort_values(col)



vc2 = d2[col].value_counts().reset_index()

vc2 = vc2.rename(columns = {col : "count", "index" : col})

vc2['percent'] = vc2['count'].apply(lambda x : 100*x/sum(vc2['count']))

vc2 = vc2.sort_values(col)



trace1 = go.Scatter(x=vc1[col], y=vc1["count"], name="TV Shows", marker=dict(color="#a678de"))

trace2 = go.Scatter(x=vc2[col], y=vc2["count"], name="Movies", marker=dict(color="#6ad49b"))

data = [trace1, trace2]

layout = go.Layout(title="Content added over the years", legend=dict(x=0.1, y=1.1, orientation="h"))

fig = go.Figure(data, layout=layout)

fig.show()
dataset.head()
dataset.director.value_counts()
col = "release_year"



vc1 = d1[col].value_counts().reset_index()

vc1 = vc1.rename(columns = {col : "count", "index" : col})

vc1['percent'] = vc1['count'].apply(lambda x : 100*x/sum(vc1['count']))

vc1 = vc1.sort_values(col)



vc2 = d2[col].value_counts().reset_index()

vc2 = vc2.rename(columns = {col : "count", "index" : col})

vc2['percent'] = vc2['count'].apply(lambda x : 100*x/sum(vc2['count']))

vc2 = vc2.sort_values(col)



trace1 = go.Bar(x=vc1[col], y=vc1["count"], name="TV Shows", marker=dict(color="#a678de"))

trace2 = go.Bar(x=vc2[col], y=vc2["count"], name="Movies", marker=dict(color="#6ad49b"))

data = [trace1, trace2]

layout = go.Layout(title="Content added over the years", legend=dict(x=0.1, y=1.1, orientation="h"))

fig = go.Figure(data, layout=layout)

fig.show()
col = 'month_added'

vc1 = d1[col].value_counts().reset_index()

vc1 = vc1.rename(columns = {col : "count", "index" : col})

vc1['percent'] = vc1['count'].apply(lambda x : 100*x/sum(vc1['count']))

vc1 = vc1.sort_values(col)



trace1 = go.Bar(x=vc1[col], y=vc1["count"], name="TV Shows", marker=dict(color="#a678de"))

data = [trace1]

layout = go.Layout(title="In which month, the conent is added the most?", legend=dict(x=0.1, y=1.1, orientation="h"))

fig = go.Figure(data, layout=layout)

fig.show()
d3 = df[df["director"] == "Raúl Campos, Jan Suter"]

d4 = df[df["director"] == "Marcus Raboy"]
col = 'release_year'

vc1 = d3[col].value_counts().reset_index()

vc1 = vc1.rename(columns = {col : "count", "index" : col})

vc1['percent'] = vc1['count'].apply(lambda x : 100*x/sum(vc1['count']))

vc1 = vc1.sort_values(col)



trace1 = go.Bar(x=vc1[col], y=vc1["count"], name="TV Shows", marker=dict(color="#a678de"))

data = [trace1]

layout = go.Layout(title="Director : Raúl Campos, Jan Suter Published Film on Netflix 2016-2018 Years", legend=dict(x=0.1, y=1.1, orientation="h"))

fig = go.Figure(data, layout=layout)

fig.show()
small = dataset.sort_values("release_year", ascending = False)

small = small[small['duration'] != ""] #W Duration

small[['title', "release_year"]][:15]
small = df.sort_values("release_year", ascending = True)

small = small[small['duration'] != ""]

small[['title', "release_year"]][:15]
dataset.info()
dataset.tail()
small = dataset.sort_values("release_year", ascending = True)

small = small[small['duration']  == "90"]

small[['title', "release_year", "duration"]][:5]
import plotly.figure_factory as ff

x1 = d2['duration'].fillna(0.0).astype(float)

fig = ff.create_distplot([x1], ['a'], bin_size=0.7, curve_type='normal', colors=["#6ad49b"])

fig.update_layout(title_text='Distplot with Normal Distribution')

fig.show()
df.head()
col = 'season_count'

vc1 = d1[col].value_counts().reset_index()

vc1 = vc1.rename(columns = {col : "count", "index" : col})

vc1['percent'] = vc1['count'].apply(lambda x : 100*x/sum(vc1['count']))

vc1 = vc1.sort_values(col)



trace1 = go.Bar(x=vc1[col], y=vc1["count"], name="TV Shows", marker=dict(color="#a678de"))

data = [trace1]

layout = go.Layout(title="Seasons", legend=dict(x=0.1, y=1.1, orientation="h"))

fig = go.Figure(data, layout=layout)

fig.show()

d1 = df[df["type"] == "TV Show"]

d2 = df[df["type"] == "Movie"]



# Yearly Tv Show publishing  Rate we can sure for 2020, 2020 best rate of publishing TV Shows



col = 'release_year'

vc1 = d1[col].value_counts().reset_index()

vc1 = vc1.rename(columns = {col : "count", "index" : col})

vc1['percent'] = vc1['count'].apply(lambda x : 100*x/sum(vc1['count']))

vc1 = vc1.sort_values(col)



trace1 = go.Bar(x=vc1[col], y=vc1["count"], name="TV Shows", marker=dict(color="#a678de"))

data = [trace1]

layout = go.Layout(title="Seasons", legend=dict(x=0.1, y=1.1, orientation="h"))

fig = go.Figure(data, layout=layout)

fig.show()
col = 'release_year'

vc1 = d2[col].value_counts().reset_index()

vc1 = vc1.rename(columns = {col : "count", "index" : col})

vc1['percent'] = vc1['count'].apply(lambda x : 100*x/sum(vc1['count']))

vc1 = vc1.sort_values(col)



trace1 = go.Bar(x=vc1[col], y=vc1["count"], name="TV Shows", marker=dict(color="#a678de"))

data = [trace1]

layout = go.Layout(title="Movies", legend=dict(x=0.1, y=1.1, orientation="h"))

fig = go.Figure(data, layout=layout)

fig.show()
col = "rating"



vc1 = d1[col].value_counts().reset_index()

vc1 = vc1.rename(columns = {col : "count", "index" : col})

vc1['percent'] = vc1['count'].apply(lambda x : 100*x/sum(vc1['count']))

vc1 = vc1.sort_values(col)



vc2 = d2[col].value_counts().reset_index()

vc2 = vc2.rename(columns = {col : "count", "index" : col})

vc2['percent'] = vc2['count'].apply(lambda x : 100*x/sum(vc2['count']))

vc2 = vc2.sort_values(col)



trace1 = go.Bar(x=vc1[col], y=vc1["count"], name="TV Shows", marker=dict(color="#a678de"))

trace2 = go.Bar(x=vc2[col], y=vc2["count"], name="Movies", marker=dict(color="#6ad49b"))

data = [trace1, trace2]

layout = go.Layout(title="Content added over the years", legend=dict(x=0.1, y=1.1, orientation="h"))

fig = go.Figure(data, layout=layout)

fig.show()
df.head()
d4 = df[df["country"] == "United States"]

d5 = df[df["country"] == "United Kingdom"]



col = "rating"



vc1 = d4[col].value_counts().reset_index()

vc1 = vc1.rename(columns = {col : "count", "index" : col})

vc1['percent'] = vc1['count'].apply(lambda x : 100*x/sum(vc1['count']))

vc1 = vc1.sort_values(col)



vc2 = d5[col].value_counts().reset_index()

vc2 = vc2.rename(columns = {col : "count", "index" : col})

vc2['percent'] = vc2['count'].apply(lambda x : 100*x/sum(vc2['count']))

vc2 = vc2.sort_values(col)



trace1 = go.Bar(x=vc1[col], y=vc1["count"], name="United States", marker=dict(color="#a678de"))

trace2 = go.Bar(x=vc2[col], y=vc2["count"], name="United Kingdom", marker=dict(color="#6ad49b"))

data = [trace1, trace2]

layout = go.Layout(title="Rating: USA and UK ", legend=dict(x=0.1, y=1.1, orientation="h"))

fig = go.Figure(data, layout=layout)

fig.show()
df[df["type"] == "TV Show"]
d1 = df[df["type"] == "TV Show"]

d2 = df[df["type"] == "Movie"]



import collections

col = "listed_in"

categories = ", ".join(d2['listed_in']).split(", ")

counter_list = collections.Counter(categories).most_common(50)

labels = [_[0] for _ in counter_list][::-1]

values = [_[1] for _ in counter_list][::-1]

trace1 = go.Bar(y=labels, x=values, orientation="h", name="Movies", marker=dict(color="#a678de"))



data = [trace1]

layout = go.Layout(title="Content added over the years for Movies ", legend=dict(x=0.1, y=1.1, orientation="h"))

fig = go.Figure(data, layout=layout)

fig.show()
col = "listed_in"

categories = ", ".join(d1['listed_in']).split(", ")

counter_list = collections.Counter(categories).most_common(50)

labels = [_[0] for _ in counter_list][::-1]

values = [_[1] for _ in counter_list][::-1]

trace1 = go.Bar(y=labels, x=values, orientation="h", name="Shows", marker=dict(color="#a678de"))



data = [trace1]

layout = go.Layout(title="Content added over the years for TV Shows ", legend=dict(x=0.1, y=1.1, orientation="h"))

fig = go.Figure(data, layout=layout)

fig.show()
country_codes = {'afghanistan': 'AFG',

 'albania': 'ALB',

 'algeria': 'DZA',

 'american samoa': 'ASM',

 'andorra': 'AND',

 'angola': 'AGO',

 'anguilla': 'AIA',

 'antigua and barbuda': 'ATG',

 'argentina': 'ARG',

 'armenia': 'ARM',

 'aruba': 'ABW',

 'australia': 'AUS',

 'austria': 'AUT',

 'azerbaijan': 'AZE',

 'bahamas': 'BHM',

 'bahrain': 'BHR',

 'bangladesh': 'BGD',

 'barbados': 'BRB',

 'belarus': 'BLR',

 'belgium': 'BEL',

 'belize': 'BLZ',

 'benin': 'BEN',

 'bermuda': 'BMU',

 'bhutan': 'BTN',

 'bolivia': 'BOL',

 'bosnia and herzegovina': 'BIH',

 'botswana': 'BWA',

 'brazil': 'BRA',

 'british virgin islands': 'VGB',

 'brunei': 'BRN',

 'bulgaria': 'BGR',

 'burkina faso': 'BFA',

 'burma': 'MMR',

 'burundi': 'BDI',

 'cabo verde': 'CPV',

 'cambodia': 'KHM',

 'cameroon': 'CMR',

 'canada': 'CAN',

 'cayman islands': 'CYM',

 'central african republic': 'CAF',

 'chad': 'TCD',

 'chile': 'CHL',

 'china': 'CHN',

 'colombia': 'COL',

 'comoros': 'COM',

 'congo democratic': 'COD',

 'Congo republic': 'COG',

 'cook islands': 'COK',

 'costa rica': 'CRI',

 "cote d'ivoire": 'CIV',

 'croatia': 'HRV',

 'cuba': 'CUB',

 'curacao': 'CUW',

 'cyprus': 'CYP',

 'czech republic': 'CZE',

 'denmark': 'DNK',

 'djibouti': 'DJI',

 'dominica': 'DMA',

 'dominican republic': 'DOM',

 'ecuador': 'ECU',

 'egypt': 'EGY',

 'el salvador': 'SLV',

 'equatorial guinea': 'GNQ',

 'eritrea': 'ERI',

 'estonia': 'EST',

 'ethiopia': 'ETH',

 'falkland islands': 'FLK',

 'faroe islands': 'FRO',

 'fiji': 'FJI',

 'finland': 'FIN',

 'france': 'FRA',

 'french polynesia': 'PYF',

 'gabon': 'GAB',

 'gambia, the': 'GMB',

 'georgia': 'GEO',

 'germany': 'DEU',

 'ghana': 'GHA',

 'gibraltar': 'GIB',

 'greece': 'GRC',

 'greenland': 'GRL',

 'grenada': 'GRD',

 'guam': 'GUM',

 'guatemala': 'GTM',

 'guernsey': 'GGY',

 'guinea-bissau': 'GNB',

 'guinea': 'GIN',

 'guyana': 'GUY',

 'haiti': 'HTI',

 'honduras': 'HND',

 'hong kong': 'HKG',

 'hungary': 'HUN',

 'iceland': 'ISL',

 'india': 'IND',

 'indonesia': 'IDN',

 'iran': 'IRN',

 'iraq': 'IRQ',

 'ireland': 'IRL',

 'isle of man': 'IMN',

 'israel': 'ISR',

 'italy': 'ITA',

 'jamaica': 'JAM',

 'japan': 'JPN',

 'jersey': 'JEY',

 'jordan': 'JOR',

 'kazakhstan': 'KAZ',

 'kenya': 'KEN',

 'kiribati': 'KIR',

 'north korea': 'PRK',

 'south korea': 'KOR',

 'kosovo': 'KSV',

 'kuwait': 'KWT',

 'kyrgyzstan': 'KGZ',

 'laos': 'LAO',

 'latvia': 'LVA',

 'lebanon': 'LBN',

 'lesotho': 'LSO',

 'liberia': 'LBR',

 'libya': 'LBY',

 'liechtenstein': 'LIE',

 'lithuania': 'LTU',

 'luxembourg': 'LUX',

 'macau': 'MAC',

 'macedonia': 'MKD',

 'madagascar': 'MDG',

 'malawi': 'MWI',

 'malaysia': 'MYS',

 'maldives': 'MDV',

 'mali': 'MLI',

 'malta': 'MLT',

 'marshall islands': 'MHL',

 'mauritania': 'MRT',

 'mauritius': 'MUS',

 'mexico': 'MEX',

 'micronesia': 'FSM',

 'moldova': 'MDA',

 'monaco': 'MCO',

 'mongolia': 'MNG',

 'montenegro': 'MNE',

 'morocco': 'MAR',

 'mozambique': 'MOZ',

 'namibia': 'NAM',

 'nepal': 'NPL',

 'netherlands': 'NLD',

 'new caledonia': 'NCL',

 'new zealand': 'NZL',

 'nicaragua': 'NIC',

 'nigeria': 'NGA',

 'niger': 'NER',

 'niue': 'NIU',

 'northern mariana islands': 'MNP',

 'norway': 'NOR',

 'oman': 'OMN',

 'pakistan': 'PAK',

 'palau': 'PLW',

 'panama': 'PAN',

 'papua new guinea': 'PNG',

 'paraguay': 'PRY',

 'peru': 'PER',

 'philippines': 'PHL',

 'poland': 'POL',

 'portugal': 'PRT',

 'puerto rico': 'PRI',

 'qatar': 'QAT',

 'romania': 'ROU',

 'russia': 'RUS',

 'rwanda': 'RWA',

 'saint kitts and nevis': 'KNA',

 'saint lucia': 'LCA',

 'saint martin': 'MAF',

 'saint pierre and miquelon': 'SPM',

 'saint vincent and the grenadines': 'VCT',

 'samoa': 'WSM',

 'san marino': 'SMR',

 'sao tome and principe': 'STP',

 'saudi arabia': 'SAU',

 'senegal': 'SEN',

 'serbia': 'SRB',

 'seychelles': 'SYC',

 'sierra leone': 'SLE',

 'singapore': 'SGP',

 'sint maarten': 'SXM',

 'slovakia': 'SVK',

 'slovenia': 'SVN',

 'solomon islands': 'SLB',

 'somalia': 'SOM',

 'south africa': 'ZAF',

 'south sudan': 'SSD',

 'spain': 'ESP',

 'sri lanka': 'LKA',

 'sudan': 'SDN',

 'suriname': 'SUR',

 'swaziland': 'SWZ',

 'sweden': 'SWE',

 'switzerland': 'CHE',

 'syria': 'SYR',

 'taiwan': 'TWN',

 'tajikistan': 'TJK',

 'tanzania': 'TZA',

 'thailand': 'THA',

 'timor-leste': 'TLS',

 'togo': 'TGO',

 'tonga': 'TON',

 'trinidad and tobago': 'TTO',

 'tunisia': 'TUN',

 'turkey': 'TUR',

 'turkmenistan': 'TKM',

 'tuvalu': 'TUV',

 'uganda': 'UGA',

 'ukraine': 'UKR',

 'united arab emirates': 'ARE',

 'united kingdom': 'GBR',

 'united states': 'USA',

 'uruguay': 'URY',

 'uzbekistan': 'UZB',

 'vanuatu': 'VUT',

 'venezuela': 'VEN',

 'vietnam': 'VNM',

 'virgin islands': 'VGB',

 'west bank': 'WBG',

 'yemen': 'YEM',

 'zambia': 'ZMB',

 'zimbabwe': 'ZWE'}



## countries 

from collections import Counter

colorscale = ["#f7fbff", "#ebf3fb", "#deebf7", "#d2e3f3", "#c6dbef", "#b3d2e9", "#9ecae1",

    "#85bcdb", "#6baed6", "#57a0ce", "#4292c6", "#3082be", "#2171b5", "#1361a9",

    "#08519c", "#0b4083", "#08306b"

]

    

def geoplot(abc):

    country_with_code, country = {}, {}

    shows_countries = ", ".join(abc['country'].dropna()).split(", ")

    for c,v in dict(Counter(shows_countries)).items():

        code = ""

        if c.lower() in country_codes:

            code = country_codes[c.lower()]

        country_with_code[code] = v

        country[c] = v



    data = [dict(

            type = 'choropleth',

            locations = list(country_with_code.keys()),

            z = list(country_with_code.values()),

            colorscale = [[0,"rgb(5, 30, 172)"],[0.65,"rgb(40, 80, 140)"],[0.75,"rgb(30, 100, 245)"],\

                        [0.80,"rgb(90, 140, 235)"],[0.9,"rgb(120, 137, 247)"],[1,"rgb(120, 220, 220)"]],

            autocolorscale = False,

            reversescale = True,

            marker = dict(

                line = dict (

                    color = 'gray',

                    width = 0.5

                ) ),

            colorbar = dict(

                autotick = True,

                title = ''),

          ) ]



    layout = dict(

        title = '',

        geo = dict(

            showframe = False,

            showcoastlines = False,

            projection = dict(

                type = 'Mercator'

            )

        )

    )



    fig = dict( data=data, layout=layout )

    iplot( fig, validate=False, filename='d3-world-map' )

    return country



country_vals = geoplot(df)

tabs = Counter(country_vals).most_common(25)



labels = [_[0] for _ in tabs][::-1]

values = [_[1] for _ in tabs][::-1]

trace1 = go.Bar(y=labels, x=values, orientation="h", name="", marker=dict(color="#a678de"))



data = [trace1]

layout = go.Layout(title="Countries with most content", height=700, legend=dict(x=0.1, y=1.1, orientation="h"))

fig = go.Figure(data, layout=layout)

df.head()
import matplotlib.patheffects as path_effects

year_data = df['year_added'].value_counts().sort_index().loc[:2019]

type_data = df.groupby('type')['year_added'].value_counts().sort_index().unstack().fillna(0).T.loc[:2019] # you can split movie and tv show values with by years sample: 2019:TV show 2018:movie



fig, ax = plt.subplots(1,1, figsize=(28, 15))

ax.plot(year_data.index, year_data,  color="white", linewidth=5, label='Total', path_effects=[path_effects.SimpleLineShadow(),

                       path_effects.Normal()])

ax.plot(type_data.index, type_data['Movie'], color='skyblue', linewidth=5, label='Movie', path_effects=[path_effects.SimpleLineShadow(),

                       path_effects.Normal()])

ax.plot(type_data.index, type_data['TV Show'], color='salmon', linewidth=5, label='TV Show', path_effects=[path_effects.SimpleLineShadow(),

                       path_effects.Normal()])



ax.set_xlim(2006, 2020)

ax.set_ylim(-40, 2700)



t = [

    2008,

    2010.8,

    2012.1,

    2013.1,

    2015.7,

    2016.1,

    2016.9

]



events = [

    "Launch Streaming Video\n2007.1",

    "Expanding Streaming Service\nStarting with Candata | 2010.11",

    "Expanding to Europe\n2012.1",

    "First Original Content\n2013.2",

    "Expanding to Japan\n2015.9",

    "Original targeting Kids\n2016/1",

    "Offline Playback Features to all of Users\n2016/11"

]



up_down = [ 

    100,

    110,

    280,

    110,

    0,

    0,

    0

]



left_right = [

    -1,

    -0,

    -0,

    -0,

    -1,

    -1,

    -1.6,

    

]



for t_i, event_i, ud_i, lr_i in zip(t, events, up_down, left_right):

    ax.annotate(event_i,

                xy=(t_i + lr_i, year_data[int(t_i)] * (int(t_i+1)-t_i) + year_data[int(t_i)+1]  * (t_i-int(t_i)) + ud_i),

                xytext=(0,0), textcoords='offset points',

                va="center", ha="center",

                color="w", fontsize=16,

                bbox=dict(boxstyle='round4', pad=0.5, color='#303030', alpha=0.90))

    

    # A proportional expression to draw the middle of the year

    ax.scatter(t_i, year_data[int(t_i)] * (int(t_i+1)-t_i) + year_data[int(t_i)+1]  * (t_i-int(t_i)), color='#E50914', s=300)



ax.set_facecolor((0.4, 0.4, 0.4))

ax.set_title("Why Netflix's Conetents Count Soared?", position=(0.23, 1.0+0.03), fontsize=30, fontweight='bold')

ax.yaxis.set_tick_params(labelsize=20)

ax.xaxis.set_tick_params(labelsize=20)

plt.legend(loc='upper left', fontsize=20)



plt.show()
type_data.head(2)
year_data = df['year_added'].value_counts().sort_index().loc[:2019]

type_data = df.groupby('rating')['year_added'].value_counts().sort_index().unstack().fillna(0).T.loc[:2019] # you can split movie and tv show values with by years sample: 2019:TV show 2018:movie



fig, ax = plt.subplots(1,1, figsize=(28, 15))

ax.plot(year_data.index, year_data,  color="white", linewidth=5, label='Total', path_effects=[path_effects.SimpleLineShadow(),

                       path_effects.Normal()])

ax.plot(type_data.index, type_data['TV-PG'], color='skyblue', linewidth=5, label='TV-PG', path_effects=[path_effects.SimpleLineShadow(),

                       path_effects.Normal()])

ax.plot(type_data.index, type_data['TV-14'], color='salmon', linewidth=5, label='TV-14', path_effects=[path_effects.SimpleLineShadow(),

                       path_effects.Normal()])



ax.set_xlim(2006, 2020)

ax.set_ylim(-40, 2700)



t = [

    2009,

    2010.8,

    2014.1,

    2015.7,



]



events = [

    "We dont see TV-GB or 14",

    "2010-12 TV-14 little bit rise ",

    "still continue with investing process(guess)",

    "2 TV rising",



]



up_down = [ 

    100,

    110,

    280,

    110,

    0,

    0,

    0

]



left_right = [

    -1,

    -0,

    -0,

    -0,

    -1,

    -1,

    -1.6,

    

]



for t_i, event_i, ud_i, lr_i in zip(t, events, up_down, left_right):

    ax.annotate(event_i,

                xy=(t_i + lr_i, year_data[int(t_i)] * (int(t_i+1)-t_i) + year_data[int(t_i)+1]  * (t_i-int(t_i)) + ud_i),

                xytext=(0,0), textcoords='offset points',

                va="center", ha="center",

                color="w", fontsize=16,

                bbox=dict(boxstyle='round4', pad=0.5, color='#306030', alpha=0.90))

    

    # A proportional expression to draw the middle of the year

    ax.scatter(t_i, year_data[int(t_i)] * (int(t_i+1)-t_i) + year_data[int(t_i)+1]  * (t_i-int(t_i)), color='#E50918', s=300)



ax.set_facecolor((0.4, 0.4, 0.4))

ax.set_title("2 TV Channels Rise by Year", position=(0.23, 1.0+0.03), fontsize=30, fontweight='bold')

ax.yaxis.set_tick_params(labelsize=20)

ax.xaxis.set_tick_params(labelsize=20)

plt.legend(loc='upper left', fontsize=20)



plt.show()
tag = "International TV Shows"

df["relevant"] = df['listed_in'].fillna("").apply(lambda x : 1 if tag.lower() in x.lower() else 0)

small = df[df["relevant"] == 1]

small[small["country"] == "United States"][["title", "country","release_year"]].head(10)
df.head()
df2016 = df[df.release_year == 2016].iloc[:2,:] #you can see 

df2016
# prepare data frame



dtimes = df.iloc[:100,:]

stimes = dtimes.sort_values(by=['release_year'])



d1 = stimes[stimes["type"] == "TV Show"]

d2 = stimes[stimes["type"] == "Movie"]







# import graph objects as "go"

import plotly.graph_objs as go



# Creating trace1

trace1 = go.Scatter(

                    x = stimes.release_year,

                    y = d1,

                    mode = "lines",

                    name = "TV Show",

                    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),

                    text= dtimes.rating)

# Creating trace2

trace2 = go.Scatter(

                    x = stimes.release_year,

                    y = d2,

                    mode = "lines+markers",

                    name = "Movie",

                    marker = dict(color = 'rgba(80, 26, 80, 0.8)'),

                    text= dtimes.rating)

data = [trace1, trace2]

layout = dict(title = 'TV Show and Movie vs Country of Top 100 Watches',

              xaxis= dict(title= 'Release Year',ticklen= 5,zeroline= True)

             )

fig = dict(data = data, layout = layout)

iplot(fig)
df.head()
times= df[['date_added', 'type']]

de = times[times.type == "Movie"]

new_movie= de.max()

old_movie = de.min()

new_movie, old_movie
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

plt.rcParams['figure.figsize'] = (13, 13)

wordcloud = WordCloud(stopwords=STOPWORDS,background_color = 'black', width = 1000,  height = 1000, max_words = 121).generate(' '.join(df['listed_in']))

plt.imshow(wordcloud)

plt.axis('off')

plt.title('Most Popular Genere On Netflix',fontsize = 30)

plt.show()
df.head()
from plotly.offline import iplot, init_notebook_mode

d_df = df['duration'].value_counts().reset_index()



# plot

trace = go.Histogram(

                     x = d_df['duration'],

                     marker = dict(color = 'rgb(249, 6, 6)'))

layout = go.Layout(template= "plotly_dark", title = 'Total Duration - In Minutes', xaxis = dict(title = 'Time'))

fig = go.Figure(data = [trace], layout = layout)

fig.show()



c20 = df['country'].value_counts().reset_index()[:10]





# create trace1

trace1 = go.Bar(

                x = c20['index'],

                y = c20['country'],

                marker = dict(color = 'rgb(153,255,200)',

                              line=dict(color='rgb(100,0,0)',width=6.5)))

layout = go.Layout(template= "plotly_dark",title = 'TOP 10 COUNTRIES WITH MOST CONTENT' , xaxis = dict(title = 'Countries'), yaxis = dict(title = 'Count'))

fig = go.Figure(data = [trace1], layout = layout)

fig.show()
import missingno as msno

msno.matrix(df)

plt.show()
msno.bar(df)

plt.show()
df.head()
df["type"].reset_index()[:1000]
deneme=df["type"]
col = "type"

grouped = df[col].value_counts().reset_index()

grouped = grouped.rename(columns = {col : "count", "index" : col})
grouped