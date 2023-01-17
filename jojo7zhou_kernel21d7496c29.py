# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
'''Import basic modules.'''

import pandas as pd

import numpy as np





'''Customize visualization

Seaborn and matplotlib visualization.'''

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style("whitegrid")

%matplotlib inline



'''Plotly visualization .'''

import plotly.offline as py

import plotly.graph_objs as go



df = pd.read_csv("/kaggle/input/netflix-shows/netflix_titles.csv")
df
## add new features in the dataset

df["date_added"] = pd.to_datetime(df['date_added'])

df['year_added'] = df['date_added'].dt.year

df['month_added'] = df['date_added'].dt.month



df['season_count'] = df.apply(lambda x : x['duration'].split(" ")[0] if "Season" in x['duration'] else "", axis = 1)

df['duration'] = df.apply(lambda x : x['duration'].split(" ")[0] if "Season" not in x['duration'] else "", axis = 1)

df.head()
import pandas_profiling

pandas_profiling.ProfileReport(df)
dups=df.duplicated(['title','country','type','release_year'])

df[dups]
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

plt.rcParams['figure.figsize'] = (13, 13)

wordcloud = WordCloud(stopwords=STOPWORDS,background_color = 'black', width = 1000,  height = 1000, max_words = 121).generate(' '.join(df['title']))

plt.imshow(wordcloud)

plt.axis('off')

plt.title('Most Popular Words in Title',fontsize = 30)

plt.show()
from plotly.offline import init_notebook_mode, iplot

col = "type"

grouped = df[col].value_counts().reset_index()

grouped = grouped.rename(columns = {col : "count", "index" : col})



## plot

trace = go.Pie(labels=grouped[col], values=grouped['count'], pull=[0.05, 0], marker=dict(colors=["pink", "blue"]))

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
temp_df = df['rating'].value_counts().reset_index()





# create trace1

trace1 = go.Bar(

                x = temp_df['index'],

                y = temp_df['rating'],

                marker = dict(color = 'rgb(255,165,0)',

                              line=dict(color='rgb(0,0,0)',width=1.5)))

layout = go.Layout(template= "plotly_dark",title = 'MOST OF PROGRAMME ON NEYFLIX IS TV-14 & TV-MA RATED' , xaxis = dict(title = 'Rating'), yaxis = dict(title = 'Count'))

fig = go.Figure(data = [trace1], layout = layout)

fig.show()



def pie_plot(cnt_srs, title):

    labels=cnt_srs.index

    values=cnt_srs.values

    trace = go.Pie(labels=labels, 

                   values=values, 

                   title=title, 

                   hoverinfo='percent+value', 

                   textinfo='percent',

                   textposition='inside',

                   hole=0.7,

                   showlegend=True,

                   marker=dict(colors=plt.cm.viridis_r(np.linspace(0, 1, 14)),

                               line=dict(color='pink',

                                         width=2),

                              )

                  )

    return trace



py.iplot([pie_plot(df['rating'].value_counts(), 'Content Type')])
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
df1 = df[df["type"] == "TV Show"]

df2 = df[df["type"] == "Movie"]



temp_df1 = df1['rating'].value_counts().reset_index()

temp_df2 = df2['rating'].value_counts().reset_index()





# create trace1

trace1 = go.Bar(

                x = temp_df1['index'],

                y = temp_df1['rating'],

                name="TV Shows",

                marker = dict(color = 'rgb(249, 6, 6)',

                             line=dict(color='pink',width=1.5)))

# create trace2 

trace2 = go.Bar(

                x = temp_df2['index'],

                y = temp_df2['rating'],

                name = "Movies",

                marker = dict(color = 'rgb(26, 118, 255)',

                              line=dict(color='green',width=1.5)))





layout = go.Layout(template= "plotly_dark",title = 'RATING BY CONTENT TYPE' , xaxis = dict(title = 'Rating'), yaxis = dict(title = 'Count'))

fig = go.Figure(data = [trace1, trace2], layout = layout)

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

    

def geoplot(ddf):

    country_with_code, country = {}, {}

    shows_countries = ", ".join(ddf['country'].dropna()).split(", ")

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

            colorscale = [[0,"rgb(5, 10, 172)"],[0.65,"rgb(40, 60, 190)"],[0.75,"rgb(70, 100, 245)"],\

                        [0.80,"rgb(90, 120, 245)"],[0.9,"rgb(106, 137, 247)"],[1,"rgb(220, 220, 220)"]],

            autocolorscale = False,

            reversescale = True,

            marker = dict(

                line = dict (

                    color = 'gray',

                    width = 0.5

                ) ),

            colorbar = dict(

                autotick = False,

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

trace1 = go.Bar(y=labels, x=values, orientation="h", name="", marker=dict(color="red"))



data = [trace1]

layout = go.Layout(title="Countries with most content", height=700, legend=dict(x=0.1, y=1.1, orientation="h"))

fig = go.Figure(data, layout=layout)

fig.show()
temp_df = df['country'].value_counts().reset_index()[:20]





# create trace1

trace1 = go.Bar(

                x = temp_df['index'],

                y = temp_df['country'],

                marker = dict(color = 'red',

                              line=dict(color='red',width=1.5)))

layout = go.Layout(template= "plotly_dark",title = 'TOP 20 COUNTIES WITH MOST CONTENT' , xaxis = dict(title = 'Countries'), yaxis = dict(title = 'Count'))

fig = go.Figure(data = [trace1], layout = layout)

fig.show()
temp_df1 = df['release_year'].value_counts().reset_index()





# create trace1

trace1 = go.Bar(

                x = temp_df1['index'],

                y = temp_df1['release_year'],

                marker = dict(color = 'blue',

                             line=dict(color='rgb(0,0,0)',width=1.5)))

layout = go.Layout(template= "plotly_dark",title = 'CONTENT RELEASE OVER THE YEAR' , xaxis = dict(title = 'Rating'), yaxis = dict(title = 'Count'))

fig = go.Figure(data = [trace1], layout = layout)

fig.show()
df1 = df[df["type"] == "TV Show"]

df2 = df[df["type"] == "Movie"]



temp_df1 = df1['release_year'].value_counts().reset_index()

temp_df2 = df2['release_year'].value_counts().reset_index()





# create trace1

trace1 = go.Bar(

                x = temp_df1['index'],

                y = temp_df1['release_year'],

                name="TV Shows",

                marker = dict(color = 'blue'))

# create trace2 

trace2 = go.Bar(

                x = temp_df2['index'],

                y = temp_df2['release_year'],

                name = "Movies",

                marker = dict(color = 'red'))





layout = go.Layout(template= "plotly_dark",title = 'CONTENT RELEASE OVER THE YEAR BY CONTENT TYPE' , xaxis = dict(title = 'Year'), yaxis = dict(title = 'Count'))

fig = go.Figure(data = [trace1, trace2], layout = layout)

fig.show()
trace = go.Histogram(

                     x = df['duration'],

                     xbins=dict(size=0.5),

                     marker = dict(color = 'rgb(26, 118, 255)'))

layout = go.Layout(template= "plotly_dark", title = 'Distribution of Movies Duration', xaxis = dict(title = 'Minutes'))

fig = go.Figure(data = [trace], layout = layout)

fig.show()