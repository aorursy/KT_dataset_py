# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from plotly.offline import init_notebook_mode, plot, iplot
import plotly as py
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import pycountry

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
df=pd.read_csv("/kaggle/input/suicide-rates-overview-1985-to-2016/master.csv")
df.columns=['country', 'year', 'gender', 'age', 'suicides_no', 'population',
       'suicides_100k_pop', 'country-year', 'HDI_for_year',
       'gdp_for_year ', 'gdp_per_capita', 'generation']
df.columns
df.corr()
f,ax = plt.subplots(figsize=(16, 16))
sns.heatmap(df.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()
df.head(10)
df["year"].unique()
f,ax1 = plt.subplots(figsize =(20,10))
sns.pointplot(x=df.groupby('year')['suicides_100k_pop'].mean().index,y=df.groupby('year')['suicides_100k_pop'].mean().values,data=df,color='lime',alpha=0.8)
plt.xlabel('Years',fontsize = 15,color='blue')
plt.ylabel('Suicides per 100K',fontsize = 15,color='blue')
plt.title('Global Suicides(per 100K)-trend over time 1985-2016',fontsize = 20,color='blue')
plt.grid()
ASIA = ['Afghanistan',
'Bangladesh',
'Bhutan',
'Brunei',
'Burma',
'Cambodia',
'China',
'East Timor',
'Hong Kong',
'India',
'Indonesia',
'Iran',
'Japan',
'Republic of Korea',
'Laos',
'Macau',
'Malaysia',
'Maldives',
'Mongolia',
'Nepal',
'Pakistan',
'Philippines',
'Singapore',
'Sri Lanka',
'Taiwan',
'Thailand',
'Vietnam']

C_W_OF_IND_STATES=['Armenia',
'Azerbaijan',
'Belarus',
'Georgia',
'Kazakhstan',
'Kyrgyzstan',
'Moldova',
'Russian Federation',
'Tajikistan',
'Turkmenistan',
'Ukraine',
'Uzbekistan']
EASTERN_EUROPE=['Albania','Bosnia and Herzegovina','Bulgaria','Croatia','Czech Republic','Hungary','Macedonia','Poland','Romania']
EASTERN_EUROPE+=['Serbia','Slovakia','Slovenia']
LATIN_AMER_CARIB=['Anguilla',
'Antigua and Barbuda',
'Argentina',
'Aruba',
'Bahamas',
'Barbados',
'Belize',
'Bolivia',
'Brazil',
'British Virgin Is.',
'Cayman Islands',
'Chile',
'Colombia',
'Costa Rica',
'Cuba',
'Dominica',
'Dominican Republic',
'Ecuador',
'El Salvador',
'French Guiana',
'Grenada',
'Guadeloupe',
'Guatemala',
'Guyana',
'Haiti',
'Honduras',
'Jamaica',
'Martinique',
'Mexico',
'Montserrat',
'Netherlands Antilles',
'Nicaragua',
'Panama',
'Paraguay',
'Peru',
'Puerto Rico',
'Saint Kitts and Nevis',
'Saint Lucia',
'Saint Vincent and Grenadines',
'Suriname',
'Trinidad and Tobago',
'Turks and Caicos Is',
'Uruguay',
'Venezuela',
'Virgin Islands']

NEAR_EAST=['Bahrain',
'Cyprus',
'Gaza Strip',
'Iraq',
'Israel',
'Jordan',
'Kuwait',
'Lebanon',
'Oman',
'Qatar',
'Saudi Arabia',
'Syria',
'Turkey',
'United Arab Emirates',
'West Bank',
'Yemen']

NORTHERN_AFRICA=['Algeria',
'Egypt',
'Libya',
'Morocco',
'Tunisia',
'Western Sahara']
NORTHERN_AMERICA=['Bermuda',
'Canada',
'Greenland',
'St Pierre and Miquelon',
'United States']

OCEANIA=['American Samoa',
'Australia',
'Cook Islands',
'Fiji',
'French Polynesia',
'Guam',
'Kiribati',
'Marshall Islands',
'Micronesia, Fed. St.',
'Nauru',
'New Caledonia',
'New Zealand',
'N. Mariana Islands',
'Palau',
'Papua New Guinea',
'Samoa',
'Solomon Islands',
'Tonga',
'Tuvalu',
'Vanuatu',
'Wallis and Futuna']

SUB_SAHARAN_AFRICA=['Angola',
'Benin',
'Botswana',
'Burkina Faso',
'Burundi',
'Cameroon',
'Cape Verde',
'Central African Rep.',
'Chad',
'Comoros',
'Congo, Dem. Rep.',
'Congo, Repub. of the',
'Cote dIvoire',
'Djibouti',
'Equatorial Guinea',
'Eritrea',
'Ethiopia',
'Gabon',
'Gambia, The',
'Ghana',
'Guinea',
'Guinea-Bissau',
'Kenya',
'Lesotho',
'Liberia',
'Madagascar',
'Malawi',
'Mali',
'Mauritania',
'Mauritius',
'Mayotte',
'Mozambique',
'Namibia',
'Niger',
'Nigeria',
'Reunion',
'Rwanda',
'Saint Helena',
'Sao Tome & Principe',
'Senegal',
'Seychelles',
'Sierra Leone',
'Somalia',
'South Africa',
'Sudan',
'Swaziland',
'Tanzania',
'Togo',
'Uganda',
'Zambia',
'Zimbabwe']
WESTERN_EUROPE=['Andorra',
'Austria',
'Belgium',
'Denmark',
'Faroe Islands',
'Finland',
'France',
'Germany',
'Gibraltar',
'Greece',
'Guernsey',
'Iceland',
'Ireland',
'Isle of Man',
'Italy',
'Jersey',
'Liechtenstein',
'Luxembourg',
'Malta',
'Monaco',
'Netherlands',
'Norway',
'Portugal',
'San Marino',
'Spain',
'Sweden',
'Switzerland',
'United Kingdom']
def GetConti(counry):
    if counry in ASIA:
        return "ASIA"
    elif counry in C_W_OF_IND_STATES:
        return "C_W_OF_IND_STATES"
    elif counry in EASTERN_EUROPE:
        return "EASTERN_EUROPE"
    elif counry in LATIN_AMER_CARIB:
        return "LATIN_AMER_CARIB"
    elif counry in NEAR_EAST:
        return "NEAR_EAST"
    elif counry in NORTHERN_AFRICA:
        return "NORTHERN_AFRICA"
    elif counry in NORTHERN_AMERICA:
        return "NORTHERN_AMERICA"
    elif counry in OCEANIA:
        return "OCEANIA"
    elif counry in SUB_SAHARAN_AFRICA:
        return "SUB_SAHARAN_AFRICA"
    elif counry in WESTERN_EUROPE:
        return "WESTERN_EUROPE"
    else:
        return "other"
country=df["country"]
country=pd.DataFrame(country)
# list(country["country"])
df1 = pd.DataFrame({"Country": list(country["country"])})
df1['Continent'] = df1['Country'].apply(lambda x: GetConti(x))
df["continent"]=df1["Continent"]
df[df["continent"]=="other"]["country"]
continent_list=list(df['continent'].unique())
suicides_100k_pop = []
for i in continent_list:
    x = df[df['continent']==i]
    rate = sum(x.suicides_100k_pop)/len(x)
    suicides_100k_pop.append(rate)
data1 = pd.DataFrame({'Continent_list': continent_list,'suicides_100k_pop':suicides_100k_pop})

plt.figure(figsize = (15,15))
plt.subplot(2,2,1)
sns.barplot(x=df.groupby('continent')['suicides_100k_pop'].mean().index,y=df.groupby('continent')['suicides_100k_pop'].mean().values)
plt.title("Global Suicides(per 100K) by Continent")
plt.ylabel("Suicide per 100K")
plt.xlabel("Continents")
plt.xticks(rotation=90)

plt.subplot(2,2,2)
labels =data1.Continent_list
colors = ['grey','blue','red','yellow','green',"orange", "darkblue","purple","maroon","gold"]
explode = [0,0,0,0,0,0,0,0,0,0]
sizes = data1.suicides_100k_pop
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%')
plt.title('Global Suicides(per 100K) rate of Continents',color = 'blue',fontsize = 15)
plt.show()
data=pd.DataFrame()
data["gender"]=df.groupby('gender')['suicides_100k_pop'].mean().index
data["suicides_100k_pop_means"]=df.groupby('gender')['suicides_100k_pop'].mean().values

plt.figure(figsize = (20,20))
plt.subplot(3,3,1)
sns.barplot(x=data["gender"],y=data['suicides_100k_pop_means'])
plt.title("Global Suicides(per 100K) by Continent")
plt.ylabel("Suicide per 100K")
plt.xlabel("Gender")

data_male=pd.DataFrame()
data_male["year"]=df[df["gender"]=="male"].groupby('year')['suicides_100k_pop'].mean().index
data_male["suicides_100k_pop_means"]=df[df["gender"]=="male"].groupby('year')['suicides_100k_pop'].mean().values
data_female=pd.DataFrame()
data_female["year"]=df[df["gender"]=="female"].groupby('year')['suicides_100k_pop'].mean().index
data_female["suicides_100k_pop_means"]=df[df["gender"]=="female"].groupby('year')['suicides_100k_pop'].mean().values

plt.subplot(3,3,2)
sns.pointplot(x=data_male["year"],y=data_male['suicides_100k_pop_means'],data=data_male,color='lime',alpha=0.8)
plt.title('Trend over time 1985-2016 of Male Suicide')

plt.xticks(rotation=90)

plt.subplot(3,3,3)

sns.pointplot(x=data_female["year"],y=data_female['suicides_100k_pop_means'],data=data_female,color='orange',alpha=0.8)
plt.title('Trend over time 1985-2016 of Female Suicide')
plt.xticks(rotation=90)
plt.grid()
plt.show()

df.plot(kind='scatter', x='population', y='gdp_per_capita',alpha = 0.5,color = 'red')
plt.xlabel('population')              # label = name of label
plt.ylabel('gdp_per_capita')
plt.title('Population-gdp_per_capita Plot')            # title = title of plot
plt.show()

df_country_per_capita=pd.DataFrame()
df_country_per_capita["country"]=df.groupby('country')['suicides_100k_pop'].mean().index

sns.jointplot(df.groupby('country')['suicides_100k_pop'].mean().values,df.groupby('country')['gdp_per_capita'].mean().values, kind="scatter", size=7)
plt.xlabel("Suicides per 100k")
plt.ylabel("GDP per Capita($)")
plt.title("Correlation between GDP(per Capita) and suicides per 100k\n\n\n\n")
plt.show()

sns.kdeplot(df.groupby('country')['suicides_100k_pop'].mean().values, df.groupby('country')['gdp_per_capita'].mean().values, shade=False, cut=3)
plt.xlabel("Suicides per 100k")
plt.ylabel("GDP per Capita($)")
plt.show()
sns.countplot(df.generation,hue=df.gender)
plt.title('Generation hue Gender Counter')
plt.xticks(rotation=45)

plt.show()
plt.figure(figsize=(16,7))
bar_age = sns.barplot(x = 'gender', y = 'suicides_no', hue = 'age',data = df)
g = sns.lmplot(x="year", y="suicides_no", hue="generation",
               truncate=True, height=5, data=df)
g.set_axis_labels("Year", "Suicides No")
plt.show()
df_city=pd.read_csv("../input/world-cities/worldcities.csv")
df_city=df_city[df_city['capital'] == "primary"]

latitude=[]
longitude=[]
name=list(df_city['country'].unique())

for j in range(len(name)):
    for i in list(df["country"]):  
        if name[j]==i:
            latitude.append([i,df_city.iloc[j]["lat"]])
            longitude.append([i,df_city.iloc[j]["lng"]])
df_city.head()
#read country info data ['country', 'latitude', 'longitude', 'name'
def Getlatitude(a):
    for i in range(len(latitude)):
        if a == latitude[i][0]:
            return latitude[i][1]
def Getlongitude(b):
    for i in range(len(longitude)):
        if b == longitude[i][0]:
            return longitude[i][1]

df1 = pd.DataFrame({"Country": list(df["country"])})
df1['latitude'] = df1['Country'].apply(lambda x: Getlatitude(x))
df1['longitude'] = df1['Country'].apply(lambda x: Getlongitude(x))
df["latitude"]=df1['latitude']
df["longitude"]=df1['longitude']
df.head(10)

# df.info()
df["generation"].unique()
dataset = df.loc[:,["year","latitude","longitude","generation","continent","suicides_100k_pop"]]
years = [str(each) for each in list(df.year.unique())] 
dataset.info()
dataset[dataset['year'] == 1987]

# make list of types
types = ['Generation X', 'Silent', 'G.I. Generation', 'Boomers','Millenials', 'Generation Z']
custom_colors = {
    'Generation X': 'red',
    'Silent':"yellow",
    'G.I. Generation': 'blue',
    'Boomers': 'green',
    'Millenials': 'orange',
    'Generation Z': 'black'
}
# make figure
figure = {
    'data': [],
    'layout': {},
    'frames': []
}

figure['layout']['geo'] = dict(showframe=False, showland=True, showcoastlines=True, showcountries=True,
               countrywidth=1, 
              landcolor = 'rgb(217, 217, 217)',
              subunitwidth=1,
              showlakes = True,
              lakecolor = 'rgb(255, 255, 255)',
              countrycolor="rgb(5, 5, 5)")
figure['layout']['hovermode'] = 'closest'
figure['layout']['sliders'] = {
    'args': [
        'transition', {
            'duration': 300,
            'easing': 'cubic-in-out'
        }
    ],
    'initialValue': '1987',
    'plotlycommand': 'animate',
    'values': years,
    'visible': True
}
figure['layout']['updatemenus'] = [
    {
        'buttons': [
            {
                'args': [None, {'frame': {'duration': 400, 'redraw': False},
                         'fromcurrent': True, 'transition': {'duration': 300, 'easing': 'quadratic-in-out'}}],
                'label': 'Play',
                'method': 'animate'
            },
            {
                'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate',
                'transition': {'duration': 0}}],
                'label': 'Pause',
                'method': 'animate'
            }
        ],
        'direction': 'left',
        'pad': {'r': 10, 't': 87},
        'showactive': False,
        'type': 'buttons',
        'x': 0.1,
        'xanchor': 'right',
        'y': 0,
        'yanchor': 'top'
    }
]

sliders_dict = {
    'active': 0,
    'yanchor': 'top',
    'xanchor': 'left',
    'currentvalue': {
        'font': {'size': 20},
        'prefix': 'Year:',
        'visible': True,
        'xanchor': 'right'
    },
    'transition': {'duration': 900, 'easing': 'cubic-in-out'},
    'pad': {'b': 10, 't': 50},
    'len': 0.9,
    'x': 0.1,
    'y': 0,
    'steps': []
}

# make data
year = 1695
for ty in types:
    dataset_by_year = dataset[dataset['year'] == year]
    dataset_by_year_and_cont = dataset_by_year[dataset_by_year['generation'] == ty]

    data_dict = dict(
    type='scattergeo',
    lon = dataset['longitude'],
    lat = dataset['latitude'],
    hoverinfo = 'text',
    text = ty,
    mode = 'markers',
    marker=dict(
        sizemode = 'area',
        sizeref = 1,
        size= 10 ,
        line = dict(width=1,color = "white"),
        color = custom_colors[ty],
        opacity = 0.7),
)
    figure['data'].append(data_dict)
    
# make frames
for year in years:
    frame = {'data': [], 'name': str(year)}
    for ty in types:
        dataset_by_year = dataset[dataset['year'] == int(year)]
        dataset_by_year_and_cont = dataset_by_year[dataset_by_year['generation'] == ty]

        data_dict = dict(
                type='scattergeo',
                lon = dataset_by_year_and_cont['longitude'],
                lat = dataset_by_year_and_cont['latitude'],
                hoverinfo = 'text',
                text = ty,
                mode = 'markers',
                marker=dict(
                    sizemode = 'area',
                    sizeref = 1,
                    size= 10 ,
                    line = dict(width=1,color = "white"),
                    color = custom_colors[ty],
                    opacity = 0.7),
                name = ty
            )
        frame['data'].append(data_dict)

    figure['frames'].append(frame)
    slider_step = {'args': [
        [year],
        {'frame': {'duration': 300, 'redraw': False},
         'mode': 'immediate',
       'transition': {'duration': 300}}
     ],
     'label': year,
     'method': 'animate'}
    sliders_dict['steps'].append(slider_step)


figure["layout"]["autosize"]= True
figure["layout"]["title"] = "Suicides of Generations"       

figure['layout']['sliders'] = [sliders_dict]

iplot(figure)