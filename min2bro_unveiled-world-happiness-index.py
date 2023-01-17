import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode()

# import plotly.plotly as py
df1=pd.read_csv('../input/2016.csv')
# Not sure how to install pycountry in kaggle so i just downloaded all the countries code in dictionary which would be used 

#for plotting the choropleth graph



countries = {'Bahamas': 'BHS', 'Guernsey': 'GGY', 'Saint Lucia': 'LCA', 'Bahrain': 'BHR', 'Dominican Republic': 'DOM', 

             'South Africa': 'ZAF', 'Sint Maarten (Dutch part)': 'SXM', 'French Guiana': 'GUF', 'Guinea-Bissau': 'GNB', 

             'Nigeria': 'NGA', 'Indonesia': 'IDN', 'Taiwan, Province of China': 'TWN', 'Tokelau': 'TKL', 'Suriname': 'SUR', 

             'Libya': 'LBY', 'Tunisia': 'TUN', 'Russian Federation': 'RUS', 'French Polynesia': 'PYF', 'Yemen': 'YEM', 

             'Nicaragua': 'NIC', 'Moldova, Republic of': 'MDA', 'Argentina': 'ARG', 'Cocos (Keeling) Islands': 'CCK', 

             'Cayman Islands': 'CYM', 'Cameroon': 'CMR', 'Paraguay': 'PRY', 'Ireland': 'IRL', 'Azerbaijan': 'AZE', 

             'Myanmar': 'MMR', 'Central African Republic': 'CAF', 'Åland Islands': 'ALA', 'Fiji': 'FJI', 'Monaco': 'MCO',

             'Austria': 'AUT', 'Germany': 'DEU', "Lao People's Democratic Republic": 'LAO', 'Cuba': 'CUB', 'Mauritius': 'MUS', 

             'Réunion': 'REU', 'Andorra': 'AND', 'Congo, The Democratic Republic of the': 'COD', 'Nauru': 'NRU', 'Maldives': 'MDV', 

             'Barbados': 'BRB', 'Sri Lanka': 'LKA', 'Bangladesh': 'BGD', 'France': 'FRA', 'Guadeloupe': 'GLP',

             'Micronesia, Federated States of': 'FSM', 'Colombia': 'COL', 'Kiribati': 'KIR', 'Brazil': 'BRA',

             'Hong Kong': 'HKG', 'Somalia': 'SOM', 'Malawi': 'MWI', 'Viet Nam': 'VNM', 'Italy': 'ITA',

             'Saint Helena, Ascension and Tristan da Cunha': 'SHN', 'Bhutan': 'BTN', 'Armenia': 'ARM', 

             'Slovakia': 'SVK', 'Peru': 'PER', 'Tuvalu': 'TUV', 'San Marino': 'SMR', 'Chile': 'CHL', 

             'Montenegro': 'MNE', 'Mali': 'MLI', 'Cook Islands': 'COK', 'Korea, Republic of': 'KOR',

             'Bonaire, Sint Eustatius and Saba': 'BES', 'Aruba': 'ABW', 'Slovenia': 'SVN', 'Cyprus': 'CYP', 

             'Namibia': 'NAM', 'Vanuatu': 'VUT', 'Kazakhstan': 'KAZ', 'United Kingdom': 'GBR',

             'American Samoa': 'ASM', 'Uzbekistan': 'UZB', 'Papua New Guinea': 'PNG', 'Zambia': 'ZMB',

             'Virgin Islands, U.S.': 'VIR', 'Guatemala': 'GTM', 'Macao': 'MAC', 'Norway': 'NOR', 'Croatia': 'HRV', 

             'Turks and Caicos Islands': 'TCA', 'Iraq': 'IRQ', 'Rwanda': 'RWA', 'Cabo Verde': 'CPV',

             'Western Sahara': 'ESH', 'New Zealand': 'NZL', 'Greece': 'GRC', 'Saint Martin (French part)': 'MAF', 

             'Trinidad and Tobago': 'TTO', 'Equatorial Guinea': 'GNQ', 'Netherlands': 'NLD', 'Samoa': 'WSM', 

             'Spain': 'ESP', 'Mauritania': 'MRT', 'Malaysia': 'MYS', 'Benin': 'BEN', "Korea, Democratic People's Republic of": 'PRK', 

             'Montserrat': 'MSR', 'Bolivia, Plurinational State of': 'BOL', 'Solomon Islands': 'SLB', 'Liberia': 'LBR', 

             'Panama': 'PAN', 'Kuwait': 'KWT', 'Singapore': 'SGP', 'Djibouti': 'DJI', 'Faroe Islands': 'FRO', 'Palau': 'PLW',

             'Nepal': 'NPL', 'Marshall Islands': 'MHL', 'Gambia': 'GMB', 'Tonga': 'TON', 'Morocco': 'MAR', 

             'Holy See (Vatican City State)': 'VAT', 'Jersey': 'JEY', 'Honduras': 'HND', 'Wallis and Futuna': 'WLF', 

             'Lebanon': 'LBN', 'Isle of Man': 'IMN', 'Antarctica': 'ATA', 'United States Minor Outlying Islands': 'UMI', 

             'Chad': 'TCD', 'Angola': 'AGO', 'Comoros': 'COM', 'Macedonia, Republic of': 'MKD', 'Saudi Arabia': 'SAU', 

             'Iceland': 'ISL', 'India': 'IND', 'Anguilla': 'AIA', 'Heard Island and McDonald Islands': 'HMD', 

             'French Southern Territories': 'ATF', 'Uruguay': 'URY', 'Malta': 'MLT', 'Northern Mariana Islands': 'MNP', 

             'Guam': 'GUM', 'South Georgia and the South Sandwich Islands': 'SGS', 

             'Syrian Arab Republic': 'SYR', 'Egypt': 'EGY', 'Hungary': 'HUN', 'Mexico': 'MEX',

             'Madagascar': 'MDG', 'Antigua and Barbuda': 'ATG', 'Lesotho': 'LSO', 'Jamaica': 'JAM',

             'Christmas Island': 'CXR', 'Martinique': 'MTQ', 'Philippines': 'PHL', 'Greenland': 'GRL',

             'Sudan': 'SDN', 'Falkland Islands (Malvinas)': 'FLK', 'Senegal': 'SEN', 'United States': 'USA',

             'Norfolk Island': 'NFK', 'Gabon': 'GAB', 'Luxembourg': 'LUX', 'Saint Pierre and Miquelon': 'SPM',

             'Congo': 'COG', 'Ethiopia': 'ETH', "Côte d'Ivoire": 'CIV', 'Serbia': 'SRB', 'Mongolia': 'MNG',

             'Haiti': 'HTI', 'Jordan': 'JOR', 'El Salvador': 'SLV', 'Estonia': 'EST', 'Canada': 'CAN', 'Georgia': 'GEO', 

             'Bulgaria': 'BGR', 'Cambodia': 'KHM', 'Sweden': 'SWE', 'Costa Rica': 'CRI', 'Romania': 'ROU', 'Saint Barthélemy': 'BLM',

             'Mozambique': 'MOZ', 'Kyrgyzstan': 'KGZ', 'Timor-Leste': 'TLS', 'Tajikistan': 'TJK', 'Brunei Darussalam': 'BRN',

             'Kenya': 'KEN', 'Finland': 'FIN', 'Zimbabwe': 'ZWE', 'Denmark': 'DNK', 'Turkmenistan': 'TKM', 'Niger': 'NER',

             'Burkina Faso': 'BFA', 'Thailand': 'THA', 'Palestine, State of': 'PSE', 'Ghana': 'GHA', 'Uganda': 'UGA',

             'Seychelles': 'SYC', 'Saint Kitts and Nevis': 'KNA', 'Mayotte': 'MYT', 'Afghanistan': 'AFG', 

             'Sao Tome and Principe': 'STP', 'Venezuela, Bolivarian Republic of': 'VEN', 'Iran, Islamic Republic of': 'IRN', 

             'Liechtenstein': 'LIE', 'Sierra Leone': 'SLE', 'Pakistan': 'PAK', 'Tanzania, United Republic of': 'TZA',

             'United Arab Emirates': 'ARE', 'Turkey': 'TUR', 'Japan': 'JPN', 'Belize': 'BLZ', 'British Indian Ocean Territory': 'IOT', 

             'Botswana': 'BWA', 'Swaziland': 'SWZ', 'Albania': 'ALB', 'Algeria': 'DZA', 'Burundi': 'BDI', 'Israel': 'ISR',

             'Portugal': 'PRT', 'Bouvet Island': 'BVT', 'Niue': 'NIU', 'Belgium': 'BEL', 'Belarus': 'BLR', 'Latvia': 'LVA', 

             'Qatar': 'QAT', 'Dominica': 'DMA', 'Eritrea': 'ERI', 'Oman': 'OMN', 'Saint Vincent and the Grenadines': 'VCT',

             'Svalbard and Jan Mayen': 'SJM', 'Guyana': 'GUY', 'Virgin Islands, British': 'VGB', 'Gibraltar': 'GIB', 

             'Poland': 'POL', 'Togo': 'TGO', 'Guinea': 'GIN', 'Pitcairn': 'PCN', 'Australia': 'AUS', 'Czechia': 'CZE', 

             'South Sudan': 'SSD', 'Bermuda': 'BMU', 'Ukraine': 'UKR', 'New Caledonia': 'NCL', 'China': 'CHN', 

             'Switzerland': 'CHE', 'Curaçao': 'CUW', 'Grenada': 'GRD'}
# Function for Converting the Country Name to 3 digits code for Choropleth map



# import pycountry



# countries = {}

# for country in pycountry.countries:

#     countries[country.name] = country.alpha_3





def getcountrycode(countryname):

    return(countries.get(countryname, 'Unknown code') )
df1['Code']=df1['Country'].apply(getcountrycode)
df1[df1['Code']=='Unknown code'].count()
df1=df1[df1['Code']!='Unknown code']
# df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2014_world_gdp_with_codes.csv')



data = [ dict(

        type = 'choropleth',

        locations = df1['Code'],

        z = df1['Happiness Rank'],

        text = df1['Country'],

        colorscale = [[0,"rgb(5, 10, 172)"],[0.35,"rgb(40, 60, 190)"],[0.5,"rgb(70, 100, 245)"],\

            [0.6,"rgb(90, 120, 245)"],[0.7,"rgb(106, 137, 247)"],[1,"rgb(220, 220, 220)"]],

        autocolorscale = False,

        reversescale = True,

        marker = dict(

            line = dict (

                color = 'rgb(120,120,120)',

                width = 0.5

            ) ),

        colorbar = dict(

            autotick = False,

            tickprefix = '',

            title = 'Happiness<br>Rank'),

      ) ]





layout = dict(

    title = '2016 Happiness Index:',

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
# Top Five happiest countries in world and their Happiness Score

sns.set_style("dark")

df2=df1.sort_values(by=['Happiness Rank'],ascending=True).head(5)

ax=sns.barplot(x='Country',y='Happiness Score',data=df2)

ax.set(xlabel='Country', ylabel='Happiness Score')
#Heatmap to find correlation between each of the features 

%matplotlib inline

import seaborn as sns

import matplotlib.pyplot as pl

import numpy as np

f, ax = pl.subplots(figsize=(10, 8))

corr = df1.corr()

sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),

            square=True, ax=ax)
#Box Plot of Happiness Score by Each Region

sns.set_context("notebook",font_scale=1.0)

plt.figure(figsize=(15,10))

sns.boxplot(x="Region", y="Happiness Score", data=df1)

plt.xticks(rotation=50)
#Happiness Score & Life Expectancy



g = sns.PairGrid(df1, vars=["Happiness Score", "Health (Life Expectancy)"], size=5)

g.map(plt.scatter)
# Happiness Score & Economy

g = sns.PairGrid(df1, vars=["Happiness Score", "Economy (GDP per Capita)"], size=5)

g.map(plt.scatter)