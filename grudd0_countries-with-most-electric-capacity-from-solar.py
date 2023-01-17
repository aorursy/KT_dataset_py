%matplotlib inline

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', None)

pd.set_option('max_colwidth',-1)

from plotly.offline import download_plotlyjs, init_notebook_mode, iplot

from plotly.graph_objs import *

init_notebook_mode()
df = pd.read_csv('../input/all_energy_statistics.csv')

df.columns = ['country','commodity','year','unit','quantity','footnotes','category']
df_solar = df[df.commodity.str.contains

                  ('Electricity - total net installed capacity of electric power plants, solar')]

df_max = df_solar.groupby(pd.Grouper(key='country'))['quantity'].max()

df_max = df_max.sort_values(ascending=False)

df_max = df_max[:6]

df_max.index.values;
commodity_string = 'Electricity - total net installed capacity of electric power plants, solar'

df_max = df[df.commodity.str.contains

            (commodity_string)].groupby(pd.Grouper(key='country'))['quantity'].max().sort_values(ascending=False)[:6]

range = np.arange(2000,2015)

dict_major = {}

for c in df_max.index.values:

    read_index = df_solar[df_solar.commodity.str.contains(commodity_string) & df_solar.country.str.contains(c + '$')].year

    read_data = df_solar[df_solar.commodity.str.contains(commodity_string) & df_solar.country.str.contains(c + '$')].quantity

    read_data.index=read_index

    prod = read_data.reindex(index=range,fill_value=0)

    dict_major.update({c:prod.values})

df_major = pd.DataFrame(dict_major)

df_major.index = range

df_major
ax = df_major.plot(kind='bar',x=df_major.index,stacked=False,figsize=(15,9))

plt.title('Solar energy production')

plt.xlabel('Year')

plt.ylabel("Megawatts")

ax.yaxis.grid(False,'minor') # turn off minor tic grid lines

ax.yaxis.grid(True,'major') # turn on major tic grid lines;
dict_alpha3 = {'Afghanistan': 'AFG',

 'Albania': 'ALB',

 'Algeria': 'DZA',

 'American Samoa': 'ASM',

 'Andorra': 'AND',

 'Angola': 'AGO',

 'Anguilla': 'AIA',

 'Antarctic Fisheries': '@@@',

 'Antigua and Barbuda': 'ATG',

 'Argentina': 'ARG',

 'Armenia': 'ARM',

 'Aruba': 'ABW',

 'Australia': 'AUS',

 'Austria': 'AUT',

 'Azerbaijan': 'AZE',

 'Bahamas': 'BHS',

 'Bahrain': 'BHR',

 'Bangladesh': 'BGD',

 'Barbados': 'BRB',

 'Belarus': 'BLR',

 'Belgium': 'BEL',

 'Belize': 'BLZ',

 'Benin': 'BEN',

 'Bermuda': 'BMU',

 'Bhutan': 'BTN',

 'Bolivia (Plur. State of)': 'BOL',

 'Bonaire, St Eustatius, Saba': 'BIH',

 'Bosnia and Herzegovina': 'BIH',

 'Botswana': 'BWA',

 'Brazil': 'BRA',

 'British Virgin Islands': 'VGB',

 'Brunei Darussalam': 'BRN',

 'Bulgaria': 'BGR',

 'Burkina Faso': 'BFA',

 'Burundi': 'BDI',

 'Cabo Verde': 'CPV',

 'Cambodia': 'KHM',

 'Cameroon': 'CMR',

 'Canada': 'CAN',

 'Cayman Islands': 'CYM',

 'Central African Rep.': 'CAF',

 'Chad': 'TCD',

 'Chile': 'CHL',

 'China': 'CHN',

 'China, Hong Kong SAR': 'HKG',

 'China, Macao SAR': 'MAC',

 'Colombia': 'COL',

 'Commonwealth of Independent States (CIS)': '@@@',

 'Comoros': 'COM',

 'Congo': 'COG',

 'Cook Islands': 'COK',

 'Costa Rica': 'CRI',

 'Croatia': 'HRV',

 'Cuba': 'CUB',

 'Curaçao': 'CUW',

 'Cyprus': 'CYP',

 'Czechia': 'CZE',

 'Czechoslovakia (former)': 'CZE',

 "Côte d'Ivoire": 'CIV',

 'Dem. Rep. of the Congo': 'COD',

 'Denmark': 'DNK',

 'Djibouti': 'DJI',

 'Dominica': 'DMA',

 'Dominican Republic': 'DOM',

 'Ecuador': 'ECU',

 'Egypt': 'EGY',

 'El Salvador': 'SLV',

 'Equatorial Guinea': 'GNQ',

 'Eritrea': 'ERI',

 'Estonia': 'EST',

 'Ethiopia': 'ETH',

 'Ethiopia, incl. Eritrea': 'ETH',

 'Faeroe Islands': 'FRO',

 'Falkland Is. (Malvinas)': 'MDV',

 'Fiji': 'FJI',

 'Finland': 'FIN',

 'France': 'FRA',

 'French Guiana': 'GUF',

 'French Polynesia': 'PYF',

 'Gabon': 'GAB',

 'Gambia': 'GMB',

 'Georgia': 'GEO',

 'German Dem. R. (former)': '@@@',

 'Germany': 'DEU',

 'Germany, Fed. R. (former)': '@@@',

 'Ghana': 'GHA',

 'Gibraltar': 'GIB',

 'Greece': 'GRC',

 'Greenland': 'GRL',

 'Grenada': 'GRD',

 'Guadeloupe': 'GLP',

 'Guam': 'GUM',

 'Guatemala': 'GTM',

 'Guernsey': 'GGY',

 'Guinea': 'GIN',

 'Guinea-Bissau': 'GNB',

 'Guyana': 'GUY',

 'Haiti': 'HTI',

 'Honduras': 'HND',

 'Hungary': 'HUN',

 'Iceland': 'ISL',

 'India': 'IND',

 'Indonesia': 'IDN',

 'Iran (Islamic Rep. of)': 'IRN',

 'Iraq': 'IRQ',

 'Ireland': 'IRL',

 'Isle of Man': 'IMN',

 'Israel': 'ISR',

 'Italy': 'ITA',

 'Jamaica': 'JAM',

 'Japan': 'JPN',

 'Jersey': 'JEY',

 'Jordan': 'JOR',

 'Kazakhstan': 'KAZ',

 'Kenya': 'KEN',

 'Kiribati': 'KIR',

 "Korea, Dem.Ppl's.Rep.": 'PRK',

 'Korea, Republic of': 'KOR',

 'Kuwait': 'KWT',

 'Kyrgyzstan': 'KGZ',

 "Lao People's Dem. Rep.": 'LAO',

 'Latvia': 'LVA',

 'Lebanon': 'LBN',

 'Lesotho': 'LSO',

 'Liberia': 'LBR',

 'Libya': 'LBY',

 'Liechtenstein': 'LIE',

 'Lithuania': 'LTU',

 'Luxembourg': 'LUX',

 'Madagascar': 'MDG',

 'Malawi': 'MWI',

 'Malaysia': 'MYS',

 'Maldives': 'MDV',

 'Mali': 'MLI',

 'Malta': 'MLT',

 'Marshall Islands': 'MHL',

 'Martinique': 'MTQ',

 'Mauritania': 'MRT',

 'Mauritius': 'MUS',

 'Mayotte': 'MYT',

 'Mexico': 'MEX',

 'Micronesia (Fed. States of)': 'FSM',

 'Mongolia': 'MNG',

 'Montenegro': 'MNE',

 'Montserrat': 'MSR',

 'Morocco': 'MAR',

 'Mozambique': 'MOZ',

 'Myanmar': 'MMR',

 'Namibia': 'NAM',

 'Nauru': 'NRU',

 'Nepal': 'NPL',

 'Neth. Antilles (former)': '@@@',

 'Netherlands': 'NLD',

 'New Caledonia': 'NCL',

 'New Zealand': 'NZL',

 'Nicaragua': 'NIC',

 'Niger': 'NER',

 'Nigeria': 'NGA',

 'Niue': 'NIU',

 'Northern Mariana Islands': 'MNP',

 'Norway': 'NOR',

 'Oman': 'OMN',

 'Other Asia': '@@@',

 'Pacific Islands (former)': '@@@',

 'Pakistan': 'PAK',

 'Palau': 'PLW',

 'Panama': 'PAN',

 'Papua New Guinea': 'PNG',

 'Paraguay': 'PRY',

 'Peru': 'PER',

 'Philippines': 'PHL',

 'Poland': 'POL',

 'Portugal': 'PRT',

 'Puerto Rico': 'PRI',

 'Qatar': 'QAT',

 'Republic of Moldova': 'MDA',

 'Romania': 'ROU',

 'Russian Federation': 'RUS',

 'Rwanda': 'RWA',

 'Réunion': 'REU',

 'Samoa': 'WSM',

 'Sao Tome and Principe': 'STP',

 'Saudi Arabia': 'SAU',

 'Senegal': 'SEN',

 'Serbia': 'SRB',

 'Serbia and Montenegro': 'SRB',

 'Seychelles': 'SYC',

 'Sierra Leone': 'SLE',

 'Singapore': 'SGP',

 'Sint Maarten (Dutch part)': 'SXM',

 'Slovakia': 'SVK',

 'Slovenia': 'SVN',

 'Solomon Islands': 'SLB',

 'Somalia': 'SOM',

 'South Africa': 'ZAF',

 'South Sudan': 'SSD',

 'Spain': 'ESP',

 'Sri Lanka': 'LKA',

 'St. Helena and Depend.': 'SHN',

 'St. Kitts-Nevis': 'KNA',

 'St. Lucia': 'LCA',

 'St. Pierre-Miquelon': 'SPM',

 'St. Vincent-Grenadines': 'VCT',

 'State of Palestine': 'PSE',

 'Sudan': 'SDN',

 'Sudan (former)': 'SDN',

 'Suriname': 'SUR',

 'Swaziland': 'SWZ',

 'Sweden': 'SWE',

 'Switzerland': 'CHE',

 'Syrian Arab Republic': 'SYR',

 'T.F.Yug.Rep. Macedonia': 'MKD',

 'Tajikistan': 'TJK',

 'Thailand': 'THA',

 'Timor-Leste': 'TLS',

 'Togo': 'TGO',

 'Tonga': 'TON',

 'Trinidad and Tobago': 'TTO',

 'Tunisia': 'TUN',

 'Turkey': 'TUR',

 'Turkmenistan': 'TKM',

 'Turks and Caicos Islands': 'TCA',

 'Tuvalu': 'TUV',

 'USSR (former)': '@@@',

 'Uganda': 'UGA',

 'Ukraine': 'UKR',

 'United Arab Emirates': 'ARE',

 'United Kingdom': 'GBR',

 'United Rep. of Tanzania': 'TZA',

 'United States': 'USA',

 'United States Virgin Is.': 'VIR',

 'Uruguay': 'URY',

 'Uzbekistan': 'UZB',

 'Vanuatu': 'VUT',

 'Venezuela (Bolivar. Rep.)': 'VEN',

 'Viet Nam': 'VNM',

 'Wallis and Futuna Is.': 'WLF',

 'Yemen': 'YEM',

 'Yemen Arab Rep. (former)': 'YEM',

 'Yemen, Dem. (former)': '@@@',

 'Yugoslavia, SFR (former)': '@@@',

 'Zambia': 'ZMB',

 'Zimbabwe': 'ZWE'}
df_codes = pd.DataFrame(df.country.transform(lambda x: dict_alpha3[x]))

df_codes.columns = ['alpha3']

df['alpha3'] = df_codes
# get data for installed electric capacity from solar

df_solar = df[df.commodity.str.contains('Electricity - total net installed capacity of electric power plants, solar')]

# split out the data for year 2014

df_solar_2014 = df_solar[df_solar.year == 2014].sort_values(ascending=False,by='quantity')[:]
# Plot solar generating capacity on world map

# this code based on example code at: https://plot.ly/python/choropleth-maps/

mags = np.array([])

code = np.array([])

for item in df_solar_2014.country:

    code = np.append(code,df_solar_2014[df_solar_2014.country == item]['alpha3'])

    mags = np.append(mags,df_solar_2014[df_solar_2014.country == item]['quantity'])

data = [ dict(

        type = 'choropleth',

        locations = code,

        locationmode = 'ISO-3',

      z = mags,

        text = df_solar_2014.country,

        colorscale = 'Greens',

        autocolorscale = False,

        reversescale = True,

        marker = dict(

            line = dict (

                color = 'rgb(180,180,180)',

                width = 0.5

            ) ),

        colorbar = dict(

            autotick = False,

            tickprefix = '',

            title = 'Solar Power<br>Megawatts'),

      )]



layout = dict(

    title = '2014 Solar Power Capacity Megawatts',

    geo = dict(

        showframe = True,

        showcoastlines = True,

        showcountries = True,

        projection = dict(

            type = 'Mercator'

        )

    )

)



fig = dict( data=data, layout=layout )

iplot( fig, validate=False, filename='world-map' )