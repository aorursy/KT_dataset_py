# Importing modules
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import plotly as py
import warnings
from plotly.offline import init_notebook_mode

# Setting up notebook
init_notebook_mode(connected=True)
warnings.filterwarnings('ignore')
plt.style.use('ggplot')
%matplotlib inline

# Importing files
PATH = '../input/'
events = pd.read_csv(f'{PATH}athlete_events.csv')
regions = pd.read_csv(f'{PATH}noc_regions.csv')

# Preprocessing steps

# Merge both input tables
data = pd.merge(events, regions, how='left', on='NOC')

# Fill up the region column of teams with 'Singapore', 'Singapore-2' and 'Singapore-1' names as 'Singapore'. 
# Also, fill up values for 'Tuvalu'
data.loc[(data.Team == 'Singapore') | (data.Team == 'Singapore-1') | (data.Team =='Singapore-2'), 'region'] = 'Singapore'
data.loc[(data.Team == 'Tuvalu'), 'region'] = 'Tuvalu'

# Fill the NaN values for Medal to 'None'
data.loc[data.Medal.isna(), 'Medal'] = 'None'

# ------------------- Define plotting functions -----------------------


def plot_age(df, pos, name):
    '''
    Function to plot the athlete count according to age ranges
    df: DataFrame
    pos: position on the subplot
    name: string, title of the plot
    '''
    plot = df.groupby(pd.cut(df.Age, np.arange(0,75,10))).Medal.count().plot.barh(ax=pos, figsize = (16,8))
    plot.set_title(name)
    return plot

def plot_age_dist(df, pos, name):
    '''
    Function to plot the athlete winning-participation ratio according to age ranges
    df: DataFrame
    pos: position on the subplot
    name: string, title of the plot
    '''
    freq = (df[df.Medal!='None'].groupby(pd.cut(df[df.Medal!='None'].Age, np.arange(10,70,10))).Medal.count() / 
           df.groupby(pd.cut(df.Age, np.arange(10,70,10))).Medal.count())
    plot = freq.plot.barh(ax=pos, figsize=(16,6))
    plot.set_title(name)
    return plot

def performance_comparison(df, country, pos, label, line_color, line_style='-'):
    '''
    Function to plot performance of countries
    df: DataFrame
    country: string, country to be analysed
    pos: position on the subplot
    label: string, label for the line corresponding to the current data and country
    line_color: string, color
    line_style: string, line_style
    '''
    performance = (df[(df.region == country) & (df.Medal!='None')].groupby('Year').ID.count() / 
            df[(df.region == country)].groupby('Year').ID.count())
    plot = performance.plot.line(ax=pos, label = label, color = line_color, linestyle = line_style, figsize=(16,8))
    return plot

codes = {'Afghanistan': 'AFG', 'Albania': 'ALB', 'Algeria': 'DZA', 'American Samoa': 'ASM',
        'Andorra': 'AND', 'Angola': 'AGO', 'Anguilla': 'AIA', 'Antigua and Barbuda': 'ATG',
        'Argentina': 'ARG', 'Armenia': 'ARM', 'Aruba': 'ABW', 'Australia': 'AUS', 'Austria': 'AUT',
        'Azerbaijan': 'AZE', 'Bahamas, The': 'BHM', 'Bahrain': 'BHR', 'Bangladesh': 'BGD',
        'Barbados': 'BRB', 'Belarus': 'BLR', 'Belgium': 'BEL', 'Belize': 'BLZ', 'Benin': 'BEN',
        'Bermuda': 'BMU', 'Bhutan': 'BTN', 'Boliva': 'BOL', 'Bolivia': 'BOL', 'Bosnia and Herzegovina': 'BIH',
        'Botswana': 'BWA', 'Brazil': 'BRA', 'British Virgin Islands': 'VGB', 'Brunei': 'BRN', 'Bulgaria': 'BGR',
        'Burkina Faso': 'BFA', 'Burma': 'MMR', 'Burundi': 'BDI', 'Cabo Verde': 'CPV', 'Cambodia': 'KHM',
        'Cameroon': 'CMR', 'Canada': 'CAN', 'Cayman Islands': 'CYM', 'Central African Republic': 'CAF',
        'Chad': 'TCD', 'Chile': 'CHL', 'China': 'CHN', 'Colombia': 'COL', 'Comoros': 'COM',
        'Congo, Democratic Republic of the': 'COD', 'Congo, Republic of the': 'COG', 'Cook Islands': 'COK',
        'Costa Rica': 'CRI', "Cote d'Ivoire": 'CIV', 'Croatia': 'HRV', 'Cuba': 'CUB', 'Curacao': 'CUW',
        'Cyprus': 'CYP', 'Czech Republic': 'CZE', 'Democratic Republic of the Congo': 'COD',
        'Denmark': 'DNK', 'Djibouti': 'DJI', 'Dominica': 'DMA', 'Dominican Republic': 'DOM', 'Ecuador': 'ECU',
        'Egypt': 'EGY', 'El Salvador': 'SLV', 'Equatorial Guinea': 'GNQ', 'Eritrea': 'ERI', 'Estonia': 'EST',
        'Ethiopia': 'ETH', 'Falkland Islands (Islas Malvinas)': 'FLK', 'Faroe Islands': 'FRO', 'Fiji': 'FJI',
        'Finland': 'FIN', 'France': 'FRA', 'French Polynesia': 'PYF', 'Gabon': 'GAB', 'Gambia, The': 'GMB',
        'Georgia': 'GEO', 'Germany': 'DEU', 'Ghana': 'GHA', 'Gibraltar': 'GIB', 'Greece': 'GRC', 'Greenland': 'GRL',
        'Grenada': 'GRD', 'Guam': 'GUM', 'Guatemala': 'GTM', 'Guernsey': 'GGY', 'Guinea': 'GIN', 'Guinea-Bissau': 'GNB',
        'Guyana': 'GUY', 'Haiti': 'HTI', 'Honduras': 'HND', 'Hong Kong': 'HKG', 'Hungary': 'HUN', 'Iceland': 'ISL',
        'India': 'IND', 'Indonesia': 'IDN', 'Iran': 'IRN', 'Iraq': 'IRQ', 'Ireland': 'IRL', 'Isle of Man': 'IMN',
        'Israel': 'ISR', 'Italy': 'ITA', 'Ivory Coast': 'CIV', 'Jamaica': 'JAM', 'Japan': 'JPN', 'Jersey': 'JEY',
        'Jordan': 'JOR', 'Kazakhstan': 'KAZ', 'Kenya': 'KEN', 'Kiribati': 'KIR', 'Korea, North': 'PRK',
        'Korea, South': 'KOR', 'Kosovo': 'KSV', 'Kuwait': 'KWT', 'Kyrgyzstan': 'KGZ', 'Laos': 'LAO',
        'Latvia': 'LVA', 'Lebanon': 'LBN', 'Lesotho': 'LSO', 'Liberia': 'LBR', 'Libya': 'LBY', 'Liechtenstein': 'LIE',
        'Lithuania': 'LTU', 'Luxembourg': 'LUX', 'Macau': 'MAC', 'Macedonia': 'MKD', 'Madagascar': 'MDG', 'Malawi': 'MWI',
        'Malaysia': 'MYS', 'Maldives': 'MDV', 'Mali': 'MLI', 'Malta': 'MLT', 'Marshall Islands': 'MHL', 'Mauritania': 'MRT',
        'Mauritius': 'MUS', 'Mexico': 'MEX', 'Micronesia, Federated States of': 'FSM', 'Moldova': 'MDA', 'Monaco': 'MCO',
        'Mongolia': 'MNG', 'Montenegro': 'MNE', 'Morocco': 'MAR', 'Mozambique': 'MOZ', 'Myanmar': 'MMR', 'Namibia': 'NAM',
        'Nepal': 'NPL', 'Netherlands': 'NLD', 'New Caledonia': 'NCL', 'New Zealand': 'NZL', 'Nicaragua': 'NIC',
        'Niger': 'NER', 'Nigeria': 'NGA', 'Niue': 'NIU', 'North Korea': 'PRK', 'Northern Mariana Islands': 'MNP',
        'Norway': 'NOR', 'Oman': 'OMN', 'Pakistan': 'PAK', 'Palau': 'PLW', 'Panama': 'PAN', 'Papua New Guinea': 'PNG',
        'Paraguay': 'PRY', 'Peru': 'PER', 'Philippines': 'PHL', 'Poland': 'POL', 'Portugal': 'PRT', 'Puerto Rico': 'PRI',
        'Qatar': 'QAT', 'Republic of Congo': 'COG', 'Romania': 'ROU', 'Russia': 'RUS', 'Rwanda': 'RWA',
        'Saint Kitts and Nevis': 'KNA', 'Saint Lucia': 'LCA', 'Saint Martin': 'MAF', 'Saint Pierre and Miquelon': 'SPM',
        'Saint Vincent and the Grenadines': 'VCT', 'Samoa': 'WSM', 'San Marino': 'SMR', 'Sao Tome and Principe': 'STP',
        'Saudi Arabia': 'SAU', 'Senegal': 'SEN', 'Serbia': 'SRB', 'Seychelles': 'SYC', 'Sierra Leone': 'SLE',
        'Singapore': 'SGP', 'Sint Maarten': 'SXM', 'Slovakia': 'SVK', 'Slovenia': 'SVN', 'Solomon Islands': 'SLB',
        'Somalia': 'SOM', 'South Africa': 'ZAF', 'South Korea': 'KOR', 'South Sudan': 'SSD', 'Spain': 'ESP', 'Sri Lanka': 'LKA',
        'Sudan': 'SDN', 'Suriname': 'SUR', 'Swaziland': 'SWZ', 'Sweden': 'SWE', 'Switzerland': 'CHE', 'Syria': 'SYR',
        'Taiwan': 'TWN', 'Tajikistan': 'TJK', 'Tanzania': 'TZA', 'Thailand': 'THA', 'Timor-Leste': 'TLS', 'Togo': 'TGO', 
        'Tonga': 'TON', 'Trinidad and Tobago': 'TTO', 'Tunisia': 'TUN', 'Turkey': 'TUR', 'Turkmenistan': 'TKM', 'Tuvalu': 'TUV',
        'UK': 'GBR', 'USA': 'USA', 'Uganda': 'UGA', 'Ukraine': 'UKR', 'United Arab Emirates': 'ARE', 'United Kingdom': 'GBR',
        'United States': 'USA', 'Uruguay': 'URY', 'Uzbekistan': 'UZB', 'Vanuatu': 'VUT', 'Venezuela': 'VEN', 'Vietnam': 'VNM',
        'Virgin Islands': 'VGB', 'West Bank': 'WBG', 'Yemen': 'YEM', 'Zambia': 'ZMB', 'Zimbabwe': 'ZWE'}

def plot_map(df, col, title='Title', bar_title=''):
    '''
    Function to plot on World Map
    df: DataFrame
    col: string, column to be used 
    title: string, title of the Map
    bar_title: string, title of the colorbar
    '''
    df['CODE'] = df.region.map(codes)

    plot_df = pd.DataFrame(df.groupby('CODE')[col].count().reset_index())

    map_data = [dict(type = 'choropleth',
                     locations = plot_df.CODE,
                     z = plot_df[col],
                     autocolorscale = False,
                     colorscale = [[0, 'rgb(166,206,227)'],
                                   [0.25, 'rgb(31,120,180)'],
                                   [0.45, 'rgb(178,223,138)'],
                                   [0.65, 'rgb(51,160,44)'],
                                   [0.85, 'rgb(251,154,153)'],
                                   [1, 'rgb(227,26,28)']],
                     marker = dict(line = dict (color = 'rgb(180,180,180)', width = 0.5)),
                     colorbar = dict(autotick = True, tickprefix = '', title = bar_title))]
    layout = dict(title = title, 
                  geo = dict(showframe = True,
                             showcoastlines = True,
                             projection = dict(type = 'Mercator')))
    fig = dict(data=map_data, layout=layout )
    py.offline.iplot(fig, validate=False)
    
# ------------- Create dataframe slices for easier analysis ------------------

# There is a disparity in how the summer and winter olympics were held in the same year before 1994, 
# when they were separated to take place in gaps of 2 years. 
# Let's separate the data according to seasons for further analysis.
summer_data = data[data.Season == 'Summer']
winter_data = data[data.Season == 'Winter']

# Separating male and female data for easier analysis
male_summer = summer_data[summer_data.Sex == 'M']
female_summer = summer_data[summer_data.Sex == 'F']
male_winter = winter_data[winter_data.Sex == 'M']
female_winter = winter_data[winter_data.Sex == 'F']
print(f'Number of participating countries: {len(data.region.unique())}')
summer_data.groupby('Year').region.nunique().plot.line(figsize=(16,8), label = 'summer', marker = 'o')
winter_data.groupby('Year').region.nunique().plot.line(figsize=(16,8), label = 'winter', marker = 'o')
plt.title('Country Participation over years', fontsize=18)
plt.legend(loc='best',fontsize=14)
plt.ylabel('Number of Countries')
plt.text(1980,75,'Moscow, 1980')
plt.text(1976,85,'Montreal 1976')
plt.text(1906,17,'Athens 1906')
plt.text(1913.25, 77,'WW I')
plt.text(1939.75,77,'WW II')
plt.plot([1912,1918],[75,75],c='g',linewidth=4)
plt.plot([1939,1945],[75,75],c='m',linewidth=4)
summer_data.groupby('Year').Event.nunique().plot.line(figsize=(16,8), label = 'summer', marker = 'o')
winter_data.groupby('Year').Event.nunique().plot.line(figsize=(16,8), label = 'winter', marker = 'o')
plt.title('Events over years', fontsize=18)
plt.legend(loc='best',fontsize=14)
plt.ylabel('Number of Events')
summer_data.groupby('Year').Name.nunique().plot.line(figsize=(16,8), label = 'summer', marker = 'o')
winter_data.groupby('Year').Name.nunique().plot.line(figsize=(16,8), label = 'winter', marker = 'o')
plt.title('Athletes Participation over years', fontsize=18)
plt.legend(loc='best',fontsize=14)
plt.ylabel('Number of Athletes')
plt.text(1980,5200,'Moscow, 1980')
plt.text(1976,6000,'Montreal 1976')
plt.text(1906,500,'Athens 1906')
plt.text(1932,2000,'Los Angeles 1932')
plt.text(1956,3300,'Melbourne 1956')
plt.text(1913.25, 6100,'WW I')
plt.text(1939.75,6100,'WW II')
plt.plot([1912,1918],[6000,6000],c='g',linewidth=4)
plt.plot([1939,1945],[6000,6000],c='m',linewidth=4)
male_summer.groupby('Year').Sex.count().plot.line(figsize=(16,8), label = 'summer (men)')
female_summer.groupby('Year').Sex.count().plot.line(figsize=(16,8), label = 'summer (women)')
plt.title('Athletes over years in Summer Olympics', fontsize=18)
plt.legend(loc='best',fontsize=14)
plt.ylabel('Number of athletes')
male_winter.groupby('Year').Sex.count().plot.line(figsize=(16,8), label = 'winter (men)')
female_winter.groupby('Year').Sex.count().plot.line(figsize=(16,8), label = 'winter (women)')
plt.title('Athletes over years in Winter Olympics', fontsize=18)
plt.legend(loc='best',fontsize=14)
plt.ylabel('Number of athletes')
plt.plot([1994,1994],[2500,900], c='g', linewidth=2)
plt.text(1994,2000,'1994 Winter Olympics', rotation = 270, fontsize=10)
sns.jointplot(kind='kde', x = 'Weight', y = 'Height',
              data = male_summer[male_summer.Medal != 'None'], color='gold',
              size = 8, xlim=(30,160), ylim=(140,210))
plt.title('Height vs Weight for Medal Winners in Summer Olympics(Men)', fontdict = {'horizontalalignment':'right'})
sns.jointplot(kind='kde', x = 'Weight', y = 'Height', 
              data = male_summer[male_summer.Medal == 'None'], color='pink',
              size = 8, xlim=(30,160), ylim=(140,210))
plt.title('Height vs Weight for non-Medal Winners in Summer Olympics(Men)', fontdict = {'horizontalalignment':'right'})
sns.jointplot(kind='kde', x = 'Weight', y = 'Height',
              data = female_summer[female_summer.Medal != 'None'], color='gold',
              size = 8, xlim=(30,160), ylim=(140,210))
plt.title('Height vs Weight for Medal winners in Summer Olympics(Women)', fontdict = {'horizontalalignment':'right'})
sns.jointplot(kind='kde', x = 'Weight', y = 'Height',
              data = female_summer[female_summer.Medal == 'None'], color='pink',
              size = 8, xlim=(30,160), ylim=(140,210))
plt.title('Height vs Weight for non-Medal winners in Summer Olympics(Women)', fontdict = {'horizontalalignment':'right'})
sns.jointplot(kind='kde', x = 'Weight', y = 'Height',
              data = male_winter[male_winter.Medal != 'None'], color='gold',
              size = 8, xlim=(30,160), ylim=(140,210))
plt.title('Height vs Weight for Medal winners in Winter Olympics(Men)', fontdict = {'horizontalalignment':'right'})
sns.jointplot(kind='kde', x = 'Weight', y = 'Height',
              data = male_winter[male_winter.Medal == 'None'], color='pink',
              size = 8, xlim=(30,160), ylim=(140,210))
plt.title('Height vs Weight for non-Medal winners in Winter Olympics(Men)', fontdict = {'horizontalalignment':'right'})
sns.jointplot(kind='kde', x = 'Weight', y = 'Height',
              data = female_winter[female_winter.Medal != 'None'], color='gold',
              size = 8, xlim=(30,160), ylim=(140,210))
plt.title('Height vs Weight for Medal winners in Winter Olympics(Women)', fontdict = {'horizontalalignment':'right'})
sns.jointplot(kind='kde', x = 'Weight', y = 'Height',
              data = female_winter[female_winter.Medal == 'None'], color='pink',
              size = 8, xlim=(30,160), ylim=(140,210))
plt.title('Height vs Weight for non-Medal winners in Winter Olympics(Women)', fontdict = {'horizontalalignment':'right'})
fig, axarr = plt.subplots(2,2,sharex=True, sharey = True)
plot_age(male_summer[male_summer.Medal!='None'], axarr[0][0], 'Men Medal Winner Ages (Summer Olympics)')
plot_age(female_summer[female_summer.Medal!='None'], axarr[0][1], 'Women Medal Winner Ages (Summer Olympics)')
plot_age(male_summer[male_summer.Medal=='None'], axarr[1][0], 'Men non-Medal Winner Ages (Summer Olympics)')
plot_age(female_summer[female_summer.Medal=='None'], axarr[1][1], 'Women non-Medal Winner Ages (Summer Olympics)')
axarr[1][0].set_xlabel('Count')
axarr[1][1].set_xlabel('Count')
fig, axarr = plt.subplots(1,2, sharey=True)
plot_age_dist(male_summer, axarr[0],'Men winning-participation ratio (Summer Olympics)')
plot_age_dist(female_summer, axarr[1],'Women winning-participation ratio (Summer Olympics)')
axarr[0].set_xlabel('Ratio')
axarr[1].set_xlabel('Ratio')
fig, axarr = plt.subplots(1,2,figsize=(16,6))
male_summer[male_summer.Age > 40].Sport.value_counts().head(5).plot.bar(ax=axarr[0], color ='m')
male_summer[(male_summer.Age > 40) & (male_summer.Medal != 'None')].Sport.value_counts().head(5).plot.bar(ax=axarr[1], color ='m')
axarr[0].title.set_text('Number of male participants over 40 (Summer Olympics)')
axarr[1].title.set_text('Number of male Medal winners over 40 (Summer Olympics)')
axarr[0].set_xlabel('Category')
axarr[1].set_xlabel('Category')
axarr[0].set_ylabel('Count')
axarr[1].set_ylabel('Count')
fig, axarr = plt.subplots(1,2,figsize=(16,6))
female_summer[female_summer.Age > 40].Sport.value_counts().head(5).plot.bar(ax=axarr[0], color ='m')
female_summer[(female_summer.Age > 40) & (female_summer.Medal != 'None')].Sport.value_counts().head(5).plot.bar(ax=axarr[1], color ='m')
axarr[0].title.set_text('Number of female participants over 40 (Summer Olympics)')
axarr[1].title.set_text('Number of female Medal winners over 40 (Summer Olympics)')
axarr[0].set_xlabel('Category')
axarr[1].set_xlabel('Category')
axarr[0].set_ylabel('Count')
axarr[1].set_ylabel('Count')
fig, axarr = plt.subplots(2,2,sharex=True, sharey = True)
plot_age(male_winter[male_winter.Medal!='None'], axarr[0][0], 'Men Medal Winner Ages (Winter Olympics)')
plot_age(female_winter[female_winter.Medal!='None'], axarr[0][1], 'Women Medal Winner Ages (Winter Olympics)')
plot_age(male_winter[male_winter.Medal=='None'], axarr[1][0], 'Men non-Medal Winner Ages (Winter Olympics)')
plot_age(female_winter[female_winter.Medal=='None'], axarr[1][1], 'Women non-Medal Winner Ages (Winter Olympics)')
axarr[1][0].set_xlabel('Count')
axarr[1][1].set_xlabel('Count')
fig, axarr = plt.subplots(1,2, sharey=True)
plot_age_dist(male_winter, axarr[0],'Men winning-participation ratio (Winter Olympics)')
plot_age_dist(female_winter, axarr[1],'Women winning-participation ratio (Winter Olympics)')
axarr[0].set_xlabel('Ratio')
axarr[1].set_xlabel('Ratio')
fig, axarr = plt.subplots(1,2,figsize=(16,6))
male_winter[male_winter.Age > 40].Sport.value_counts().head(5).plot.bar(ax=axarr[0], color ='m')
male_winter[(male_winter.Age > 40) & (male_winter.Medal != 'None')].Sport.value_counts().head(5).plot.bar(ax=axarr[1], color ='m')
axarr[0].title.set_text('Number of male participants over 40 (Summer Olympics)')
axarr[1].title.set_text('Number of male Medal winners over 40 (Summer Olympics)')
axarr[0].set_xlabel('Category')
axarr[1].set_xlabel('Category')
axarr[0].set_ylabel('Count')
axarr[1].set_ylabel('Count')
fig, axarr = plt.subplots(1,2,figsize=(16,6))
female_winter[female_winter.Age > 40].Sport.value_counts().head(5).plot.bar(ax=axarr[0], color ='m')
female_winter[(female_winter.Age > 40) & (female_winter.Medal != 'None')].Sport.value_counts().head(5).plot.bar(ax=axarr[1], color ='m')
axarr[0].title.set_text('Number of male participants over 40 (Summer Olympics)')
axarr[1].title.set_text('Number of male Medal winners over 40 (Summer Olympics)')
axarr[0].set_xlabel('Category')
axarr[1].set_xlabel('Category')
axarr[0].set_ylabel('Count')
axarr[1].set_ylabel('Count')
plot_map(male_summer,'ID','Participation in Summer Olympics (Men)','Number of participants')
plot_map(male_summer[male_summer.Medal!='None'],'ID','Medal Winners in Summer Olympics (Men)','Number of medals')
plot_map(female_summer,'ID','Participation in Summer Olympics (Women)','Number of participants')
plot_map(female_summer[female_summer.Medal!='None'],'ID','Medal Winners in Summer Olympics (Women)','Number of medals')
plot_map(male_winter,'ID','Participation in Winter Olympics (Men)','Number of participants')
plot_map(male_winter[male_winter.Medal!='None'],'ID','Medal Winners in Winter Olympics (Men)','Number of medals')
plot_map(female_winter,'ID','Participation in Winter Olympics (Women)','Number of participants')
plot_map(female_winter[female_winter.Medal!='None'],'ID','Medal Winners in Winter Olympics (Women)','Number of medals')
male_summer[(male_summer.region=='China')].groupby('Year').ID.count().plot.line(label='Men', figsize=(16,8))
female_summer[(female_summer.region=='China')].groupby('Year').ID.count().plot.line(label='Women', figsize=(16,8))
plt.legend(loc='upper left', fontsize=12)
plt.title('Men vs Women athlete count in Summer Olympics (China)')
male_summer[(male_summer.region=='China') & 
            (male_summer.Medal!='None')].groupby('Year').ID.count().plot.line(label='Men', figsize=(16,8))
female_summer[(female_summer.region=='China') & 
              (female_summer.Medal!='None')].groupby('Year').ID.count().plot.line(label='Women', figsize=(16,8))
plt.legend(loc='upper left', fontsize=12)
plt.title('Men vs Women medal count in Summer Olympics(China)')
male_winter[(male_winter.region=='China')].groupby('Year').ID.count().plot.line(label='Men', figsize=(16,8))
female_winter[(female_winter.region=='China')].groupby('Year').ID.count().plot.line(label='Women', figsize=(16,8))
plt.legend(loc='upper left', fontsize=12)
plt.title('Men vs Women athlete count in Winter Olympics (China)')
male_winter[(male_winter.region=='China') & 
            (male_winter.Medal!='None')].groupby('Year').ID.count().plot.line(label='Men', figsize=(16,8))
female_winter[(female_winter.region=='China') & 
              (female_winter.Medal!='None')].groupby('Year').ID.count().plot.line(label='Women', figsize=(16,8))
plt.legend(loc='upper left', fontsize=12)
plt.title('Men vs Women medal count in Winter Olympics(China)')
fig, axarr = plt.subplots(3,1)
performance_comparison(summer_data, 'Finland', axarr[0], 'Finland(summer)', 'r', '--')
performance_comparison(winter_data, 'Finland', axarr[0], 'Finland(winter)','r', '-')
performance_comparison(summer_data, 'Sweden', axarr[1], 'Sweden(summer)', 'b', '--')
performance_comparison(winter_data, 'Sweden', axarr[1], 'Sweden(winter)','b', '-')
performance_comparison(summer_data, 'Norway', axarr[2], 'Norway(summer)', 'g', '--')
performance_comparison(winter_data, 'Norway', axarr[2], 'Norway(winter)','g', '-')
axarr[0].legend() 
axarr[1].legend()
axarr[2].legend()