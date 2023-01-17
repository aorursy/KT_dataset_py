import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

from mpl_toolkits.basemap import Basemap
from matplotlib.colors import rgb2hex, Normalize
from matplotlib.patches import Polygon
from matplotlib.colorbar import ColorbarBase

matplotlib.rc('font', size=20)
matplotlib.rc('axes', titlesize=20)
matplotlib.rc('axes', labelsize=20)
matplotlib.rc('xtick', labelsize=20)
matplotlib.rc('ytick', labelsize=20)
matplotlib.rc('legend', fontsize=20)
matplotlib.rc('figure', titlesize=20)

%matplotlib inline
np.random.seed(42)
data = pd.read_csv("../input/honey-production/honeyproduction.csv").rename(columns={
    'state':'state_code',
    'numcol':'n_colony',
    'yieldpercol':'production_per_colony',
    'totalprod':'total_production',
    'stocks':'stock_held',
    'priceperlb':'price_per_lb',
    'prodvalue':'total_production_value'
})

state_code_to_name = {
    'AK': 'Alaska',
    'AL': 'Alabama',
    'AR': 'Arkansas',
    'AZ': 'Arizona',
    'CA': 'California',
    'CO': 'Colorado',
    'CT': 'Connecticut',
    'DC': 'District of Columbia',
    'DE': 'Delaware',
    'FL': 'Florida',
    'GA': 'Georgia',
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
    'MS': 'Mississippi',
    'MT': 'Montana',
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
    'VT': 'Vermont',
    'WA': 'Washington',
    'WI': 'Wisconsin',
    'WV': 'West Virginia',
    'WY': 'Wyoming'
}
data['consumption'] = data['total_production'] - data['stock_held']
data['state'] = data['state_code'].apply(lambda x: state_code_to_name[x])
inflation_rate = {
    1998: 1.454,
    1999: 1.423,
    2000: 1.376,
    2001: 1.339,
    2002: 1.317,
    2003: 1.288,
    2004: 1.255,
    2005: 1.214,
    2006: 1.176,
    2007: 1.143,
    2008: 1.101,
    2009: 1.105,
    2010: 1.087,
    2011: 1.054,
    2012: 1.032
}

monetized_features = ['price_per_lb', 'total_production_value']

for year in set(data['year']):
    for feature in monetized_features:
        data.loc[data['year']==year, feature] = inflation_rate[year]*data.loc[data['year']==year, feature]
data.sample(5)
data.info()
df_top10_states_consumption = data.loc[data['state'].isin(list(data[['state','price_per_lb']].groupby('state').sum().sort_values(by='price_per_lb', ascending=False).head(10).reset_index().state.values))]
df_prices_by_year = data[['year','price_per_lb']]
plt.figure(figsize=(20,5))
sns.boxplot(data=df_top10_states_consumption, x='year',y='price_per_lb')
plt.title("Prices of Honey by the Year")
plt.ylabel("Price Per Pound ($/lbs)")
plt.xlabel("Year")
plt.xticks(rotation=90)
plt.show()
top10_states_by_production = data[['state','total_production']].groupby('state').sum().sort_values(by='total_production', ascending=False).head(10).reset_index()
top10_states_by_consumption = data[['state','consumption']].groupby('state').sum().sort_values(by='consumption', ascending=False).head(10).reset_index()

f, (ax1,ax2) = plt.subplots(1,2, figsize = (30,5), sharey=True)

plt.suptitle("Top Honey Producing-Consuming States in the United States")
sns.barplot(data=top10_states_by_production, x='state', y='total_production', ax=ax1)
ax1.set(xlabel='States', ylabel='Total Honey Production \nBy State')
sns.barplot(data=top10_states_by_consumption, x='state', y='consumption', ax=ax2)
ax2.set(xlabel='States', ylabel='Total Honey Consumption \nBy State')
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=70)
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=70)
plt.show()

f, (ax1,ax2) = plt.subplots(1,2, figsize = (30,5))

states_production = data[['state','total_production']].groupby('state').sum().sort_values(by='total_production', ascending=False).reset_index()
states_production_dict = dict(zip(states_production.state, states_production.total_production))

m = Basemap(llcrnrlon=-119,llcrnrlat=20,urcrnrlon=-64,urcrnrlat=49,
            projection='lcc',lat_1=33,lat_2=45,lon_0=-95, ax=ax1)

shp_info = m.readshapefile('../input/usa-map-shape/st99_d00','states',drawbounds=True, linewidth=0.45,color='gray')

colors={}
statenames=[]
cmap = plt.cm.hot_r 
vmin = 0; vmax = 1000000000
norm = Normalize(vmin=vmin, vmax=vmax)
for shapedict in m.states_info:
    statename = shapedict['NAME']
    # skip DC and Puerto Rico.
    if statename not in ['District of Columbia','Puerto Rico','Alaska',
                         'New Hampshire','Massachusetts','Connecticut',
                        'Rhode Island','Delaware']:
        pop = states_production_dict[statename]
        colors[statename] = cmap(np.sqrt((pop-vmin)/(vmax-vmin)))[:3]
    statenames.append(statename)

for nshape,seg in enumerate(m.states):
    if statenames[nshape] not in ['District of Columbia','Puerto Rico','Alaska',
                         'New Hampshire','Massachusetts','Connecticut',
                        'Rhode Island','Delaware']:
        color = rgb2hex(colors[statenames[nshape]])
        poly = Polygon(seg,facecolor=color,edgecolor=color)
        ax1.add_patch(poly)
        

states_consumption = data[['state','consumption']].groupby('state').sum().sort_values(by='consumption', ascending=False).reset_index()
states_consumption_dict = dict(zip(states_consumption.state, states_consumption.consumption))

m = Basemap(llcrnrlon=-119,llcrnrlat=20,urcrnrlon=-64,urcrnrlat=49,
            projection='lcc',lat_1=33,lat_2=45,lon_0=-95, ax=ax2)

shp_info = m.readshapefile('../input/usa-map-shape/st99_d00','states',drawbounds=True, linewidth=0.45,color='gray')

colors={}
statenames=[]
cmap = plt.cm.hot_r 
vmin = 0; vmax = 1000000000
norm = Normalize(vmin=vmin, vmax=vmax)
for shapedict in m.states_info:
    statename = shapedict['NAME']
    # skip DC and Puerto Rico.
    if statename not in ['District of Columbia','Puerto Rico','Alaska',
                         'New Hampshire','Massachusetts','Connecticut',
                        'Rhode Island','Delaware']:
        pop = states_consumption_dict[statename]
        colors[statename] = cmap(np.sqrt((pop-vmin)/(vmax-vmin)))[:3]
    statenames.append(statename)

for nshape,seg in enumerate(m.states):
    if statenames[nshape] not in ['District of Columbia','Puerto Rico','Alaska',
                         'New Hampshire','Massachusetts','Connecticut',
                        'Rhode Island','Delaware']:
        color = rgb2hex(colors[statenames[nshape]])
        poly = Polygon(seg,facecolor=color,edgecolor=color)
        ax2.add_patch(poly)
        
plt.show()
plt.figure(figsize=(20,5))
sns.boxplot(data=df_top10_states_consumption, x='state',y='price_per_lb')
plt.title("Prices of Honey in the Top 10 Consumer States")
plt.ylabel("Price Per Pound ($/lbs)")
plt.xlabel("States")
plt.xticks(rotation=90)
plt.show()
sns.pairplot(data.loc[data['state'].isin(list(top10_states_by_production.state))], hue='state',
            diag_kind = 'kde', plot_kws = {'alpha': 0.6, 's': 80, 'edgecolor': 'k'},
            size=4, vars = ['production_per_colony','stock_held','price_per_lb','year','consumption'])
