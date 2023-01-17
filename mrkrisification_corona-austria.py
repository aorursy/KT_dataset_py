# necessary imports

import requests

from bs4 import BeautifulSoup

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
base_url = 'https://coronatracker.at/'

bundeslaender = ['wien', 'niederoesterreich', 'oberoesterreich', 'burgenland', 'steiermark','kaernten','tirol','salzburg','vorarlberg', 'austria']
# get all necessary pages (from all countries)

def get_all_tables(base_url,bundeslaender):

    data_tables = []

    for bundesland in bundeslaender:

        print(f'Fetching data for {bundesland}')

        try:

            if bundesland != 'austria':

                page = requests.get(base_url+bundesland)

            else:

                page = requests.get(base_url)

            soup = BeautifulSoup(page.content, 'html.parser')

            table = soup.findAll('table')

            #for austria relevant table is 1 - otherwise 0

            if bundesland != 'austria':

                i = 0

            else:

                i = 1

            data_tables.append(table[i])

        except Exception as e:

            print(f'Could not get data: {e}')

            

    return(data_tables)



    
all_tables = get_all_tables(base_url, bundeslaender)
def get_data_from_table(table):

    

    #extract single rows from table

    all_cells = []

    timestamps = []

    tests = []

    all_cases=[]

    active_cases=[]

    active_by_100k = []

    recovered=[]

    death=[]

    hospital=[]

    icu=[]



    for row in table.findAll("tr"):

        cell = row.findAll("td")

        all_cells.append(cell)



    #print(all_cells[3][1].get('data-order'))

    # values start from index 1 to end 

    for i in range(1, len(all_cells)):

        timestamps.append(all_cells[i][0].text)

        tests.append(all_cells[i][1].get("data-order"))

        all_cases.append(all_cells[i][2].get("data-order"))

        active_cases.append(all_cells[i][3].get("data-order"))

        active_by_100k.append(all_cells[i][4].get("data-order"))

        recovered.append(all_cells[i][5].get("data-order"))

        death.append(all_cells[i][6].get("data-order"))

        hospital.append(all_cells[i][7].get("data-order"))

        icu.append(all_cells[i][8].get("data-order"))





    # put all retrieved data to result dict

    outdict = {'timestamp':timestamps, 'tests':tests, 'all_cases': all_cases, 'active_cases': active_cases, 'active_by_100k':active_by_100k,

                   'recovered':recovered, 'death':death, 'hospital':hospital, 'icu':icu}



    return outdict
def clean_table(df):

    #cleaning data

    type_dict = {

        'tests': 'int',

     'all_cases': 'int',

     'active_cases': 'int',

     'active_by_100k': 'float',

     'recovered': 'int',

     'death': 'int',

     'hospital': 'int',

     'icu': 'int'

    }



    # converting timestamp to date and time

    #df['date'] = pd.to_datetime(df['timestamp']).dt.date

    #df['time'] = pd.to_datetime(df['timestamp']).dt.time



    # converting dtypes for each column as defined in type_dict

    for column, dtype in type_dict.items():

        df[column] = df[column].astype(dtype)



    return df
for i, table in enumerate(all_tables):

    print(f'Processing table :{bundeslaender[i]}')

    #get data (as dictionary from each scraped table)

    data_dict = get_data_from_table(table)

    #convert it to dataframe and clean it

    df = pd.DataFrame.from_dict(data_dict)

    df = clean_table(df)

    

    df.to_csv(f'{bundeslaender[i]}.csv', index=False)

    


google_mobility_url = 'https://www.gstatic.com/covid19/mobility/Region_Mobility_Report_CSVs.zip'
r = requests.get(google_mobility_url)

filename = 'mobility.zip'

f = open(filename, 'wb')

f.write(r.content)

f.close()
mobility_AT = '2020_AT_Region_Mobility_Report.csv'
from zipfile import ZipFile

    

with ZipFile(filename, 'r') as zip:

    zip.extract(mobility_AT)
df = pd.read_csv(mobility_AT)
df.info()
df.sub_region_1.unique()
df.loc[df.sub_region_1=='Lower Austria']
df.iso_3166_2_code.unique()
lower_austria_total = df.loc[df.iso_3166_2_code=='AT-3'].set_index('date')
df.columns.to_list()
plt.figure(figsize=(20,10))

sns.set_style('darkgrid')

sns.lineplot(x=lower_austria_total.index, y='workplaces_percent_change_from_baseline', data=lower_austria_total)

plt.show()
df.columns
#df[['iso_3166_2_code', 'sub_region_1']].groupby(by=['iso_3166_2_code', 'sub_region_1']).count()
mapping_dict = {'wien':'AT-9',

                'niederoesterreich': 'AT-3', 

                'oberoesterreich': 'AT-4',

                'burgenland': 'AT-1',

                'steiermark':'AT-6',

                'kaernten': 'AT-2',

                'tirol': 'AT-7',

                'salzburg': 'AT-5',

                'vorarlberg':'AT-8',

                'austria': 'AT'}
def prepare_epidemic_data(bl):

    dfbl = pd.read_csv(bl+'.csv')

    dfbl['iso_3166_2_code'] = mapping_dict[bl]

    dfbl['bundesland']=bl

    dfbl['timestamp'] = pd.to_datetime(dfbl['timestamp'], dayfirst=True)

    dfbl['date'] = dfbl.timestamp.dt.date

    dfbl.sort_values(by='date', inplace=True)

    #dfbl.set_index('date', inplace=True)

    # 30day rolling





    quantcolumns = ['tests', 'all_cases', 'active_cases','recovered', 'death', 'hospital', 'icu']

    # calculate daily net additions for quantitative features

    for q in quantcolumns:

        dfbl[f'{q}_new'] = dfbl[q].diff(periods=1)



    # calculate n day sum for quantitative columns

    days = 7

    for q in quantcolumns:

        dfbl[f'{q}_{days}_day_sum'] = dfbl[f'{q}_new'].rolling(window=days).sum()



    # calculate ratios based on the 7 day sums

    dfbl['PositiveRate_perc'] =  round(dfbl[f'all_cases_{days}_day_sum']/dfbl[f'tests_{days}_day_sum']*100,2)

    return dfbl



region = 'wien'



dfcombined = prepare_epidemic_data(region)

dfcombined['date'] = dfcombined['date'].astype('str')



if region != 'austria':

    key = mapping_dict[region]

    dfcombined2 = df.loc[df.iso_3166_2_code==key]

else:

    # whole austria has NaN in sub_region_1

    dfcombined2 = df.loc[df.sub_region_1.isnull()]



df_total = pd.merge(dfcombined, dfcombined2, on='date', suffixes=['_base','google'])

df_total['region'] = region
movement_data = ['retail_and_recreation_percent_change_from_baseline',

       'grocery_and_pharmacy_percent_change_from_baseline',

       'parks_percent_change_from_baseline',

       'transit_stations_percent_change_from_baseline',

       'workplaces_percent_change_from_baseline',

       'residential_percent_change_from_baseline']



n_days = 7



for mov in movement_data:

    df_total[f'{mov}_{n_days}_avg']=df_total[mov].rolling(window=n_days).mean()
df_total.columns
relcols = ['timestamp','retail_and_recreation_percent_change_from_baseline_7_avg',

       'grocery_and_pharmacy_percent_change_from_baseline_7_avg',

       'parks_percent_change_from_baseline_7_avg',

       'transit_stations_percent_change_from_baseline_7_avg',

       'workplaces_percent_change_from_baseline_7_avg',

       'residential_percent_change_from_baseline_7_avg']

dfmov = df_total[relcols].melt(id_vars = 'timestamp', value_vars = relcols[1:])
dfmov.head()
plt.figure(figsize=(20,10))

sns.set_palette('tab10')

sns.set_context('notebook')

g = sns.lineplot(x='timestamp', y='value', hue='variable', data=dfmov)

plt.show()
df_total.columns
corrcols = ['all_cases_7_day_sum','retail_and_recreation_percent_change_from_baseline_7_avg',

       'grocery_and_pharmacy_percent_change_from_baseline_7_avg',

       'parks_percent_change_from_baseline_7_avg',

       'transit_stations_percent_change_from_baseline_7_avg',

       'workplaces_percent_change_from_baseline_7_avg',

       'residential_percent_change_from_baseline_7_avg']


sns.heatmap(df_total[corrcols].corr(), cmap='coolwarm', annot=True)
movement_category = 'grocery_and_pharmacy_percent_change_from_baseline_7_avg'

xmax = df_total.loc[df_total[movement_category].notnull()]['all_cases_7_day_sum'].max()



g = sns.lmplot(x='all_cases_7_day_sum', y=movement_category, data=df_total,height=7, )

g.set(xlim=(None,xmax))
df_total.loc[df_total[movement_category].notnull()]['all_cases_7_day_sum'].max()
bl = 'wien'

dfbl = df_total.loc[df_total.bundesland==bl]
bins = [0,1,3,5,10,20,np.inf]

labels = ['<1%','1-3%','3-5%','5-10%','10-20%','>20%']

dfbl['pos_rate_category'] = pd.cut(dfbl.PositiveRate_perc, bins=bins, labels=labels)

dfbl['pos_rate_category'] = dfbl['pos_rate_category'].astype('category') 
dfbl.columns
sns.set_style('darkgrid')

sns.set_context('paper')

sns.set_palette('colorblind')

plt.figure(figsize=(15,10))

sns.lineplot(x='date', y='PositiveRate_perc',data=dfbl)
plotcols = ['date', 'all_cases_7_day_sum', 'pos_rate_category']

dfplot = dfbl[plotcols]
#sns.catplot(x='date', y='all_cases_7_day_sum', kind='scatter', hue='pos_rate_category', palette='coolwarm', data=dfplot, height=10, aspect=1)

#sns.barplot(x='date', y='all_cases_7_day_sum', data=dfplot, hue='pos_rate_category')

dfplot=dfbl.pivot_table(index='timestamp', columns=['pos_rate_category'], values=['all_cases_7_day_sum'])
plt.figure(figsize=(20,10))

dfplot.plot.line(figsize=(20,10), cmap='RdBu')
dfbl.loc[dfbl.timestamp>'2020-05-01'].head()
bez = pd.read_csv('https://covid19-dashboard.ages.at/data/CovidFaelle_Timeline_GKZ.csv', delimiter=";")
bez.loc[bez.Bezirk=="Mistelbach"]
bezirksstr = str(sorted(bez.Bezirk.unique().tolist())).replace("'","")[1:-1]

nrbez = len(bez.Bezirk.unique())

print(f'Im Datensatz befinden sich die {nrbez} Bezirke {bezirksstr}')
bezirk = 'Korneuburg'

bez.loc[bez.Bezirk==bezirk][['Time','AnzahlFaelle']].plot()
sorted(df.sub_region_2.dropna().unique().tolist())
df['Bezirk'] = df.sub_region_2.str.replace(' District','')
bez.head()
googlebez = df.Bezirk.dropna().unique().tolist()

bez['Bezirk_clean'] = bez.Bezirk.str.replace('(Stadt)','')
found = 0

notfound = []

for bez in argesbez:

    if bez in googlebez:

        found += 1

    else:

        notfound.append(bez)

        

print(f'Bezirke gemapped: {found}')

print(f'Nicht gefunden: {notfound}')

bez.Bezirk_clean
## Map Trial
import geopandas as gpd
austria = gpd.read_file('../input/austria-map/STATISTIK_AUSTRIA_POLBEZ_20200101Polygon.shp')
austria.plot()
import geoplot
austria['namelen'] = austria.name.str.len()
import json

#convert projection

austria.to_crs(epsg=4326)

# convert to json

austria.to_file("austria.json", driver = "GeoJSON")



with open("austria.json") as geofile:

    j_file = json.load(geofile)
import plotly.express as px
# alternative trial to get geojson file

austria = gpd.read_file('../input/austria-map/STATISTIK_AUSTRIA_POLBEZ_20200101Polygon.shp')

austria_conv = austria.to_crs(epsg=4326) # convert the coordinate reference system to lat/long

austria_json = austria_conv.__geo_interface__ #covert to geoJSON



austria.set_index('id', inplace=True)

austria_json['features'][0]['properties']
austria_conv.crs
austria['namelen'] = austria.name.str.len()

austria_data = pd.DataFrame(austria[['namelen']])

#austria_data.set_index('id', inplace=True)
import plotly.express as px



df = px.data.election()

geojson = px.data.election_geojson()



#fig = px.choropleth_mapbox(austria_data, geojson=austria_json, color="namelen",locations=austria_data.index)

fig = px.choropleth(austria_data, geojson=austria_json, color="namelen",locations=austria_data.index,

                   scope='europe')

fig.show()
import plotly.express as px



df = px.data.election()

geojson = px.data.election_geojson()



fig = px.choropleth_mapbox(df, geojson=geojson, color="Bergeron",

                           locations="district", featureidkey="properties.district",

                           center={"lat": 45.5517, "lon": -73.7073},

                           mapbox_style="carto-positron", zoom=9)

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

fig.show()
austria.set_index('id')
type(austria)
# OPTIONAL: Display using geopandas

fig, ax = plt.subplots(1,1, figsize=(20,20))

#divider = make_axes_locatable(ax)

tmp = austria.copy()

#cax = divider.append_axes("right", size="3%", pad=-1) #resize the colorbar

tmp.plot(column='namelen', ax=ax,  legend=True, 

         legend_kwds={'label': "Length of Name Dummy"})

tmp.geometry.boundary.plot(color='#BABABA', ax=ax, linewidth=0.3) #Add some borders to the geometries

ax.axis('off')