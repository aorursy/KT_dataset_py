import matplotlib as mpl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from mpl_toolkits.basemap import Basemap
import bq_helper
from bq_helper import BigQueryHelper
# https://www.kaggle.com/sohier/introduction-to-the-bq-helper-package
ghnp = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="world_bank_health_population")
bq_assistant = BigQueryHelper("bigquery-public-data", "world_bank_health_population")
bq_assistant.list_tables()
bq_assistant.table_schema('country_summary');
bq_assistant.table_schema('health_nutrition_population');
bq_assistant.head('country_series_definitions',num_rows=10);
bq_assistant.head('country_summary',num_rows=100);
bq_assistant.head("health_nutrition_population", num_rows=5);
#list  of indicators
query= """SELECT DISTINCT indicator_name, indicator_code
        FROM
        `bigquery-public-data.world_bank_health_population.health_nutrition_population`
"""
indicators= ghnp.query_to_pandas_safe(query)
query= """SELECT A.country_name, 
                A.country_code,
                A.value AS new_cases, 
                B.value AS total_population, 
                A.value/B.value*100000 AS rate,
                A.year
        FROM
        `bigquery-public-data.world_bank_health_population.health_nutrition_population` AS A
        INNER JOIN 
         `bigquery-public-data.world_bank_health_population.health_nutrition_population` AS B
        ON (A.country_code = B.country_code AND A.year=B.year )
        WHERE (A.indicator_code = 'SH.HIV.INCD' AND  B.indicator_code = 'SP.POP.TOTL') 
        ;
           
"""
new= ghnp.query_to_pandas_safe(query)
world_new = new[new.country_name=='World']
ax = plt.subplot(111)
sns.set()
ax.bar(world_new.year,world_new.rate,)
plt.title('HIV infections worldwide')
plt.ylabel('new cases per 100,000 people')
ax.set_xlim([1989,2018])
new = new[['country_name','country_code', 'rate', 'year']]

new = new.sort_values(by='year')
rate = pd.pivot_table(new, index = 'country_name',columns = ['year'])
rate =rate['rate'].reset_index()

tot_countries_list = rate.country_name.values

countries_list =['Algeria', 'Angola', 'Azerbaijan',
                'Botswana', 'Cambodia', 'Central African Republic','Chile',
                'Ethiopia', 'Kenya', 'Madagascar', 'Russian Federation','Philippines']
def graph_country(country):
    y = rate[rate['country_name']==country].iloc[0,1:].values
    x= rate.columns[1:].values
    ax= plt.subplot(1,1,1)
    ax.bar(x,y)
    ax.legend(['{}'.format(country)])
    plt.ylabel('rate per 100,000 people')
def graph_countries(countries):
    count=0
    l = len(countries)
    plt.figure(figsize = (10,2*l))
    for i in countries:
        count += 1
        y = rate[rate['country_name']==i].iloc[0,1:].values
        x= rate.columns[1:].values
        ax= plt.subplot((l/2),2,count)
        ax.bar(x,y)
        ax.legend(['{}'.format(i)])
        #plt.ylabel('rate per 100,000 people')
graph_countries(countries_list)
new_2017 = new[new.year==2017]
new_2017.set_index('country_code', inplace=True)
shapefile='../input/ne_10m_admin_0_countries'

num_colors = 10
values = new_2017['rate']
cm = plt.get_cmap('Reds')
scheme = [cm(i / num_colors) for i in range(num_colors)]
bins = np.linspace(values.min(), values.max(), num_colors)
new_2017['bin'] = np.digitize(values, bins) - 1
new_2017.sort_values('bin', ascending=False).head(10);

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, frame_on=False)
fig.suptitle('New HIV infections rate per 100,000, 2017',fontsize=15, y=0.88)
m = Basemap(lon_0=0, projection='robin')
m.drawmapboundary(color='w')

m.readshapefile(shapefile, 'units', color='#444444', linewidth=.2)
for info, shape in zip(m.units_info, m.units):
    iso3 = info['ADM0_A3']
    if iso3 not in new_2017.index:
        color = '#dddddd'
    else:
        color = scheme[new_2017.loc[iso3]['bin']]
    patches = [Polygon(np.array(shape), True)]
    pc = PatchCollection(patches)
    pc.set_facecolor(color)
    ax.add_collection(pc)

# Draw color legend.
ax_legend = fig.add_axes([0.35, 0.14, 0.3, 0.03], zorder=2)
cmap = mpl.colors.ListedColormap(scheme)
cb = mpl.colorbar.ColorbarBase(ax_legend, cmap=cmap, ticks=bins, boundaries=bins, orientation='horizontal')
cb.ax.set_xticklabels([str(round(i, 1)) for i in bins]);


plt.savefig('new_infection.png', bbox_inches='tight', pad_inches=.2)
plt.savefig('new_infections.png', bbox_inches='tight', pad_inches=.2)
