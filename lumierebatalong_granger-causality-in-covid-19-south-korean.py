# install the package we need

!pip install vincent

!pip install pmdarima

!pip install chart_studio
import seaborn as sns

import matplotlib.pyplot as plt 

import numpy as np

from pathlib import Path

import pandas as pd
# plotly standard imports

import plotly.graph_objs as go

import chart_studio.plotly as py



# Cufflinks wrapper on plotly

import cufflinks





# Display all cell outputs

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = 'all'



from plotly.offline import iplot

cufflinks.go_offline()



# Set global theme

cufflinks.set_config_file(world_readable=True, theme='ggplot')
filename = '../input/coronavirusdataset/'

route = pd.read_csv(filename+'/route.csv')
route.head()
route.info()
route['date'] = pd.to_datetime(route['date'])

route = route.set_index('date')

route.head()
#check if data have missing value

missing = route.isnull().sum()

missing
province_id = route.groupby('province')['id'].aggregate([np.sum])

city_id = route.groupby('city')['id'].aggregate([np.sum])

visit_id = route.groupby('visit')['id'].aggregate([np.sum])
province_id.iplot(kind='bar', title='total id in each province', xTitle='Province',\

                  yTitle='total id')
city_id.iplot(kind='bar', title='total id in each city', xTitle='city',\

                  yTitle='total id')
visit_id.iplot(kind='bar', title='total id in each visit', xTitle='city',\

                  yTitle='total id')


def color(id):

    minimun = int(route['id'].min())

    step = int((route['id'].max() - route['id'].min())/3)

    

    if id in range(minimun, minimun+step):

        col = 'blue'

    elif id in range(minimun+step, minimun+step*2):

        col = 'orange'

    else:

        col = 'red'

    

    return col
import folium as fl

import json

import vincent



def geospace():



    mapped = fl.Map(location=[route['latitude'].mean(),route['longitude'].mean()],zoom_start=6,\

             control_scale=True, world_copy_jump=True, no_wrap=False)



    fg_province= fl.FeatureGroup(name="Province")

    for lat, lon, name, id in zip(route['latitude'], route['longitude'], route['province'],route['id']):

        fl.Marker(location=[lat, lon], popup=(fl.Popup(name+ ' id = '+ str(id))),\

              icon = fl.Icon(color=color(id))).add_to(fg_province)

    

    fg_city = fl.FeatureGroup(name='City')

    for lat, lon, name, id in zip(route['latitude'], route['longitude'], route['city'],route['id']):

        fl.Marker(location=[lat, lon], popup=(fl.Popup(name+ ' id = '+ str(id))), \

              icon = fl.Icon(color=color(id))).add_to(fg_city)

    

    fg_visit = fl.FeatureGroup(name="Visit")

    for lat, lon, name, id in zip(route['latitude'], route['longitude'], route['visit'],route['id']):

        fl.Marker(location=[lat, lon], popup=(fl.Popup(name+ ' id = '+ str(id))),\

              icon = fl.Icon(color=color(id))).add_to(fg_visit)

    

    fg_time = fl.FeatureGroup(name='South Korean covid-19|year=2020')

    for lat, lon, name, id in zip(route['latitude'], route['longitude'], route['province'],route['id']):

    

        y=route['id'][route['province']==name]

    

        date = [d.strftime('%m/%d') for d in y.index.date] #[]

    

        

        multi_iter2 = pd.DataFrame(y.values, index=date).sort_index()

        scatter = vincent.GroupedBar(multi_iter2, height=200, width=350)

        data = json.loads(scatter.to_json())

    

        v = fl.features.Vega(data, width='100%', height='100%')

        p = fl.Popup(name)

        pop =p.add_child(v)

        fl.features.Marker(location=[lat, lon], popup=pop,icon = fl.Icon(color=color(id))).add_to(fg_time)

    

    

    

    fg_province.add_to(mapped)

    fg_city.add_to(mapped)

    fg_visit.add_to(mapped)

    fg_time.add_to(mapped)

    fl.LayerControl().add_to(mapped)

    

    return mapped   
geo_map = geospace()

geo_map.save(outfile='South Korean Coronavirus.html')
geo_map
route_daily = route[['latitude', 'longitude', 'id']]
route_daily['day'] = [d.strftime('%m/%d') for d in route.index.date]

route_daily.head() 
fig, ax = plt.subplots(1,1, figsize=(15.5,5.5))

h=sns.boxplot(x='day', y='id', data=route_daily.sort_values(by=['day']), ax=ax)

ax.set_title('South Korean COVID-19|year=2020 boxplot')
# display each time series

from pmdarima import *
fx=utils.tsdisplay(route_daily['longitude'],title='longitude')
fy =  utils.tsdisplay(route_daily['id'], title='id')
fv = utils.tsdisplay(route_daily['latitude'], title='latitude')
ts_route = route_daily.copy()

ts_route = ts_route.reset_index()

ts_route = ts_route.sort_values(by=['date'])

ts_route.head()
def is_GrangerCause(data=None, maxlag=30):

    """This function find if x2 Granger cause x1 vis versa """    

    from statsmodels.tsa.stattools import grangercausalitytests

    gc = grangercausalitytests(data, maxlag=maxlag, verbose=False)

    

    for i in range(maxlag):

        x=gc[i+1][0]

        p1 = x['lrtest'][1] # pvalue for lr test

        p2 = x['ssr_ftest'][1] # pvalue for ssr ftest

        p3 = x['ssr_chi2test'][1] #pvalue for ssr_chi2test

        p4 = x['params_ftest'][1] #pvalue for 'params_ftest'

        

        condition = ((p1 < 0.05 and p2 < 0.05) and (p3 < 0.05 and p4 < 0.05))

        

        if condition == True:

            cols = data.columns

            print('Yes: {} Granger causes {}'.format(cols[0], cols[1]))

            print('maxlag = {}\nResults: {}'.format(i, x))

            break

            

        else:

            if i == maxlag - 1:

                cols = data.columns

                print('No: {} does not Granger cause {}'.format(cols[0], cols[1]))
is_GrangerCause(data = ts_route[['longitude', 'id']])
is_GrangerCause(data = ts_route[['id','longitude']])
is_GrangerCause(data = ts_route[['id','latitude']])
is_GrangerCause(data = ts_route[['latitude','id']])
is_GrangerCause(data = ts_route[['latitude','longitude']])
is_GrangerCause(data = ts_route[['longitude','latitude']])