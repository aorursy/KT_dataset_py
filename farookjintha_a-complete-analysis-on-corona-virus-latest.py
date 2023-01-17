#Import the necessary libraries

import numpy as np 

import pandas as pd 



#Visualisation

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set()

from plotly.offline import init_notebook_mode, iplot

import plotly.graph_objs as go

import plotly.offline as py

import pycountry

py.init_notebook_mode(connected = True)

import folium

from folium import plugins





%config InlineBackend.figure_format = 'retina'



plt.rcParams['figure.figsize'] = 8, 5









import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



import warnings

warnings.filterwarnings('ignore')





#Display Markdown formatted output such as bold, italic bold and so on...

from IPython.display import Markdown

def bold(string):

    display(Markdown(string))
data= pd.read_csv("../input/novel-corona-virus-2019-dataset/2019_nCoV_data.csv",)

data.head()
data.info()
data['Last Update'] = data['Last Update'].apply(pd.to_datetime)

data['Date'] = data['Date'].apply(pd.to_datetime)

data.drop(['Sno'],axis=1,inplace=True)



data.head()
bold("** Countries, territories or areas with reported confirmed cases, Deaths, Recovered of 2019-nCoV (13-02-2020)**")

from datetime import date

data_Feb13 = data[data['Date']>pd.Timestamp(date(2020, 2, 13))]

data_Feb13.head()

bold('**Present Gobal condition: confirmed, death and recovered**')

print('Globally Confirmed Cases: ',data_Feb13['Confirmed'].sum())

print('Global Deaths: ',data_Feb13['Deaths'].sum())

print('Globally Recovered Cases: ',data_Feb13['Recovered'].sum())
bold("** Country-wise data on Corona effects upto 13-Feb-2020 **")

temp = data_Feb13.groupby('Country')['Confirmed','Deaths','Recovered'].sum().reset_index()



cm = sns.light_palette("blue", as_cmap=True)



# Set CSS properties for th elements in dataframe

th_props = [('font-size', '11px'),('text-align', 'center'),('font-weight', 'bold'),('color', '#6d6d6d'),('background-color', '#f7f7f9')]



## Set CSS properties for td elements in dataframe

td_props = [('font-size', '11px'),('color', 'grey')]



# Set table styles

styles = [

  dict(selector="th", props=th_props),

  dict(selector="td", props=td_props)

  ]



(temp.style

  .background_gradient(cmap=cm, subset=["Confirmed","Deaths","Recovered"])

  .highlight_max(subset=["Confirmed","Deaths","Recovered"])

  .set_caption('*China Have most confirmed, deaths & recovered cases.')

  .set_table_styles(styles))



countries = data_Feb13['Country'].unique().tolist()

print("\nTotal countries affected by virus: ",len(countries))
d = data['Date'][-1:].astype('str')

year = int(d.values[0].split('-')[0])

month = int(d.values[0].split('-')[1])

day = int(d.values[0].split('-')[2].split()[0])



from datetime import date

data_latest = data_Feb13[data_Feb13['Date'] > pd.Timestamp(date(year,month,day))]

data_latest.head()
Number_of_countries = len(data_latest['Country'].value_counts())





cases = pd.DataFrame(data_latest.groupby('Country')['Confirmed'].sum())

cases['Country'] = cases.index

cases.index=np.arange(1,Number_of_countries+1)



global_cases = cases[['Country','Confirmed']]
import pandas as pd

world_coordinates = pd.read_csv("../input/world-coordinates/world_coordinates.csv")

world_data = pd.merge(world_coordinates,global_cases,on='Country')
world_map = folium.Map(location=[10, -20], zoom_start=2.3,tiles='Stamen Toner')



for lat, lon, value, name in zip(world_data['latitude'], world_data['longitude'], world_data['Confirmed'], world_data['Country']):

    folium.CircleMarker([lat, lon],

                        radius=10,

                        popup = ('<strong>Country</strong>: ' + str(name).capitalize() + '<br>'

                                '<strong>Confirmed Cases</strong>: ' + str(value) + '<br>'),

                        color='red',

                        

                        fill_color='red',

                        fill_opacity=0.7 ).add_to(world_map)

world_map
d = data_Feb13['Date'][-1:].astype('str')

year = int(d.values[0].split('-')[0])

month = int(d.values[0].split('-')[1])

day = int(d.values[0].split('-')[2].split()[0])



from datetime import date

data_latest = data[data['Date'] > pd.Timestamp(date(year,month,day))]

data_latest.head()
china_Feb13 = data_Feb13[data_Feb13['Country']=='Mainland China'][["Province/State","Confirmed","Deaths","Recovered"]]



bold("**Present Scenario of China Condition of 2019-nCoV (13-02-2020)**")



cm = sns.light_palette("lightblue", as_cmap=True)



# Set CSS properties for th elements in dataframe

th_props = [

  ('font-size', '11px'),

  ('text-align', 'center'),

  ('font-weight', 'bold'),

  ('color', '#6d6d6d'),

  ('background-color', '#f7f7f9')

  ]



## Set CSS properties for td elements in dataframe

td_props = [

  ('font-size', '11px'),

  ('color', 'grey')

   ]



# Set table styles

styles = [

  dict(selector="th", props=th_props),

  dict(selector="td", props=td_props)

  ]



(china_Feb13.style

  .background_gradient(cmap=cm, subset=["Confirmed","Deaths","Recovered"])

  .highlight_max(subset=["Confirmed","Deaths","Recovered"])

  .set_table_styles(styles))
f, ax = plt.subplots(figsize=(12, 8))

sns.set_color_codes("pastel")

sns.barplot(x="Confirmed", y="Province/State", data=china_Feb13[1:],

            label="Confirmed", color="b")



sns.set_color_codes("muted")

sns.barplot(x="Recovered", y="Province/State", data=china_Feb13[1:],

            label="Recovered", color="g")



sns.set_color_codes("deep")

sns.barplot(x="Deaths", y="Province/State", data=china_Feb13[1:],

            label="Deaths", color="r")



# Add a legend and informative axis label

ax.set_title('Confirmed vs Recovered vs Death figures of Provinces of China other than Hubei', fontsize=20, fontweight='bold', position=(0.53, 1.05))

ax.legend(ncol=3, loc="lower right", frameon=True)

ax.set(xlim=(0, 1300), ylabel="",

       xlabel="Stats")

sns.despine(left=True, bottom=True)
fig = go.Figure()



ch_map_data = pd.DataFrame({

   'State':list(china_Feb13['Province/State']),

   'lat':[30.58333,30.29365,23.116667,33.57,25.97,28.655758,29.562778,31.863889,36.790556,30.666667,32.061667,31.408447,39.928819,

          24.513333,38.041389,22.816667,34.346335,25.038889,40.743394,46.583333,20.045833,37.869444,36.057006,39.266667,26.25,38.468056,

          40.652222,43.807347,43.850833,36.625541,29.65],

   'lon':[114.266667,120.161419,113.25,114.03,113.4,115.905049,106.552778,117.280833,118.063333,104.066667,118.777778,121.489563,

          116.388869,117.655556,114.478611,108.316667,108.718164,102.718333,120.816702,125,110.341667,112.560278,103.839868,117.8,

          105.933333,106.273056,109.822222,87.630506,126.560278,101.75739,91.1,],

   'Confirmed':list(china_Feb13['Confirmed']),

   'Recovered':list(china_Feb13['Recovered']),

   'Deaths':list(china_Feb13['Deaths'])

})





fig.add_trace(go.Scattergeo(

        lat=ch_map_data['lat'],

        lon=ch_map_data['lon'],

        mode='markers',

        marker=dict(

            size=15,

            color='black',

            opacity=0.7

        ),

         showlegend=False

    ))



fig.add_trace(go.Scattergeo(

        lat=ch_map_data['lat'],

        lon=ch_map_data['lon'],

        name = 'Confirmed',

        mode='markers',

        marker=dict(

            size=10,

            color='orange',

            opacity=0.9

        ),

        text=ch_map_data[['State','Confirmed']],

        hoverinfo='text',

    

    ))



fig.update_layout(

        autosize=True,

        hovermode='closest',

        showlegend=True,

        title_text = '<b>China states with reported Confirmed cases of 2019-nCoV,<br>(13-02-2020)</b>',

        font=dict(family="Arial, Balto, Courier New, Droid Sans",color='blue'),

        geo = go.layout.Geo(

        scope = 'asia',

        showframe = False,

        showcountries = True,

        landcolor = "rgb(229, 229, 229)",

        countrycolor = "black",

        ))



 

fig.show()



fig = go.Figure()





ch_re = ch_map_data[ch_map_data['Recovered']> 0]

fig.add_trace(go.Scattergeo(

        lat=ch_re['lat'],

        lon=ch_re['lon'],

        name = 'Recovered',

        mode='markers',

        marker=dict(

            size=15,

            color='green',

            opacity=0.8

        ),

        text=ch_re[['State','Recovered']],

        hoverinfo='text'

    ))



ch_de = ch_map_data[ch_map_data['Deaths']> 0]

fig.add_trace(go.Scattergeo(

        lat=ch_de['lat'],

        lon=ch_de['lon'],

        name = 'Deaths',

        mode='markers',

        marker=dict(

            size=10,

            color='red',

            opacity=0.8

        ),

        text=ch_de[['State','Deaths']],

        hoverinfo='text'

    ))



fig.update_layout(

        autosize=True,

        hovermode='closest',

        showlegend=True,

        title_text = '<b>China states with reported Recovered, Deaths cases of 2019-nCoV,<br>(13-02-2020)</b>',

        font=dict(family="Arial, Balto, Courier New, Droid Sans",color='blue'),

        geo = go.layout.Geo(

        scope = 'asia',

        showframe = False,

        showcountries = True,

        landcolor = "rgb(229, 229, 229)",

        countrycolor = "black",

        ))



 

fig.show()