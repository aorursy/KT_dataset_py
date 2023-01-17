import jovian

import plotly.graph_objects as go

import pandas as pd



df = pd.read_csv('us_cities_population.csv')



covid = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/earthquakes-23k.csv')



dc = pd.read_csv('DataCenters.csv')

dc['text'] = dc['BuildingCode'] + '-' + dc['BuildingID'].astype(str)+ ', '+dc['City_ID']+', '+dc['zip'].astype(str)

df['text'] = df['city'] + '<br>Population ' + (df['population']/1e6).astype(str)+' million'

limits = [(0,24561),(24562,28590),(28591,28624),(28625,28655),(28656,28674)]

risk = ["Low Risk: 0-0.9","Medium Low Risk: .1-.49","Medium Risk: .50-.99","Med-High Risk: 1-4.99","High Risk: 5-20"]

colors = ["lightgrey","lightyellow","yellow","orange","crimson"]

dc_color = ["black"]

cities = []

scale = 17500



fig = go.Figure()



for i in range(len(limits)):

    lim = limits[i]

    r = risk[i]

    df_sub = df[lim[0]:lim[1]]

    fig.add_trace(go.Scattergeo(

        locationmode = 'USA-states',

        lon = df_sub['lng'],

        lat = df_sub['lat'],

        text = df_sub['text'],

        marker = dict(

            size = df_sub['population']/scale,

            color = colors[i],

            line_color='lightyellow',

            line_width=0.25,

            sizemode = 'area'

        ),

        name = '{0} Million'.format(r[0:])))

    

fig.add_trace(go.Scattergeo(

        lon = dc['long'],

        lat = dc['lat'],

        text = dc['text'],

        marker = dict(color = 'lightblue',line_color='blue',line_width=0.75),

        mode = 'markers',

        marker_color = 'blue',

        name = 'Data Center'

        ))



fig.update_layout(

        title_text = 'US City Population Size Risk - Data Centers',

        showlegend = True,

        geo = dict(

            scope = 'usa',

            landcolor = 'rgb(217, 217, 217)',

        )

    

    )



fig.show()
import plotly.graph_objects as go



import pandas as pd



df = pd.read_csv('us_cities_density.csv')



covid = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/earthquakes-23k.csv')



dc = pd.read_csv('DataCenters.csv')

dc['text'] = dc['BuildingCode'] + '-' + dc['BuildingID'].astype(str)+ ', '+dc['City_ID']+', '+dc['zip'].astype(str)

df['text'] = df['city'] + '<br>Population Density' + df['density'].astype(str)

limits = [(0,19999),(20000,24999),(25000,27499),(27500,28549),(28550,28674)]

risk = ["Low Risk: 0-0.9","Medium Low Risk: .1-.49","Medium Risk: .50-.99","Med-High Risk: 1-4.99","High Risk: 5-20"]

colors = ["lightgrey","lightyellow","yellow","orange","crimson"]

dc_color = ["black"]

cities = []

scale = 50



fig = go.Figure()







for i in range(len(limits)):

    lim = limits[i]

    r = risk[i]

    df_sub = df[lim[0]:lim[1]]

    fig.add_trace(go.Scattergeo(

        locationmode = 'USA-states',

        lon = df_sub['lng'],

        lat = df_sub['lat'],

        text = df_sub['text'],

        marker = dict(

            size = df_sub['density']/scale,

            color = colors[i],

            line_color='lightyellow',

            line_width=0.25,

            sizemode = 'area'

        ),

        name = '{0} Million'.format(r[0:])))  



fig.add_trace(go.Scattergeo(

        lon = dc['long'],

        lat = dc['lat'],

        text = dc['text'],

        marker = dict(color = 'lightblue',line_color='blue',line_width=0.75),

        mode = 'markers',

        marker_color = 'blue',

        name = 'Data Center'

        ))



fig.update_layout(

        title_text = 'US City Population Size Risk - Data Centers',

        showlegend = True,

        geo = dict(

            scope = 'usa',

            landcolor = 'rgb(217, 217, 217)',

        )

    

    )



fig.show()
import plotly.graph_objects as go



import pandas as pd



df = pd.read_csv('us_cities_density.csv')



covid = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/earthquakes-23k.csv')



dc = pd.read_csv('DataCenters.csv')

dc['text'] = dc['BuildingCode'] + '-' + dc['BuildingID'].astype(str)+ ', '+dc['City_ID']+', '+dc['zip'].astype(str)

df['text'] = df['city'] + '<br>Population Density ' + df['density'].astype(str)

limits = [(0,14999),(15000,24999),(25000,27999),(28000,28624),(28625,28674)]

risk = ["Low Risk","Medium-Low Risk","Medium Risk","Med-High Risk","High Risk"]

colors = ["lightgrey","lightyellow","yellow","orange","crimson"]

dc_color = ["black"]

cities = []

scale = 100

fig = go.Figure()

fig = go.Figure(go.Densitymapbox(lat=df.lat, lon=df.lng, z=df.density, radius=10))

fig.update_geos(fitbounds="locations")

fig.update_layout(mapbox_style="open-street-map", mapbox_center_lon=-97, mapbox_center_lat=37)

fig.update_geos(

    visible=False, resolution=110, scope="usa",

    showcountries=True, countrycolor="Black",

    showsubunits=True, subunitcolor="Blue"

)



fig.update_layout(

        title_text = 'US Population Density Plot - FB Data Centers',

        showlegend = True,

        geo = dict(

            scope = 'usa',

            landcolor = 'rgb(217, 217, 217)',

        )

    

    )



fig.show()