import pandas as pd

import numpy as np



import matplotlib.pyplot as plt

from mpl_toolkits.basemap import Basemap
mount = pd.read_csv('../input/Mountains.csv')
mount['lat'] = mount.Coordinates.apply(lambda x: x.split()[0][0:2]+'.'+x.split()[0][3:5]).astype('float')
mount['lont'] = mount.Coordinates.apply(lambda x: x.split()[1][0:2]+'.'+x.split()[1][3:5])
mount.replace('10.°5', '101.52', inplace=True)

mount['lont']= mount['lont'].astype('float')
mount.head()
plt.figure(1, figsize=(20,10))

m = Basemap(projection='merc',

             llcrnrlat=-60,

             urcrnrlat=70,

             llcrnrlon=-180,

             urcrnrlon=180,

             lat_ts=0,

             resolution='h')



m.drawcoastlines()

m.drawcountries()

m.fillcontinents(color = 'gainsboro')

m.drawmapboundary(fill_color='steelblue')

x, y = m(mount['lont'].tolist(), mount['lat'].tolist())

m.plot(x, y, 

            'o',                    # marker shape

            c="#1292db", lw=0, alpha=1, zorder=5         # marker size

            )

plt.title("World's_Highest_Mountains")

plt.show()
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot



init_notebook_mode()

scl = [ [0,"rgb(5, 10, 172)"],[0.35,"rgb(40, 60, 190)"],[0.5,"rgb(70, 100, 245)"],\

    [0.6,"rgb(90, 120, 245)"],[0.7,"rgb(106, 137, 247)"],[1,"rgb(220, 220, 220)"] ]



data = [ dict(

        type = 'scattergeo',

        

        lon = mount['lont'],

        lat = mount['lat'],

        text = mount['Mountain'],

        mode = 'markers',

        marker = dict( 

            size = 8, 

            opacity = 0.8,

            reversescale = True,

            autocolorscale = False,

            symbol = 'diamond',

            line = dict(

                width=1,

                color='rgba(102, 102, 102)'

            ),

            colorscale = scl,

            cmin = 0,

            color = mount['Height (m)'],

            cmax = mount['Height (m)'].max(),

            colorbar=dict(

                title="High"

            )

        ))]



layout = dict(

        title = "World's_Highest_Mountains",

        colorbar = True,   

        geo = dict(

            scope='world',

            projection=dict( type='orthographic' ),

            showland = True,

            landcolor = "rgb(250, 250, 250)",

            subunitcolor = "rgb(217, 217, 217)",

            countrycolor = "rgb(217, 217, 217)",

            countrywidth = 0.5,

            subunitwidth = 0.5,

            showsubunits = True,

            showcountries = True,

            

        ),

    )



fig = dict( data=data, layout=layout )

iplot( fig, validate=False, filename='d3-mounts' )