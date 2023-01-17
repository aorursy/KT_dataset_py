import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import codecs
import base64
%matplotlib inline
from IPython.display import HTML, display
from matplotlib import animation,rc
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
from mpl_toolkits.basemap import Basemap
import plotly.tools as tls
import time
import warnings
warnings.filterwarnings('ignore')
terror_data = pd.read_csv('../input/globalterrorismdb_0617dist.csv', encoding='ISO-8859-1',
                          usecols=[0, 1, 2, 3, 8,9, 10,11,12,13, 14, 35, 82, 98, 101])
terror_data = terror_data.rename(
    columns={'eventid':'id', 'iyear':'year', 'imonth':'month', 'iday':'day',
             'country_txt':'country', 'provstate':'state', 'targtype1_txt':'target',
             'weaptype1_txt':'weapon', 'nkill':'fatalities', 'nwound':'injuries'})

terror_data = terror_data[np.isfinite(terror_data.latitude)]
terror_data = terror_data.sort_values(['country'], ascending = False)
terror_data.columns
plt.figure(figsize=(15,8))
europe = terror_data[terror_data["region_txt"].isin(["Eastern Europe", "Western Europe"])]

EU = Basemap(projection='mill', llcrnrlat = 10, urcrnrlat = 75, llcrnrlon = -15, urcrnrlon = 70, resolution = 'l')
#EU.drawcoastlines()
EU.etopo()
EU.drawcountries()
EU.drawstates()

x, y = EU(list(europe["longitude"].astype("float")), list(europe["latitude"].astype(float)))
EU.plot(x, y, "bo", markersize = 2, alpha = 0.6, color = 'blue')

plt.title('Terror Attacks on Europe (1970-2016)')
plt.show()
weaptype = terror_data[terror_data["region_txt"]=='Western Europe'].groupby('weapon')['id'].count().reset_index().sort_values('id', ascending=False)
weaptype 
# These coordinates form the bounding box of Germany
bot, top, left, right = -10.75, 4, 35, 44 # just to zoom in to only Germany
spain_map = Basemap(projection='merc', resolution='l',    llcrnrlat=left,    llcrnrlon=bot,    urcrnrlat=right,    urcrnrlon=top)
spain = terror_data[terror_data["country"] == "Spain"]
x, y = spain_map(list(spain["longitude"].astype("float")), list(spain["latitude"].astype(float)))


fig = plt.figure(figsize=(20,10))  # predefined figure size, change to your liking. 
plt.title('Terror Attacks in Spain 1970-2016')
# But doesn't matter if you save to any vector graphics format though (e.g. pdf)
ax = fig.add_axes([0.05,0.05,0.9,0.85])

# add county shapes from http://www.gadm.org/download
#spain_map.drawcoastlines()
#spain_map.drawcountries()
#spain_map.drawstates()
spain_map.drawmapboundary()
spain_map.fillcontinents(color='lightblue')

spain_map.plot(x,y,'.', markersize = 10, color='red', alpha=0.5)
               
cities = terror_data[terror_data["country"] == "Spain"].city.replace('Unknown', 'Bilbao').values
dead = terror_data[terror_data["country"] == "Spain"].fatalities.fillna(0).values
suma = []
for i in range(len(cities)):
    if dead[i]>3:
        suma.append([cities[i], dead[i]])
suma.insert(0, ['City', 'Fatalities'])

s = '''<html>
  <head>
    <script type='text/javascript' src='https://www.gstatic.com/charts/loader.js'></script>
    <script type='text/javascript'>
     google.charts.load('current', {
       'packages': ['geochart'],
       // Note: you will need to get a mapsApiKey for your project.
       // See: https://developers.google.com/chart/interactive/docs/basic_load_libs#load-settings
       'mapsApiKey': 'AIzaSyD-9tSrke72PouQMnMX-a7eZSW0jkFMBWY'
     });
     google.charts.setOnLoadCallback(drawMarkersMap);

      function drawMarkersMap() {
      var data = google.visualization.arrayToDataTable(%s
 );

      var options = {
        region: 'ES',
        displayMode: 'markers',
        colorAxis: {colors: ['green', 'red']}
      };

      var chart = new google.visualization.GeoChart(document.getElementById('chart_div'));
      chart.draw(data, options);
    };
    </script>
  </head>
  <body>
    <div id="chart_div" style="width: 900px; height: 500px;"></div>
  </body>
</html>''' % suma
from IPython.core.display import display, HTML
display(HTML(s))
#Thanks to I,Code.
terror_data = pd.read_csv('../input/globalterrorismdb_0617dist.csv', encoding='ISO-8859-1',
                          usecols=[0, 1, 2, 3, 8,9, 10,11,12,13, 14, 35, 82, 98, 101])
terror_data = terror_data.rename(
    columns={'eventid':'id', 'iyear':'year', 'imonth':'month', 'iday':'day',
             'country_txt':'country', 'provstate':'state', 'targtype1_txt':'target',
             'weaptype1_txt':'weapon', 'nkill':'fatalities', 'nwound':'injuries'})

terror_data = terror_data[np.isfinite(terror_data.latitude)]

fig = plt.figure(figsize = (10,6))
def animate(Year):
    ax = plt.axes()
    ax.clear()
    ax.set_title('Animation Of Terrorist Activities'+'\n'+'Year:' +str(Year))
    m6 = Basemap(projection='mill',llcrnrlat=-80,urcrnrlat=80, llcrnrlon=-180,urcrnrlon=180,lat_ts=20,resolution='c')
    lat6=list(terror_data[terror_data['year']==Year].latitude)
    long6=list(terror_data[terror_data['year']==Year].longitude)
    x6,y6=m6(long6,lat6)
    m6.scatter(x6, y6,s=[(fatalities+injuries)*0.1 for fatalities,injuries in zip(terror_data[terror_data['year']==Year].fatalities,terror_data[terror_data['year']==Year].injuries)],color = 'r')
    m6.drawcoastlines()
    m6.drawcountries()
    m6.fillcontinents(zorder = 1,alpha=0.4)
    m6.drawmapboundary()
ani = animation.FuncAnimation(fig,animate,list(terror_data.year.unique()), interval = 1500)    
ani.save('animation.gif', writer='imagemagick', fps=1)
plt.close(1)
filename = 'animation.gif'
video = io.open(filename, 'r+b').read()
encoded = base64.b64encode(video)
HTML(data='''<img src="data:image/gif;base64,{0}" type="gif" />'''.format(encoded.decode('ascii')))