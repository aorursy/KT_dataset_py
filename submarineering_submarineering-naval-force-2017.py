# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from IPython.core.display import display, HTML

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/GlobalFirePower.csv')
sub = df[['Country', 'Submarines']].sort_values('Submarines', ascending=False)[0:40]['Submarines'].values

country = df[['Country', 'Submarines']].sort_values('Submarines', ascending=False)[0:40]['Country'].values



suma = []

for i in range(len(sub)):

   suma.append([country[i], sub[i]])

suma.insert(0, ['Country', 'Submarines'])
s = '''<html>

  <head>

    <script type="text/javascript" src="https://www.gstatic.com/charts/loader.js"></script>

    <script type="text/javascript">

      google.charts.load('current', {

        'packages':['geochart'],

        // Note: you will need to get a mapsApiKey for your project.

        // See: https://developers.google.com/chart/interactive/docs/basic_load_libs#load-settings

        'mapsApiKey': 'AIzaSyD-9tSrke72PouQMnMX-a7eZSW0jkFMBWY'

      });

      google.charts.setOnLoadCallback(drawRegionsMap);



      function drawRegionsMap() {

        var data = google.visualization.arrayToDataTable(%s);



        var options = {colorAxis: {colors: ['yellow', 'red']},

        backgroundColor: 'lightblue'};



        var chart = new google.visualization.GeoChart(document.getElementById('regions_div'));



        chart.draw(data, options);

      }

    </script>

  </head>

  <body>

    <div id="regions_div" style="width: 900px; height: 500px;"></div>

  </body>

</html>

      ''' % suma
display(HTML(s))