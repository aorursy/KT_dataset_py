# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



#plotting libraries

import plotly

from plotly.graph_objs import *

import plotly.offline as ply

plotly.offline.init_notebook_mode()



#Insert your mapbox access token here; this is a dummy token

mapbox_access_token='3427409832rhj2efkwjhr9wefhfihw84'



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
bikes=pd.read_csv('../input/bikes.csv')



bikes.head()
for col in bikes.columns:

    print(col,len(bikes[col].unique()))
bikes=bikes.drop(['bonus','banking','contract_name'], axis=1)
groupby_lat_lng=pd.DataFrame(bikes.groupby(['lat','lng']).sum())
groupby_lat_lng.head()
groupby_lat_lng=groupby_lat_lng.reset_index()



groupby_lat_lng.head()
groupby_lat_lng['available_bike_stands_ratio']=groupby_lat_lng['available_bike_stands']/groupby_lat_lng['bike_stands']
data = Data([

    Scattermapbox(

        lat=groupby_lat_lng['lat'],

        lon=groupby_lat_lng['lng'],

        mode='markers',

        marker=Marker(

            size=groupby_lat_lng['available_bike_stands_ratio']*25

        ),

        text=['Available bike stands: '+i for i in np.char.mod('%d',groupby_lat_lng['available_bike_stands'])]

    )

])



layout = Layout(

    autosize=True,

    hovermode='closest',

    mapbox=dict(

        accesstoken=mapbox_access_token,

        zoom=11.5,

        center=dict(

            lat=groupby_lat_lng['lat'].mean(),

            lon=groupby_lat_lng['lng'].mean()

        ),

    ),

)



ply.iplot(dict(data=data, layout=layout))