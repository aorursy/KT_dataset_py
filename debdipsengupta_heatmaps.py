# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly.express as px #graphic library
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
data=pd.read_csv("/kaggle/input/us-accidents/US_Accidents_June20.csv").query("City=='Los Angeles'")
df = data[(data.Severity >= 3)]
df.head(5)
import pandas as pd
data=pd.read_csv("/kaggle/input/us-accidents/US_Accidents_June20.csv").query("City=='Los Angeles'")
df = data[(data.Severity >= 3)]
import plotly.express as px #graphic library
fig = px.density_mapbox(df, lat='Start_Lat', lon='Start_Lng', radius=5, zoom=7,
                        color_continuous_scale=px.colors.sequential.YlOrRd,
                        mapbox_style="stamen-terrain")
fig.update_layout(
        title = 'Los Angeles - Accidents',
)
fig.show()
import pandas as pd
data=pd.read_csv("/kaggle/input/us-accidents/US_Accidents_June20.csv").query("City=='Dallas'")
df = data[(data.Severity >= 3)]
import plotly.express as px #graphic library
fig = px.density_mapbox(df, lat='Start_Lat', lon='Start_Lng', radius=5, zoom=7,
                        color_continuous_scale=px.colors.sequential.YlOrRd,
                        mapbox_style="stamen-terrain")
fig.update_layout(
        title = 'Dallas - Accidents',
)
fig.show()
import pandas as pd
data=pd.read_csv("/kaggle/input/us-accidents/US_Accidents_June20.csv").query("City=='Atlanta'")
df = data[(data.Severity >= 3)]
import plotly.express as px #graphic library
fig = px.density_mapbox(df, lat='Start_Lat', lon='Start_Lng', radius=5, zoom=7,
                        color_continuous_scale=px.colors.sequential.YlOrRd,
                        mapbox_style="stamen-terrain")
fig.update_layout(
        title = 'Atlanta - Accidents',
)
fig.show()
import pandas as pd
data=pd.read_csv("/kaggle/input/us-accidents/US_Accidents_June20.csv").query("City=='Columbus'")
df = data[(data.Severity >= 3)]
import plotly.express as px #graphic library
fig = px.density_mapbox(df, lat='Start_Lat', lon='Start_Lng', radius=5, zoom=7,
                        color_continuous_scale=px.colors.sequential.YlOrRd,
                        mapbox_style="stamen-terrain")
fig.update_layout(
        title = 'Columbus,OH - Accidents',
)
fig.show()
import pandas as pd
data=pd.read_csv("/kaggle/input/us-accidents/US_Accidents_June20.csv").query("City=='Saint Louis'")
df = data[(data.Severity >= 3)]
import plotly.express as px #graphic library
fig = px.density_mapbox(df, lat='Start_Lat', lon='Start_Lng', radius=5, zoom=7,
                        color_continuous_scale=px.colors.sequential.YlOrRd,
                        mapbox_style="stamen-terrain")
fig.update_layout(
        title = 'Saint Louis - Accidents',
)
fig.show()
import pandas as pd
data=pd.read_csv("/kaggle/input/us-accidents/US_Accidents_June20.csv").query("City=='San Antonio'")
df = data[(data.Severity >= 3)]
import plotly.express as px #graphic library
fig = px.density_mapbox(df, lat='Start_Lat', lon='Start_Lng', radius=5, zoom=7,
                        color_continuous_scale=px.colors.sequential.YlOrRd,
                        mapbox_style="stamen-terrain")
fig.update_layout(
        title = 'San Antonio - Accidents',
)
fig.show()