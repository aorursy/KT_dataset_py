from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from plotly.offline import init_notebook_mode, iplot 

import plotly.graph_objs as go

import plotly.offline as py

import plotly.express as px

import pycountry

py.init_notebook_mode(connected=True)



# Graphics in retina format 

%config InlineBackend.figure_format = 'retina' 



# Increase the default plot size and set the color scheme

plt.rcParams['figure.figsize'] = 8, 5



# Disable warnings in Anaconda

import warnings

warnings.filterwarnings('ignore')
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

nRowsRead = 1000 # specify 'None' if want to read whole file

# Coursera AI GSI Percentile and Category.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows

df = pd.read_csv('/kaggle/input/Coursera AI GSI Percentile and Category.csv', delimiter=',', nrows = nRowsRead)

df.dataframeName = 'Coursera AI GSI Percentile and Category.csv'

nRow, nCol = df1.shape

print(f'There are {nRow} rows and {nCol} columns')
df.head()
df.info()

plt.figure(figsize=(20,12))



sns.countplot(x="region", data=df)
plt.figure(figsize=(20,12))



sns.countplot(x="incomegroup", data=df)
plt.figure(figsize=(20,12))



sns.countplot(x="region", data=df,hue='percentile_category')
plt.figure(figsize=(20,12))

sns.distplot(df['percentile_rank'], hist=True, kde=True)
df.competency_id.value_counts()

# creating different dataframes based on the competency Ids

df_AI = df[df['competency_id'] == 'artificial-intelligence']

df_Stats_prog = df[df['competency_id'] == 'statistical-programming']

df_Stats = df[df['competency_id'] == 'statistics']

df_SE = df[df['competency_id'] == 'software-engineering']

df_Math = df[df['competency_id'] == 'fields-of-mathematics']

df_ML = df[df['competency_id'] == 'machine-learning']
def map(data):

    """

    function to plot a world map of the competency ids, distributed regionwise

    """

    

    fig = go.Figure(data=go.Choropleth(

        locations = data['iso3'],

        z = data['percentile_rank'],

        text = data['percentile_category'],

        colorscale = "Rainbow",

        autocolorscale=False,

        reversescale=True,

        marker_line_color='darkgray',

        marker_line_width=0.5,

        #colorbar_tickprefix = '$',

        colorbar_title = 'Skill Index (1 is highest)'))



    fig.update_layout(

            title_text= data['competency_id'].iloc[0].title() +" "+'Skill Index in 2019',

            geo=dict(

                  showframe=False,

                  showcoastlines=False,

                  projection_type='equirectangular'))



    fig.show()
map(df_AI)
