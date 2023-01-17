# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # this is used for the plot the graph 

import seaborn as sns # used for plot interactive graph.

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import warnings

warnings.filterwarnings('ignore')

from pylab import rcParams

# figure size in inches

%matplotlib inline



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/global-significant-earthquake-database-from-2150bc/Worldwide-Earthquake-database.csv',index_col=0)
df.head()
df.shape
df.info()
df.EQ_PRIMARY.isna().sum()
#missing data

total = df.isnull().sum().sort_values(ascending=False)

percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

cols = missing_data.head(10).index

df1 = df.drop(cols,axis="columns")
df1.isna().sum()
fig = plt.figure(figsize=(16,8))

x = df1.groupby("YEAR")["COUNTRY"].count()

plt.plot(x,linestyle='solid',marker='o',label="Count of Recorded Earthquakes")

plt.xlabel('Year', fontsize=20)

plt.ylabel('Count', fontsize=20)

plt.title('Count of Earthquakes', fontsize=20)

plt.style.use('fivethirtyeight')

plt.grid(True)

plt.legend()

plt.tight_layout()

fig = plt.figure(figsize=(16,8)) 

labels = ['No = No Tsunami','Yes = Tsunami']

sizes = df1['FLAG_TSUNAMI'].value_counts(sort = True)

plt.pie(sizes,labels=labels,autopct='%1.1f%%', shadow=True)

plt.title('% of Earthquakes resulting in Tsunami',size = 20)

plt.legend()

plt.show()



fig = plt.figure(figsize=(16,8)) 

country = df1.groupby("COUNTRY")["YEAR"]

country.count().sort_values(ascending=False).head(20).plot(kind="barh")

plt.title('Most number of Earthquakes', fontsize=20)
fig = plt.figure(figsize=(16,8)) 

country_Tsunami = df1[df1["FLAG_TSUNAMI"] == "Yes"].groupby("COUNTRY")["YEAR"]

country_Tsunami.count().sort_values(ascending=False).head(20).plot(kind="barh")

plt.title('Most number of Tsunamis', fontsize=20)
#Plotting on the WorldMap using plotly

x = df.groupby("COUNTRY")["INTENSITY"].mean().sort_values()

data = dict(type = 'choropleth',

            locations = x.index,

            locationmode = 'country names',

            colorscale= 'Portland',

            text= x.index,

            z=x,

            colorbar = {'title': 'Mean Earthquake Intensity', 'len':200,'lenmode':'pixels' })

layout = dict(geo = {'scope':'world'},title="Mean Earthquake Intensity around the world",width=1200, height=600)

col_map = go.Figure(data = [data],layout = layout)

col_map.show()
fig = plt.figure(figsize=(16,8)) 

country = df1.groupby("COUNTRY")["TOTAL_DEATHS"].sum()

country.sort_values(ascending=False).head(20).plot(kind="barh")

plt.title('Most number of Deaths due to Earthquake', fontsize=20)
df1["TOTAL_DEATHS"].describe()
df2 = df[["FLAG_TSUNAMI","YEAR","MONTH","DAY","HOUR","MINUTE","SECOND","FOCAL_DEPTH","EQ_PRIMARY","INTENSITY","COUNTRY","STATE","LOCATION_NAME","LATITUDE","LONGITUDE","REGION_CODE","DEATHS","TOTAL_DEATHS","TOTAL_INJURIES"]]
df2.head()
fig = plt.figure(figsize=(16,8)) 

df2.groupby("LOCATION_NAME")["EQ_PRIMARY"].count().sort_values(ascending=False).head(20).plot(kind="barh")

plt.title('Location with Most number of Earthquakes', fontsize=20)