import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.offline as py

import plotly.graph_objs as go

from plotly.offline import iplot

import plotly.express as px
data= pd.read_csv('../input/pandas-bokeh/long_data_.csv')

data.head()
def year(x):

    return int(x[6:10])

data['Year']=data['Dates'].apply(year)
data.head()
perc = data.loc[:,["Year","States",'Usage']]

perc['total_usage'] = perc.groupby([perc.States,perc.Year])['Usage'].transform('sum')

perc.drop('Usage', axis=1, inplace=True)

perc = perc.drop_duplicates()

perc = perc.sort_values("Year",ascending =True)

fig=px.bar(perc,x='States', y="total_usage", animation_frame="Year", 

           animation_group="States", color="States", hover_name="States")

fig.show()
states=np.array(data['States'])

from wordcloud import WordCloud

cloud=WordCloud(width=800, height=400)

cloud.generate(" ".join(states))

cloud.to_image()
fig_px = px.scatter_mapbox(data, lat="latitude", lon="longitude",hover_name="States",zoom=9, height=300)

fig_px.update_layout(mapbox_style="open-street-map",margin={"r":0,"t":0,"l":0,"b":0})



fig_px.show()
plt.figure(figsize=(10,8))

visual= sns.scatterplot(x='longitude', y='latitude', data=data, hue='States')

visual.legend(loc='center left', bbox_to_anchor=(1.25, 0.5), ncol=2);
plt.figure(figsize=(15,10))

sns.countplot(data.Regions)
data1=pd.read_csv("../input/pandas-bokeh/dataset_tk.csv",index_col='Unnamed: 0',parse_dates=['Unnamed: 0'])

data1.head()
data1.columns
data1["Punjab"][:].plot(figsize=(15,10),legend=True,color='green')
data1["Haryana"][:].plot(figsize=(15,10),legend=True,color='red')

data1["Rajasthan"][:].plot(figsize=(15,10),legend=True,color='c')