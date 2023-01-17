import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.offline as py

from plotly.offline import init_notebook_mode, iplot

import plotly.graph_objs as go

import plotly.offline as offline

from plotly import tools

import plotly.figure_factory as ff

import plotly.express as px
data=pd.read_csv("../input/suicide-rates-overview-1985-to-2016/master.csv")

data.head()
data.describe()
data.country.value_counts()[:5]
# don't parse dates while running this code

perc = data.loc[:,["year","country",'suicides_no']]

perc['total_suicides'] = perc.groupby([perc.country,perc.year])['suicides_no'].transform('sum')

perc.drop('suicides_no', axis=1, inplace=True)

perc = perc.drop_duplicates()

perc = perc[(perc['year']>=1990.0) & (perc['year']<=2012.0)]

perc = perc.sort_values("year",ascending = False)



top_countries = ['Mauritius','Austria','Iceland','Netherlands',"Republic of Korea"] 

perc = perc.loc[perc['country'].isin(top_countries)]

perc = perc.sort_values("year")

fig=px.bar(perc,x='country', y="total_suicides", animation_frame="year", 

           animation_group="country", color="country", hover_name="country")

fig.show()
data.age.value_counts().plot(kind='pie',shadow=True,startangle=90,explode=(0,0.1,0,0,.2,0),figsize=(15,10))
data=pd.read_csv("../input/suicide-rates-overview-1985-to-2016/master.csv",parse_dates=["year"],index_col="year")

data.head()
data['gdp_per_capita ($)'][:'2000'].plot(figsize=(15,10),legend=True,color='r')

data['gdp_per_capita ($)']['2000':].plot(figsize=(15,10),legend=True,color='g')

plt.legend(["Before 2000","After 2000"])