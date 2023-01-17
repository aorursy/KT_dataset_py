import pandas as pd

import plotly.graph_objs as go 

from plotly.offline import init_notebook_mode,iplot,plot

init_notebook_mode(connected=True)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import numpy as np
df_Homelessness = pd.read_csv('../input/2007-2016-Homelessnewss-USA.csv',thousands=',')
df_Homelessness.head(10)
df_Homelessness.info()
df_Homelessness.isnull().sum()
df_Homelessness.dtypes
df_Homelessness['Year'] = pd.to_datetime(df_Homelessness['Year'])
df_Homelessness.dtypes
df_Homelessness.nunique()
df_Homelessness.Measures.unique()
df_Homelessness_2016 = df_Homelessness[df_Homelessness['Year']== '1/1/2016']
df_Homelessness_2016_Total =  df_Homelessness_2016[df_Homelessness_2016['Measures']== 'Total Homeless']
df_Homelessness_2016_Total.head()
s_2016_Total = df_Homelessness_2016_Total.groupby(['State'])['Count'].max()

s_2016_Total= s_2016_Total.sort_values(ascending=False)



df_2016_Total = pd.DataFrame(s_2016_Total)

df_2016_Total.reset_index(inplace=True)

df_2016_Total.shape
df_2016_Total.head()
f,ax = plt.subplots(figsize=(18,11))

sns.barplot(x='Count',y='State',data=df_2016_Total,lw =2.5);

ax.set(ylabel='States',xlabel= 'Total Homeless Population Count',);

plt.title('Total Homeless Population Count In 2016 Sorted by Count',fontsize=(18));

major_ticks = np.arange(0, 80000, 5000)

plt.xticks(major_ticks);

sns.despine()
f,ax = plt.subplots(figsize=(16,8))

sns.set_style("whitegrid")

sns.boxplot(x='Count',data=df_2016_Total)





ax.set(ylabel='...',xlabel= 'Total Homelessness Count For U.S. States');

plt.title('Total Homeless Population Count In 2016 For U.S.',fontsize=(18));



# Set major ticks for x axis

major_ticks = np.arange(0, 80000, 5000)

plt.xticks(major_ticks);

plt.xlim(0,78000);

df_Homelessness_2016_Total[df_Homelessness_2016_Total['Count'] > 10000]
df_Homelessness_2016_Total[df_Homelessness_2016_Total['Count'] == df_2016_Total.Count.min()]
data = dict(type='choropleth',

            colorscale = 'Jet',

            reversescale = True,

            locations = df_2016_Total['State'],

            z = df_2016_Total['Count'],

            locationmode = 'USA-states',

            text = df_2016_Total['State'],

            marker = dict(line = dict(color = 'rgb(255,255,255)',width = 1)),

            colorbar = {'title':'Total Homelessness Count For U.S.'}

            ) 



layout = dict(title = 'Total Homeless Population Count In 2016 For U.S.',

              geo = dict(scope='usa',

                         showlakes = True,

                         lakecolor = 'rgb(85,173,240)')

             )
choromap = go.Figure(data = [data],layout = layout)

iplot(choromap,validate=False)