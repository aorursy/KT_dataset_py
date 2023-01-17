import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
sns.set_style('darkgrid')

plt.style.use('seaborn-darkgrid')
from dask.distributed import Client

client = Client()
import dask.dataframe as dd
df = dd.read_csv('/kaggle/input/london-police-records/london-street.csv')
df.count().compute()
df.head()
df['COUNTER']=1
df['YEAR'] = df['Month'].apply(lambda x: str(x).split('-')[0],meta=('Month', 'object'))
df.columns
df.tail()
plt.figure(figsize=(12,6))

df.groupby('Month').count().compute()['COUNTER'].plot.line(marker='o')

plt.title("Total Number of Reports by Year and Month")
df['Month'].nunique().compute()
total_reports_crimes=df.groupby('Crime type').count().compute()['COUNTER'].sort_values(ascending=False)

total_reports_crimes = total_reports_crimes.reset_index()
total_reports_crimes['%Rep'] = total_reports_crimes['COUNTER']/np.sum(total_reports_crimes['COUNTER'])

total_reports_crimes['%Pareto'] = np.cumsum(total_reports_crimes['%Rep'])

total_reports_crimes
crimes_df=df.groupby(['Crime type','Month']).count().compute()['COUNTER'].reset_index()
crimes_df.head()
chart = sns.catplot(x="Month",y='COUNTER',col="Crime type",col_wrap=4,kind='bar',data=crimes_df,palette='winter')

chart.set_xticklabels(rotation=90)
df = df.categorize('Crime type')

c_matrix=df.pivot_table(index='Month',columns='Crime type',values='COUNTER',aggfunc='sum').compute()

c_matrix.head()
crime_vso_locations=df[df['Crime type']=="Violence and sexual offences"].groupby(['Latitude','Longitude']).sum().compute()['COUNTER'].reset_index()

crime_vso_locations.sort_values(by='COUNTER',ascending=False,inplace=True)

crime_vso_locations=crime_vso_locations.head(25)
import plotly.express as px

fig=px.scatter_mapbox(crime_vso_locations, lat="Latitude", lon="Longitude",color="COUNTER",size="COUNTER",

                      title="Top 25 locations with the highest volume of reports related to Violence and Sexual Offences",

                      color_continuous_scale=px.colors.sequential.thermal)

fig.update_layout(mapbox_style="carto-positron")

fig.show()

#another mapbox styles available:

#open-street-map

#carto-darkmatter
crime_asb_locations=df[df['Crime type']=="Anti-social behaviour"].groupby(['Latitude','Longitude']).sum().compute()['COUNTER'].reset_index()

crime_asb_locations.sort_values(by='COUNTER',ascending=False,inplace=True)

crime_asb_locations=crime_asb_locations.head(25)
import plotly.express as px

fig=px.scatter_mapbox(crime_asb_locations, lat="Latitude", lon="Longitude",color="COUNTER",size="COUNTER",

                      title="Top 25 locations with the highest volume of reports related to Antisocial Behaviour",

                      color_continuous_scale=px.colors.sequential.Plasma)

fig.update_layout(mapbox_style="carto-positron")

fig.show()