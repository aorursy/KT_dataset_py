# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

from pandas import DataFrame

from matplotlib import pyplot as plt

from matplotlib import style

import seaborn as sns

import numpy as np

import plotly_express as px

import plotly.graph_objects as go

from datetime import datetime, timedelta

import pandas_profiling

import altair as alt



import warnings

warnings.filterwarnings('ignore')
data = pd.read_csv("/kaggle/input/us-border-crossing-data/Border_Crossing_Entry_Data.csv")

data.head()
print(data.info())
data['Date'] = pd.to_datetime(data['Date'])   # convert objecg to datetime
data.columns = data.columns.str.lower()  # change columns name to lower case to make it easy:
# reset dataframe: add year & day of the week for further analysis

data['year'] = data['date'].dt.strftime('%Y')

data['day_of_week'] = data['date'].dt.day_name()



data.head(3)
# The busiest border

busy_border = data.groupby("border").value.sum().reset_index()

busy_border.sort_values(["value"], axis=0, 

                 ascending=[False], inplace=True)  

busy_border
# Global treemap breakdown

fig = px.treemap(data, path=['border','state','measure'], values='value',

                  color='value', hover_data=['state'],

                  color_continuous_scale='RdBu')



fig.update_layout(title="US inbound crossing with Canada and Mexico: State level",

                  width=780, height=600, uniformtext=dict(minsize=10, mode='hide'))  



fig.show()
# The transportation way of inbound crossing for each US border

# visualization with Seaborn



border_tick=range(2)



fig2, ax2 = plt.subplots(figsize=(18,8))



ax2 = sns.barplot(x=data['border'], y=data['value'], hue=data['measure'], data=data, log=True)

ax2.set_xlabel('US borders', size=14, fontweight='bold')

ax2.set_ylabel('Inbound crossing value', size=14, fontweight='bold')

plt.legend(loc="upper left", frameon=False, ncol=3, fontsize=12)

plt.title("US inbound crossing analysis: Measure VS Border", size=17, color="indianred", fontweight='bold')



# bar chart styling

sns.despine(left=True)  # remove upper and left line



ax2.set_axisbelow(True)  

ax2.yaxis.grid(True, color='#EEEEEE')

ax2.xaxis.grid(False)



plt.show()
# Now, let's look at state level yearly transportation evolution



tran = data.groupby(['measure', 'state','year']).value.sum().reset_index()



fig = px.scatter(tran, x="year", y="value", color="measure", facet_col="state",

       facet_col_wrap=5)



fig.update_layout(title="US inbound crossing measure from Canada and Mexico border 1996-2020", 

                  width=780, height=600)



fig.update_xaxes(showgrid=False)



fig.show()
# Use pivot table for day of week analysis 



# regroup related data

week_table = data.groupby(['border', 'day_of_week']).value.count().reset_index()



# Create pivot table

pivot_week = pd.pivot_table(data, 'value', ['border','state','year'], 'day_of_week')



# Reorder column

pivot_week = pivot_week[['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday','Saturday', 'Sunday']]

pivot_week.head()
sns.set()

pd.pivot_table(data, 'value', ['state'], 'day_of_week').plot(kind= 'bar', width=0.9, alpha=0.9, figsize=(22, 10), title="US inbound crossing weekday level", fontsize=15)

plt.ylabel("crossing number", fontsize=18, color='indianred', fontweight='bold')

plt.xlabel("US states", fontsize=18, color='indianred', fontweight='bold')

plt.title('US inbound crossing weekday level', fontsize=25, color='indianred', fontweight='bold')

plt.legend(fancybox=True, shadow=True, fontsize=16)
# create new column which will return "illegal" if matches certain condition for measure.

data['law'] = data.measure.apply(lambda x: 'illegal' if x == 'Truck Containers Empty' or x == 'Truck Containers Full' or x =='Rail Containers Empty' or x == 'Rail Containers Full' else 'legal')



# Then, extract all categorical data: illegal crossing

illegal_data = data[data.law == 'illegal']

illegal_data2 = illegal_data.groupby(['year', 'border']).value.sum().reset_index()

illegal_data2.head()
fig = px.area(illegal_data2, x="year", y="value", color="border")



fig.update_layout(title="US illegal inbound crossing from Canada and Mexico border 1996-2020", 

                  width=780, height=500)



fig.update_xaxes(showgrid=False)



fig.show()
# legal VS illegal crossing analysis: state level



measure_ill = data[data.law == 'illegal']

measure_ill = measure_ill.groupby(['border', 'measure', 'state', 'year']).value.sum().reset_index()

measure_ill.head()



# set pivot table: columns should be illega crossing measure

pivot_illegal = pd.pivot_table(measure_ill, 'value', ['state'], 'measure')

pivot_illegal.head(5)
# plot pivot table outcome with Seaborn



sns.set()

pd.pivot_table(measure_ill, 'value', ['state'], 'measure').plot(kind= 'bar', width=0.88, figsize=(22, 10), title="US inbound crossing weekday level", fontsize=15)

plt.ylabel("crossing number", fontsize=18, color='indianred', fontweight='bold')

plt.xlabel("US states", fontsize=18, color='indianred', fontweight='bold')

plt.title('US inbound crossing illegal transportation', fontsize=25, color='indianred', fontweight='bold')

plt.legend(fancybox=True, shadow=True, fontsize=16)
# Extract legal dataset

all_legal = data[data.law == 'legal']



# Regroup and rename measure elements in dataset

all_legal.loc[(all_legal['measure'] == 'Bus Passengers') | (all_legal['measure'] == 'Train Passengers') | (all_legal['measure'] == 'Buses') | (all_legal['measure'] == 'Trains'), 'measure'] = 'public'

all_legal.loc[(all_legal['measure'] == 'Personal Vehicle Passengers') | (all_legal['measure'] == 'Personal Vehicles') | (all_legal['measure'] == 'Pedestrians') | (all_legal['measure'] == 'Trucks'), 'measure'] = 'private'

all_legal.head()
# create final sorted table: pu_pri

pu_pri = all_legal[['year', 'state', 'measure', 'value']]



# create pivot table show private & public

pivot_pupri = pd.pivot_table(pu_pri, 'value', ['state','year'], 'measure')



pivot_pupri = pivot_pupri.reset_index()   # reconvert pivot table back to dataframe



# Calculate percentage of public 

pivot_pupri['per_public'] = pivot_pupri.apply(lambda row: row['public']/(row['private']+row['public']), axis=1)

pivot_pupri.head()
# Now, visualize public measure percentage



fig = px.scatter(pivot_pupri, x="year", y="per_public", facet_col="state",

       facet_col_wrap=5)



fig.update_layout(title="public transportation percentage from 1966 to 2020", 

                  width=780, height=600)



fig.layout.yaxis2.update(matches=None)  # matches=none, reset y value

fig.update_xaxes(showgrid=False)

fig.show()
# Explore Alaska data

alaska = data.groupby(['year', 'measure', 'state']).value.sum().reset_index()

alaska.head()



alaska = alaska[alaska.state == 'AK'].reset_index()

alaska.head()



ooo = alaska[(alaska.year == '2017') | (alaska.year == '2018') | (alaska.year == '2019')]

nnn = ooo[(ooo.measure == 'Bus Passengers') | (ooo.measure == 'Train Passengers') | (ooo.measure == 'Buses') | (ooo.measure == 'Trains')]  

nnn.head(12)
# Create pivot table



AK = pd.pivot_table(nnn, 'value', ['year', 'state'], 'measure').reset_index()

AK.head()