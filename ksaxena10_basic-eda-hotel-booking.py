# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from plotly import tools

import chart_studio.plotly as py

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.figure_factory as ff

from IPython.display import HTML, Image



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
hotel_df = pd.read_csv('/kaggle/input/hotel-booking-demand/hotel_bookings.csv')
df = hotel_df.head(10)

df = df.loc[:, ['hotel','lead_time', "arrival_date_year", 'children', 'babies', 'meal', 'country']]

table = ff.create_table(df)



iplot(table, filename='pandas_table')
hotel_df.info()
df.describe().T
def summary(df):

    

    types = df.dtypes

    counts = df.apply(lambda x: x.count())

    uniques = df.apply(lambda x: [x.unique()])

    nas = df.apply(lambda x: x.isnull().sum())

    distincts = df.apply(lambda x: x.unique().shape[0])

    missing = (df.isnull().sum() / df.shape[0]) * 100

    sk = df.skew()

    krt = df.kurt()

    

    print('Data shape:', df.shape)



    cols = ['Type', 'Total count', 'Null Values', 'Distinct Values', 'Missing Ratio', 'Skewness', 'Kurtosis']

    dtls = pd.concat([types, counts, nas, distincts, missing, sk, krt], axis=1, sort=False)

  

    dtls.columns = cols

    return dtls
details = summary(hotel_df)

table = ff.create_table(details)

iplot(table, filename='pandas_table')
from IPython.display import Image

Image('/kaggle/input/symmimg/651px-Relationship_between_mean_and_median_under_different_skewness.png')
hotel_df.columns
fig = ff.create_distplot([hotel_df.lead_time],['lead_time'],bin_size=10)

iplot(fig, filename='Lead Time Distplot')
hotel_df.head()
trace = go.Histogram(x=hotel_df['arrival_date_month'], marker=dict(color='rgb(0, 0, 100)'))



layout = go.Layout(

    title="Month wise count of bookings"

)



fig = go.Figure(data=go.Data([trace]), layout=layout)

iplot(fig, filename='histogram-freq-counts')
# get number of acutal guests by country

country_data = pd.DataFrame(hotel_df.loc[hotel_df["is_canceled"] == 0]["country"].value_counts())

country_data.index.name = "country"

country_data.rename(columns={"country": "Number of Guests"}, inplace=True)

total_guests = country_data["Number of Guests"].sum()

country_data["Guests in %"] = round(country_data["Number of Guests"] / total_guests * 100, 2)

table = ff.create_table(country_data.head())

iplot(table, filename='pandas_table')
# show on map

import plotly.express as px

guest_map = px.choropleth(country_data,

                    locations=country_data.index,

                    color=country_data["Guests in %"], 

                    hover_name=country_data.index, 

                    color_continuous_scale=px.colors.sequential.Plasma,

                    title="Home country of guests")

guest_map.show()
import plotly.express as px

fig = px.scatter(hotel_df, x="arrival_date_month", y="lead_time", animation_frame="arrival_date_year", animation_group="reserved_room_type",

           size="adults", color="reserved_room_type", hover_name="reserved_room_type", facet_col="customer_type")

fig.show()