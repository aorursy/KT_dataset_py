# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



import plotly.express as px

        

# Any results you write to the current directory are saved as output.
# loading data with pandas read_csv

df = pd.read_csv('/kaggle/input/community-mobility-data-for-covid19/community_mobility_change.csv')
# size checking 

df.shape

# there are 768352 rows and 6 columns
#check snippet

df.head()
# check data type df.info()
# change format date to date and time 

df.date = pd.to_datetime(df.date)
df.info()
df.head(100)
# filter location for Indonesia. There are 300 data points, from 16 February 2020 until 5 April 2020

df[df.location=='Indonesia'].head(300)
df_indo = df[df.location=='Indonesia']

df_indo.groupby(['mobility_type']).count()
fig = px.line(df_indo, x="date", y="mobility_change", color='mobility_type')

fig.update_layout(

    title="Starting from 15 March 2020, More Indonesians are staying home.",

    xaxis_title="Date",

    yaxis_title="Mobility Changes",

    font=dict(

        family="Lato, monospace",

        size=14,

        color="#7f7f7f"

    )

)

fig.add_shape(

        # Line Vertical

        dict(

            type="line",

            x0='2020-03-15',

            y0=-0.6,

            x1='2020-03-15',

            y1=0.3,

            line=dict(

                color="RoyalBlue",

                width=3,

                dash="dashdot"

            )

))

fig.show()

df_indokor = df[(df.location=='Indonesia')|(df.location=='South Korea')]

df_indokor
df[df.loc_type=='parent'].groupby(['location']).count()
fig = px.line(df_indokor[df_indokor.mobility_type=='Residential'], 

              x="date", y="mobility_change", color='location')

fig.update_layout(

    title="Indonesian vs South Korea changes to stay at home is +15% vs +7% Respectively",

    xaxis_title="Date",

    yaxis_title="Mobility Changes",

    font=dict(

        family="Lato, monospace",

        size=14,

        color="#7f7f7f"

    )

)

fig.add_shape(

        # Line Vertical

        dict(

            type="line",

            x0='2020-03-15',

            y0=-0.6,

            x1='2020-03-15',

            y1=0.3,

            line=dict(

                color="RoyalBlue",

                width=3,

                dash="dashdot"

            )

))

fig.show()