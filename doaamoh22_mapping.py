# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pycountry

import plotly.express as px



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df= pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/2019_nCoV_data.csv')

df.drop('Sno',axis=1, inplace=True)
df.head()
df.info()
countries = {}

for country in pycountry.countries:

    countries[country.name] = country.alpha_3

df["iso_alpha"] = df["Country"].map(countries.get)



data = df.groupby("iso_alpha").sum().reset_index()

fig_Confirmed = px.choropleth(data, locations="iso_alpha",

                    color='Confirmed',

                     hover_name="iso_alpha",

                    color_continuous_scale=px.colors.sequential.Plasma)

fig_Confirmed.show()

fig_Deaths = px.choropleth(data, locations="iso_alpha",

                    color='Deaths',

                     hover_name="iso_alpha",

                    color_continuous_scale=px.colors.sequential.Plasma)

fig_Deaths.show()

fig_Recovered = px.scatter_geo(data, locations="iso_alpha",color="Recovered",

                     hover_name="iso_alpha",size="Recovered"

                     )

fig_Recovered.show()