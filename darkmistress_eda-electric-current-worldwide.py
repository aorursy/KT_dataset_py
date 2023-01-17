import pandas as pd

import numpy as np

import seaborn as sns
acelec = pd.read_csv("../input/electric-current-worldwide/Electric Current Worldwide.csv", encoding = "iso-8859-1")

acelec.head()
acelec_1 = acelec.drop(['Note(s)'], axis=1)

acelec_1
acelec_2 = acelec_1.drop([0], axis=0)

acelec_2
acelec['Frequency of current'].value_counts()
acelec_2['Frequency stability'] = acelec_2['Frequency stability'].replace('No','no')
acelec_2
for col in acelec_2.columns:

    pct_missing = np.mean(acelec_2[col].isnull())

    print('{} - {}%'.format(col, round(pct_missing*100)))
acelec_2['Type(s) of plug'] = acelec_2['Type(s) of plug'].fillna('C,E')

acelec_2['Frequency of current'] = acelec_2['Frequency of current'].fillna(50)

acelec_2['Type of current'] = acelec_2['Type of current'].replace('a.c','a.c.')
for col in acelec_2.columns:

    pct_missing = np.mean(acelec_2[col].isnull())

    print('{} - {}%'.format(col, round(pct_missing*100)))
acelec_2['Frequency of current'] = acelec_2['Frequency of current'].str.extract('(\d+)').astype('float')

acelec_2['Number of wires'] = acelec_2['Number of wires'].str.extract('(\d+)').astype('float')
# shape and data types of the data

print(acelec_2.info())
import pycountry

len(pycountry.countries)

list(pycountry.countries)[0]
import plotly.express as px

df = acelec_2

fig = px.area(df, x="Country", y="Frequency of current", color="Type of current",

      line_group="Country")

fig.show()
df = acelec_2

fig = px.histogram(df, x="Country", y="Nominal voltage", color="Number of phases", marginal="rug",

                   hover_data=df.columns)

fig.show()
df = acelec_2

fig = px.sunburst(df, path=['Country'], values='Number of wires',

                  color='Type(s) of plug', hover_data=['Country'],

                  color_continuous_scale='RdBu')

fig.show()
df = acelec_2

fig = px.bar(df, x="Frequency stability", y="Country", color="Frequency stability", title="Frequency stability in each Country")

fig.show()