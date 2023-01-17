import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns



import plotly.graph_objects as go

import plotly.express as px



import pycountry
space_missions = pd.read_csv("/kaggle/input/all-space-missions-from-1957/Space_Corrected.csv")

df = space_missions.copy()
df.head()
df.info()
values = [['Company Name', 'Location', 'Datum', 'Detail', 'Status Rocket', 'Rocket', 'Status Mission'], #1st col

  ["Name of the Company", "Place where the Mission Started","Date of the Mission Started",

  "Information About Rocket", "Information About Status of Rocket. It is Active or Retired",

  "Cost of the Mission", "Information About Status of Mission. It wasa Success or Not"]]





fig = go.Figure(data=[go.Table(

  columnwidth = [80,400],

  header = dict(

    values = [['<b>COLUMN NAME</b>'],

                  ['<b>DESCRIPTION</b>']],

    line_color='darkslategray',

    fill_color='royalblue',

    align=['left','center'],

    font=dict(color='white', size=12),

    height=40

  ),

  cells=dict(

    values=values,

    line_color='darkslategray',

    fill=dict(color=['paleturquoise', 'white']),

    align=['left', 'center'],

    font_size=12,

    height=30)

    )

])

fig.show()
df = df.drop(["Unnamed: 0", "Unnamed: 0.1"], axis = 1)
df.rename(columns={" Rocket": "Rocket"}, inplace = True)
df['Rocket'] = df['Rocket'].str.replace(',', '')

df['Rocket'] = df['Rocket'].astype(np.float32)

df['Rocket'] = df['Rocket'] * 1000000



df['Year'] = pd.to_datetime(df['Datum']).apply(lambda x: x.year)

df['Month'] = pd.to_datetime(df['Datum']).apply(lambda x: x.month)
df["country"] = df["Location"].apply(lambda x: x.strip().split(", ")[-1])
countries_list = list()

frequency_list = list()

test = df.groupby("country")["Company Name"].unique()

for i in test.iteritems():

    countries_list.append(i[0])

    frequency_list.append(len(i[1]))

    

companies = pd.DataFrame(list(zip(countries_list, frequency_list)), columns =['Country', 'Company Number'])

companies = companies.sort_values("Company Number", ascending=False)
fig = go.Figure(data=[go.Table(

    header=dict(values=list(companies.columns),

                fill_color='paleturquoise',

                align='left'),

    cells=dict(values=[companies["Country"].head(7), companies["Company Number"].head(7)],

               fill_color='lavender',

               align='left'))

])



fig.update_layout(title="Countries List which Have More Than One Corp.")

fig.show()
companies["IsoAlpha3"] = companies.Country[:7].apply(lambda x: pycountry.countries.search_fuzzy(x)[0].alpha_3)



fig = px.scatter_geo(companies, locations="IsoAlpha3", size="Company Number")

fig.show()
df = df[(df["country"] == "USA") | (df["country"] == "Russia") | (df["country"] == "China")]

df.head()
test = pd.DataFrame(df.groupby(["country","Company Name"])["Location"].count())

test.rename(columns={"Location":"Mission Number"}, inplace=True)
test = test.reset_index(level=[0,1])

fig = px.bar(test, x="Mission Number", y="country",

             color='Company Name', text="Company Name")

fig.update_layout(

    title='Mission Numbers by Countries and Corp Names',

    yaxis=dict(

        title='Countries',

        titlefont_size=16,

        tickfont_size=14,

    ),

)

fig.show()
a = pd.DataFrame(df.groupby(["country","Company Name","Status Mission"]).Location.count())

a = a.reset_index(level=[0,1,2])



fig = px.sunburst(a, path=["country", 'Company Name', 'Status Mission'], values='Location',color="Company Name")

fig.show()
b = pd.DataFrame(df.groupby(["country", "Company Name", "Status Rocket"])["Location"].count())

b = b.reset_index(level=[0,1,2])

b.rename(columns={"Location":"Numbers"}, inplace=True)

b = b[b["Status Rocket"] == "StatusActive"]

fig = px.bar(b, x="country", y="Numbers", color = "Company Name", title="Active Space Missions Number")

fig.show()
df_2 = df.dropna() # I've said in first-looking to data to remove null values. While we using Rocket column.
test = pd.DataFrame(df_2.groupby(["country", "Company Name"])["Rocket"].sum())

test = test.reset_index(level=[0,1])



fig = px.bar(test, x='country', y='Rocket', color ='Company Name')

fig.show()
test2 = pd.DataFrame(df_2.groupby(["country", "Company Name"])["Location"].count())

test2 = test2.reset_index(level=[0,1])



test["Mission Number"] = test2["Location"]

test["Amount for Each Mission"] = test["Rocket"] / test["Mission Number"]



fig = px.bar(test, x='country', y='Amount for Each Mission', color ='Company Name')

fig.show()
test = pd.DataFrame(df.groupby(["country","Location"])["Location"].count())

test.rename(columns={"Location": "Mission Number"}, inplace = True)

test = test.reset_index(level=[0,1])

test = test.sort_values("Mission Number", ascending = False)

fig = px.bar(test, x='Mission Number', y='Location', color ='country')

fig.show()
test = pd.DataFrame(df.groupby(["country", "Month"])["Location"].count())

test = test.reset_index(level=[0,1])



fig = px.bar(test, x='Month', y='Location', color ='country')

fig.show()
test = pd.DataFrame(df.groupby(["country", "Year"])["Location"].count())

test = test.reset_index(level=[0,1])



fig = px.bar(test, x='Year', y='Location', color ='country')

fig.show()