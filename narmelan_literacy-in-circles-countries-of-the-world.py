import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

print("Good to go")
#Loading the data ready for examination



file_pathway = "../input/countries-of-the-world/countries of the world.csv"

data = pd.read_csv(file_pathway, decimal=',')

data.columns = (["Country","Region","Population","Area","Density","Coastline","Migration","IMR","GDP","Literacy","Phones",

                 "Arable","Crops","Other","Climate","Birthrate","Deathrate","Agriculture","Industry","Service"])



data.Country = data.Country.astype('category')

data.Region = data.Region.astype('category')

data.Population = data.Population.astype('int')

data.IMR = data.IMR.astype(float)

data.Literacy = data.Literacy.astype(float)

data.Phones = data.Phones.astype(float)

data.Birthrate = data.Birthrate.astype(float)

data.Deathrate = data.Deathrate.astype(float)

# Selecting our attributes of interest



subset=data[['Country','Region','Population','IMR','GDP','Literacy','Phones','Birthrate','Deathrate']]
# All missing values for our attributes of interest



subset.isna().sum()
# All tuples in our database with missing values for literacy



unknown_literacy=pd.isnull(subset["Literacy"])

subset[unknown_literacy]
plt.figure(figsize=(10,5))

sns.kdeplot(data=subset['Literacy'], shade=True)
low_lit=subset.sort_values(by='Literacy', ascending=True)

low_lit.head(10)
high_lit=subset.sort_values(by='Literacy', ascending=False)

high_lit.head(10)
fig = px.scatter(subset, x="Literacy", y="IMR", size="Population", color="Region",

           hover_name="Country", log_x=True, log_y=True, size_max=100, width=950, height=600)

fig.update_layout(showlegend=False,

    xaxis={'title':'Literacy Rate (%age)',},

    yaxis={'title':'Infant Moratality (per 1000 births)'})

fig.show()
subset.nlargest(5,'IMR') 

subset.nsmallest(5,'IMR') 
fig = px.scatter(subset, x="Literacy", y="Birthrate", size="Population", color="Region",

                 hover_name="Country", log_x=True, log_y=True, size_max=100, width=950, height=600)

fig.update_layout(showlegend=False, 

    xaxis={'title':'Literacy Rate (%age)',},

    yaxis={'title':'Birthrate (per 1000)'})

fig.show()
subset.nlargest(5,'Birthrate')
subset.nsmallest(5,'Birthrate')
fig = px.scatter(subset, x="Literacy", y="Deathrate", size="Population", color="Region",

                 hover_name="Country", log_x=True, log_y=True, size_max=100, width=950, height=600)

fig.update_layout(showlegend=False,

    xaxis={'title':'Literacy Rate (%age)',},

    yaxis={'title':'Deathrate (per 1000) '})

fig.show()
subset.nlargest(5,'Deathrate')
subset.nsmallest(5,'Deathrate')
fig = px.scatter(subset, x="Literacy", y="Phones", size="Population", color="Region",

                 hover_name="Country", log_x=True, log_y=True, size_max=100, width=950, height=600)

fig.update_layout(showlegend=False,

    xaxis={'title':'Literacy Rate (%age)',},

    yaxis={'title':'Phones (per 1000)'})

fig.show()
subset.nlargest(5,'Phones')
subset.nsmallest(5,'Phones')
fig = px.scatter(subset, x="Literacy", y="GDP", size="Population", color="Region",

           hover_name="Country", log_x=True,log_y=True, size_max=100, width=950, height=600)

fig.update_layout(showlegend=False,

    xaxis={

        'title':'Literacy Rate (%age)',},

    yaxis={'title':'GDP per capita'})



fig.show()
subset.nlargest(5,'GDP')
subset.nsmallest(5,'GDP')