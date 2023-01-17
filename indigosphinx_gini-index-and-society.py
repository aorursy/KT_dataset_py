import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px
#Here I am reading the csv files into a pandas dataframe

data = pd.read_csv("../input/gini-index/Gini_index.csv")

happiness = pd.read_csv("../input/gini-index/happiness_index.csv")
#I am still trying to find a way to incorperate this data into the project,  



read_file = pd.read_excel("../input/homicide-total-count/homicide_total_rate_and_count.xls")

read_file.to_csv ("homicide_rate.csv",  

                  index = None, 

                  header=True)

homicide_rate = pd.read_csv("homicide_rate.csv")

homicide_rate.head()

homicide_rate = homicide_rate.loc[homicide_rate.groupby('Territory').Year.idxmax()]

homicide_rate.head()

homicide = homicide_rate[["Value", "Territory", "Year"]]

homicide.columns = ['Murders', 'Country', 'Year']

homicide.head()
grouped_gini = data.groupby(by='Country Name')
data.head()
data.columns = ['Country', 'Country Code', 'Year', 'Value']
data['Country'].unique()
Australia = data.loc[data['Country'] == "Australia"]

US = data.loc[data['Country'] == "United States"]
South_Africa = data.loc[data['Country'] == "South Africa"]
plt.plot(Australia['Year'], Australia['Value'])
plt.plot(South_Africa['Year'], South_Africa['Value'])
plt.plot(US['Year'], US['Value'])
lastest_gini = data.loc[data.groupby('Country').Year.idxmax()]
lastest_gini.head()
sorted_gini = lastest_gini.sort_values("Value", ascending=False)
sorted_gini.head(60)
most_equality = lastest_gini.sort_values("Value", ascending=True)

most_equality.head(60)
sorted_gini.columns = ['Country', 'Country Code', 'Year', 'Gini Coefficient']
happiness.head()
happiness = happiness[happiness.columns[0:3]]
happiness.columns = ['Happiness Rank', 'Country', 'Happiness Score']
merged = pd.merge(sorted_gini, happiness, on='Country')
merged.head()
fig = px.scatter(merged, x="Gini Coefficient", y="Happiness Score", text="Country", log_x=True, size_max=25, color="Gini Coefficient")

fig.update_traces(textposition='top center')

fig.update_layout(title_text='Gini Coefficient vs Happiness Score', title_x=0.5)

fig.show()
merged2 = merged[merged.Country != 'Azerbaijan']
fig = px.scatter(merged2, x="Gini Coefficient", y="Happiness Score", text="Country", log_x=True, size_max=25, color="Gini Coefficient")

fig.update_traces(textposition='top center')

fig.update_layout(title_text='Income Inequality vs Happiness Score', title_x=0.5)

fig.show()
sns.regplot(x="Gini Coefficient", y="Happiness Score", data=merged2)