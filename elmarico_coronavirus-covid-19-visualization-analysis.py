# Importing libraries



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))





df = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv")



print(df.head())



# Quick overview of data



df.describe()

df.info()

df.isnull().sum()

df.head()
df.rename(columns={"SNo" : "serial_number",

           "ObservationDate" : "observation_date",

           "Province/State" : "state",

           "Country/Region" : "country",

           "Last Update" : "last_update",

           "Confirmed" : "confirmed",

           "Deaths" : "deaths",

           "Recovered" : "recovered"},

          inplace=True)
# Grouping confirmed, recovered and death cases per country

grouped_country = df.groupby(["country"] ,as_index=False)["confirmed","recovered","deaths"].last().sort_values(by="confirmed",ascending=False)



# Using just first 10 countries with most cases

most_common_countries = grouped_country.head(10)
# FUNCTION TO SHOW ACTUAL VALUES ON BARPLOT



def show_valushowes_on_bars(axs):

    def _show_on_single_plot(ax):

        for p in ax.patches:

            _x = p.get_x() + p.get_width() / 2

            _y = p.get_y() + p.get_height()

            value = '{:.2f}'.format(p.get_height())

            ax.text(_x, _y, value, ha="center")



    if isinstance(axs, np.ndarray):

        for idx, ax in np.ndenumerate(axs):

            _show_on_single_plot(ax)

    else:

        _show_on_single_plot(axs)

# Barplot: confirmed,recovered and death cases per country



plt.figure(figsize=(13,23))



# AXIS 1

plt.subplot(311)

vis_1 = sns.barplot(x="country",y="confirmed",data=most_common_countries)

vis_1.set_title("Confirmed cases")

show_valushowes_on_bars(vis_1)



# AXIS 2

plt.subplot(312)

vis_2 = sns.barplot(x="country",y="recovered",data=most_common_countries,palette="bright")

vis_2.set_title("Recovered cases")

show_valushowes_on_bars(vis_2)



# AXIS 3

plt.subplot(313)

vis_3 = sns.barplot(x="country",y="deaths",data=most_common_countries,palette="dark")

vis_3.set_title("Death cases")

show_valushowes_on_bars(vis_3)

plt.show()

fig_1 = px.bar(most_common_countries,x="country", y="confirmed",color="deaths",text="recovered",title="Countries with most cases.")

fig_1.show()
# Function returns interactive lineplot of confirmed cases on specific country





def getGrowthPerCountryInteractive(countryName):

    country = df[df["country"] == countryName]



    fig_1 = px.line(country, x="observation_date", y="confirmed", title=(countryName + " confirmed cases."))

    fig_2 = px.line(country, x="observation_date", y="deaths", title=(countryName +" death cases"))

    fig_3 = px.line(country, x="observation_date", y="recovered", title=(countryName + " recovered cases"))



    fig_1.show()

    fig_2.show()

    fig_3.show()
getGrowthPerCountryInteractive("Italy")
china = df[df["country"] == "Mainland China"]

fig_2 = px.bar(china,x="state",y="confirmed",color="recovered",text="deaths")

fig_2.show()
hubei = china[china["state"] == "Hubei"]



fig_3 = px.line(hubei,x="observation_date",y="confirmed",title="Hubei, China. Confirmed cases.")

fig_3.show()



fig_4 = px.bar(hubei, x="observation_date",y="confirmed",color="recovered",text="deaths", title="Hubei, China.")

fig_4.show()
china_states = china[china["state"] != "Hubei"]



fig_5 = px.bar(china_states,x="state",y="confirmed", title="Confirmed cases in other states of China.")

fig_5.show()
fig_6 = px.bar(china_states, x="state",y="recovered",color="deaths",title="Recovered vs Deaths in other states of China.")

fig_6.show()