import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

%matplotlib inline

plt.style.use("ggplot")
path = '/kaggle/input/ntt-data-global-ai-challenge-06-2020/'

data=pd.read_csv(path+"COVID-19_and_Price_dataset.csv")

countries=list()

for i in range(len(data.columns)):

    if len(data.columns[i].split('_total_c'))>=2:

        countries.append(data.columns[i].split('_')[0])

#data[countries[1]+'_total_cases']   

len(np.unique(countries))==len(countries)
countries[1:20]
active_data=pd.read_csv('https://datahub.io/core/covid-19/r/countries-aggregated.csv')

active_data['Active']=active_data['Confirmed']-active_data['Recovered']-active_data['Deaths']

active_data.tail()
plt.plot(active_data.groupby("Date")["Active"].sum().values)

plt.xlabel("Days")

plt.ylabel("Active Cases in The World")

plt.show()
sample_countries=["US","Brazil","China","France","Iran","Japan","Italy","Russia","Spain","Turkey","United Arab Emirates","United Kingdom","India","Germany"]

fig, axs = plt.subplots(len(sample_countries),figsize=(15,80))

for i in range(len(sample_countries)):

    axs[i].plot(active_data[active_data['Country']==sample_countries[i]]['Active'].values)

    axs[i].set_title(sample_countries[i])

    axs[i].set_xlabel("Days")

    axs[i].set_ylabel("Active Cases")
active_df=active_data.pivot(index='Date', columns='Country', values='Active')

active_df.tail(10)
all_data=pd.merge(data,active_df,how="left", on='Date')

all_data=all_data.fillna(0)

all_data.tail()