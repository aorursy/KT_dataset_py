#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



import warnings
warnings.filterwarnings("ignore")
#loading data
path = "../input/covid19-global-forecasting-week-3/"
path2 = "../input/stringency-data/"
data = pd.read_csv(path+"train.csv")
data.head()
#stringency index
stringency = pd.read_excel(path2+"stringencyindex_hxl.xlsx", sheet_name=None)

stringency = stringency["StringencyIndex"]
stringency.head()
country_code = pd.read_csv(path2+"country_code.csv")
country_code.head()
stringency_with_name = stringency.merge(country_code, left_on="country_code", right_on="alpha-3")
stringency_with_name.head()
np.unique(stringency_with_name.name)
stringency = stringency_with_name[["name", "date_value", "stringency_actual"]].rename({"name":"Country_Region", "date_value": "Date"})
stringency.head()
data = data.merge(stringency, left_on=["Country_Region", "Date"], right_on=["name", "date_value"], how="left")

data["isCountry"] = data.Province_State.isna()
data.Province_State[data.Province_State.isna()] = ""
data["Region"]= data.Country_Region 
data["Region"][~data.isCountry]= data.Country_Region + "-" + data.Province_State

#new confirmed column
data["Id_by_Region"] = data[["Region", "Id"]].groupby("Region").cumcount()
data["PreviousConfirmed"] = data["ConfirmedCases"].shift()
data.loc[ data.Id_by_Region==0, "PreviousConfirmed"] = 0
data["NewConfirmed"]= data.ConfirmedCases - data.PreviousConfirmed

#new fatalities column
data["Id_by_Region"] = data[["Region", "Id"]].groupby("Region").cumcount()
data["PreviousFatalities"] = data["Fatalities"].shift()
data.loc[ data.Id_by_Region==0,"PreviousFatalities"] = 0
data["NewFatalities"]= data.Fatalities - data.PreviousFatalities

days = np.unique(data.Date)
n_days = len(days)
data
selected_countries = ["Spain", "Italy", "Germany", "Singapore", 'Korea, South']
c = ["red", "blue", "green", "yellow", "purple"]

fig, ax = plt.subplots(1,2, figsize=(15,5))

for i, country in enumerate(selected_countries):
    
    confirmed_country = data[data.Region==country].NewConfirmed
    stringency_country = data[data.Region==country].stringency_actual
    ax[0].scatter(stringency_country, confirmed_country, c=c[i])
    ax[0].set_xticks(np.arange(0, n_days,20 ))
    
ax[0].legend(selected_countries)
ax[0].set_title("Confirmed cases")
ax[0].grid()

for i, country in enumerate(selected_countries):
    
    fatalities_country = data[data.Region==country].NewFatalities
    stringency_country = data[data.Region==country].stringency_actual
    ax[1].scatter(stringency_country, fatalities_country, c=c[i])
    ax[1].set_xticks(np.arange(0, n_days,20 ))
    
ax[1].legend(selected_countries)
ax[1].set_title("Fatalities")
ax[1].grid()
selected_countries = ["Spain", "Italy", "Germany"]
c = ["red", "blue", "green", "yellow", "purple"]
legend = []
fig, ax = plt.subplots(1,2, figsize=(15,5))

for i, country in enumerate(selected_countries):
    
    confirmed_country = data[data.Region==country].NewConfirmed
    stringency_country = data[data.Region==country].stringency_actual*100
    ax[0].plot(days, confirmed_country, c=c[i])
    ax[0].plot(days, stringency_country,  "--", c=c[i])
    ax[0].set_xticks(np.arange(0, n_days,20 ))
    legend.append(country)
    legend.append(country+"-Stringeny")


ax[0].legend(selected_countries)
ax[0].set_title("Confirmed cases")
ax[0].grid()

for i, country in enumerate(selected_countries):
    
    fatalities_country = data[data.Region==country].NewFatalities
    stringency_country = data[data.Region==country].stringency_actual*10
    ax[1].plot(days, fatalities_country, c=c[i])
    ax[1].plot(days, stringency_country,  "--", c=c[i])
    ax[1].set_xticks(np.arange(0, n_days,20 ))
    legend.append(country)
    legend.append(country+"-Stringeny")
    
ax[1].legend(legend)
ax[1].set_title("Fatalities")
ax[1].grid()
selected_countries = ["Singapore"]
c = ["red", "blue"]
legend = []
fig, ax = plt.subplots(1,2, figsize=(15,5))

for i, country in enumerate(selected_countries):
    
    confirmed_country = data[data.Region==country].NewConfirmed
    stringency_country = data[data.Region==country].stringency_actual
    ax[0].plot(days, confirmed_country, c=c[i])
    ax[0].plot(days, stringency_country,  "--", c=c[i])
    ax[0].set_xticks(np.arange(0, n_days,20 ))
    legend.append(country)
    legend.append(country+"-Stringeny")
    
ax[0].legend(legend)
ax[0].set_title("Confirmed cases")
ax[0].grid()

for i, country in enumerate(selected_countries):
    
    fatalities_country = data[data.Region==country].NewFatalities
    stringency_country = data[data.Region==country].stringency_actual
    ax[1].plot(days, fatalities_country, c=c[i])
    ax[1].plot(days, stringency_country,  "--", c=c[i])
    ax[1].set_xticks(np.arange(0, n_days,20 ))
    legend.append(country)
    legend.append(country+"-Stringeny")
    
ax[1].legend(legend)
ax[1].set_title("Fatalities")
ax[1].grid()
selected_countries = ["Spain", "Italy", "Germany", "Singapore"]
c = ["red", "blue", "green", "yellow", "purple"]
legend = []
fig = plt.figure()

for i, country in enumerate(selected_countries):
    
    confirmed_country = data[data.Region==country].NewConfirmed
    stringency_country = data[data.Region==country].stringency_actual*100
    plt.plot(days, stringency_country,  "--", c=c[i])
    plt.xticks(np.arange(0, n_days,20 ))
    legend.append(country+"-Stringeny")
plt.grid()
plt.legend(legend)
