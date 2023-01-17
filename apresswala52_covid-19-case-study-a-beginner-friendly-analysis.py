!pip install pycountry_convert 

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import folium

import pycountry_convert as pc

import warnings

from datetime import datetime, timedelta, date

warnings.filterwarnings('ignore')



%matplotlib inline
# importing covid19 confirmed cases datatset

df_confirmed = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')



# renaming some columns

df_confirmed = df_confirmed.rename(columns={"Country/Region": "Country", "Province/State": "State"})

df_confirmed.head()
# importing covid19 deaths datatset

df_deaths = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')



# renaming some columns

df_deaths = df_deaths.rename(columns={"Country/Region": "Country", "Province/State": "State"})

df_deaths.head()
# importing covid19 datatset

df_covid19 = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/web-data/data/cases_country.csv")



# dropping columns that are not required

df_covid19.drop(['Last_Update', 'People_Tested', 'People_Hospitalized', 'UID', 'ISO3'], inplace=True, axis=1)



# changing column name

df_covid19 = df_covid19.rename(columns={"Country_Region": "Country"})

df_covid19.head()
# Breif info about the dataset

df_covid19.info()
# Changing the conuntry names as required by pycountry_convert library

df_confirmed.loc[df_confirmed['Country'] == "US", "Country"] = "USA"

df_deaths.loc[df_deaths['Country'] == "US", "Country"] = "USA"

df_covid19.loc[df_covid19['Country'] == "US", "Country"] = "USA"





df_confirmed.loc[df_confirmed['Country'] == 'Korea, South', "Country"] = 'South Korea'

df_deaths.loc[df_deaths['Country'] == 'Korea, South', "Country"] = 'South Korea'

df_covid19.loc[df_covid19['Country'] == "Korea, South", "Country"] = "South Korea"



df_confirmed.loc[df_confirmed['Country'] == 'Taiwan*', "Country"] = 'Taiwan'

df_deaths.loc[df_deaths['Country'] == 'Taiwan*', "Country"] = 'Taiwan'

df_covid19.loc[df_covid19['Country'] == "Taiwan*", "Country"] = "Taiwan"



df_confirmed.loc[df_confirmed['Country'] == 'Congo (Kinshasa)', "Country"] = 'Democratic Republic of the Congo'

df_deaths.loc[df_deaths['Country'] == 'Congo (Kinshasa)', "Country"] = 'Democratic Republic of the Congo'

df_covid19.loc[df_covid19['Country'] == "Congo (Kinshasa)", "Country"] = "Democratic Republic of the Congo"



df_confirmed.loc[df_confirmed['Country'] == "Cote d'Ivoire", "Country"] = "Côte d'Ivoire"

df_deaths.loc[df_deaths['Country'] == "Cote d'Ivoire", "Country"] = "Côte d'Ivoire"

df_covid19.loc[df_covid19['Country'] == "Cote d'Ivoire", "Country"] = "Côte d'Ivoire"



df_confirmed.loc[df_confirmed['Country'] == "Reunion", "Country"] = "Réunion"

df_deaths.loc[df_deaths['Country'] == "Reunion", "Country"] = "Réunion"

df_covid19.loc[df_covid19['Country'] == "Reunion", "Country"] = "Réunion"



df_confirmed.loc[df_confirmed['Country'] == 'Congo (Brazzaville)', "Country"] = 'Republic of the Congo'

df_deaths.loc[df_deaths['Country'] == 'Congo (Brazzaville)', "Country"] = 'Republic of the Congo'

df_covid19.loc[df_covid19['Country'] == "Congo (Brazzaville)", "Country"] = "Republic of the Congo"



df_confirmed.loc[df_confirmed['Country'] == 'Bahamas, The', "Country"] = 'Bahamas'

df_deaths.loc[df_deaths['Country'] == 'Bahamas, The', "Country"] = 'Bahamas'

df_covid19.loc[df_covid19['Country'] == "Bahamas, The", "Country"] = "Bahamas"



df_confirmed.loc[df_confirmed['Country'] == 'Gambia, The', "Country"] = 'Gambia'

df_deaths.loc[df_deaths['Country'] == 'Gambia, The', "Country"] = 'Gambia'

df_covid19.loc[df_covid19['Country'] == "Gambia, The", "Country"] = "Gambia"
continents = {

    'NA': 'North America',

    'SA': 'South America', 

    'AS': 'Asia',

    'OC': 'Australia',

    'AF': 'Africa',

    'EU' : 'Europe',

    'OTH' : 'Others'

}
# function to find the continent of the country supplied

def country_to_continent(country):

    try:

        continent_code = pc.country_alpha2_to_continent_code(pc.country_name_to_country_alpha2(country))

    except:

        continent_code = 'OTH'

    return continents[continent_code]
# extracting the countries columns from all the 3 datasets 

countries_covid19 = np.asarray(df_covid19["Country"])

countries_confirmed = np.asarray(df_confirmed["Country"])

countries_deaths = np.asarray(df_deaths["Country"])
# applying the above function to all the 3 datasets to find the continents of the respective countries

df_covid19.insert(1,"Continent",  [country_to_continent(country) for country in countries_covid19])

df_confirmed.insert(1,"Continent",  [country_to_continent(country) for country in countries_confirmed])

df_deaths.insert(1,"Continent",  [country_to_continent(country) for country in countries_deaths])
df_global = df_covid19.drop(['Country', 'Continent', 'Lat', 'Long_', 'Incident_Rate', 'Mortality_Rate'], axis=1)
df_global_cases = pd.DataFrame(pd.to_numeric(df_global.sum()), dtype=np.float64).transpose()

df_global_cases['Mortality_Rate'] = np.round((df_global_cases["Deaths"]/df_global_cases["Confirmed"])*100,2)

df_global_cases
labels =  [df_global_cases.columns[i]+ "\n" + str(int(df_global_cases.values[0][i])) for i in range(1,4)]

values = [df_global_cases.values[0][i] for i in range(1,4)]

plt.figure(figsize=(8,8))

plt.pie(values, labels=labels, autopct='%1.2f%%', pctdistance=0.85, labeldistance=1.1, textprops = {'fontsize':12})

my_circle = plt.Circle( (0,0), 0.7, color='white')

p = plt.gcf()

p.gca().add_artist(my_circle)

plt.text(0, 0, "Total\nConfirmed Cases \n"+str(int(df_global_cases.values[0][0])), horizontalalignment='center', verticalalignment='center', size=18)

plt.show()
confirmed_cases = df_confirmed.drop(['Lat', 'Long', 'State', 'Country', 'Continent'], axis=1)

cases = confirmed_cases.sum().tolist()

cases = np.asarray(cases)



death_cases = df_deaths.drop(['Lat', 'Long', 'State', 'Country', 'Continent'], axis=1)

deaths = death_cases.sum().tolist()

deaths = np.asarray(deaths)



dates = confirmed_cases.columns

d = [datetime.strptime(date,'%m/%d/%y').strftime("%d %b") for date in dates]
plt.figure(figsize=(8,8))

marker_style_confirmed = dict(c="darkcyan", linewidth=6, linestyle='-', marker='o', markersize=6, markerfacecolor='#ffffff')

marker_style_death = dict(c="crimson", linewidth=6, linestyle='-', marker='o', markersize=6, markerfacecolor='#ffffff')

plt.plot(d, cases, label = 'Confirmed', **marker_style_confirmed)

plt.plot(d, deaths, label = 'Deaths', **marker_style_death)

plt.fill_between(d, cases, color='darkcyan', alpha=0.3)

plt.fill_between(d, deaths, color='crimson', alpha=0.3)

plt.xlabel("Date", fontsize = 15)

plt.ylabel("No. of Cases",fontsize = 15)

plt.title("COVID Cases: WorldWide", fontsize = 18)

plt.legend(loc= "best", fontsize = 15)

plt.grid(alpha=0.8)

plt.xticks(list(np.arange(0,len(d),int(len(d)/5))))

plt.yticks(np.arange(0, max(cases), 10**(len(str(int(max(cases))))-1)))

plt.show()
daily_cases = np.nan_to_num(df_confirmed.sum()[5:].diff())

f = plt.figure(figsize=(15,10))

date = np.arange(0,len(daily_cases))

marker_style = dict(linewidth=2, linestyle='-', marker='o',markersize=5)

plt.plot(date, daily_cases/1000,"-.",color="blue",**marker_style)



# Grid Settings

plt.grid(lw = 1, ls = '-', c = "0.85", which = 'major')

plt.grid(lw = 1, ls = '-', c = "0.95", which = 'minor')



#Title

plt.title("COVID-19 Daily Confirmed Cases - Worldwide",{'fontsize':24})



# Axis Label

plt.xlabel("Days",fontsize =18)

plt.ylabel("Number of Daily New Cases (Thousand)",fontsize =18)



plt.show()
daily_deaths = np.nan_to_num(df_deaths.sum()[5:].diff())

f = plt.figure(figsize=(15,10))

date = np.arange(0,len(daily_deaths))

marker_style = dict(linewidth=2, linestyle='-', marker='o',markersize=5)

plt.plot(date, daily_deaths/1000,"-.",color="red",**marker_style)



# Grid Settings

plt.grid(lw = 1, ls = '-', c = "0.85", which = 'major')

plt.grid(lw = 1, ls = '-', c = "0.95", which = 'minor')



#Title

plt.title("COVID-19 Daily Deaths - Worldwide",{'fontsize':24})



# Axis Label

plt.xlabel("Days",fontsize =18)

plt.ylabel("Number of Daily Deaths (Thousand)",fontsize =18)



plt.show()
df_continents = df_covid19.drop(['Country', 'Lat', 'Long_', 'Incident_Rate', 'Mortality_Rate'], axis=1)
df_continents_cases = df_continents.groupby('Continent').sum()

df_continents_cases['Mortality_Rate'] = np.round((df_continents_cases["Deaths"]/df_continents_cases["Confirmed"])*100,2)

df_continents_cases.drop(['Others'], inplace=True)

df_continents_cases
labels = list(df_continents_cases.index)

sizes = df_continents_cases['Confirmed'].values

plt.figure(figsize=(8,8))

plt.pie(sizes, labels=labels, autopct='%1.2f%%', pctdistance=0.85, labeldistance=1.1, textprops = {'fontsize':10.5})

my_circle = plt.Circle( (0,0), 0.7, color='white')

p = plt.gcf()

p.gca().add_artist(my_circle)

plt.text(0, 0, "Continent wise \n Distribution of Cases", horizontalalignment='center', verticalalignment='center', size=18)

plt.show()
df_confirmed_continents = df_confirmed.groupby('Continent').sum()

df_confirmed_continents = df_confirmed_continents[df_confirmed_continents.index!='Others']

df_confirmed_continents.drop(['Lat', 'Long'], inplace=True, axis=1)



df_deaths_continents = df_deaths.groupby('Continent').sum()

df_deaths_continents = df_deaths_continents[df_deaths_continents.index!='Others']

df_deaths_continents.drop(['Lat', 'Long'], inplace=True, axis=1)



dates = df_confirmed_continents.columns

d = [datetime.strptime(date,'%m/%d/%y').strftime("%d %b") for date in dates]
marker_style_confirmed = dict(c="darkcyan", linewidth=6, linestyle='-', marker='o', markersize=6, markerfacecolor='#ffffff')

marker_style_death = dict(c="crimson", linewidth=6, linestyle='-', marker='o', markersize=6, markerfacecolor='#ffffff')

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15,15))

plt.subplots_adjust(top = 1.2, bottom = 0.1)

i=0

for rows in axes:

    for ax1 in rows:

        ax1.plot(d, df_confirmed_continents.iloc[i], label = 'Confirmed', **marker_style_confirmed)

        ax1.plot(d, df_deaths_continents.iloc[i], label = 'Deaths', **marker_style_death)

        ax1.fill_between(d, df_confirmed_continents.iloc[i], color='darkcyan', alpha=0.3)

        ax1.fill_between(d, df_deaths_continents.iloc[i], color='crimson', alpha=0.3)

        ax1.set_xlabel("Dates", fontsize = 12)

        ax1.set_ylabel("No. of Cases",fontsize = 12)

        ax1.set_title("COVID Cases: "+df_deaths_continents.index[i], fontsize = 15)

        ax1.legend(loc= "best", fontsize = 12)

        ax1.grid(which='major', linewidth = 0.3)

        ax1.set_xticks(list(np.arange(0,len(d),int(len(d)/5))))

        i+=1
daily_cases_continents = df_confirmed.groupby('Continent').sum().diff(axis=1).replace(np.nan,0)

daily_cases_continents = daily_cases_continents[daily_cases_continents.index!='Others']

f = plt.figure(figsize=(20,12))

ax = f.add_subplot(111)

for i,continent in enumerate(daily_cases_continents.index):

    t = daily_cases_continents.loc[daily_cases_continents.index == continent].values[0]

    t = t[t>=0]

    date = np.arange(0,len(t[:]))

    plt.plot(date,t/1000,'-o',label = continent,linewidth =2, markevery=[-1])



# Grid Settings

plt.grid(lw = 1, ls = '-', c = "0.85", which = 'major')

plt.grid(lw = 1, ls = '-', c = "0.95", which = 'minor')



#Title

plt.title("COVID-19 Daily Confirmed Cases in Continents",{'fontsize':24})



# Axis Label

plt.xlabel("Days",fontsize =18)

plt.ylabel("Number of Daily Confirmed Cases (Thousand)",fontsize =18)



# Legend

plt.legend(fontsize=18)



plt.show()
daily_deaths_continents = df_deaths.groupby('Continent').sum().diff(axis=1).replace(np.nan,0)

daily_deaths_continents = daily_deaths_continents[daily_deaths_continents.index!='Others']

f = plt.figure(figsize=(20,12))

ax = f.add_subplot(111)

for i,continent in enumerate(daily_deaths_continents.index):

    t = daily_deaths_continents.loc[daily_deaths_continents.index == continent].values[0]

    t = t[t>=0]

    date = np.arange(0,len(t[:]))

    plt.plot(date,t/1000,'-o',label = continent,linewidth =2, markevery=[-1])



# Grid Settings

plt.grid(lw = 1, ls = '-', c = "0.85", which = 'major')

plt.grid(lw = 1, ls = '-', c = "0.95", which = 'minor')



#Title

plt.title("COVID-19 Daily Deaths in Continents",{'fontsize':24})



# Axis Label

plt.xlabel("Days",fontsize =18)

plt.ylabel("Number of Daily Deaths (Thousand)",fontsize =18)



# Legend

plt.legend(fontsize=18)



plt.show()
df_continents_cases['Latitude'] = [6.426117205286786, 44.94789322476297, -25.734968546496344, 44.94789322476297, 56.51520886670177, -31.065922730080157]

df_continents_cases['Longitude'] = [18.2766152761759, 95.7503726784575, 134.489562782425, 28.2490403487619, -92.32043635079269, -60.7921128171538]

df_continents_cases.head()
world_map = folium.Map(location=[10,0], tiles="cartodbpositron", zoom_start=2, max_zoom=6, min_zoom=2)

for i in range(0, len(df_continents_cases)):

    folium.Circle(

        location=[df_continents_cases.iloc[i]['Latitude'], df_continents_cases.iloc[i]['Longitude']],

        tooltip = "<h5 style='text-align:center;font-weight: bold'>"+df_continents_cases.index[i]+"</h5>"+

                    "<hr style='margin:10px;'>"+

                    "<ul style='color: #444;list-style-type:circle;align-item:left;padding-left:20px;padding-right:20px'>"+

        "<li>Active: "+str(df_continents_cases['Active'][i])+"</li>"+

        "<li>Confirmed: "+str(df_continents_cases['Confirmed'][i])+"</li>"+

        "<li>Deaths:   "+str(df_continents_cases['Deaths'][i])+"</li>"+

        "</ul>",

        radius=(int((np.log(df_continents_cases['Confirmed'][i]+1.00001)))+0.2)*50000,

        color='#ff6600',

        fill_color='#ff8533',

        fill=True).add_to(world_map)

world_map
df_country = df_covid19.drop(['Continent', 'Lat', 'Long_', 'Incident_Rate', 'Mortality_Rate'], axis=1)

df_country.index = df_country["Country"]

df_country.drop(['Country'], axis=1, inplace=True)

df_country.fillna(0,inplace=True)
df_country['Mortality_Rate'] = np.round((df_country["Deaths"]/df_country["Confirmed"])*100,2)

df_country
# function for plotting horizontal bar plot

def horizontal_barplot(x, y, title, xlabel, ylabel, color):

    fig = plt.figure(figsize = (10,5))

    fig.add_subplot(111)

    plt.axes(axisbelow = True)

    plt.barh(x.index[-10:], y.values[-10:], color = color)

    plt.tick_params(size = 5, labelsize = 13)

    plt.xlabel(xlabel, fontsize = 18)

    plt.ylabel(ylabel,fontsize = 18)

    plt.title(title,fontsize = 20)

    plt.grid(alpha = 0.3)

    plt.show()
horizontal_barplot(x = df_country.sort_values('Confirmed')["Confirmed"], 

                   y = df_country.sort_values('Confirmed')["Confirmed"], 

                   title = "Top 10 Countries - Confirmed Cases", 

                   xlabel = "Confirmed Cases", 

                   ylabel = "Countries", 

                   color = "blue")
horizontal_barplot(x = df_country.sort_values('Active')["Active"], 

                   y = df_country.sort_values('Active')["Active"], 

                   title = "Top 10 Countries - Active Cases", 

                   xlabel = "Active Cases", 

                   ylabel = "Countries", 

                   color = "orange")
horizontal_barplot(x = df_country.sort_values('Recovered')["Recovered"], 

                   y = df_country.sort_values('Recovered')["Recovered"], 

                   title = "Top 10 Countries - Recovered Cases", 

                   xlabel = "Recovered", 

                   ylabel = "Countries", 

                   color = "limegreen")
horizontal_barplot(x = df_country.sort_values('Deaths')["Deaths"], 

                   y = df_country.sort_values('Deaths')["Deaths"], 

                   title = "Top 10 Countries - Deaths", 

                   xlabel = "Deaths", 

                   ylabel = "Countries", 

                   color = "red")
df_countries = df_covid19.drop(['Continent', 'Incident_Rate', 'Mortality_Rate'], axis=1)

df_countries.index = df_countries["Country"]

df_countries.fillna(0,inplace=True)

df_countries.head()
world_map = folium.Map(location=[10,0], tiles="cartodbpositron", zoom_start=2, max_zoom=6, min_zoom=2)

for i in range(0, len(df_countries)):

    folium.Circle(

        location=[df_countries.iloc[i]['Lat'], df_countries.iloc[i]['Long_']],

        tooltip = "<h5 style='text-align:center;font-weight: bold'>"+df_countries.index[i]+"</h5>"+

                    "<hr style='margin:10px;'>"+

                    "<ul style='color: #444;list-style-type:circle;align-item:left;padding-left:20px;padding-right:20px'>"+

        "<li>Active: "+str(df_countries['Active'][i])+"</li>"+

        "<li>Confirmed: "+str(df_countries['Confirmed'][i])+"</li>"+

        "<li>Deaths:   "+str(df_countries['Deaths'][i])+"</li>"+

        "</ul>",

        radius=(int((np.log(df_countries['Confirmed'][i]+1.00001)))+0.2)*50000,

        color='#ff6600',

        fill_color='#ff8533',

        fill=True).add_to(world_map)

world_map
case_nums_country = df_confirmed.groupby("Country").sum().drop(['Lat','Long'],axis =1).apply(lambda x: x[x > 0].count(), axis =0)

d = [datetime.strptime(date,'%m/%d/%y').strftime("%d %b") for date in case_nums_country.index]



f = plt.figure(figsize=(14,8))

f.add_subplot(111)

marker_style = dict(c="crimson",linewidth=6, linestyle='-', marker='o',markersize=6, markerfacecolor='#ffffff')

plt.plot(d, case_nums_country,**marker_style)

plt.tick_params(labelsize = 14)

plt.xticks(list(np.arange(0,len(d),int(len(d)/5))),d[:-1:int(len(d)/5)]+[d[-1]])



plt.xlabel("Dates",fontsize=18)

plt.ylabel("Number of Countries",fontsize=18)

plt.grid(alpha = 0.3)

plt.show()
df_countries_cases = df_confirmed.groupby(["Country"]).sum()

df_countries_deaths = df_deaths.groupby(["Country"]).sum()



df_countries_cases.drop(['Lat', 'Long'], inplace=True, axis=1)

df_countries_deaths.drop(['Lat', 'Long'], inplace=True, axis=1)



df_countries_cases = df_countries_cases.sort_values(df_confirmed.columns[-1],ascending = False)[:20]



dates = df_countries_cases.columns

d = [datetime.strptime(date,'%m/%d/%y').strftime("%d %b") for date in dates]
marker_style_confirmed = dict(c="darkcyan", linewidth=6, linestyle='-', marker='o', markersize=6, markerfacecolor='#ffffff')

marker_style_death = dict(c="crimson", linewidth=6, linestyle='-', marker='o', markersize=6, markerfacecolor='#ffffff')

fig, axes = plt.subplots(nrows=10, ncols=2, figsize=(15,15))

plt.subplots_adjust(top = 4.0)

i=0

for rows in axes:

    for ax1 in rows:

        ax1.plot(d, df_countries_cases.iloc[i], label = 'Confirmed', **marker_style_confirmed)

        ax1.plot(d, df_countries_deaths[df_countries_deaths.index == df_countries_cases.index[i]].values[0], label = 'Deaths', **marker_style_death)

        ax1.fill_between(d, df_countries_cases.iloc[i], color='darkcyan', alpha=0.3)

        ax1.fill_between(d, df_countries_deaths[df_countries_deaths.index == df_countries_cases.index[i]].values[0], color='crimson', alpha=0.3)

        ax1.set_xlabel("Dates", fontsize = 12)

        ax1.set_ylabel("No. of Cases",fontsize = 12)

        ax1.set_title("COVID Cases: "+df_countries_cases.index[i], fontsize = 15)

        ax1.legend(loc= "best", fontsize = 12)

        ax1.set_xticks(list(np.arange(0,len(d),int(len(d)/5))))

        ax1.grid(which='major', linewidth = 0.3)

        i+=1
temp = df_confirmed.groupby('Country').sum().diff(axis=1).sort_values(df_confirmed.columns[-1],ascending=False).head(10).replace(np.nan,0)

f = plt.figure(figsize=(20,12))

ax = f.add_subplot(111)

for i,country in enumerate(temp.index):

    t = temp.loc[temp.index ==country].values[0]

    t = t[t>=0]

    date = np.arange(0,len(t[:]))

    plt.plot(date,t/1000,'-o',label = country,linewidth =2, markevery=[-1])



# Grid Settings

plt.grid(lw = 1, ls = '-', c = "0.85", which = 'major')

plt.grid(lw = 1, ls = '-', c = "0.95", which = 'minor')



#Title

plt.title("COVID-19 Daily Confirmed Cases in Top 10 Countries",{'fontsize':24})



# Axis Label

plt.xlabel("Days",fontsize =18)

plt.ylabel("Number of Daily Confirmed Cases (Thousand)",fontsize =18)



# Legend

plt.legend(fontsize=18)



plt.show()
temp = df_deaths.groupby('Country').sum().diff(axis=1).sort_values(df_deaths.columns[-1],ascending=False).head(10).replace(np.nan,0)

f = plt.figure(figsize=(20,12))

ax = f.add_subplot(111)

for i,country in enumerate(temp.index):

    t = temp.loc[temp.index ==country].values[0]

    t = t[t>=0]

    date = np.arange(0,len(t[:]))

    plt.plot(date,t/1000,'-o',label = country,linewidth =2, markevery=[-1])



# Grid

plt.grid(lw = 1, ls = '-', c = "0.85", which = 'major')

plt.grid(lw = 1, ls = '-', c = "0.95", which = 'minor')



#Title

plt.title("COVID-19 Daily Deaths in Top 10 Countries",{'fontsize':24})



# Axis Label

plt.xlabel("Days",fontsize =18)

plt.ylabel("Number of Daily Death Cases (Thousand)",fontsize =18)



# Legend

plt.legend(fontsize=18)



plt.show()