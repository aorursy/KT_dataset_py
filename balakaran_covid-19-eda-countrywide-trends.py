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



# Any results you write to the current directory are saved as output.
# Install pycountry_convert

!pip install pycountry_convert 

!pip install folium
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from matplotlib import ticker 

import pycountry_convert as pc

import folium

from datetime import datetime, timedelta,date

from scipy.interpolate import make_interp_spline, BSpline

import plotly.express as px

import json, requests



%matplotlib inline
# Retriving Dataset

df_confirmed = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')

df_deaths = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')



df_covid19 = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/web-data/data/cases_country.csv")
df_confirmed.head()
df_deaths.head()
df_covid19.head()
df_confirmed = df_confirmed.rename(columns={"Province/State":"state","Country/Region": "country"})

df_deaths = df_deaths.rename(columns={"Province/State":"state","Country/Region": "country"})

df_covid19 = df_covid19.rename(columns={"Country_Region": "country"})
# Changing the conuntry names as required by pycountry_convert Lib



df_confirmed.loc[df_confirmed['country'] == "US", "country"] = "USA"

df_deaths.loc[df_deaths['country'] == "US", "country"] = "USA"

df_covid19.loc[df_covid19['country'] == "US", "country"] = "USA"



df_confirmed.loc[df_confirmed['country'] == 'Korea, South', "country"] = 'South Korea'

df_deaths.loc[df_deaths['country'] == 'Korea, South', "country"] = 'South Korea'

df_covid19.loc[df_covid19['country'] == "Korea, South", "country"] = "South Korea"



df_confirmed.loc[df_confirmed['country'] == 'Taiwan*', "country"] = 'Taiwan'

df_deaths.loc[df_deaths['country'] == 'Taiwan*', "country"] = 'Taiwan'

df_covid19.loc[df_covid19['country'] == "Taiwan*", "country"] = "Taiwan"



df_confirmed.loc[df_confirmed['country'] == 'Congo (Kinshasa)', "country"] = 'Democratic Republic of the Congo'

df_deaths.loc[df_deaths['country'] == 'Congo (Kinshasa)', "country"] = 'Democratic Republic of the Congo'

df_covid19.loc[df_covid19['country'] == "Congo (Kinshasa)", "country"] = "Democratic Republic of the Congo"



df_confirmed.loc[df_confirmed['country'] == "Cote d'Ivoire", "country"] = "Côte d'Ivoire"

df_deaths.loc[df_deaths['country'] == "Cote d'Ivoire", "country"] = "Côte d'Ivoire"

df_covid19.loc[df_covid19['country'] == "Cote d'Ivoire", "country"] = "Côte d'Ivoire"



df_confirmed.loc[df_confirmed['country'] == "Reunion", "country"] = "Réunion"

df_deaths.loc[df_deaths['country'] == "Reunion", "country"] = "Réunion"

df_covid19.loc[df_covid19['country'] == "Reunion", "country"] = "Réunion"



df_confirmed.loc[df_confirmed['country'] == 'Congo (Brazzaville)', "country"] = 'Republic of the Congo'

df_deaths.loc[df_deaths['country'] == 'Congo (Brazzaville)', "country"] = 'Republic of the Congo'

df_covid19.loc[df_covid19['country'] == "Congo (Brazzaville)", "country"] = "Republic of the Congo"



df_confirmed.loc[df_confirmed['country'] == 'Bahamas, The', "country"] = 'Bahamas'

df_deaths.loc[df_deaths['country'] == 'Bahamas, The', "country"] = 'Bahamas'

df_covid19.loc[df_covid19['country'] == "Bahamas, The", "country"] = "Bahamas"



df_confirmed.loc[df_confirmed['country'] == 'Gambia, The', "country"] = 'Gambia'

df_deaths.loc[df_deaths['country'] == 'Gambia, The', "country"] = 'Gambia'

df_covid19.loc[df_covid19['country'] == "Gambia, The", "country"] = "Gambia"



# getting all countries

countries = np.asarray(df_confirmed["country"])

countries1 = np.asarray(df_covid19["country"])



# Continent_code to Continent_names

continents = {

    'NA': 'North America',

    'SA': 'South America', 

    'AS': 'Asia',

    'OC': 'Australia',

    'AF': 'Africa',

    'EU' : 'Europe',

    'na' : 'Others'

}



# Defininng Function for getting continent code for country.

def country_to_continent_code(country):

    try:

        return pc.country_alpha2_to_continent_code(pc.country_name_to_country_alpha2(country))

    except :

        return 'na'



#Collecting Continent Information

df_confirmed.insert(2,"continent", [continents[country_to_continent_code(country)] for country in countries[:]])

df_deaths.insert(2,"continent",  [continents[country_to_continent_code(country)] for country in countries[:]])

df_covid19.insert(1,"continent",  [continents[country_to_continent_code(country)] for country in countries1[:]])
df_confirmed = df_confirmed.replace(np.nan, '', regex=True)

df_deaths = df_deaths.replace(np.nan, '', regex=True)

def plot_params(ax,axis_label = None, plt_title = None, label_size=15, axis_fsize = 15, title_fsize = 20, scale = 'linear' ):

    # Tick-Parameters

    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())

    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    ax.tick_params(which='both', width=1,labelsize=label_size)

    ax.tick_params(which='major', length=6)

    ax.tick_params(which='minor', length=3, color='0.8')



    # Grid

    plt.grid(lw = 1, ls = '-', c = "0.7", which = 'major')

    plt.grid(lw = 1, ls = '-', c = "0.9", which = 'minor')



    # Plot Title

    plt.title( plt_title,{'fontsize':title_fsize})

    

    # Yaxis sacle

    plt.yscale(scale)

    

    # Plot Axes Labels

    xl = plt.xlabel(axis_label[0],fontsize = axis_fsize)

    yl = plt.ylabel(axis_label[1],fontsize = axis_fsize)

    
def visualize_covid_cases(confirmed, deaths, continent=None , country = None , state = None, period = None, figure = None, scale = "linear"):

    x = 0

    if figure == None:

        f = plt.figure(figsize=(10,10))

        # Sub plot

        ax = f.add_subplot(111)

    else :

        f = figure[0]

        # Sub plot

        ax = f.add_subplot(figure[1],figure[2],figure[3])

    

    plt.tight_layout(pad=10, w_pad=5, h_pad=5)

    

    stats = [confirmed, deaths]

    label = ["Confirmed", "Deaths"]

    

    if continent != None:

        params = ["continent",continent]

    elif country != None:

        params = ["country",country]

    else: 

        params = ["All", "All"]



    for i,stat in enumerate(stats):

        if params[1] == "All" :

            cases = np.sum(np.asarray(stat.iloc[:,5:]),axis = 0)[x:]

        else :

            cases = np.sum(np.asarray(stat[stat[params[0]] == params[1]].iloc[:,5:]),axis = 0)[x:]

        date = np.arange(1,cases.shape[0]+1)[x:]

        plt.plot(date,cases,label = label[i]+" (Total : "+str(cases[-1])+")")



    if params[1] == "All" :

        Total_confirmed = np.sum(np.asarray(stats[0].iloc[:,5:]),axis = 0)[x:]

        Total_deaths = np.sum(np.asarray(stats[1].iloc[:,5:]),axis = 0)[x:]

    else :

        Total_confirmed =  np.sum(np.asarray(stats[0][stat[params[0]] == params[1]].iloc[:,5:]),axis = 0)[x:]

        Total_deaths = np.sum(np.asarray(stats[1][stat[params[0]] == params[1]].iloc[:,5:]),axis = 0)[x:]

        

    #text = "From "+stats[0].columns[4]+" to "+stats[0].columns[-2]+"\n"

    #text += "Mortality rate : "+ str(int(Total_deaths[-1]/(Total_confirmed[-1])*10000)/100)+"\n"

    #text += "Last 5 Days:\n"

    #text += "Confirmed : " + str(Total_confirmed[-1] - Total_confirmed[-6])+"\n"

    #text += "Deaths : " + str(Total_deaths[-1] - Total_deaths[-6])+"\n"

    text = ''

    text += "Last 24 Hours:\n"

    text += "Confirmed : " + str(Total_confirmed[-1] - Total_confirmed[-2])+"\n"

    text += "Deaths : " + str(Total_deaths[-1] - Total_deaths[-2])+"\n"

    

    plt.text(0.02, 0.78, text, fontsize=15, horizontalalignment='left', verticalalignment='top', transform=ax.transAxes,bbox=dict(facecolor='white', alpha=0.4))

    

    # Plot Axes Labels

    axis_label = ["Days ("+df_confirmed.columns[5]+" - "+df_confirmed.columns[-1]+")","No of Cases"]

    

    # Plot Parameters

    plot_params(ax,axis_label,scale = scale)

    

    # Plot Title

    if params[1] == "All" :

        plt.title("COVID-19 Cases World",{'fontsize':25})

    else:   

        plt.title("COVID-19 Cases for "+params[0]+" "+params[1] ,{'fontsize':25})

        

    # Legend Location

    l = plt.legend(loc= "best",fontsize = 15)

    

    if figure == None:

        plt.show()

        
def get_total_cases(cases, country = "All"):

    if(country == "All") :

        return np.sum(np.asarray(cases.iloc[:,5:]),axis = 0)[-1]

    else :

        return np.sum(np.asarray(cases[cases["country"] == country].iloc[:,5:]),axis = 0)[-1]

    
def get_mortality_rate(confirmed,deaths, continent = None, country = None):

    if continent != None:

        params = ["continent",continent]

    elif country != None:

        params = ["country",country]

    else :

        params = ["All", "All"]

    

    if params[1] == "All" :

        Total_confirmed = np.sum(np.asarray(confirmed.iloc[:,5:]),axis = 0)

        Total_deaths = np.sum(np.asarray(deaths.iloc[:,5:]),axis = 0)

        mortality_rate = np.round((Total_deaths/Total_confirmed)*100,2)

    else :

        Total_confirmed =  np.sum(np.asarray(confirmed[confirmed[params[0]] == params[1]].iloc[:,5:]),axis = 0)

        Total_deaths = np.sum(np.asarray(deaths[deaths[params[0]] == params[1]].iloc[:,5:]),axis = 0)

        mortality_rate = np.round((Total_deaths/Total_confirmed)*100,2)

    

    return np.nan_to_num(mortality_rate)
def dd(date1,date2):

    return (datetime.strptime(date1,'%m/%d/%y') - datetime.strptime(date2,'%m/%d/%y')).days
df_countries_cases = df_covid19.copy().drop(['Lat','Long_','continent','Last_Update'],axis =1)

df_countries_cases.index = df_countries_cases["country"]

df_countries_cases = df_countries_cases.drop(['country'],axis=1)

df_countries_cases.head()



df_continents_cases = df_covid19.copy().drop(['Lat','Long_','country','Last_Update'],axis =1)

df_continents_cases = df_continents_cases.groupby(["continent"]).sum()
pd.DataFrame(df_countries_cases.sum()).transpose().style.background_gradient(cmap='Wistia',axis=1)

df_continents_cases.style.background_gradient(cmap='Wistia')
df_countries_cases.sort_values('Confirmed', ascending= False).style.background_gradient(cmap='Wistia')

f = plt.figure(figsize=(10,6))

f.add_subplot(111)



inds = df_countries_cases.sort_values('Confirmed')["Confirmed"].index[-10:].append(df_countries_cases.loc[df_countries_cases.index == 'India']['Confirmed'].index)

vals = np.append(df_countries_cases.sort_values('Confirmed')["Confirmed"].values[-10:],df_countries_cases.loc[df_countries_cases.index == 'India']['Confirmed'].values)



plt.barh(inds, vals, color = 'b')

plt.tick_params(size=5,labelsize = 13)

plt.xlabel("Confirmed Cases",fontsize=18)

plt.title("India and Top10 Countries",fontsize=20)

plt.grid(alpha=0.3)

for i, v in enumerate(vals):                                                   # i varies from 1 to length(vals)

    plt.text(v, i, ' '+str(v), va = 'center', color='blue', fontweight='bold')
f = plt.figure(figsize=(10,6))

f.add_subplot(111)



inds = df_countries_cases.sort_values('Deaths')['Deaths'].index[-10:].append(df_countries_cases.loc[df_countries_cases.index == 'India']['Deaths'].index)

vals = np.append(df_countries_cases.sort_values('Deaths')['Deaths'].values[-10:],df_countries_cases.loc[df_countries_cases.index == 'India']['Deaths'].values)



plt.barh(inds, vals, color = 'b')

plt.tick_params(size=5,labelsize = 13)

plt.xlabel("Deaths",fontsize=18)

plt.title("India and Top10 Countries",fontsize=20)

plt.grid(alpha=0.3)

for i, v in enumerate(vals):                                                   # i varies from 1 to length(vals)

    plt.text(v, i, ' '+str(v), va = 'center', color='blue', fontweight='bold')
df_countries_cases['Mortality'] = df_countries_cases['Deaths'] / df_countries_cases['Confirmed']

f = plt.figure(figsize=(10,6))

f.add_subplot(111)



inds = df_countries_cases.sort_values('Mortality')['Mortality'].index[-10:].append(df_countries_cases.loc[df_countries_cases.index == 'India']['Mortality'].index)

vals = np.append(df_countries_cases.sort_values('Mortality')['Mortality'].values[-10:],df_countries_cases.loc[df_countries_cases.index == 'India']['Mortality'].values)



plt.barh(inds, vals, color = 'b')

plt.tick_params(size=5,labelsize = 13)

plt.xlabel("Death Rate",fontsize=18)

plt.title("India and Top10 Countries",fontsize=20)

plt.grid(alpha=0.3)

for i, v in enumerate(vals):                                                   # i varies from 1 to length(vals)

    plt.text(v, i, ' '+str(round(v*100, 2))+'%', va = 'center', color='blue', fontweight='bold')
world_map = folium.Map(location=[10,0], tiles="cartodbpositron", zoom_start=2, max_zoom=6, min_zoom=2)

for i in range(0,len(df_confirmed)):

    folium.Circle(

        location=[df_confirmed.iloc[i]['Lat'], df_confirmed.iloc[i]['Long']],

        tooltip = "<h5 style='text-align:center;font-weight: bold'>"+df_confirmed.iloc[i]['country']+"</h5>"+

                    "<div style='text-align:center;'>"+str(np.nan_to_num(df_confirmed.iloc[i]['state']))+"</div>"+

                    "<hr style='margin:10px;'>"+

                    "<ul style='color: #444;list-style-type:circle;align-item:left;padding-left:20px;padding-right:20px'>"+

        "<li>Confirmed: "+str(df_confirmed.iloc[i,-1])+"</li>"+

        "<li>Deaths:   "+str(df_deaths.iloc[i,-1])+"</li>"+

        "<li>Mortality Rate:   "+str(np.round(df_deaths.iloc[i,-1]/(df_confirmed.iloc[i,-1]+1.00001)*100,2))+"</li>"+

        "</ul>"

        ,

        radius=(int((np.log10(df_confirmed.iloc[i,-1]+1.00001)))+0.2)*50000,

        color = '#2e41bf',

        fill_color = '#3a75bd',

        fill=True).add_to(world_map)



world_map
df_countries = df_confirmed.groupby(["country"]).sum()

df_countries = df_countries.sort_values(df_countries.columns[-1],ascending = False)

countries = df_countries.index[0:10].append(pd.Index(['India'])) # conver 'India' to index before appending



cols = 2

rows = int(np.ceil(countries.shape[0]/cols))

f = plt.figure(figsize=(20,8*rows))

for i,country in enumerate(countries):

    visualize_covid_cases(df_confirmed, df_deaths, country = country,figure = [f,rows,cols, i+1])



plt.show()
tt = df_confirmed.groupby('country').sum().drop(["Lat","Long"],axis =1).sort_values(df_confirmed.columns[-1], ascending= False)

countrylist = ['Italy','Spain','Iran','France', 'USA', 'United Kingdom', 'Netherlands','Canada','India']

temp = tt.loc[countrylist]



threshold = 20

f = plt.figure(figsize=(14,10))

ax = f.add_subplot(111)

for i,country in enumerate(temp.index):

    if i >= 10:

        if country != "India":

            continue

    x = 40

    t = temp.loc[temp.index== country].values[0]

    t = t[t>threshold][:x]

     

    date = np.arange(0,len(t[:x]))

    xnew = np.linspace(date.min(), date.max(), 10)

    spl = make_interp_spline(date, t, k=1)  # type: BSpline

    power_smooth = spl(xnew)

    

        

    if country == 'India':

        marker_style = dict(linewidth=4, linestyle='-', marker='o',markersize=10, markerfacecolor='r')

        plt.plot(date,t,"-.",label = country,**marker_style)

    

    elif country == 'United Kingdom':

        marker_style = dict(linewidth=4, linestyle='-', marker='o',markersize=10, markerfacecolor='b')

        plt.plot(date,t,"-.",label = country,**marker_style)

        

    elif country == 'Canada':

        marker_style = dict(linewidth=4, linestyle='-', marker='o',markersize=10, markerfacecolor='y')

        plt.plot(date,t,"-.",label = country,**marker_style)

        

    else:

        plt.plot(xnew,power_smooth,label = country,linewidth = 3)



plt.tick_params(labelsize = 14)        

plt.xticks(np.arange(0,60,7),[ "Day "+str(i) for i in range(60)][::7])     



# Reference lines 

x = np.arange(0,20)

y = 2**(x+np.log2(threshold))

plt.plot(x,y,"--",linewidth =2,color = "gray")

plt.annotate("No. of cases doubles every day",(x[-2],y[-1]),xycoords="data",fontsize=14,alpha = 0.5)



x = np.arange(0,36)

y = 2**(x/2+np.log2(threshold))

plt.plot(x,y,"--",linewidth =2,color = "gray")

plt.annotate(".. every socend day",(x[-3],y[-1]),xycoords="data",fontsize=14,alpha = 0.5)



x = np.arange(0,36)

y = 2**(x/3+np.log2(threshold))

plt.plot(x,y,"--",linewidth =2,color = "blue")

plt.annotate(".. every third day",(x[-3],y[-1]),color='Blue', xycoords="data",fontsize=14,alpha = 0.8)



x = np.arange(0,36)

y = 2**(x/7+np.log2(threshold))

plt.plot(x,y,"--",linewidth =2,color = "gray")

plt.annotate(".. every week",(x[-3],y[-1]),xycoords="data",fontsize=14,alpha = 0.5)



x = np.arange(0,36)

y = 2**(x/30+np.log2(threshold))

plt.plot(x,y,"--",linewidth =2,color = "gray")

plt.annotate(".. every month",(x[-3],y[-1]),xycoords="data",fontsize=14,alpha = 0.5)





# India is following trend similar to doulbe the cases in 4 days but it may increase the rate 

x = np.arange(0,36)

y = 2**(x/4+np.log2(threshold))

plt.plot(x,y,"--",linewidth =2,color = "Red")

plt.annotate(".. every 4 days",(x[-3],y[-1]),color="Red",xycoords="data",fontsize=14,alpha = 0.8)



# plot Params

plt.xlabel("Days",fontsize=17)

plt.ylabel("Number of Confirmed Cases",fontsize=17)

plt.title("Trend Comparision of India and Canada with Top-6 Countries",fontsize=22)

plt.legend(loc = "upper left")

plt.yscale("log")

plt.grid(which="major")

plt.savefig('Trend Comparision with India.png')

plt.show()
tt = df_deaths.groupby('country').sum().drop(["Lat","Long"],axis =1).sort_values(df_deaths.columns[-1], ascending= False)

countrylist = ['Italy','Spain','China','Iran','France', 'USA', 'United Kingdom', 'India']

temp = tt.loc[countrylist]



tt.head(10)      # top 10
tt = df_deaths.groupby('country').sum().drop(["Lat","Long"],axis =1).sort_values(df_deaths.columns[-1], ascending= False)

countrylist = ['Italy','Spain','Iran','France', 'USA', 'United Kingdom', 'Netherlands','Canada','India']

temp = tt.loc[countrylist]



threshold = 5

f = plt.figure(figsize=(16,10))

ax = f.add_subplot(111)

for i,country in enumerate(temp.index):

    if i > 20:                                                       # plot top 20 countries

        break

    x = 80                                                      

    t = temp.loc[temp.index== country].values[0]

    t = t[t>threshold][:x]                                           # plot x days from the day the value is higher than the threshold

     

    date = np.arange(0,len(t[:x]))

    xnew = np.linspace(date.min(), date.max(), 10)

    spl = make_interp_spline(date, t, k=1)                           # type: BSpline

    power_smooth = spl(xnew)

    

    if country == 'India':

        marker_style = dict(linewidth=4, linestyle='-', marker='o',markersize=10, markerfacecolor='r')

        plt.plot(date,t,"-.",label = country,**marker_style)

    

    elif country == 'United Kingdom':

        marker_style = dict(linewidth=4, linestyle='-', marker='o',markersize=10, markerfacecolor='b')

        plt.plot(date,t,"-.",label = country,**marker_style)

        

    elif country == 'Canada':

        marker_style = dict(linewidth=4, linestyle='-', marker='o',markersize=10, markerfacecolor='y')

        plt.plot(date,t,"-.",label = country,**marker_style)

        

    elif country == 'USA':

        marker_style = dict(linewidth=4, linestyle='-', marker='o',markersize=10, markerfacecolor='m')

        plt.plot(date,t,"-.",label = country,**marker_style)

        

    elif country == 'Netherlands':

        marker_style = dict(linewidth=4, linestyle='-', marker='o',markersize=10, markerfacecolor='c')

        plt.plot(date,t,"-.",label = country,**marker_style)

        

    else:

        plt.plot(xnew, power_smooth, label = country, linewidth = 3)

    

plt.tick_params(labelsize = 14)        

plt.xticks(np.arange(0,x,7),[ "Day "+str(i) for i in range(x)][::7])     



plt.axvline(x=13, c = 'b', alpha = 0.8)     # UK lockdown 23 March 2020

plt.annotate('UK Lockdown 23rd March', (13,60000), color = 'Blue', fontsize=14, rotation = 90)

plt.annotate('Day 0: At least 5 fatalities', (0,40000), color = 'k', fontsize=14, rotation = 90)





# Reference lines 

#x = np.arange(0,18)

#y = 2**(x+np.log2(threshold))

#plt.plot(x,y,"--",linewidth =2,color = "gray")

#plt.annotate("No. of cases doubles every day",(x[-2],y[-1]),xycoords="data",fontsize=14,alpha = 0.5)



#x = np.arange(0,26)

#y = 2**(x/2+np.log2(threshold))

#plt.plot(x,y,"--",linewidth =2,color = "Red")

#plt.annotate("Doubles every 2nd day",(x[-3],y[-1]),color='Red',xycoords="data",fontsize=14,alpha = 0.8)



x = np.arange(0,42)

y = 2**(x/3+np.log2(threshold))

plt.plot(x,y,"--",linewidth =2,color = "red")

plt.annotate(".. every third day",(x[-3],y[-1]),color='red',xycoords="data",fontsize=14,alpha = 0.8)



x = np.arange(0,68)

y = 2**(x/5+np.log2(threshold))

plt.plot(x,y,"--",linewidth =2,color = "Blue")

plt.annotate(".. every fifth day",(x[-3],y[-1]),color = 'Blue', xycoords="data",fontsize=14,alpha = 0.5)



#x = np.arange(0,36)

#y = 2**(x/7+np.log2(threshold))

#plt.plot(x,y,"--",linewidth =2,color = "gray")

#plt.annotate(".. every week",(x[-3],y[-1]),xycoords="data",fontsize=14,alpha = 0.5)



#x = np.arange(0,36)

#y = 2**(x/30+np.log2(threshold))

#plt.plot(x,y,"--",linewidth =2,color = "gray")

#plt.annotate(".. every month",(x[-3],y[-1]),xycoords="data",fontsize=14,alpha = 0.5)



# plot Params

plt.xlabel("Days",fontsize=17)

plt.ylabel("Number of Deaths",fontsize=17)

plt.title("Trend Comparision",fontsize=22)

plt.legend(loc = "upper left")

#plt.yscale("log")

plt.grid(which="major")

plt.show()

tt = df_deaths.groupby('country').sum().drop(["Lat","Long"],axis =1).sort_values(df_deaths.columns[-1], ascending= False)

countrylist = ['Italy','Spain','Iran','France', 'USA', 'United Kingdom', 'Netherlands','Canada','India']

temp = tt.loc[countrylist]



threshold = 5

f = plt.figure(figsize=(18,10))

ax = f.add_subplot(111)

for i,country in enumerate(temp.index):

    if i > 20:                                                       # plot top 20 countries

        break

    x = 80                                                      

    t = temp.loc[temp.index== country].values[0]

    t = t[t>threshold][:x]                                           # plot x days from the day the value is higher than the threshold

     

    date = np.arange(0,len(t[:x]))

    xnew = np.linspace(date.min(), date.max(), 10)

    spl = make_interp_spline(date, t, k=1)                           # type: BSpline

    power_smooth = spl(xnew)

    

    if country == 'India':

        marker_style = dict(linewidth=4, linestyle='-', marker='o',markersize=10, markerfacecolor='r')

        plt.plot(date,t,"-.",label = country,**marker_style)

    

    elif country == 'United Kingdom':

        marker_style = dict(linewidth=4, linestyle='-', marker='o',markersize=10, markerfacecolor='b')

        plt.plot(date,t,"-.",label = country,**marker_style)

        

    elif country == 'Canada':

        marker_style = dict(linewidth=4, linestyle='-', marker='o',markersize=10, markerfacecolor='y')

        plt.plot(date,t,"-.",label = country,**marker_style)

        

    elif country == 'USA':

        marker_style = dict(linewidth=4, linestyle='-', marker='o',markersize=10, markerfacecolor='m')

        plt.plot(date,t,"-.",label = country,**marker_style)

        

    elif country == 'Netherlands':

        marker_style = dict(linewidth=4, linestyle='-', marker='o',markersize=10, markerfacecolor='c')

        plt.plot(date,t,"-.",label = country,**marker_style)

        

    else:

        plt.plot(xnew, power_smooth, label = country, linewidth = 3)

    

plt.tick_params(labelsize = 14)        

plt.xticks(np.arange(0,x,7),[ "Day "+str(i) for i in range(x)][::7])     



plt.axvline(x=13, c = 'b', alpha = 0.8)     # UK lockdown 23 March 2020

plt.annotate('UK Lockdown 23rd March', (13,200000), color = 'Blue', fontsize=14, rotation = 90)

plt.annotate('Day 0: At least 5 fatalities', (0,1000), color = 'k', fontsize=14, rotation = 90)





# Reference lines 

#x = np.arange(0,18)

#y = 2**(x+np.log2(threshold))

#plt.plot(x,y,"--",linewidth =2,color = "gray")

#plt.annotate("No. of cases doubles every day",(x[-2],y[-1]),xycoords="data",fontsize=14,alpha = 0.5)



x = np.arange(0,32)

y = 2**(x/2+np.log2(threshold))

plt.plot(x,y,"--",linewidth =2,color = "Red")

plt.annotate("Doubles every 2nd day",(x[-3],y[-1]),color='Red',xycoords="data",fontsize=14,alpha = 0.8)



x = np.arange(0,46)

y = 2**(x/3+np.log2(threshold))

plt.plot(x,y,"--",linewidth =2,color = "grey")

plt.annotate(".. every third day",(x[-3],y[-1]),color='grey',xycoords="data",fontsize=14,alpha = 0.8)



#x = np.arange(0,36)

#y = 2**(x/4+np.log2(threshold))

#plt.plot(x,y,"--",linewidth =2,color = "Blue")

#plt.annotate(".. every fourth day",(x[-3],y[-1]),color='Blue', xycoords="data",fontsize=14,alpha = 0.5)



x = np.arange(0,48)

y = 2**(x/5+np.log2(threshold))

plt.plot(x,y,"--",linewidth =2,color = "Blue")

plt.annotate(".. every fifth day",(x[-3],y[-1]),color = 'Blue', xycoords="data",fontsize=14,alpha = 0.5)



#x = np.arange(0,36)

#y = 2**(x/7+np.log2(threshold))

#plt.plot(x,y,"--",linewidth =2,color = "gray")

#plt.annotate(".. every week",(x[-3],y[-1]),xycoords="data",fontsize=14,alpha = 0.5)



#x = np.arange(0,36)

#y = 2**(x/30+np.log2(threshold))

#plt.plot(x,y,"--",linewidth =2,color = "gray")

#plt.annotate(".. every month",(x[-3],y[-1]),xycoords="data",fontsize=14,alpha = 0.5)



# plot Params

plt.xlabel("Days",fontsize=17)

plt.ylabel("Number of Deaths",fontsize=17)

plt.title("Trend Comparision",fontsize=22)

plt.legend(loc = "upper left")

plt.yscale("log")

plt.grid(which="major")

#plt.savefig('Trend Comparision with India.png')

plt.show()
tt = df_deaths.groupby('country').sum().drop(["Lat","Long"],axis =1).sort_values(df_deaths.columns[-1], ascending= False)

countrylist = ['USA','United Kingdom']

temp = tt.loc[countrylist].T.diff(1)[48:]

plt.figure(figsize=(18,9))

#temp['USA'].plot(kind = 'bar', color = 'blue')

temp['USA'].plot(kind = 'bar', color = 'lightblue')

temp['USA'].rolling(window=7).min().plot(linewidth=3, color = 'orange')

temp['USA'].rolling(window=7).max().plot(linewidth=3, color = 'orange')

temp['USA'].rolling(window=7).mean().plot(linewidth=3, color = 'red')



plt.tick_params(size = 5, labelsize = 13, rotation = 90)

plt.xlabel("Time (Days)",fontsize=18)

plt.title("Daily Reported US Deaths",fontsize=20)

plt.grid(alpha=0.3)

vals = temp['USA']

for i, v in enumerate(vals):                                                   # i varies from 1 to length(vals)

    plt.text(i-0.1, v+200, str(v)[:-2], va = 'center', color='blue', fontweight='bold', rotation = 90)



tt = df_deaths.groupby('country').sum().drop(["Lat","Long"],axis =1).sort_values(df_deaths.columns[-1], ascending= False)

countrylist = ['USA','United Kingdom']

temp = tt.loc[countrylist].T.diff(1)[48:]

plt.figure(figsize=(18,9))

temp['United Kingdom'].plot(kind = 'bar', color = 'lightblue')

temp['United Kingdom'].rolling(window=7).min().plot(linewidth=3, color = 'orange', label = '7-day rolling Min')

temp['United Kingdom'].rolling(window=7).max().plot(linewidth=3, color = 'orange', label = '7-day rolling Max')

temp['United Kingdom'].rolling(window=7).mean().plot(linewidth=3, color = 'red', label = '7-day rolling Mean')

plt.fill_between(temp.index, temp['United Kingdom'].rolling(window=7).min(), 

                 temp['United Kingdom'].rolling(window=7).max(), color = 'cyan', alpha = 0.1)



plt.tick_params(size = 5, labelsize = 13, rotation = 90)

plt.xlabel("Time (Days)",fontsize=18)

plt.title("Daily Reported UK Deaths",fontsize=20)

plt.grid(alpha=0.3)

plt.legend()

vals = temp['United Kingdom']

for i, v in enumerate(vals):                                                   # i varies from 1 to length(vals)

    plt.text(i-0.1, v+45, str(v)[:-2], va = 'center', color='blue', fontweight='bold', rotation = 90)

tt = df_deaths.groupby('country').sum().drop(["Lat","Long"],axis =1).sort_values(df_deaths.columns[-1], ascending= False)

countrylist = ['USA','United Kingdom','Netherlands', 'India', 'Canada', 'France', 'Spain', 'Italy']

temp = tt.loc[countrylist].T.diff(1)[38:]

temp = temp.apply(lambda x: x/sum(x))

#plt.figure(figsize=(18,9))

#temp['USA'].plot()

#temp['United Kingdom'].plot()

#temp['India'].plot()

#temp['France'].plot()

temp[['Italy','Spain']].plot(kind='bar',figsize=(18,9))

#temp['Spain'].plot(kind=',bar')

plt.tick_params(size = 5, labelsize = 13)

plt.xlabel("Time (Days)",fontsize=18)

plt.title("Daily Deaths (Normalized by Total)",fontsize=20)

plt.grid(alpha=0.3)

plt.legend()

#vals = temp['United Kingdom']

#for i, v in enumerate(vals):                                                   # i varies from 1 to length(vals)

#    plt.text(i-0.1, v+35, str(v)[:-2], va = 'center', color='blue', fontweight='bold', rotation = 90)

tt = df_deaths.groupby('country').sum().drop(["Lat","Long"],axis =1).sort_values(df_deaths.columns[-1], ascending= False)

countrylist = ['USA','United Kingdom','Netherlands', 'India', 'Canada', 'France', 'Spain', 'Italy']

temp = tt.loc[countrylist].T.diff(1)[38:]

temp = temp.apply(lambda x: x/sum(x))

#plt.figure(figsize=(18,9))

#temp['USA'].plot()

#temp['United Kingdom'].plot()

#temp['India'].plot()

#temp['France'].plot()

temp[['Italy','Netherlands']].plot(kind='bar',figsize=(18,9))

#temp['Spain'].plot(kind=',bar')

plt.tick_params(size = 5, labelsize = 13)

plt.xlabel("Time (Days)",fontsize=18)

plt.title("Daily Deaths (Normalized by Total)",fontsize=20)

plt.grid(alpha=0.3)

plt.legend()

#vals = temp['United Kingdom']

#for i, v in enumerate(vals):                                                   # i varies from 1 to length(vals)

#    plt.text(i-0.1, v+35, str(v)[:-2], va = 'center', color='blue', fontweight='bold', rotation = 90)

tt = df_deaths.groupby('country').sum().drop(["Lat","Long"],axis =1).sort_values(df_deaths.columns[-1], ascending= False)

countrylist = ['USA','Canada','India']

temp = tt.loc[countrylist].T.diff(1)[48:]

plt.figure(figsize=(18,9))

temp['Canada'].plot(kind = 'bar', color = 'lightblue')

#temp['Canada'].rolling(window=7).min().plot(linewidth=3, color = 'orange')

#temp['Canada'].rolling(window=7).max().plot(linewidth=3, color = 'orange')

temp['Canada'].rolling(window=7).mean().plot(linewidth=3, color = 'red')

#temp['Canada'].plot(kind = 'bar', color = 'blue')

plt.tick_params(size = 5, labelsize = 13, rotation = 90)

plt.xlabel("Time (Days)",fontsize=18)

plt.title("Daily Reported Canada Deaths",fontsize=20)

plt.grid(alpha=0.3)

vals = temp['Canada']

for i, v in enumerate(vals):                                                   # i varies from 1 to length(vals)

    plt.text(i-0.1, v+10, str(v)[:-2], va = 'center', color='blue', fontweight='bold', rotation = 90)

tt = df_deaths.groupby('country').sum().drop(["Lat","Long"],axis =1).sort_values(df_deaths.columns[-1], ascending= False)

countrylist = ['USA','United Kingdom','India']

temp = tt.loc[countrylist].T.diff(1)[48:]

plt.figure(figsize=(18,9))

#temp['India'].plot(kind = 'bar', color = 'blue')

#temp['India'].rolling(window=7).mean().plot()



temp['India'].plot(kind = 'bar', color = 'lightblue')

temp['India'].rolling(window=7).min().plot(linewidth=3, color = 'orange')

temp['India'].rolling(window=7).max().plot(linewidth=3, color = 'orange')

temp['India'].rolling(window=7).mean().plot(linewidth=3, color = 'red')



plt.tick_params(size = 5, labelsize = 13, rotation = 90)

plt.xlabel("Time (Days)",fontsize=18)

plt.title("Daily Reported India Deaths",fontsize=20)

plt.grid(alpha=0.3)

vals = temp['India']

for i, v in enumerate(vals):                                                   # i varies from 1 to length(vals)

    plt.text(i-0.1, v+2, str(v)[:-2], va = 'center', color='blue', fontweight='bold', rotation = 90)

tt = df_deaths.groupby('country').sum().drop(["Lat","Long"],axis =1).sort_values(df_deaths.columns[-1], ascending= False)

countrylist = ['USA','United Kingdom','Netherlands', 'India', 'Canada', 'France', 'Spain', 'Italy']

temp = tt.loc[countrylist].T.diff(1)[47:]

temp = temp.apply(lambda x: x/sum(x))

#plt.figure(figsize=(18,9))

#temp['USA'].plot()

#temp['United Kingdom'].plot()

#temp['India'].plot()

#temp['France'].plot()

temp[['Canada','India']].plot(kind='bar',figsize=(18,9))

#temp['Spain'].plot(kind=',bar')

plt.tick_params(size = 5, labelsize = 13)

plt.xlabel("Time (Days)",fontsize=18)

plt.title("Daily Deaths (Normalized by Total)",fontsize=20)

plt.grid(alpha=0.3)

plt.legend()
tt = df_deaths.groupby('country').sum().drop(["Lat","Long"],axis =1).sort_values(df_deaths.columns[-1], ascending= False)

countrylist = ['USA','United Kingdom','Netherlands', 'India', 'Canada', 'France', 'Spain', 'Italy']

temp = tt.loc[countrylist].T.diff(1)[38:]

temp = temp.apply(lambda x: x/sum(x))

#plt.figure(figsize=(18,9))

#temp['USA'].plot()

#temp['United Kingdom'].plot()

#temp['India'].plot()

#temp['France'].plot()

temp[['Italy','India']].plot(kind='bar',figsize=(18,9))

#temp['Spain'].plot(kind=',bar')

plt.tick_params(size = 5, labelsize = 13)

plt.xlabel("Time (Days)",fontsize=18)

plt.title("Daily Deaths (Normalized by Total)",fontsize=20)

plt.grid(alpha=0.3)

plt.legend()
tt = df_deaths.groupby('country').sum().drop(["Lat","Long"],axis =1).sort_values(df_deaths.columns[-1], ascending= False)

countrylist = ['USA','United Kingdom','Netherlands', 'India', 'Canada', 'France', 'Spain', 'Italy']

temp = tt.loc[countrylist].T.diff(1)[38:]

temp = temp.apply(lambda x: x/sum(x))

#plt.figure(figsize=(18,9))

#temp['USA'].plot()

#temp['United Kingdom'].plot()

#temp['India'].plot()

#temp['France'].plot()

temp[['Italy','United Kingdom']].plot(kind='bar',figsize=(18,9))

#temp['Spain'].plot(kind=',bar')

plt.tick_params(size = 5, labelsize = 13)

plt.xlabel("Time (Days)",fontsize=18)

plt.title("Daily Deaths (Normalized by Total)",fontsize=20)

plt.grid(alpha=0.3)

plt.legend()

tt = df_deaths.groupby('country').sum().drop(["Lat","Long"],axis =1).sort_values(df_deaths.columns[-1], ascending= False)

countrylist = ['USA','United Kingdom','Netherlands', 'India', 'Canada', 'France', 'Spain', 'Italy']

temp = tt.loc[countrylist].T.diff(1)[48:]

temp = temp.apply(lambda x: x/sum(x))

#plt.figure(figsize=(18,9))

#temp['USA'].plot()

#temp['United Kingdom'].plot()

#temp['India'].plot()

#temp['France'].plot()

temp[['USA','United Kingdom']].plot(kind='bar',figsize=(18,9))

#temp['Spain'].plot(kind=',bar')

plt.tick_params(size = 5, labelsize = 13)

plt.xlabel("Time (Days)",fontsize=18)

plt.title("Daily Deaths (Normalized by Total)",fontsize=20)

plt.grid(alpha=0.3)

plt.legend()

#vals = temp['United Kingdom']

#for i, v in enumerate(vals):                                                   # i varies from 1 to length(vals)

#    plt.text(i-0.1, v+35, str(v)[:-2], va = 'center', color='blue', fontweight='bold', rotation = 90)

tt = df_deaths.groupby('country').sum().drop(["Lat","Long"],axis =1).sort_values(df_deaths.columns[-1], ascending= False)

countrylist = ['USA','United Kingdom','Netherlands', 'India', 'Canada', 'France', 'Spain', 'Italy']

temp = tt.loc[countrylist].T.diff(1)[38:]

temp = temp.apply(lambda x: x/sum(x))

#plt.figure(figsize=(18,9))

#temp['USA'].plot()

#temp['United Kingdom'].plot()

#temp['India'].plot()

#temp['France'].plot()

temp[['USA','Italy']].plot(kind='bar',figsize=(18,9))

#temp['Spain'].plot(kind=',bar')

plt.tick_params(size = 5, labelsize = 13)

plt.xlabel("Time (Days)",fontsize=18)

plt.title("Daily Deaths (Normalized by Total)",fontsize=20)

plt.grid(alpha=0.3)

plt.legend()

#vals = temp['United Kingdom']

#for i, v in enumerate(vals):                                                   # i varies from 1 to length(vals)

#    plt.text(i-0.1, v+35, str(v)[:-2], va = 'center', color='blue', fontweight='bold', rotation = 90)

tt = df_deaths.groupby('country').sum().drop(["Lat","Long"],axis =1).sort_values(df_deaths.columns[-1], ascending= False)

countrylist = ['USA','United Kingdom','Netherlands', 'India', 'Canada', 'France', 'Spain', 'Italy']

temp = tt.loc[countrylist].T.diff(1)[38:]

temp = temp.apply(lambda x: x/sum(x))

#temp['ts'] = pd.to_datetime(temp.index)

#temp.set_index('ts', inplace = True)



#plt.figure(figsize=(18,9))

#temp['USA'].plot()

#temp['United Kingdom'].plot()

#temp['India'].plot()

#temp['France'].plot()temp[['Italy','United Kingdom']].plot(kind='bar',figsize=(18,9))

#temp[['Italy','United Kingdom']].plot(kind='bar',figsize=(18,9))



f, ax = plt.subplots(figsize=(18,9))

temp[['Italy','United Kingdom']].rolling(1).mean(center=True).plot(kind = 'area', figsize = (18, 9), stacked = False, alpha = 0.5, ax = ax)

temp[['Italy','United Kingdom']].plot(kind='bar',figsize=(18,9), ax = ax)



plt.tick_params(size = 5, labelsize = 13)

plt.xlabel("Time (Days)",fontsize=18)

plt.title("Daily Deaths (Normalized by Total)",fontsize=20)

plt.grid(alpha=0.3)

plt.legend()

df_continents= df_confirmed.groupby(["continent"]).sum()

continents = df_continents.sort_values(df_continents.columns[-1],ascending = False).index

continents = ["All"]+list(continents)



cols =1

rows = 2

axis_label = ["Days ("+df_confirmed.columns[5]+" - "+df_confirmed.columns[-1]+")","Mortality Rate (of 100)"]



f = plt.figure(figsize=(15,10*rows))



#SubPlot 1

ax = f.add_subplot(211)

mortality_rate = get_mortality_rate(df_confirmed,df_deaths,continent=continents[0])

plt.plot(np.arange(1,mortality_rate.shape[0]+1),mortality_rate,label = "World : Current Mortality Rate "+str(mortality_rate[-1]))



plt_title = "COVID-19: World Mortality Rate Curve"

plot_params(ax,axis_label,plt_title)

# Legend Location

l = plt.legend(loc= "best")



plt.show()

india_data_json = requests.get('https://api.rootnet.in/covid19-in/unofficial/covid19india.org/statewise').json()

df_india = pd.io.json.json_normalize(india_data_json['data']['statewise'])

df_india = df_india.set_index("state")
df_india.sort_values('deaths', ascending= False)[:5].style.background_gradient(cmap='Wistia')
# Adding Location data (Latitude,Longitude)

df_india.insert(0,"Lat", [10.8505,19.7515,15.3173,18.1124,22.2587,26.8467,27.0238,28.7041,31.1471,13.0827,29.0588,22.9734,34.152588,33.7782,15.9129,22.9868,30.7333,25.0961,21.2787,30.0668,31.1048,15.2993,20.9517,11.9416,24.6637,23.1645,11.7401,26.2006,25.4670,23.9408,28.2180,23.6102,26.1584,27.5330,20.1809,20.4283,10.5667])

df_india.insert(1,"Long",[76.2711,75.7139,75.7139,79.0193,71.1924,80.9462,74.2179,77.1025,75.3412,80.2707,76.0856,78.6569,77.577049,76.5762,79.7400,87.8550,76.7794,85.3131,81.8661,79.0193,77.1734,74.1240,85.0985,79.8083,93.9063,92.9376,92.6586,92.9376,91.3662,91.9882,94.7278,85.2799,94.5624,88.5122,73.0169,72.8397,72.6417])
df_india.head(20)
df_india.head(10)
# url = "https://raw.githubusercontent.com/Subhash9325/GeoJson-Data-of-Indian-States/master/Indian_States"

# state_json = requests.get(url).json()



#india = folium.Map(location=[23,80], zoom_start=5,max_zoom=6,min_zoom=5,height=800,width="80%")

#for i in range(0,len(df_india.index)):

#    folium.Circle(

#        location=[df_india.iloc[i]['Lat'], df_india.iloc[i]['Long']],

#        tooltip = "<h5 style='text-align:center;font-weight: bold'>"+df_india.iloc[i].name+"</h5>"+

#                    "<hr style='margin:10px;'>"+

#                    "<ul style='color: #444;list-style-type:circle;align-item:left;padding-left:20px;padding-right:20px'>"+

#        "<li>Confirmed: "+str(df_india.iloc[i]['confirmed'])+"</li>"+

#        "<li>Active:   "+str(df_india.iloc[i]['active'])+"</li>"+

#        "<li>Recovered:   "+str(df_india.iloc[i]['recovered'])+"</li>"+

#        "<li>Deaths:   "+str(df_india.iloc[i]['deaths'])+"</li>"+

#        

#        # add small value to avoid divide by zero

#        "<li>Mortality Rate:   "+str(np.round(100*df_india.iloc[i,-2]/(df_india.iloc[i,-4]+0.00001),2))+"</li>"+  

#        "</ul>"

#        ,

#        radius=(int(np.log2(df_india.iloc[i]['confirmed']+1)))*15000,

#        color='#ff6600',

#        fill_color='#ff8533',

#        fill=True).add_to(india)



#india
