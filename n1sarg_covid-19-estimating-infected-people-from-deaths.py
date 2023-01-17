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
import matplotlib.pyplot as plt

import altair as alt

from datetime import timedelta, datetime, date

%config InlineBackend.figure_format = 'retina'



chart_width = 550

chart_height= 400
#hide

def plot(data, type1, levels):

    data_countries_pc2 = data.copy()

    for i in range(0,len(countries)):

        data_countries_pc2[i] = data_countries_pc2[i].reset_index()

        data_countries_pc2[i]['n_days'] = data_countries_pc2[i].index

        if type1 == "scatter":

            data_countries_pc2[i]['cases'] = data_countries_pc2[i]["total_cases"]

        data_countries_pc2[i]['infected'] = data_countries_pc2[i]["total_infected"]

    data_plot = data_countries_pc2[0]

    for i in range(1, len(countries)):    

        data_plot = pd.concat([data_plot, data_countries_pc2[i]], axis=0)

    

    if type1 == "scatter":

        data_plot["45_line"] = data_plot["cases"]



    # Plot it using Altair

    source = data_plot

    

    if levels == True:

        ylabel = "Total"

    else :

        ylabel = "Per Million"



    scales = alt.selection_interval(bind='scales')

    selection = alt.selection_multi(fields=['location'], bind='legend')



    if type1 == "line": 

        base = alt.Chart(source, title =  "Estimated Infected Population By Country").encode(

            x = alt.X('n_days:Q', title = "Days since outbreak"),

            y = alt.Y("infected:Q",title = ylabel),

            color = alt.Color('location:N', legend=alt.Legend(title="Country", labelFontSize=15, titleFontSize=17),

                             scale=alt.Scale(scheme='tableau20'))

        )

        

        shades = base.mark_area().encode(

            x='n_days:Q',

            y='total_infected_lower:Q',

            y2='total_infected_upper:Q',

            opacity = alt.condition(selection, alt.value(0.2), alt.value(0.05))

        )

    

        lines = base.mark_line().encode(

            opacity = alt.condition(selection, alt.value(1), alt.value(0.1))

        ).add_selection(

            scales

        ).add_selection(

            selection

        ).properties(

            width=chart_width,

            height=chart_height

        )

        return(

        ( lines + shades)

        .configure_title(fontSize=20)

        .configure_axis(labelFontSize=15,titleFontSize=18)

        )

    

    if levels == True:

        ylabel = "Infected"

        xlabel = "Cases"

    else :

        ylabel = "Per Million Infected"

        xlabel = "Per Million Cases"

        

    if type1 == "scatter":

        base = alt.Chart(source, title = "COVID-19 Cases VS Infected").encode(

            x = alt.X('cases:Q', title = xlabel),

            y = alt.Y("infected:Q",title = ylabel),

            color = alt.Color('location:N', legend=alt.Legend(title="Country", labelFontSize=15, titleFontSize=17),

                             scale=alt.Scale(scheme='tableau20')),

            opacity = alt.condition(selection, alt.value(1), alt.value(0.1))

        )



        

        scatter = base.mark_point().add_selection(

            scales

        ).add_selection(

            selection

        ).properties(

            width=chart_width,

            height=chart_height

        )



        line_45 = alt.Chart(source).encode(

            x = "cases:Q",

            y = alt.Y("45_line:Q",  scale=alt.Scale(domain=(0, max(data_plot["infected"])))),

        ).mark_line(color="grey", strokeDash=[3,3])

        

        return(

        (scatter + line_45)

        .configure_title(fontSize=20)

        .configure_axis(labelFontSize=15,titleFontSize=18)

        )
#hide 

# Get data on deaths D_t

data = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv", 

                   error_bad_lines=False)

data = data.drop(columns=["Lat", "Long"])

data = data.melt(id_vars= ["Province/State", "Country/Region"])

data = pd.DataFrame(data.groupby(['Country/Region', "variable"]).sum())

data.reset_index(inplace=True)  

data = data.rename(columns={"Country/Region": "location", "variable": "date", "value": "total_deaths"})

data['date'] =pd.to_datetime(data.date)

data = data.sort_values(by = "date")

data.loc[data.location == "US","location"] = "United States"

data.loc[data.location == "Korea, South","location"] = "South Korea"



#hide

# Get data and clean it

data_cases = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv", error_bad_lines=False)



data_cases = data_cases.drop(columns=["Lat", "Long"])

data_cases = data_cases.melt(id_vars= ["Province/State", "Country/Region"])

data_cases = pd.DataFrame(data_cases.groupby(['Country/Region', "variable"]).sum())

data_cases.reset_index(inplace=True)  

data_cases = data_cases.rename(columns={"Country/Region": "location", "variable": "date", "value": "total_cases"})

data_cases['date'] =pd.to_datetime(data_cases.date)

data_cases = data_cases.sort_values(by = "date")

data_cases.loc[data_cases.location == "US","location"] = "United States"

data_cases.loc[data_cases.location == "Korea, South","location"] = "South Korea"

# Add countries

countries = ["China", "Italy", "Spain", "France", "United Kingdom", "Germany", 

             "Portugal", "United States", "Singapore","South Korea", "Japan", 

             "Brazil","Iran", "India", "Switzerland", "Canada", "Australia", 

             "Russia", "Belarus", "Ukraine"]



data_final = pd.merge(data,

                 data_cases

                 )

data_final["CFR"] = data_final["total_deaths"]/data_final["total_cases"]





data_final["total_infected"] = np.NaN

data_final = data_final.sort_values(by = ['location', 'date'])

data_final = data_final.reset_index(drop = True)





for j in countries:

    for i in data_final["date"].unique()[0:-8]:

        data_final.loc[(data_final.date == i) & (data_final.location == j), "total_infected"] = data_final.loc[(data_final.date == i + np.timedelta64(8, 'D')) & (data_final.location == j), "total_deaths"].iloc[0]/data_final.loc[(data_final.date == i + np.timedelta64(8, 'D')) & (data_final.location == j), "CFR"].iloc[0]

        



# Estimate growth rate of infected, g        

data_final['infected_g'] = np.log(data_final['total_infected'])

data_final['infected_g'] = data_final['infected_g'].diff() 



# Estimate number of infected given g

today = data_final.date.iloc[-1]

for j in countries:

    for i in range(7,-1,-1):

        data_final.loc[(data_final.location == j) & (data_final.date == today - timedelta(i)), "total_infected"] = data_final.loc[data_final.location == j, "total_infected"].iloc[-i-2]*(1+data_final.loc[data_final.location == j, "infected_g"][-12:-8].aggregate(func = "mean"))

    





# Upper Bound

data_final["total_infected_upper"] = np.NaN

data_final = data_final.sort_values(by = ['location', 'date'])

data_final = data_final.reset_index(drop = True)

for j in countries:

    for i in data_final["date"].unique()[0:-8]:

        data_final.loc[(data_final.date == i) & (data_final.location == j), "total_infected_upper"] = data_final.loc[(data_final.date == i + np.timedelta64(8, 'D')) & (data_final.location == j), "total_deaths"].iloc[0]/(data_final.loc[(data_final.date == i + np.timedelta64(8, 'D')) & (data_final.location == j), "CFR"].iloc[0]*0.7)

# Estimate growth rate of infected, g        

data_final['infected_g'] = np.log(data_final['total_infected_upper'])

data_final['infected_g'] = data_final['infected_g'].diff() 

# Estimate number of infected given g 

today = data_final.date.iloc[-1]

for j in countries:

    for i in range(7,-1,-1):

        data_final.loc[(data_final.location == j) & (data_final.date == today - timedelta(i)), "total_infected_upper"] = data_final.loc[data_final.location == j, "total_infected_upper"].iloc[-i-2]*(1+data_final.loc[data_final.location == j, "infected_g"][-12:-8].aggregate(func = "mean"))



# Lower Bound

data_final["total_infected_lower"] = np.NaN

data_final = data_final.sort_values(by = ['location', 'date'])

data_final = data_final.reset_index(drop = True)

for j in countries:

    for i in data_final["date"].unique()[0:-8]:

        data_final.loc[(data_final.date == i) & (data_final.location == j), "total_infected_lower"] = data_final.loc[(data_final.date == i + np.timedelta64(8, 'D')) & (data_final.location == j), "total_deaths"].iloc[0]/(data_final.loc[(data_final.date == i + np.timedelta64(8, 'D')) & (data_final.location == j), "CFR"].iloc[0]*1.3)

# Estimate growth rate of infected, g        

data_final['infected_g'] = np.log(data_final['total_infected_lower'])

data_final['infected_g'] = data_final['infected_g'].diff() 

# Estimate number of infected given g 

today = data_final.date.iloc[-1]

for j in countries:

    for i in range(7,-1,-1):

        data_final.loc[(data_final.location == j) & (data_final.date == today - timedelta(i)), "total_infected_lower"] = data_final.loc[data_final.location == j, "total_infected_lower"].iloc[-i-2]*(1+data_final.loc[data_final.location == j, "infected_g"][-12:-8].aggregate(func = "mean"))

data_final.loc[data_final.total_infected_lower < data_final.total_cases, "total_infected_lower"] = data_final.loc[data_final.total_infected_lower < data_final.total_cases, "total_cases"]





data_pc = data_final[['location', 'date', 'total_infected', 'total_infected_lower', 'total_infected_upper']].copy()



countries = ['India',"China", "Italy", "Spain", "France", "United Kingdom", "Germany", 

             "Portugal", "United States", "Singapore","South Korea", "Japan", 

             "Brazil","Iran"]

data_countries = []

data_countries_pc = []



for i in countries:

    data_pc.loc[data_pc.location == i,"total_infected"] = data_pc.loc[data_pc.location == i,"total_infected"]



# Get each country time series

filter1 = data_pc["total_infected"] > 1

for i in countries:

    filter_country = data_pc["location"]== i

    data_countries_pc.append(data_pc[filter_country & filter1])      
#hide_input

# Plot estimated absolute number of infected

plot1 = plot(data_countries_pc, "line", True)

#plot1.save("../images/covid-estimate-infections.png")

plot1
#hide_input    

label = 'Estimated Infected'

temp = pd.concat([x.copy() for x in data_countries_pc]).loc[lambda x: x.date >= '3/1/2020']



metric_name = f'{label}'

temp.columns = ['Country', 'Date', metric_name, "Lower Bound Estimates", "Upper Bound Estimates"]

temp.loc[:, "Estimated Infected"] = temp.loc[:, "Estimated Infected"].round(0).map('{:,.0f}'.format) 

temp.loc[:, "Lower Bound Estimates"] = temp.loc[:, "Lower Bound Estimates"].round(0).map('{:,.0f}'.format) 

temp.loc[:, "Upper Bound Estimates"] = temp.loc[:, "Upper Bound Estimates"].round(0).map('{:,.0f}'.format) 

temp.groupby('Country').last()
data_pc = data_final[['location', 'date', 'total_cases', 'total_infected']].copy()



countries = ['India',"China", "Italy", "Spain", "France", "United Kingdom", "Germany", 

             "Portugal", "United States", "Singapore","South Korea", "Japan", 

             "Brazil","Iran"]

data_countries = []

data_countries_pc = []



for i in countries:

    data_pc.loc[data_pc.location == i,"total_infected"] = data_pc.loc[data_pc.location == i,"total_infected"]

    data_pc.loc[data_pc.location == i,"total_cases"] = data_pc.loc[data_pc.location == i,"total_cases"]

    # get each country time series

filter1 = data_pc["total_infected"] > 1

for i in countries:

    filter_country = data_pc["location"]== i

    data_countries_pc.append(data_pc[filter_country & filter1])





plot(data_countries_pc, "scatter", True)
label1 = 'Observed Cases'

label2 = 'Estimated Infected'

temp = pd.concat([x.copy() for x in data_countries_pc]).loc[lambda x: x.date >= '3/1/2020']



metric_name1 = f'{label1}'

metric_name2 = f'{label2}'

temp.columns = ['Country', 'Date', metric_name1, metric_name2]

# temp.loc[:, 'month'] = temp.date.dt.strftime('%Y-%m')

temp.loc[:, "Observed Cases"] = temp.loc[:, "Observed Cases"].round(0).map('{:,.0f}'.format)

temp.loc[:, "Estimated Infected"] = temp.loc[:, "Estimated Infected"].round(0).map('{:,.0f}'.format)

temp.groupby('Country').last()