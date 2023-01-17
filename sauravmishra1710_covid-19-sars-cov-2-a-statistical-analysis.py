# import the necessary libraries



import numpy as np 

import pandas as pd

from datetime import date

import matplotlib.pyplot as plt

import seaborn as sns

from IPython.display import Markdown

import plotly.graph_objs as go

import plotly.offline as py

from plotly.subplots import make_subplots

import plotly.express as px

import pycountry

import folium 

from folium import plugins









%config InlineBackend.figure_format = 'retina'

py.init_notebook_mode(connected=True)





# Utility Functions



'''Display markdown formatted output like bold, italic bold etc.'''

def formatted_text(string):

    display(Markdown(string))
# Utility Plotting Functions



def plotDailyReportedCasesOverTime(df, country):

    # confirmed

    fig = px.bar(df, x="Date", y="Confirmed")

    layout = go.Layout(

        title=go.layout.Title(text="Daily count of confirmed cases in "+ country, x=0.5),

        font=dict(size=14),

        width=800,

        height=500,

        xaxis_title = "Date",

        yaxis_title = "Confirmed cases")



    fig.update_layout(layout)

    fig.show()



    # deaths

    fig = px.bar(df, x="Date", y="Deaths")

    layout = go.Layout(

        title=go.layout.Title(text="Daily count of reported deaths in "+ country, x=0.5),

        font=dict(size=14),

        width=800,

        height=500,

        xaxis_title = "Date",

        yaxis_title = "Deaths Reported")



    fig.update_layout(layout)

    fig.show()



    # recovered

    fig = px.bar(df, x="Date", y="Recovered")

    layout = go.Layout(

        title=go.layout.Title(text="Daily count of recovered cases in "+ country, x=0.5),

        font=dict(size=14),

        width=800,

        height=500,

        xaxis_title = "Date",

        yaxis_title = "Recovered Cases")



    fig.update_layout(layout)

    fig.show()

    

     # active

    fig = px.bar(df, x="Date", y="Active")

    layout = go.Layout(

        title=go.layout.Title(text="Daily count of active cases in "+ country, x=0.5),

        font=dict(size=14),

        width=800,

        height=500,

        xaxis_title = "Date",

        yaxis_title = "Active Cases")



    fig.update_layout(layout)

    fig.show()



    

    

# Cases over time

def scatterPlotCasesOverTime(df, country):

    plot = make_subplots(rows=2, cols=2, subplot_titles=("Comfirmed", "Deaths", "Recovered", "Active"))



    subPlot1 = go.Scatter(

                    x=df['Date'],

                    y=df['Confirmed'],

                    name="Confirmed",

                    line_color='orange',

                    opacity=0.8)



    subPlot2 = go.Scatter(

                    x=df['Date'],

                    y=df['Deaths'],

                    name="Deaths",

                    line_color='red',

                    opacity=0.8)



    subPlot3 = go.Scatter(

                    x=df['Date'],

                    y=df['Recovered'],

                    name="Recovered",

                    line_color='green',

                    opacity=0.8)

    

    subPlot4 = go.Scatter(

                    x=df['Date'],

                    y=df['Active'],

                    name="Active",

                    line_color='blue',

                    opacity=0.8)



    plot.append_trace(subPlot1, 1, 1)

    plot.append_trace(subPlot2, 1, 2)

    plot.append_trace(subPlot3, 2, 1)

    plot.append_trace(subPlot4, 2, 2)

    plot.update_layout(template="ggplot2", title_text = country + '<b> - Spread of the nCov Over Time</b>')



    plot.show()
# Import the data

nCoV_data = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv")



# Data Glimpse

nCoV_data.head()
# Data Info

nCoV_data.info()
# Convert 'Last Update' column to datetime object

nCoV_data['Last Update'] = nCoV_data['Last Update'].apply(pd.to_datetime)

nCoV_data['ObservationDate'] = nCoV_data['ObservationDate'].apply(pd.to_datetime)



# Also drop the 'SNo' and the 'Date' columns

nCoV_data.drop(['SNo'], axis=1, inplace=True)



# Fill the missing values in 'Province/State' with the 'Country' name.

nCoV_data['Province/State'] = nCoV_data['Province/State'].replace(np.nan, nCoV_data['Country/Region'])



# Data Glimpse

nCoV_data.head()
# Active Case = confirmed - deaths - recovered

nCoV_data['Active'] = nCoV_data['Confirmed'] - nCoV_data['Deaths'] - nCoV_data['Recovered']
# Check the Data Info again

nCoV_data.info()
# Lets rename the columns - 'Province/State' and 'Last Update' to remove the '/' and space respectively.

nCoV_data.rename(columns={'Last Update': 'LastUpdate', 'Province/State': 'State', 'Country/Region': 'Country', 'ObservationDate': 'Date'}, inplace=True)



# Data Glimpse

nCoV_data.head()
# Lets check the total #Countries affected by nCoV



nCoV_Countries = nCoV_data['Country'].unique().tolist()

print('\n')

print(nCoV_Countries)

print("\n------------------------------------------------------------------")

print("\nTotal countries affected by nCoV: ",len(nCoV_Countries))
# Convert 'Mainland China' to 'China'

nCoV_data['Country'] = np.where(nCoV_data['Country'] == 'Mainland China', 'China', nCoV_data['Country'])



# Check the # countries again

nCoV_Countries = nCoV_data['Country'].unique().tolist()

print('\n')

print(nCoV_Countries)

print("\n------------------------------------------------------------------")

print("\nTotal countries affected by nCoV: ",len(nCoV_Countries))
nCoV_data['Day'] = nCoV_data['LastUpdate'].apply(lambda x:x.day)

nCoV_data['Hour'] = nCoV_data['LastUpdate'].apply(lambda x:x.hour)



# Data Glimpse

nCoV_data.head()
formatted_text('***Confirmed Cases vs Day***')

plt.figure(figsize=(16,6))

sns.barplot(x='Day',y='Confirmed',data=nCoV_data, order=nCoV_data.Day.unique().tolist())

plt.title('Distribution of total confirmed cases on every day basis starting from 22nd Jan')
formatted_text('***Death Toll vs Day***')

plt.figure(figsize=(16,6))

sns.barplot(x='Day',y='Deaths',data=nCoV_data, order=nCoV_data.Day.unique().tolist())

plt.title('Distribution of total death toll on every day basis starting from 22nd Jan')
formatted_text('***Recovered Cases vs Day***')

plt.figure(figsize=(16,6))

sns.barplot(x='Day',y='Recovered',data=nCoV_data, order=nCoV_data.Day.unique().tolist())

plt.title('Distribution of total recovered cases on every day basis starting from 22nd Jan')
# Make the latest data extraction generic. As the data is getting updated on a daily (hourly) basis, 

# the below code would work without needing to be updated to extract the latest data.

# We will here extract the year, month and day from the last reported case and use it.

strDate = nCoV_data['Date'][-1:].astype('str')

year = int(strDate.values[0].split('-')[0])

month = int(strDate.values[0].split('-')[1])

day = int(strDate.values[0].split('-')[2].split()[0])



formatted_text('***Last reported case date-time***')

print(strDate)

print(year)

print(month)

print(strDate.values[0].split('-')[2].split())

print(pd.Timestamp(date(year,month,day)).date())
latest_nCoV_data = nCoV_data[nCoV_data['LastUpdate'] > pd.Timestamp(date(year,month,day)).date()]



# Data Glimpse

latest_nCoV_data.tail()
# Getting the latest numbers



formatted_text('***Latest Numbers Globaly***')

print('Confirmed Cases around the globe : ',latest_nCoV_data['Confirmed'].sum())

print('Deaths Confirmed around the globe: ',latest_nCoV_data['Deaths'].sum())

print('Recovered Cases around the globe : ',latest_nCoV_data['Recovered'].sum())

print('Total Active Cases around the globe : ',latest_nCoV_data['Active'].sum())
formatted_text('***Countries Affected WorldWide as per the current date -***')

allCountries = latest_nCoV_data['Country'].unique().tolist()

print(allCountries)



print("\nTotal countries affected by virus: ",len(allCountries))
CountryWiseData = pd.DataFrame(latest_nCoV_data.groupby('Country')['Confirmed', 'Deaths', 'Recovered', 'Active'].sum())

CountryWiseData['Country'] = CountryWiseData.index

CountryWiseData.index = np.arange(1, len(allCountries)+1)



CountryWiseData = CountryWiseData[['Country','Confirmed', 'Deaths', 'Recovered', 'Active']]



formatted_text('***Country wise Analysis of ''Confirmed'', ''Deaths'', ''Recovered'', ''Active'' Cases***')

CountryWiseData
date_wise_data = nCoV_data[["Date","Confirmed","Deaths","Recovered", "Active"]]

date_wise_data.head()
date_wise_data = date_wise_data.groupby(["Date"]).sum().reset_index()



# strip off the time part from date for day-wise distribution

date_wise_data.Date = date_wise_data.Date.apply(lambda x:x.date())



formatted_text('***Day wise distribution (WorldWide) for Confirmed, Deaths and Recovered Cases***')

date_wise_data
formatted_text('***Day wise distribution (WorldWide) for Confirmed, Deaths and Recovered Cases***')

date_wise_data.plot('Date',['Confirmed', 'Deaths', 'Recovered', 'Active'],figsize=(10,10), rot=30)
global_data_over_time = date_wise_data.groupby('Date')['Confirmed','Deaths','Recovered', 'Active'].sum().reset_index()



scatterPlotCasesOverTime(global_data_over_time, "<b>Global</b>")
plotDailyReportedCasesOverTime(global_data_over_time, "all over World")
china_latest_data = latest_nCoV_data[latest_nCoV_data['Country']=='China'][["State","Confirmed","Deaths","Recovered", "Active"]]



# Reset Index

china_latest_data.reset_index(drop=True, inplace=True)

china_latest_data.index = pd.RangeIndex(start=1, stop=len(china_latest_data['State']) + 1, step=1)



formatted_text('***Numbers in China for Confirmed, Deaths, Recovered and Active Cases***')



# Data Glimpse

china_latest_data
china_latest_data.plot('State',['Confirmed', 'Deaths', 'Recovered', 'Active'],kind='bar',figsize=(20,15), fontsize=15)
Hubei = china_latest_data[china_latest_data.State=='Hubei']

Hubei = Hubei[['Confirmed','Deaths','Recovered', 'Active']] # Remove the state column as it does not have any numeric data

Hubei = Hubei.iloc[0]

#Hubei



plt.figure(figsize=(12,12))



Hubei.plot( kind='pie'

           , autopct='%1.1f%%'

           , shadow=True

           , startangle=10)



plt.title('nCov Distribution - Hubei',size=25)

plt.legend(loc = "upper right"

           , fontsize = 10

           , ncol = 1 

           , fancybox = True

           , framealpha = 0.80

           , shadow = True

           , borderpad = 1);
plot = make_subplots(rows=1, cols=3, subplot_titles=("Comfirmed", "Deaths", "Recovered", "Active"))



# to plot the spread over time, we would need the data distribution spread over time starting from 22nd Jan 2020

# so we will extract the sub-set from the original data.

hubei_data_over_time = nCoV_data[nCoV_data['State'] == 'Hubei']



scatterPlotCasesOverTime(hubei_data_over_time, "<b>Hubei</b>")
rest_of_China = china_latest_data[china_latest_data['State'] !='Hubei'][["State", "Confirmed","Deaths","Recovered", "Active"]]



# Reset Index to start from 1

rest_of_China.reset_index(drop=True, inplace=True)

rest_of_China.index = pd.RangeIndex(start=1, stop=len(rest_of_China['State'])+1, step=1)



formatted_text('***Numbers in rest of China for Confirmed, Deaths, Recovered and Active Cases***')



# Data Glimpse

rest_of_China
rest_of_China.plot('State',['Confirmed', 'Deaths', 'Recovered', 'Active'],kind='bar',figsize=(20,15), fontsize=15)
formatted_text('***Most number of Confirmed Cases Outside of Hubei***')

print(rest_of_China[rest_of_China['Confirmed'] > 500])
plot = make_subplots(rows=1, cols=3, subplot_titles=("Comfirmed", "Deaths", "Recovered", "Active"))



# to plot the spread over time, we would need the data distribution spread over time starting from 22nd Jan 2020

# so we will extract the sub-set from the original data.

chinese_data_over_time = nCoV_data[(nCoV_data['Country'] == 'China') & (nCoV_data['State'] != 'Hubei')]

chinese_data_over_time = chinese_data_over_time.groupby('Date')['Confirmed','Deaths','Recovered', 'Active'].sum().reset_index()



scatterPlotCasesOverTime(chinese_data_over_time, "<b>Rest of China</b>")
plotDailyReportedCasesOverTime(chinese_data_over_time, "Rest of China")
top10 = CountryWiseData.sort_values('Confirmed',ascending=False)[:10]

top10
fig, axs  = plt.subplots(1,4, figsize=(24, 6))



ax_x = top10['Country']



ax_y0 = top10['Confirmed']

ax_y1 = top10['Deaths']

ax_y2 = top10['Recovered']

ax_y3 = top10['Active']



axs[0].bar(ax_x, ax_y0)

axs[0].set_xlabel('Top 10 Country')

axs[0].set_ylabel('Confirmed Cases')

axs[0].title.set_text('Confirmed')



axs[1].bar(ax_x, ax_y1)

axs[1].set_xlabel('Top 10 Country')

axs[1].set_ylabel('Death Cases')

axs[1].title.set_text('Deaths')



axs[2].bar(ax_x, ax_y2)

axs[2].set_xlabel('Top 10 Country')

axs[2].set_ylabel('Recovered Cases')

axs[2].title.set_text('Recovered')



axs[3].bar(ax_x, ax_y2)

axs[3].set_xlabel('Top 10 Country')

axs[3].set_ylabel('Active Cases')

axs[3].title.set_text('Active')



for ax in axs:

    ax.tick_params('x', labelrotation=90)

    ax.grid(axis='both')



plt.subplots_adjust(hspace = 0.3)

plt.subplots_adjust(wspace = 0.5)

plt.subplots_adjust(top = 0.8)



fig.suptitle("Covid19 - Top 10 Countries", fontsize = 24)

plt.show()
rest_of_world = CountryWiseData[CountryWiseData['Country'] !='China'][["Country", "Confirmed","Deaths","Recovered", "Active"]]



# Reset Index

rest_of_world.reset_index(drop=True, inplace=True)

rest_of_world.index = pd.RangeIndex(start=1, stop=len(CountryWiseData['Country']), step=1)



formatted_text('***Numbers in rest of world for Confirmed, Deaths and Recovered Cases***')



# Data Glimpse

rest_of_world
formatted_text('***Most number of Confirmed Cases Outside of China***')

print(rest_of_world[rest_of_world['Confirmed'] > 20])
rest_of_world.plot('Country',['Confirmed', 'Deaths', 'Recovered', 'Active'],figsize=(25,12), fontsize=10)

tick_labels = rest_of_world['Country']

plt.xticks(range(0, len(rest_of_world.Country) + 1) , tick_labels, rotation=90)
plot = make_subplots(rows=1, cols=3, subplot_titles=("Comfirmed", "Deaths", "Recovered", "Active"))



# to plot the spread over time, we would need the data distribution spread over time starting from 22nd Jan 2020

# so we will extract the sub-set from the original data.

rest_of_world_over_time = nCoV_data[(nCoV_data['Country'] != 'China')]

rest_of_world_over_time = rest_of_world_over_time.groupby('Date')['Confirmed','Deaths','Recovered', 'Active'].sum().reset_index()



scatterPlotCasesOverTime(rest_of_world_over_time, "<b>Rest of World</b>")
# Confirmed

fig = px.sunburst(latest_nCoV_data.sort_values(by='Confirmed', ascending=False).reset_index(drop=True), 

                 path=["Country", "State"], values="Confirmed", height=700,

                 title='Number of Confirmed cases reported',

                 color_discrete_sequence = px.colors.qualitative.Prism)

fig.data[0].textinfo = 'label+text+value'

fig.show()
# Recovered

fig = px.sunburst(latest_nCoV_data.sort_values(by='Recovered', ascending=False).reset_index(drop=True), 

                 path=["Country", "State"], values="Recovered", height=700,

                 title='Number of Recovered cases',

                 color_discrete_sequence = px.colors.qualitative.Prism)

fig.data[0].textinfo = 'label+text+value'

fig.show()
# Deaths

fig = px.sunburst(latest_nCoV_data.sort_values(by='Deaths', ascending=False).reset_index(drop=True), 

                 path=["Country", "State"], values="Deaths", height=700,

                 title='Number of Deaths reported',

                 color_discrete_sequence = px.colors.qualitative.Prism)

fig.data[0].textinfo = 'label+text+value'

fig.show()
covid_19_USA = nCoV_data[nCoV_data['Country'] == 'US']

covid_19_USA = covid_19_USA.groupby("Date")["Confirmed", "Deaths", "Recovered", "Active"].sum().reset_index()



# covid_19_USA = covid_19_USA[covid_19_USA['State'] != 'Grand Princess']

# covid_19_USA = covid_19_USA[covid_19_USA['State'] != 'Diamond Princess']

# #covid_19_USA = covid_19_USA[covid_19_USA['State'] != 'Guam']



formatted_text('***USA Numbers -***')



# Data Glimpse

covid_19_USA.head()
# cases over time - confirmed vs deaths

plotDailyReportedCasesOverTime(covid_19_USA, "USA")
# USA - Cases over time

scatterPlotCasesOverTime(covid_19_USA, "<b>USA</b>")
covid_19_ITALY = nCoV_data[nCoV_data['Country'] == 'Italy']

covid_19_ITALY = covid_19_ITALY.groupby("Date")["Confirmed", "Deaths", "Recovered", "Active"].sum().reset_index()



formatted_text('***ITALY Numbers -***')



# Data Glimpse

covid_19_ITALY.head()
# cases over time - confirmed vs deaths

plotDailyReportedCasesOverTime(covid_19_ITALY, "ITALY")
# ITALY - Cases over time

scatterPlotCasesOverTime(covid_19_ITALY, "<b>ITALY</b>")
covid_19_France = nCoV_data[nCoV_data['Country'] == 'France']

covid_19_France = covid_19_France.groupby("Date")["Confirmed", "Deaths", "Recovered", "Active"].sum().reset_index()



formatted_text('***FRANCE Numbers -***')



# Data Glimpse

covid_19_France.head()
# cases over time - confirmed vs deaths

plotDailyReportedCasesOverTime(covid_19_France, "FRANCE")
# FRANCE - Cases over time

scatterPlotCasesOverTime(covid_19_France, "<b>FRANCE</b>")
covid_19_Spain = nCoV_data[nCoV_data['Country'] == 'Spain']

covid_19_Spain = covid_19_Spain.groupby("Date")["Confirmed", "Deaths", "Recovered", "Active"].sum().reset_index()



formatted_text('***Spain Numbers -***')



# Data Glimpse

covid_19_Spain.head()
# cases over time - confirmed vs deaths

plotDailyReportedCasesOverTime(covid_19_Spain, "SPAIN")

# SPAIN - Cases over time

scatterPlotCasesOverTime(covid_19_Spain, "<b>SPAIN</b>")
covid_19_UK = nCoV_data[nCoV_data['Country'] == 'UK']

covid_19_UK = covid_19_UK.groupby("Date")["Confirmed", "Deaths", "Recovered", "Active"].sum().reset_index()



formatted_text('***UK Numbers -***')



# Data Glimpse

covid_19_UK.head()
# cases over time - confirmed vs deaths

plotDailyReportedCasesOverTime(covid_19_UK, "UK")

# UK - Cases over time

scatterPlotCasesOverTime(covid_19_UK, "<b>UK</b>")
covid_19_Iran = nCoV_data[nCoV_data['Country'] == 'Iran']

covid_19_Iran = covid_19_Iran.groupby("Date")["Confirmed", "Deaths", "Recovered", "Active"].sum().reset_index()



formatted_text('***IRAN Numbers -***')



# Data Glimpse

covid_19_Iran.head()
# cases over time - confirmed vs deaths

plotDailyReportedCasesOverTime(covid_19_Iran, "IRAN")
# IRAN - Cases over time

scatterPlotCasesOverTime(covid_19_Iran, "<b>IRAN</b>")
covid_19_SKorea = nCoV_data[nCoV_data['Country'] == 'South Korea']

covid_19_SKorea = covid_19_SKorea.groupby("Date")["Confirmed", "Deaths", "Recovered", "Active"].sum().reset_index()



formatted_text('***South Korea Numbers -***')



# Data Glimpse

covid_19_SKorea.head()
# cases over time - confirmed vs deaths

plotDailyReportedCasesOverTime(covid_19_SKorea, "South Korea")
# South Korea - Cases over time

scatterPlotCasesOverTime(covid_19_SKorea, "<b>South Korea</b>")
SKor_Covid_19 = pd.read_csv('/kaggle/input/coronavirusdataset/Case.csv')

#SKor_Covid_19.head()



### Convert 'from other city', '-' in cities to 'Others'

SKor_Covid_19['city'] = np.where(SKor_Covid_19['city'] == '-', 'Others', SKor_Covid_19['city'])

SKor_Covid_19['city'] = np.where(SKor_Covid_19['city'] == 'from other city', 'Others', SKor_Covid_19['city'])



SKor_Covid_19['latitude'] = np.where(SKor_Covid_19['latitude'] == '-', '37.00', SKor_Covid_19['latitude'])

SKor_Covid_19['longitude'] = np.where(SKor_Covid_19['longitude'] == '-', '127.30', SKor_Covid_19['longitude'])



SKor_Covid_19
SKorea_citywise_data = pd.DataFrame(SKor_Covid_19.groupby(['city'], as_index=False)['confirmed'].sum())

fig = px.bar(SKorea_citywise_data.sort_values('confirmed', ascending=False), 

             x="confirmed", y="city", title='Total Confirmed Cases', text='confirmed', orientation='h', 

             width=2000, height=700, range_x = [0, max(SKorea_citywise_data['confirmed']) + 2])

fig.update_traces(marker_color='#0726ed', opacity=0.8, textposition='outside')



fig.update_layout(plot_bgcolor='rgb(208, 236, 245)')

fig.show()
SKor_patient_info = pd.read_csv("../input/coronavirusdataset/PatientInfo.csv")

SKor_patient_info.head()
plt.figure(figsize=(12, 8))

sns.countplot(y = "infection_case",

              data=SKor_patient_info,

              order=list(SKor_patient_info["infection_case"].value_counts().index))

plt.title("Cause of infection", fontsize=16)

plt.xlabel("Count", fontsize=16)

plt.ylabel("Reason of infection", fontsize=16)

plt.xticks(fontsize=12)

plt.yticks(fontsize=12)

plt.show()
fig = go.Figure()

country_list = list(top10.Country)



for i in range(len(country_list)):

    country = country_list[i]

    country_df = nCoV_data[nCoV_data.Country == country]

    country_df = pd.DataFrame(country_df.groupby("Date")["Confirmed", "Deaths", "Recovered", "Active"].sum().reset_index())



    fig.add_trace(go.Scatter(x=country_df['Date'], y=country_df['Confirmed'],

                    mode='lines+markers',name=country))



fig.update_layout(title_text='Top 10 Worst Affected Country - Total Confirmed Cases',plot_bgcolor='rgb(225,230,255)')

fig.show()
fig = go.Figure()

country_list = list(top10.Country)



for i in range(len(country_list)):

    country = country_list[i]

    country_df = nCoV_data[nCoV_data.Country == country]

    country_df = pd.DataFrame(country_df.groupby("Date")["Confirmed", "Deaths", "Recovered", "Active"].sum().reset_index())



    fig.add_trace(go.Scatter(x=country_df['Date'], y=country_df['Deaths'],

                    mode='lines+markers',name=country))



fig.update_layout(title_text='Top 10 Worst Affected Country - Total Deaths Reported',plot_bgcolor='rgb(225,230,255)')

fig.show()
fig = go.Figure()

country_list = list(top10.Country)



for i in range(len(country_list)):

    country = country_list[i]

    country_df = nCoV_data[nCoV_data.Country == country]

    country_df = pd.DataFrame(country_df.groupby("Date")["Confirmed", "Deaths", "Recovered", "Active"].sum().reset_index())



    fig.add_trace(go.Scatter(x=country_df['Date'], y=country_df['Recovered'],

                    mode='lines+markers',name=country))



fig.update_layout(title_text='Top 10 Worst Affected Country - Total Recovered Cases',plot_bgcolor='rgb(225,230,255)')

fig.show()
fig = go.Figure()

country_list = list(top10.Country)



for i in range(len(country_list)):

    country = country_list[i]

    country_df = nCoV_data[nCoV_data.Country == country]

    country_df = pd.DataFrame(country_df.groupby("Date")["Confirmed", "Deaths", "Recovered", "Active"].sum().reset_index())



    fig.add_trace(go.Scatter(x=country_df['Date'], y=country_df['Active'],

                    mode='lines+markers',name=country))



fig.update_layout(title_text='Top 10 Worst Affected Country - Currrent Active Cases',plot_bgcolor='rgb(225,230,255)')

fig.show()