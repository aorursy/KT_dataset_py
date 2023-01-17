# Essential libraries
import numpy as np
import pandas as pd

# Data visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import squarify
# hide warnings
import warnings
warnings.filterwarnings('ignore')

pd.options.display.float_format
# Reading dataset
# importing dataset
githublink='https://raw.githubusercontent.com/ms4hafiz/COVID19/master/covid_19_complete.csv'
covid_tbl_original = pd.read_csv(githublink, parse_dates=['Date'])

covid_tbl_original.tail()
# Remove lat and lon since they will create duplicate for some countries
covid_tbl_original.drop(['Lat','Long'],axis=1,inplace=True)
covid_tbl_original.shape
# Cleaning country names
# Cleaning country names
# cases columns
cases= ['Confirmed', 'Deaths', 'Recovered', 'Active']

# Generate new column Active
# Active Case = confirmed - (deaths + recovered)
covid_tbl_original['Active'] = covid_tbl_original['Confirmed'] - covid_tbl_original['Deaths'] - covid_tbl_original['Recovered']

# Cleaning data
covid_tbl_original['Country/Region'] = covid_tbl_original['Country/Region'].replace('Mainland China', 'China')
covid_tbl_original['Country/Region'] = covid_tbl_original['Country/Region'].replace('Iran (Islamic Republic of)', 'Iran')
covid_tbl_original['Country/Region'] = covid_tbl_original['Country/Region'].replace('Dominican Republic', 'Dominica')
covid_tbl_original['Country/Region'] = covid_tbl_original['Country/Region'].replace('Timor-Leste', 'East Timor')
covid_tbl_original['Country/Region'] = covid_tbl_original['Country/Region'].replace('Russian Federation', 'Russia')
covid_tbl_original['Country/Region'] = covid_tbl_original['Country/Region'].replace('Viet Nam', 'Vietnam')
covid_tbl_original['Country/Region'] = covid_tbl_original['Country/Region'].replace('Congo (Kinshasa)', 'Republic of the Congo')
covid_tbl_original['Country/Region'] = covid_tbl_original['Country/Region'].replace('occupied Palestinian territory', 'Palestine')
covid_tbl_original['Country/Region'] = covid_tbl_original['Country/Region'].replace('United Kingdom', 'UK')
covid_tbl_original['Country/Region'] = covid_tbl_original['Country/Region'].replace('West Bank and Gaza', 'Palestine')


# Rename Country/Region
covid_tbl_original.rename({"Country/Region":'Country'},axis=1,inplace=True)

# imputing missing values with 0s
covid_tbl_original[cases] = covid_tbl_original[cases].fillna(0)
covid_tbl_original['Closed']=covid_tbl_original.Deaths+covid_tbl_original.Recovered
covid_tbl_original.head()
# latest
world_latest = covid_tbl_original[covid_tbl_original['Date'] == max(covid_tbl_original['Date'])].reset_index()

# latest cumulative
world_latest_grouped = world_latest.groupby('Country')['Confirmed', 'Deaths', 'Recovered', 'Active','Closed'].sum().reset_index()

# latest cumulative
world_country_date_grouped = covid_tbl_original.groupby(['Country','Date'])['Confirmed', 'Deaths', 'Recovered', 'Active','Closed'].sum().reset_index()
world_latest=world_latest_grouped
# latest Date of reporting
max_date=pd.to_datetime(max(covid_tbl_original.Date))
max_date=max_date.strftime("%d-%b-%Y")
summary=pd.DataFrame({"Item":"Last Date Of Reporting","Number/Cases/Date":[max_date]})

# of countries
number_of_countries=len(world_latest_grouped.Country.drop_duplicates())
summryTemp=pd.DataFrame({"Item":"Total countries reported cases","Number/Cases/Date":[number_of_countries]})
summary=pd.concat([summary,summryTemp])

# total confirmed cases
total_confirmed=world_latest_grouped['Confirmed'].sum()
summryTemp=pd.DataFrame({"Item":"Total confirmed cases","Number/Cases/Date":[str(int(total_confirmed))]})
summary=pd.concat([summary,summryTemp])

# total recoveries
total_recoveries=world_latest_grouped['Recovered'].sum()
summryTemp=pd.DataFrame({"Item":"Total recovered cases","Number/Cases/Date":[total_recoveries]})
summary=pd.concat([summary,summryTemp])

# total deaths
total_deaths=world_latest_grouped['Deaths'].sum()
summryTemp=pd.DataFrame({"Item":"Total deaths","Number/Cases/Date":[total_deaths]})
summary=pd.concat([summary,summryTemp])

# total active cases
total_active=world_latest_grouped['Active'].sum()
summryTemp=pd.DataFrame({"Item":"Total active cases","Number/Cases/Date":[str(int(total_active))]})
summary=pd.concat([summary,summryTemp])

# total closed cases
total_closed=world_latest_grouped['Closed'].sum()
summryTemp=pd.DataFrame({"Item":"Total closed cases","Number/Cases/Date":[str(int(total_closed))]})
summary=pd.concat([summary,summryTemp])

# percentage of closed cases
percent_closed=100*total_closed/total_confirmed
summryTemp=pd.DataFrame({"Item":"Percentage of closed cases","Number/Cases/Date":[str(round(percent_closed,2))]})
summary=pd.concat([summary,summryTemp])

# percentage of deaths out of confirmed cases
percent_deaths=100*total_deaths/total_confirmed
summryTemp=pd.DataFrame({"Item":"Percentage of deaths out of confirmed cases","Number/Cases/Date":[str(round(percent_deaths,2))]})
summary=pd.concat([summary,summryTemp])

# percentage of deaths out of closed cases
percent_deaths_closed=100*total_deaths/total_closed
summryTemp=pd.DataFrame({"Item":"Percentage of deaths out of closed cases","Number/Cases/Date":[str(round(percent_deaths_closed,2))]})
summary=pd.concat([summary,summryTemp])

# percentage of recovered cases
percent_recovered=100*total_recoveries/total_confirmed
summryTemp=pd.DataFrame({"Item":"Percentage of recovered cases","Number/Cases/Date":[str(round(percent_recovered,2))]})
summary=pd.concat([summary,summryTemp])

# percentage of active cases
percent_active=100*total_active/total_confirmed
summryTemp=pd.DataFrame({"Item":"Percentage of active cases","Number/Cases/Date":[str(round(percent_active,2))]})
summary=pd.concat([summary,summryTemp])

summary=summary.set_index('Item')
summary.style.background_gradient(cmap='coolwarm')

world='World'
confirmed=world_latest_grouped['Confirmed'].sum()
deaths=world_latest_grouped['Deaths'].sum()
recovered=world_latest_grouped['Recovered'].sum()
active=world_latest_grouped['Active'].sum()
closed=world_latest_grouped['Closed'].sum()

world_latest_grouped_total=pd.DataFrame({"Country":[world],'Confirmed':[confirmed],'Deaths':[deaths],'Recovered':[recovered],'Active':[active],'Closed':[closed]})
world_latest_grouped1 = world_latest_grouped[['Country','Confirmed', 'Deaths', 'Recovered', 'Active','Closed']]
world_latest_grouped1=pd.concat([world_latest_grouped_total,world_latest_grouped1])[['Country','Confirmed', 'Deaths', 'Recovered', 'Closed','Active']]
world_latest_grouped1.set_index('Country',inplace=True)
world_latest_grouped1.sort_values(["Deaths"],ascending=False).style.background_gradient(cmap='coolwarm')
# New data from 

world_date_grouped = covid_tbl_original.groupby('Date')['Confirmed', 'Deaths', 'Recovered', 'Active','Closed'].sum().reset_index()

# Confirmed cases
world_date_grouped['DailyConfirmed']=world_date_grouped['Confirmed'].sub(world_date_grouped['Confirmed'].shift())
# First day total confirmed cases 

world_date_grouped.iloc[0,6]=555
world_date_grouped['DailyConfirmed']=world_date_grouped['DailyConfirmed'].astype(int)


# Deaths dailay
world_date_grouped['DailyDeaths']=world_date_grouped['Deaths'].sub(world_date_grouped['Deaths'].shift())

# First day total deaths  
world_date_grouped.iloc[0,7]=17
world_date_grouped['DailyDeaths']=world_date_grouped['DailyDeaths'].astype(int)

# Recovered dailay
world_date_grouped['DailyRecovered']=world_date_grouped['Recovered'].sub(world_date_grouped['Recovered'].shift())
# First day total deaths  
world_date_grouped.iloc[0,8]=28
world_date_grouped['DailyRecovered']=world_date_grouped['DailyRecovered'].astype(int)

# Active dailay
world_date_grouped['DailyActive']=world_date_grouped['Active'].sub(world_date_grouped['Active'].shift())
# First day total deaths  
world_date_grouped.iloc[0,9]=510
world_date_grouped['DailyActive']=world_date_grouped['DailyActive'].astype(int)

# Calculating daily Case Fatality Rate
world_date_grouped['CFR']=world_date_grouped['Deaths']/world_date_grouped['Confirmed']

# Calculating daily Case Recovery Rate
world_date_grouped['CRR']=world_date_grouped['Recovered']/world_date_grouped['Confirmed']

# Calculating daily Case Active Rate
world_date_grouped['CAR']=world_date_grouped['Active']/world_date_grouped['Confirmed']

# Calculating case fatality rate from closed cases
world_date_grouped['CCFR']=world_date_grouped['Deaths']/world_date_grouped['Closed']


# convert Date to Date
world_date_grouped['Date']=world_date_grouped['Date'].dt.date
covid_daily=world_date_grouped
fig = plt.figure(figsize=(20,7),dpi=100)
ax1 = fig.add_subplot(111)
chart=sns.barplot(x = covid_daily.Date, y = covid_daily.DailyConfirmed,color="grey")
chart.set_xticklabels(
    chart.get_xticklabels(), 
    rotation=90, 
    horizontalalignment='right',
    fontweight='light',
    fontsize='large'

)

plt.title("Figure 1: Daily reported confimed cases",fontsize=18, fontweight='bold')
plt.show()
fig = plt.figure(figsize=(20,7),dpi=100)
ax1 = fig.add_subplot(111)
chart=sns.barplot(x = covid_daily.Date, y = covid_daily.DailyDeaths,color="r")
chart.set_xticklabels(
    chart.get_xticklabels(), 
    rotation=90, 
    horizontalalignment='right',
    fontweight='light',
    fontsize='large'

)
ax2 = ax1.twinx()
plt.plot(covid_daily.CFR, color='r')
# ax3 = ax1.twinx()
# plt.plot(covid_daily.CRR, color='b')
ax2.grid(False)
plt.title("Figure 2: Daily reported deaths and case fatality rate (CFR) out of confirmed cases",fontsize=18, fontweight='bold')
plt.show(block=False)
fig = plt.figure(figsize=(20,7),dpi=100)
ax1 = fig.add_subplot(111)
chart=sns.barplot(x = covid_daily.Date, y = covid_daily.DailyDeaths,color="y")
chart.set_xticklabels(
    chart.get_xticklabels(), 
    rotation=90, 
    horizontalalignment='right',
    fontweight='light',
    fontsize='large'

)
ax2 = ax1.twinx()
plt.plot(covid_daily.CCFR, color='r')
# ax3 = ax1.twinx()
# plt.plot(covid_daily.CRR, color='b')
ax2.grid(False)

plt.title("Figure 3: Daily reported deaths and case fatality rate (CFR) out of closed cases",fontsize=18, fontweight='bold')
plt.show()
fig = plt.figure(figsize=(20,7),dpi=100)
ax1 = fig.add_subplot(111)
chart=sns.barplot(x = covid_daily.Date, y = covid_daily.DailyRecovered,color="g")
chart.set_xticklabels(
    chart.get_xticklabels(), 
    rotation=90, 
    horizontalalignment='right',
    fontweight='light',
)

plt.title("Figure 4: Daily reported recoveries and recovery rate",fontsize=18, fontweight='bold')
ax2 = ax1.twinx()
plt.plot(covid_daily.CRR, color='r')
ax2.grid(False)
plt.show()
# Top 30 countries with highest confirmed cases
world_latest=world_latest.sort_values(by='Confirmed',ascending=False)
countries_order_by_highest_confirmed_cases=world_latest
plt.figure(figsize=(15,8))
squarify.plot(sizes=world_latest['Confirmed'].head(30), label=world_latest['Country'].head(30))
plt.axis('off')
plt.title("Figure 5: Top 30 countries with highest confirmed cases",fontsize=18, fontweight='bold')
plt.show(block=False)

# Top 30 countries with highest deaths
world_latest=world_latest.sort_values(by='Deaths',ascending=False)

plt.figure(figsize=(15,8))
squarify.plot(sizes=world_latest['Deaths'].head(30), label=world_latest['Country'].head(30))
plt.axis('off')
plt.title("Figure 6: Top 30 countries with highest deaths report",fontsize=18, fontweight='bold')
plt.show(block=False)

# Top 30 countries with highest active cases
world_latest=world_latest.sort_values(by='Active',ascending=False)

plt.figure(figsize=(15,8))
squarify.plot(sizes=world_latest['Active'].head(30), label=world_latest['Country'].head(30))
plt.axis('off')
plt.title("Figure 7: Top 30 countries with highest active cases",fontsize=18, fontweight='bold')
plt.show(block=False)

# Top 30 countries with highest cloased cases

world_latest=world_latest.sort_values(by='Closed',ascending=False)

plt.figure(figsize=(15,8))
squarify.plot(sizes=world_latest['Closed'].head(30), label=world_latest['Country'].head(30))
plt.axis('off')
plt.title("Figure 8: Top 30 countries with highest closed cases",fontsize=18, fontweight='bold')
plt.show(block=False)

# Top 30 countries with highest cloased cases

world_latest['percent_of_active_cases']=100*world_latest['Active'].divide(world_latest['Confirmed'])
world_latest=world_latest.sort_values(by='percent_of_active_cases',ascending=True)
world_latest1=world_latest[['Country','percent_of_active_cases']].head(30)
world_latest1.sort_values(["percent_of_active_cases"],ascending=True).style.background_gradient(cmap='coolwarm')
# Top 30 countries with highest fatality rate out of confirmed cases

# world_latest['percent_of_deaths']=100*world_latest['Deaths']/world_latest['Confirmed']
world_latest['percent_of_deaths']=100*world_latest['Deaths'].divide(world_latest['Confirmed'])

world_latest=world_latest.sort_values(by='percent_of_deaths',ascending=False)

plt.figure(figsize=(15,8))
squarify.plot(sizes=world_latest['percent_of_deaths'].head(30), label=world_latest['Country'].head(30))
plt.axis('off')
plt.title("Figure 10: Top 30 countries with case fatality rate out of confirmed cases",fontsize=18, fontweight='bold')
plt.show(block=False)
                                                         
# Top 30 countries with highest fatality rate out of closed cases

world_latest['percent_of_deaths_cls']=100*world_latest['Deaths']/world_latest['Closed']
world_latest=world_latest.sort_values(by='percent_of_deaths_cls',ascending=False)

plt.figure(figsize=(15,8))
squarify.plot(sizes=world_latest['percent_of_deaths_cls'].head(30), label=world_latest['Country'].head(30))
plt.axis('off')
plt.title("Figure 11: Top 30 countries with case fatality rate out of closed cases",fontsize=18, fontweight='bold')
plt.show(block=False)
                                                         
country_level=world_country_date_grouped[world_country_date_grouped['Deaths']>=0]
country_level['CFR']=world_country_date_grouped['Deaths']/world_country_date_grouped['Confirmed']
country_level['CRR']=world_country_date_grouped['Recovered']/world_country_date_grouped['Confirmed']
country_level['CAR']=world_country_date_grouped['Active']/world_country_date_grouped['Confirmed']
country_level['CCFR']=world_country_date_grouped['Deaths']/world_country_date_grouped['Closed']
country_level['Date']=country_level['Date'].dt.date

# fill name with 0 
country_level.fillna(0)

# converting column to rows for easy charting and processing

# Deaths
country_level_deaths= country_level[['Country','Date','Deaths']]
country_level_deaths['Type']='Numer of deaths'
country_level_deaths=country_level_deaths.rename({'Deaths':'Cases'},axis=1)

# # Confirmed cases
# country_level_confirmed= country_level[['Country','Date','Confirmed']]
# country_level_confirmed['Type']='Confirmed'
# country_level_confirmed=country_level_confirmed.rename({'Confirmed':'Cases'},axis=1)


# Recovered cases
country_level_recovered= country_level[['Country','Date','Recovered']]
country_level_recovered['Type']='Number of recoveries'
country_level_recovered=country_level_recovered.rename({'Recovered':'Cases'},axis=1)


# Active cases
country_level_Active= country_level[['Country','Date','Active']]
country_level_Active['Type']='Number of active cases'
country_level_Active=country_level_Active.rename({'Active':'Cases'},axis=1)

# converting column to rows for easy charting and processing (for percentage)

# Case Fatality Rate from confirmed
country_level_CFR= country_level[['Country','Date','CFR']]
country_level_CFR['Type']='Case Fatality Rate/Confirmed'
country_level_CFR=country_level_CFR.rename({'CFR':'Percentage'},axis=1)

# Case Fatality Rate from closed
country_level_CCFR= country_level[['Country','Date','CCFR']]
country_level_CCFR['Type']='Case Fatality Rate/Closed'
country_level_CCFR=country_level_CCFR.rename({'CCFR':'Percentage'},axis=1)


# Recovered cases rate
country_level_CRR= country_level[['Country','Date','CRR']]
country_level_CRR['Type']='Recovered Cases Rate'
country_level_CRR=country_level_CRR.rename({'CRR':'Percentage'},axis=1)


# Active cases rate
country_level_CAR= country_level[['Country','Date','CAR']]
country_level_CAR['Type']='Active Cases Rate'
country_level_CAR=country_level_CAR.rename({'CAR':'Percentage'},axis=1)

country_level_cases=pd.concat([country_level_deaths,country_level_Active,country_level_recovered])

country_level_percent=pd.concat([country_level_CFR,country_level_CCFR,country_level_CRR,country_level_CAR])

# Impute null values to zero for countries where percentage is null
country_level_percent=country_level_percent.fillna(0)

# Getting list of countries
countries_list = list(countries_order_by_highest_confirmed_cases.Country.values)[:60]

# Setting subplosts
f, axes = plt.subplots(len(countries_list),2,figsize=(22,len(countries_list)*5),dpi=100) 


for i,j in zip(countries_list,range(0,len(countries_list))):
    # Charts for column 1
    palette = sns.color_palette("magma", 3)
    chart = sns.lineplot(x="Date", y="Cases",
                  hue="Type",style='Type',
                  palette=palette, data=country_level_cases[country_level_cases['Country']==i],ax=axes[j][0])
    chart.set_xticklabels(
        chart.get_xticklabels(90), 
        rotation=90, 
        minor=True,
        verticalalignment=True,
        horizontalalignment='right',
        fontweight='light',
        fontsize='large'
    )
    chart.set_title("Covid-19 reporte cases in "+i,fontsize=14,fontweight='bold')
    
    # Charts for column 1
    palette2 = sns.color_palette("magma", 4)
    chart2 = sns.lineplot(x="Date", y="Percentage",
                  hue="Type",style='Type',
                  palette=palette2, data=country_level_percent[country_level_percent['Country']==i],ax=axes[j][1])
    chart2.set_xticklabels(
        chart.get_xticklabels(), 
        rotation=90, 
        minor=True,
        verticalalignment=True,
        horizontalalignment='right',
        fontweight='light',
        fontsize='large'
    )
    chart2.set_title("Covid-19 percentages of cases in "+i,fontsize=14,fontweight='bold')

plt.tight_layout()
plt.legend(loc='best')
plt.show(block=False)
# Getting list of countries
countries_list = list(countries_order_by_highest_confirmed_cases.Country.values)[60:120]

# Setting subplosts
f, axes = plt.subplots(len(countries_list),2,figsize=(22,len(countries_list)*5),dpi=200) 


for i,j in zip(countries_list,range(0,len(countries_list))):
    # Charts for column 1
    palette = sns.color_palette("magma", 3)
    chart = sns.lineplot(x="Date", y="Cases",
                  hue="Type",style='Type',
                  palette=palette, data=country_level_cases[country_level_cases['Country']==i],ax=axes[j][0])
    chart.set_xticklabels(
        chart.get_xticklabels(90), 
        rotation=90, 
        minor=True,
        verticalalignment=True,
        horizontalalignment='right',
        fontweight='light',
        fontsize='large'
    )
    chart.set_title("Covid-19 reporte cases in "+i,fontsize=14,fontweight='bold')
    
    # Charts for column 1
    palette2 = sns.color_palette("magma", 4)
    chart2 = sns.lineplot(x="Date", y="Percentage",
                  hue="Type",style='Type',
                  palette=palette2, data=country_level_percent[country_level_percent['Country']==i],ax=axes[j][1])
    chart2.set_xticklabels(
        chart.get_xticklabels(), 
        rotation=90, 
        minor=True,
        verticalalignment=True,
        horizontalalignment='right',
        fontweight='light',
        fontsize='large'
    )
    chart2.set_title("Covid-19 percentages of cases in "+i,fontsize=14,fontweight='bold')

plt.tight_layout()
plt.legend(loc='best')
plt.show(block=False)
# Getting list of countries
countries_list = list(countries_order_by_highest_confirmed_cases.Country.values)[120:180]

# Setting subplosts
f, axes = plt.subplots(len(countries_list),2,figsize=(22,len(countries_list)*5),dpi=200) 


for i,j in zip(countries_list,range(0,len(countries_list))):
    # Charts for column 1
    palette = sns.color_palette("magma", 3)
    chart = sns.lineplot(x="Date", y="Cases",
                  hue="Type",style='Type',
                  palette=palette, data=country_level_cases[country_level_cases['Country']==i],ax=axes[j][0])
    chart.set_xticklabels(
        chart.get_xticklabels(90), 
        rotation=90, 
        minor=True,
        verticalalignment=True,
        horizontalalignment='right',
        fontweight='light',
        fontsize='large'
    )
    chart.set_title("Covid-19 reporte cases in "+i,fontsize=14,fontweight='bold')
    
    # Charts for column 1
    palette2 = sns.color_palette("magma", 4)
    chart2 = sns.lineplot(x="Date", y="Percentage",
                  hue="Type",style='Type',
                  palette=palette2, data=country_level_percent[country_level_percent['Country']==i],ax=axes[j][1])
    chart2.set_xticklabels(
        chart.get_xticklabels(), 
        rotation=90, 
        minor=True,
        verticalalignment=True,
        horizontalalignment='right',
        fontweight='light',
        fontsize='large'
    )
    chart2.set_title("Covid-19 percentages of cases in "+i,fontsize=14,fontweight='bold')

plt.tight_layout()
plt.legend(loc='best')
plt.show(block=False)
# countries_order_by_highest_confirmed_cases.shape
# Getting list of countries
countries_list = list(countries_order_by_highest_confirmed_cases.Country.values)[180:]

# Setting subplosts
f, axes = plt.subplots(len(countries_list),2,figsize=(22,len(countries_list)*5),dpi=200) 


for i,j in zip(countries_list,range(0,len(countries_list))):
    # Charts for column 1
    palette = sns.color_palette("magma", 3)
    chart = sns.lineplot(x="Date", y="Cases",
                  hue="Type",style='Type',
                  palette=palette, data=country_level_cases[country_level_cases['Country']==i],ax=axes[j][0])
    chart.set_xticklabels(
        chart.get_xticklabels(90), 
        rotation=90, 
        minor=True,
        verticalalignment=True,
        horizontalalignment='right',
        fontweight='light',
        fontsize='large'
    )
    chart.set_title("Covid-19 reporte cases in "+i,fontsize=14,fontweight='bold')
    
    # Charts for column 1
    palette2 = sns.color_palette("magma", 4)
    chart2 = sns.lineplot(x="Date", y="Percentage",
                  hue="Type",style='Type',
                  palette=palette2, data=country_level_percent[country_level_percent['Country']==i],ax=axes[j][1])
    chart2.set_xticklabels(
        chart.get_xticklabels(), 
        rotation=90, 
        minor=True,
        verticalalignment=True,
        horizontalalignment='right',
        fontweight='light',
        fontsize='large'
    )
    chart2.set_title("Covid-19 percentages of cases in "+i,fontsize=14,fontweight='bold')

plt.tight_layout()
plt.legend(loc='best')
plt.show(block=False)