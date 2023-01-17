import pandas as pd
import os
file='../input/novel-corona-virus-2019-dataset/covid_19_data.csv'
covid = pd.read_csv(file, parse_dates = ['ObservationDate','Last Update'])
covid=covid[['ObservationDate','Province/State','Country/Region','Confirmed','Deaths','Recovered']].rename(columns={'ObservationDate':'Date'})

covid[['Province/State']]=covid[['Province/State']].fillna('')
covid['Active']=covid['Confirmed']-covid['Deaths']-covid['Recovered']
covid[['Confirmed','Deaths','Recovered','Active']]=covid[['Confirmed','Deaths','Recovered','Active']].fillna(0)
covid['Country/Region']=covid['Country/Region'].str.replace('Mainland China','China')

covid.loc[covid['Province/State']=='Macau','Country/Region']='Macau'
covid.loc[covid['Province/State']=='Hong Kong','Country/Region']='Hong Kong'
covid
latest=covid.groupby(['Country/Region','Province/State'])[['Confirmed', 'Deaths', 'Recovered', 'Active','Date']].max().sort_values(by='Confirmed',ascending=False).sort_values('Confirmed',ascending=False)
temp=latest.groupby(['Date']).sum().reset_index()
temp=temp[temp['Date']==max(temp['Date'])].reset_index(drop=True)
temp
import plotly.express as px

t=temp.melt(id_vars="Date", value_vars=['Active', 'Deaths', 'Recovered'],var_name='Type',value_name='number')
fig=px.treemap(t,path=['Type'],values='number')
fig
latest=covid.groupby(['Country/Region','Province/State'])[['Confirmed', 'Deaths', 'Recovered', 'Active','Date']].max().sort_values(by='Confirmed',ascending=False).sort_values('Confirmed',ascending=False)
latest.style.background_gradient(cmap='Blues')
covid_country=covid.groupby(['Date','Country/Region']).sum().loc[covid.Date.max()].sort_values('Confirmed',ascending=False)[['Confirmed','Deaths','Recovered','Active']].reset_index()
covid_country.style.background_gradient(cmap='Blues')
covid_country[['Country/Region','Deaths']].sort_values('Deaths',ascending=False).reset_index(drop=True).style.background_gradient(cmap='Reds')
covid_country[covid_country['Recovered']==0].sort_values('Confirmed',ascending=False).reset_index(drop=True).style.background_gradient(cmap='Greens')
covid_country[covid_country['Confirmed']==covid_country['Deaths']]
covid_country[covid_country['Confirmed']==covid_country['Recovered']]
covid_country[covid_country['Confirmed']==covid_country['Recovered']+covid_country['Deaths']]
china=latest.loc['China'].reset_index()[['Province/State','Confirmed','Deaths','Recovered','Active']].sort_values('Confirmed',ascending=False)
china.style.background_gradient(cmap='Blues')
china[china['Recovered']==0]
china[china['Confirmed']==china['Deaths']]
china[china['Confirmed']==china['Recovered']].style.background_gradient(cmap='Greens')
china[china['Confirmed']==china['Recovered']+china['Deaths']].style.background_gradient(cmap='Greens')
import plotly.express as px
import numpy as np

covid_log_country=covid_country.copy()
covid_log_country.loc[covid_log_country["Confirmed"]==0,'Confirmed']=.0001


fig = px.choropleth(covid_log_country, locations="Country/Region", 
                    locationmode='country names', color=np.log(covid_log_country["Confirmed"]), 
                    hover_name="Country/Region", hover_data=['Confirmed'],
                    color_continuous_scale="Blues", 
                    title='Countries with Confirmed Cases')
fig.update(layout_coloraxis_showscale=True)
fig.show()
formated_gdf=covid.groupby(['Date','Country/Region'])[['Confirmed','Deaths','Recovered','Active']].sum()

formated_gdf = formated_gdf.reset_index()
formated_gdf['Date'] = pd.to_datetime(formated_gdf['Date'])
formated_gdf['Date'] = formated_gdf['Date'].dt.strftime('%m/%d/%Y')
formated_gdf['size'] = formated_gdf['Confirmed'].pow(0.3)

fig = px.scatter_geo(formated_gdf, locations="Country/Region", locationmode='country names', 
                     color="Confirmed", size='size', hover_name="Country/Region", 
                     range_color= [0, max(formated_gdf['Confirmed'])+2], 
                     projection="natural earth", animation_frame="Date", 
                     title='Spread over time')
fig.update(layout_coloraxis_showscale=False)
fig.show()
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
months = mdates.MonthLocator()
months_fmt = mdates.DateFormatter('%b-%e')

temp=covid.groupby('Date').sum()

fig,ax=plt.subplots(figsize=(11,6))
ax.plot(temp)

plt.title('World COVID-19 Historical Trends',fontdict={'fontsize':18})
ax.legend(temp.columns, loc=2, fontsize=12)
ax.xaxis.set_major_locator(plt.MaxNLocator(12))
ax.xaxis.set_major_formatter(months_fmt)
plt.xticks(rotation=45, fontsize=12)
plt.show()
top5=list(covid_country['Country/Region'].head())
top5.extend(['South Korea'])

temp=covid.groupby(['Country/Region','Date'])[['Confirmed','Deaths','Recovered','Active']].sum().loc[top5].unstack(level=0)
top5
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
months = mdates.MonthLocator()
months_fmt = mdates.DateFormatter('%b-%e')

confirm=temp.Confirmed

fig,ax=plt.subplots(figsize=(11,6))
ax.plot(confirm)

plt.title('Confirmed COVID-19 Cases',fontdict={'fontsize':18})
ax.legend(confirm.columns, loc=2, fontsize=12)
ax.xaxis.set_major_locator(plt.MaxNLocator(12))
ax.xaxis.set_major_formatter(months_fmt)
plt.xticks(rotation=45, fontsize=12)
plt.show()
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
months = mdates.MonthLocator()
months_fmt = mdates.DateFormatter('%b-%e')

death=temp.Deaths

fig,ax=plt.subplots(figsize=(11,6))
ax.plot(death)

plt.title('Deaths from COVID-19',fontdict={'fontsize':18})
ax.legend(death.columns, loc=2, fontsize=12)
ax.xaxis.set_major_locator(plt.MaxNLocator(12))
ax.xaxis.set_major_formatter(months_fmt)
plt.xticks(rotation=45, fontsize=12)
plt.show()
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
months = mdates.MonthLocator()
months_fmt = mdates.DateFormatter('%b-%e')

recover=temp.Recovered

fig,ax=plt.subplots(figsize=(11,6))
ax.plot(recover)

plt.title('Recoveries from COVID-19',fontdict={'fontsize':18})
ax.legend(recover.columns, loc=2, fontsize=12)
ax.xaxis.set_major_locator(plt.MaxNLocator(12))
ax.xaxis.set_major_formatter(months_fmt)
plt.xticks(rotation=45, fontsize=12)
plt.show()
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
months = mdates.MonthLocator()
months_fmt = mdates.DateFormatter('%b-%e')

active=temp.Active

fig,ax=plt.subplots(figsize=(11,6))
ax.plot(active)

plt.title('Active Cases of COVID-19',fontdict={'fontsize':18})
ax.legend(active.columns, loc=2, fontsize=12)
ax.xaxis.set_major_locator(plt.MaxNLocator(12))
ax.xaxis.set_major_formatter(months_fmt)
plt.xticks(rotation=45, fontsize=12)
plt.show()
top5=list(covid_country['Country/Region'].head())
top5.extend(['South Korea'])
temp=covid.groupby(['Country/Region','Date']).sum()[['Confirmed','Deaths','Recovered','Active']].reset_index()
temp['Day to Day']=temp.groupby(['Country/Region'])['Confirmed'].shift()
temp['Day to Day']=temp['Confirmed']-temp['Day to Day']
temp=temp.loc[temp['Country/Region'].isin(top5)].groupby(['Country/Region','Date']).sum()['Day to Day'].unstack(level=0)
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
months = mdates.MonthLocator()
months_fmt = mdates.DateFormatter('%b-%e')

confirm=temp

fig,ax=plt.subplots(figsize=(11,6))
ax.plot(confirm)

plt.title('Day to Day Change in Confirmed Cases',fontdict={'fontsize':18})
ax.legend(confirm.columns, loc=2, fontsize=12)
ax.xaxis.set_major_locator(plt.MaxNLocator(12))
ax.xaxis.set_major_formatter(months_fmt)
plt.xticks(rotation=45, fontsize=12)
plt.show()
top5=list(covid_country['Country/Region'].head())
top5.extend(['South Korea'])

temp=covid.groupby(['Country/Region','Date'])[['Confirmed','Deaths','Recovered','Active']].sum().reset_index()
temp['Day to Day']=temp.groupby(['Country/Region'])['Deaths'].shift()
temp['Day to Day']=temp['Deaths']-temp['Day to Day']
temp=temp.loc[temp['Country/Region'].isin(top5)].groupby(['Country/Region','Date']).sum()['Day to Day'].unstack(level=0)

months = mdates.MonthLocator()
months_fmt = mdates.DateFormatter('%b-%e')

death=temp

fig,ax=plt.subplots(figsize=(11,6))
ax.plot(death)

plt.title('Day to Day Change in Deaths',fontdict={'fontsize':18})
ax.legend(death.columns, loc=2, fontsize=12)
ax.xaxis.set_major_locator(plt.MaxNLocator(12))
ax.xaxis.set_major_formatter(months_fmt)
plt.xticks(rotation=45, fontsize=12)
plt.show()
top5=list(covid_country['Country/Region'].head())
top5.extend(['South Korea'])

temp=covid.groupby(['Country/Region','Date'])[['Confirmed','Deaths','Recovered','Active']].sum().reset_index()
temp['Day to Day']=temp.groupby(['Country/Region'])['Recovered'].shift()
temp['Day to Day']=temp['Recovered']-temp['Day to Day']
temp=temp.loc[temp['Country/Region'].isin(top5)].groupby(['Country/Region','Date']).sum()['Day to Day'].unstack(level=0)

months = mdates.MonthLocator()
months_fmt = mdates.DateFormatter('%b-%e')

recover=temp

fig,ax=plt.subplots(figsize=(11,6))
ax.plot(recover)

plt.title('Day to Day Change in Recovered',fontdict={'fontsize':18})
ax.legend(recover.columns, loc=2, fontsize=12)
ax.xaxis.set_major_locator(plt.MaxNLocator(12))
ax.xaxis.set_major_formatter(months_fmt)
plt.xticks(rotation=45, fontsize=12)
plt.show()
top5=list(covid_country['Country/Region'].head())
top5.extend(['South Korea'])

temp=covid.groupby(['Country/Region','Date'])[['Confirmed','Deaths','Recovered','Active']].sum().reset_index()
temp['Day to Day']=temp.groupby(['Country/Region'])['Active'].shift()
temp['Day to Day']=temp['Active']-temp['Day to Day']
temp=temp.loc[temp['Country/Region'].isin(top5)].groupby(['Country/Region','Date']).sum()['Day to Day'].unstack(level=0)

months = mdates.MonthLocator()
months_fmt = mdates.DateFormatter('%b-%e')

active=temp

fig,ax=plt.subplots(figsize=(11,6))
ax.plot(active)

plt.title('Day to Day Change in Active Cases',fontdict={'fontsize':18})
ax.legend(active.columns, loc=2, fontsize=12)
ax.xaxis.set_major_locator(plt.MaxNLocator(12))
ax.xaxis.set_major_formatter(months_fmt)
plt.xticks(rotation=45, fontsize=12)
plt.show()
file='../input/world-populations/worldpopulation.csv'

population=pd.read_csv(file,header=4,usecols=['Country Name','2018'])
population.rename(columns={'2018':'Population'},inplace=True)
from fuzzywuzzy import fuzz

def match_name(name, list_names,list_pops, min_score=0):
    # -1 score incase we don't get any matches
    max_score = -1
    # Returning empty name for no match as well
    max_name = ""
    pop=None
    # Iternating over all names in the other
    for name2,pop2 in zip(list_names,list_pops):
        #Finding fuzzy match score
        score = fuzz.ratio(name, name2)
        # Checking if we are above our threshold and have a better score
        if (score > min_score) & (score > max_score):
            max_name = name2
            pop=pop2
            max_score = score
    return (max_name, max_score,pop)
# List for dicts for easy dataframe creation
dict_list = []
# iterating over our players without salaries found above
for name in covid_country['Country/Region']:
    # Use our method to find best match, we can set a threshold here
    match = match_name(name, population['Country Name'],population['Population'], 75)
    
    # New dict for storing data
    dict_ = {}
    dict_.update({"Country/Region" : name})
    dict_.update({"Country Name" : match[0]})
    dict_.update({"Score" : match[1]})
    dict_.update({"Population" : match[2]})
    dict_list.append(dict_)
    
merge_table = pd.DataFrame(dict_list)
# Display results
merge_table
covid_pop=pd.merge(covid_country,merge_table,left_on='Country/Region',right_on='Country Name',how='left').drop(columns=['Country/Region_y','Country Name']).rename(columns={'Country/Region_x':'Country/Region'})

country=dict()
countries=list()
populations=list()
for country_region in covid_country['Country/Region']:
    for country_name,country_pop in zip(population['Country Name'],population['Population']):
        
        if country_region in country_name[:len(country_region)+1]:
            countries.append(country_region)
            populations.append(country_pop)
            #country_pop[country_region]=country_pop
            
country_population={'Country':countries,'Population':populations}  

country_population=pd.DataFrame(country_population)

duplicates=list(country_population.groupby('Country').count().loc[country_population.groupby('Country').Population.count()>1].index)
merge_table2=country_population[~country_population['Country'].isin(duplicates)]
covid_missing=covid_pop[covid_pop.Population.isna()].reset_index(drop=True)

covid_fill=pd.merge(covid_missing,merge_table2,left_on='Country/Region',right_on='Country',how='left').drop(columns=['Population_x','Country']).rename(columns={'Population_y':'Population'})
covid_fill=covid_fill[['Country/Region','Population']]

covid_pop=pd.merge(covid_pop,covid_fill,on='Country/Region',how='left')
covid_pop.Population_x.fillna(covid_pop.Population_y,inplace=True)
covid_pop.drop(columns=['Population_y','Score'],inplace=True)
covid_pop.rename(columns={'Population_x':'Population'},inplace=True)
covid_pop.sort_values('Population',ascending=False)

covid_pop.loc[covid_pop['Country/Region']=='US','Population']=327167434
covid_pop.loc[covid_pop['Country/Region']=='South Korea','Population']=25549819
covid_pop=covid_pop.sort_values('Confirmed',ascending=False)
covid_pop.loc[covid_pop['Country/Region'].isin(top5)][['Country/Region','Population']].reset_index(drop=True)
covid_country_pop=covid_pop[['Country/Region','Population']].set_index('Country/Region')

covid_populations=covid.set_index('Country/Region').join(covid_country_pop)
covid_populations.reset_index(inplace=True)
covid_cntry_pop=covid_populations.groupby(['Country/Region','Date','Population'])[['Confirmed','Deaths','Recovered','Active']].sum().reset_index()

covid_cntry_pop['Confirmed Cases Per Million']=(covid_cntry_pop['Confirmed']/covid_cntry_pop['Population'])*1000000
covid_cntry_pop['Deaths Per Million']=(covid_cntry_pop['Deaths']/covid_cntry_pop['Population'])*1000000
covid_cntry_pop['Recovered Per Million']=(covid_cntry_pop['Recovered']/covid_cntry_pop['Population'])*1000000
covid_cntry_pop['Active Cases Per Million']=(covid_cntry_pop['Active']/covid_cntry_pop['Population'])*1000000
covid_cntry_pop.drop(columns='Population',inplace=True)
covid_cntry_pop=covid_cntry_pop.groupby(['Country/Region','Date']).max()
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
months = mdates.MonthLocator()
months_fmt = mdates.DateFormatter('%b-%e')

confirm=covid_cntry_pop.loc[top5]['Confirmed Cases Per Million'].unstack(level=0)

fig,ax=plt.subplots(figsize=(11,6))
ax.plot(confirm)

plt.title('Confirmed COVID-19 Cases Per Million (Adjusted Pop)',fontdict={'fontsize':18})
ax.legend(confirm.columns, loc=2, fontsize=12)
ax.xaxis.set_major_locator(plt.MaxNLocator(12))
ax.xaxis.set_major_formatter(months_fmt)
plt.xticks(rotation=45, fontsize=12)
plt.show()
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
months = mdates.MonthLocator()
months_fmt = mdates.DateFormatter('%b-%e')

temp=covid_cntry_pop.loc[top5]['Deaths Per Million'].unstack(level=0)

fig,ax=plt.subplots(figsize=(11,6))
ax.plot(temp)

plt.title('COVID-19 Deaths Per Million (Adjusted Pop)',fontdict={'fontsize':18})
ax.legend(temp.columns, loc=2, fontsize=12)
ax.xaxis.set_major_locator(plt.MaxNLocator(12))
ax.xaxis.set_major_formatter(months_fmt)
plt.xticks(rotation=45, fontsize=12)
plt.show()
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
months = mdates.MonthLocator()
months_fmt = mdates.DateFormatter('%b-%e')

temp=covid_cntry_pop.loc[top5]['Recovered Per Million'].unstack(level=0)

fig,ax=plt.subplots(figsize=(11,6))
ax.plot(temp)

plt.title('COVID-19 Recoveries Per Million (Adjusted Pop)',fontdict={'fontsize':18})
ax.legend(temp.columns, loc=2, fontsize=12)
ax.xaxis.set_major_locator(plt.MaxNLocator(12))
ax.xaxis.set_major_formatter(months_fmt)
plt.xticks(rotation=45, fontsize=12)
plt.show()
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
months = mdates.MonthLocator()
months_fmt = mdates.DateFormatter('%b-%e')

temp=covid_cntry_pop.loc[top5]['Active Cases Per Million'].unstack(level=0)

fig,ax=plt.subplots(figsize=(11,6))
ax.plot(temp)

plt.title('Active COVID-19 Cases Per Million (Adjusted Pop)',fontdict={'fontsize':18})
ax.legend(temp.columns, loc=2, fontsize=12)
ax.xaxis.set_major_locator(plt.MaxNLocator(12))
ax.xaxis.set_major_formatter(months_fmt)
plt.xticks(rotation=45, fontsize=12)
plt.show()