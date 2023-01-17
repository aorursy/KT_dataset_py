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
import seaborn as sns
import plotly.express as px
%matplotlib inline
#covid_19_df = pd.read_excel("/kaggle/input/covid19geographicdistributionworldwide/COVID-19-geographic-disbtribution-worldwide.xls")
covid_19_df = pd.read_csv("/kaggle/input/covid19geographicdistributionworldwide/COVID-19-geographic-disbtribution-worldwide.csv",encoding = "ISO-8859-1")
#covid_19_df = covid_19_df.drop('Unnamed: 0',axis=1)

#Auto update
direct_covid_19_df = pd.read_csv("https://opendata.ecdc.europa.eu/covid19/casedistribution/csv",encoding = "ISO-8859-1")
covid_19_df = direct_covid_19_df


covid_19_df.columns = ['DateRep','Day','Month','Year','Cases','Deaths','Countries and territories','GeoId','Alpha-3 code','Population','ContinentExp','C']
#covid_19_df['DateRep'] = pd.to_datetime(covid_19_df['DateRep'],format='%Y-%m-%d')
covid_19_df['DateRep'] = pd.to_datetime(covid_19_df['DateRep'],format='%d/%m/%Y')
#covid_19_df['DateRep'] = pd.to_datetime(covid_19_df['DateRep'],format='%m/%d/%Y')

countries_iso = pd.read_csv("/kaggle/input/covid19geographicdistributionworldwide/Countries_ISO.csv",index_col=False)
countries_iso = countries_iso.drop('Unnamed: 0',axis=1)

countries_population = pd.read_csv("/kaggle/input/covid19geographicdistributionworldwide/Total_population_by_Country_ISO3_Year_2018.csv",index_col=False)
countries_population = countries_population.drop('Unnamed: 0',axis=1)

countries_temperature = pd.read_csv('/kaggle/input/covid19geographicdistributionworldwide/Countries_ISO3_monthly_AVG_temperature.csv')
countries_temperature.columns = ['Alpha-3 code', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug','Sep', 'Oct', 'Nov', 'Dec']
countries_temperature = countries_temperature.set_index('Alpha-3 code')

countries_bed = pd.read_csv('/kaggle/input/covid19geographicdistributionworldwide/Hospital_beds_by_country.csv')
#Capitalize first char from CountryExp
covid_19_df['Countries and territories'] = covid_19_df['Countries and territories'].apply(lambda country: str.capitalize(country))

covid_19_df['DateRep'] = pd.to_datetime(covid_19_df['DateRep'])
covid_19_df.head()
def highlight_max_yellow(s):
    is_max = s == s.max()
    return ['background-color: yellow' if v else '' for v in is_max]

def highlight_max(data, color='yellow'):
    attr = 'background-color: {}'.format(color)
    if data.ndim == 1:  # Series from .apply(axis=0) or axis=1
        is_max = data == data.max()
        return [attr if v else '' for v in is_max]
    else:  # from .apply(axis=None)
        is_max = data == data.max().max()
        return pd.DataFrame(np.where(is_max, attr, ''),
                            index=data.index, columns=data.columns)
    
def highlight_max_all(s):
    is_max = s == s.max()
    return ['background-color: #f59d71' if v else '' for v in is_max]


def highlight_min(data):
    color_min= '#b5f5d4' #green   
    attr = 'background-color: {}'.format(color_min)

    if data.ndim == 1:  # Series from .apply(axis=0) or axis=1
        is_min = data == data.min()
        return [attr if v else '' for v in is_min]
    else: 
        is_min = data.groupby(level=0).transform('min') == data
        return pd.DataFrame(np.where(is_min, attr, ''),
                            index=data.index, columns=data.columns)
    
covid_19_df.set_index('DateRep',inplace=True)
cv19_countries_day = covid_19_df.groupby(by=['DateRep','Countries and territories']).sum()[['Cases','Deaths']]

Total_confirmed = cv19_countries_day.groupby('DateRep').sum()[['Cases','Deaths']].sum()['Cases']
Total_deaths = cv19_countries_day.groupby('DateRep').sum()[['Cases','Deaths']].sum()['Deaths']

dicc = {'Total Confirmed' : Total_confirmed, 'Total Deaths' : Total_deaths, 'Death Rate %' : round((Total_deaths/Total_confirmed)*100,2)}
total = pd.DataFrame(dicc,index=['Counter'])[['Total Confirmed','Total Deaths','Death Rate %']]


total.style.set_properties(**{
    'background-color': 'white',
    'font-size': '20pt',
    'color' : 'red'
})


covid19_total = covid_19_df[['Countries and territories','Cases']].groupby(by='Countries and territories').sum().sort_values(by='Cases',ascending=False)
covid19_total.columns=['Cases']
covid19_total_d = covid_19_df[['Countries and territories','Deaths']].groupby(by='Countries and territories').sum().sort_values(by='Deaths',ascending=False)
covid19_total_d.columns=['Deaths']
cv19_countries_day = covid_19_df.groupby(by=['DateRep']).sum()[['Cases','Deaths']]
cv19_countries_day['Cases'].cumsum().plot(figsize=(15,6),label='Confirmed cases',marker='o')
cv19_countries_day['Deaths'].cumsum().plot(label="Deaths",marker='+')
plt.legend()
plt.xlabel('Date')
plt.show()

#cv19_countries_day.sort_values(by='DateRep',ascending = False)

cv19_countries_day['Deaths'].cumsum().plot(figsize=(15,6),color='red',label="Deaths",marker='+')
plt.legend()
plt.xlabel('Date')
plt.show()
#Top 10 cases
t10 = pd.concat([covid19_total.head(10),covid19_total_d.head(10)],axis=1,sort=False).head(10)
covi19_total = covid19_total.reset_index()

px.bar(data_frame=covid19_total.head(20),x=covid19_total.head(20).index,y=covid19_total.head(20))

fig = px.bar(covid19_total, 
             y=covid19_total.head(20), x=covid19_total.head(20).index, #color='NewConfCases', 
             labels={'x':'Country','y':'Nº Confirmed'},
             log_y=True, template='plotly_white', title='Confirmed Cases')
fig.show()

fig = px.bar(covid19_total_d, 
             y=covid19_total_d.head(20), x=covid19_total_d.head(20).index, #color='Cases', 
             labels={'x':'Country','y':'Nº Deaths'},
             log_y=True, template='plotly_white', title='Deaths')
fig.show()
#Cases

d_map = covid_19_df.groupby(by=['Countries and territories','GeoId']).sum().sort_values(by='Deaths',ascending=False) \
.reset_index()[['Countries and territories','GeoId','Cases','Deaths']]
d_map.columns = ['Country','Alpha-2 code','Cases','Deaths']

d_map = pd.merge(d_map,countries_iso,on='Alpha-2 code',how='right')[['Country_x','Alpha-2 code','Cases','Deaths','Alpha-3 code']]

fig = px.scatter_geo(d_map.head(20), locations="Alpha-3 code", color="Cases",
                     hover_name="Country_x", size="Cases",
                     projection="equirectangular", title="Worldwide Main Cases")
fig.show()

#Deaths

fig = px.scatter_geo(d_map.head(20), locations="Alpha-3 code", color="Deaths",
                     hover_name="Country_x", size="Deaths",
                     projection="equirectangular", title="Worldwide Main Death Cases")
fig.show()


t10['Death ratio'] = round((t10['Deaths'] / t10['Cases']) *100,2)
t10.sort_values(by='Death ratio',ascending = False)
t10f = t10[['Death ratio']].sort_values(by='Death ratio',ascending=False).dropna()
t10f.style.apply(highlight_max, color='red', axis=None)
fig = px.bar(t10f,x=t10f.index, y=t10f['Death ratio'] , labels={'x':'Countries'}, title="Top 8 rate Deaths/Cases" , width=800, template='ggplot2')
fig.show()
covid19_change_global = cv19_countries_day.cumsum()
covid19_change_global[['Cases Day','Deaths Day']] = cv19_countries_day[['Cases','Deaths']]
covid19_change_global = covid19_change_global.pct_change(1)
covid19_change_global = covid19_change_global.sort_values(by='DateRep',ascending=False)
covid19_change_global = covid19_change_global.replace([np.inf, -np.inf], np.nan)
covid19_change_global = covid19_change_global.fillna(0)
covid19_change_global = round(covid19_change_global*100,2)
covid19_change_global = covid19_change_global.reset_index()

covid19_change_global_d = cv19_countries_day.cumsum()
covid19_change_global_d[['Cases Day','Deaths Day']] = cv19_countries_day[['Cases','Deaths']]
covid19_change_global_d = covid19_change_global_d.pct_change(1)
covid19_change_global_d = covid19_change_global_d.sort_values(by='DateRep',ascending=False)
covid19_change_global_d = covid19_change_global_d.replace([np.inf, -np.inf], np.nan)
covid19_change_global_d = covid19_change_global_d.fillna(0)
covid19_change_global_d = round(covid19_change_global_d*100,2)
covid19_change_global_d = covid19_change_global_d.reset_index()

px.bar(data_frame=covid19_change_global,x=covid19_change_global['DateRep'],y=covid19_change_global['Cases'], \
       color='Cases',
       labels={'Cases':'Date','Deaths':'% change'},
       title='Cases: Global change percentage per day')

px.bar(data_frame=covid19_change_global_d,x=covid19_change_global_d['DateRep'],y=covid19_change_global_d['Deaths'], \
       color='Deaths',
       labels={'DateRep':'Date','Deaths':'% change'},
       title='Deaths: Global change percentage per day')


"""
Calculate change by country for the 15 first
"""

Impacted_countries = covid_19_df[['Countries and territories','Cases']].sort_values(by=['DateRep','Cases'],ascending=False).head(15)['Countries and territories']
Impacted_countries

top_impact = pd.DataFrame()

for country in Impacted_countries:
    top_impact[country] = covid_19_df[covid_19_df['Countries and territories']==country]['Cases']


top_impact = top_impact.reset_index().sort_values(by='DateRep',ascending=True) 
    
    
#Normalize

#top_impact_norm = top_impact/top_impact.iloc[0] * 100

#Australia 0 cases correction (mean between days 25.03 and 27.03)
#top_impact['Australia'][1] = 671.5

growth_impact_day = top_impact.set_index('DateRep').pct_change(1).reset_index().sort_values(by='DateRep',ascending=False)
growth_impact_day = round(growth_impact_day.set_index('DateRep')*100,2).head(10)
growth_impact_day.style.apply(highlight_max_all).apply(highlight_min)

last_gi = growth_impact_day.iloc[0].sort_values(ascending = False)

last_gi = pd.DataFrame(data=[last_gi],index=[0],columns=last_gi.index)
plt.figure(figsize=(18,5))
splot = sns.barplot(x='index',y=0,data=last_gi.T.reset_index())
plt.title('Growth rate last day')
plt.ylabel('Pct. change')
plt.xlabel('Country')
plt.xticks(rotation=45)
for p in splot.patches:
    splot.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
plt.show()

sns.set_style('whitegrid')
top_impact.set_index('DateRep').tail(30).plot(figsize=(18,8))
plt.ylabel('Nº cases')
plt.xlabel('Date')
plt.show()
"""
Calculate change by country for the 15 first
"""

Impacted_countries_d = covid_19_df[['Countries and territories','Deaths']].sort_values(by=['DateRep','Deaths'],ascending=False).head(15)['Countries and territories']

top_impact_d = pd.DataFrame()

for country in Impacted_countries_d:
    top_impact_d[country] = covid_19_df[covid_19_df['Countries and territories']==country]['Deaths']


top_impact_d = top_impact_d.reset_index().sort_values(by='DateRep',ascending=True) 
    
#Normalize


#top_impact_norm = top_impact/top_impact.iloc[0] * 100
#Growth impact (Change: 1 day , Scope: 10 days)

#growth_impact_d_day = top_impact_d.pct_change(1).iloc[1:]
#growth_impact_d_day = round(growth_impact_d_day*100,2).head(10)
#growth_impact_d_day = growth_impact_d_day.replace([np.inf, -np.inf], np.nan)
#growth_impact_d_day = growth_impact_d_day.fillna(0)
#growth_impact_d_day.style.apply(highlight_max_all).apply(highlight_min)


growth_impact_d_day = top_impact_d.set_index('DateRep').pct_change(1).reset_index().sort_values(by='DateRep',ascending=False)
growth_impact_d_day = growth_impact_d_day.replace([np.inf, -np.inf], np.nan)
growth_impact_d_day = round(growth_impact_d_day.set_index('DateRep')*100,2).head(10)
growth_impact_d_day.style.apply(highlight_max_all).apply(highlight_min)

last_gi = growth_impact_d_day.iloc[0].sort_values(ascending = False)

last_gi = pd.DataFrame(data=[last_gi],index=[0],columns=last_gi.index)
plt.figure(figsize=(18,5))
splot = sns.barplot(x='index',y=0,data=last_gi.T.reset_index())
plt.title('Deaths: Growth rate last day')
plt.ylabel('Pct. change')
plt.xlabel('Country')
plt.xticks(rotation=45)
for p in splot.patches:
    splot.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
plt.show()

sns.set_style('whitegrid')
top_impact_d.set_index('DateRep').tail(30).plot(figsize=(18,8))
plt.ylabel('Nº deaths')
plt.xlabel('Date')
plt.show()

mask = np.zeros_like(growth_impact_day.corr())
mask[np.triu_indices_from(mask)] = True
plt.figure(figsize=(9,7))
sns.heatmap(growth_impact_day.corr(),mask=mask,cmap='YlGnBu', annot = True, fmt='.1g')
plt.show()
countries_population = countries_population[['Country Name','Country Code','2018']]
countries_population.columns = ['Country Name','Alpha-3 code','Population 2018']
data_w_population = pd.merge(d_map,countries_population,on='Alpha-3 code',how='inner')[['Country_x','Alpha-2 code','Cases','Deaths','Alpha-3 code','Population 2018']]
data_w_population['Cases by Population x100000'] = round((data_w_population['Cases'] / data_w_population['Population 2018'])*100000,2)
data_w_population['Deaths by Population x100000'] = round((data_w_population['Deaths'] / data_w_population['Population 2018'])*100000,2)

data_w_population_c = data_w_population[['Country_x','Alpha-3 code','Cases by Population x100000']].sort_values(by='Cases by Population x100000',ascending = False).head(30)
data_w_population_d = data_w_population[['Country_x','Alpha-3 code','Deaths by Population x100000']].sort_values(by='Deaths by Population x100000',ascending = False).head(30)

#Remove San Marino 
data_w_population_c = data_w_population_c[2:]
data_w_population_d = data_w_population_d[1:]
px.bar(data_frame=data_w_population_c,x=data_w_population_c['Country_x'],y=data_w_population_c['Cases by Population x100000'], \
       color='Cases by Population x100000',
       labels={'Country_x':'Country','Cases by Population x100000':'Cases (for each 100000 people)'},
       title='Cases per country population (1 case for each 100000 people)')
px.bar(data_frame=data_w_population_d,x=data_w_population_d['Country_x'],y=data_w_population_d['Deaths by Population x100000'], \
       color='Deaths by Population x100000',
       labels={'Country_x':'Country','Deaths by Population x100000':'Deaths (for each 100000 people)'},
       title='Deaths per country population (1 death for each 100000 people)')

Italy_df = covid_19_df[covid_19_df['Alpha-3 code']=='ITA']
Spain_df = covid_19_df[covid_19_df['Alpha-3 code']=='ESP']
USA_df = covid_19_df[covid_19_df['Alpha-3 code']=='USA']

Italy_df = Italy_df.sort_values(by='DateRep',ascending=True)
Italy_df['Cases-5-days-SMA']=Italy_df['Cases'].rolling(window=5).mean()
Italy_df['Deaths-5-days_SMA']=Italy_df['Deaths'].rolling(window=5).mean()
Italy_df = Italy_df.sort_values(by='DateRep',ascending=False)

Spain_df = Spain_df.sort_values(by='DateRep',ascending=True)
Spain_df['Cases-5-days-SMA']=Spain_df['Cases'].rolling(window=5).mean()
Spain_df['Deaths-5-days_SMA']=Spain_df['Deaths'].rolling(window=5).mean()
Spain_df = Spain_df.sort_values(by='DateRep',ascending=False)

USA_df = USA_df.sort_values(by='DateRep',ascending=True)
USA_df['Cases-5-days-SMA']=USA_df['Cases'].rolling(window=5).mean()
USA_df['Deaths-5-days_SMA']=USA_df['Deaths'].rolling(window=5).mean()
USA_df = USA_df.sort_values(by='DateRep',ascending=False)



#Create combo chart
fig, ax1 = plt.subplots(figsize=(14,8))
color = 'tab:green'
#bar plot creation
ax1.set_title('Italy: Cases 5 Days SMA', fontsize=16)
ax1.set_xlabel('Date', fontsize=16)
ax1.set_ylabel('Cases', fontsize=16)
ax1 = sns.barplot(x='DateRep', y='Cases', data = Italy_df.reset_index()[:50], palette='winter')

ax1.set_xticklabels(
    ax1.get_xticklabels(minor=True), 
    rotation=45, 
    horizontalalignment='right',
    fontweight='light',
    fontsize='x-large'
)
ax1.tick_params(axis='y')

#specify we want to share the same x-axis
ax2 = ax1.twiny()
color = 'tab:red'
#line plot creation
#ax2.set_ylabel('5 days SMA', fontsize=16)
ax2 = sns.lineplot(x='DateRep', y='Cases-5-days-SMA', data=Italy_df.reset_index()[:50], color=color)
ax2.tick_params(axis='y', color=color)
plt.show()




#Create combo chart
fig, ax1 = plt.subplots(figsize=(14,8))
color = 'tab:green'
#bar plot creation
ax1.set_title('Italy: Deaths 5 Days SMA', fontsize=16)
ax1.set_xlabel('Date', fontsize=16)
ax1.set_ylabel('Deaths', fontsize=16)
ax1 = sns.barplot(x='DateRep', y='Deaths', data = Italy_df.reset_index()[:50], palette='winter')

ax1.set_xticklabels(
    ax1.get_xticklabels(minor=True), 
    rotation=45, 
    horizontalalignment='right',
    fontweight='light',
    fontsize='x-large'
)
ax1.tick_params(axis='y')
#specify we want to share the same x-axis
ax2 = ax1.twiny()
color = 'tab:red'
#line plot creation
ax2.set_ylabel('5 days SMA', fontsize=16)
ax2 = sns.lineplot(x='DateRep', y='Deaths-5-days_SMA', data = Italy_df.reset_index()[:50], color=color)
ax2.tick_params(axis='y', color=color)
#show plot
plt.show()



#Create combo chart
fig, ax1 = plt.subplots(figsize=(14,8))
color = 'tab:green'
#bar plot creation
ax1.set_title('Spain: Cases 5 Days SMA', fontsize=16)
ax1.set_xlabel('Date', fontsize=16)
ax1.set_ylabel('Cases', fontsize=16)
ax1 = sns.barplot(x='DateRep', y='Cases', data = Spain_df.reset_index()[:40], palette='spring')

ax1.set_xticklabels(
    ax1.get_xticklabels(minor=True), 
    rotation=45, 
    horizontalalignment='right',
    fontweight='light',
    fontsize='x-large'
)
ax1.tick_params(axis='y')
#specify we want to share the same x-axis
ax2 = ax1.twiny()
color = 'tab:red'
#line plot creation
ax2.set_ylabel('5 days SMA', fontsize=16)
ax2 = sns.lineplot(x='DateRep', y='Cases-5-days-SMA', data = Spain_df.reset_index()[:40], color=color)
ax2.tick_params(axis='y', color=color)
#show plot
plt.show()



#Create combo chart
fig, ax1 = plt.subplots(figsize=(14,8))
color = 'tab:green'
#bar plot creation
ax1.set_title('Spain: Deaths 5 Days SMA', fontsize=16)
ax1.set_xlabel('Date', fontsize=16)
ax1.set_ylabel('Deaths', fontsize=16)
ax1 = sns.barplot(x='DateRep', y='Deaths', data = Spain_df.reset_index()[:40], palette='spring')

ax1.set_xticklabels(
    ax1.get_xticklabels(minor=True), 
    rotation=45, 
    horizontalalignment='right',
    fontweight='light',
    fontsize='x-large'
)
ax1.tick_params(axis='y')
#specify we want to share the same x-axis
ax2 = ax1.twiny()
color = 'tab:red'
#line plot creation
ax2.set_ylabel('5 days SMA', fontsize=16)
ax2 = sns.lineplot(x='DateRep', y='Deaths-5-days_SMA', data = Spain_df.reset_index()[:40], color=color)
ax2.tick_params(axis='y', color=color)
#show plot
plt.show()



#Create combo chart
fig, ax1 = plt.subplots(figsize=(14,8))
color = 'tab:green'
#bar plot creation
ax1.set_title('USA: Cases 5 Days SMA', fontsize=16)
ax1.set_xlabel('Date', fontsize=16)
ax1.set_ylabel('Cases', fontsize=16)
ax1 = sns.barplot(x='DateRep', y='Cases', data = USA_df.reset_index()[:40], palette='Blues')

ax1.set_xticklabels(
    ax1.get_xticklabels(minor=True), 
    rotation=45, 
    horizontalalignment='right',
    fontweight='light',
    fontsize='x-large'
)
ax1.tick_params(axis='y')
#specify we want to share the same x-axis
ax2 = ax1.twiny()
color = 'tab:red'
#line plot creation
ax2.set_ylabel('5 days SMA', fontsize=16)
ax2 = sns.lineplot(x='DateRep', y='Cases-5-days-SMA', data = USA_df.reset_index()[:40], color=color)
ax2.tick_params(axis='y', color=color)
#show plot
plt.show()



#Create combo chart
fig, ax1 = plt.subplots(figsize=(14,8))
color = 'tab:green'
#bar plot creation
ax1.set_title('USA: Deaths 5 Days SMA', fontsize=16)
ax1.set_xlabel('Date', fontsize=16)
ax1.set_ylabel('Deaths', fontsize=16)
ax1 = sns.barplot(x='DateRep', y='Deaths', data = USA_df.reset_index()[:40], palette='Blues')

ax1.set_xticklabels(
    ax1.get_xticklabels(minor=True), 
    rotation=45, 
    horizontalalignment='right',
    fontweight='light',
    fontsize='x-large'
)
ax1.tick_params(axis='y')
#specify we want to share the same x-axis
ax2 = ax1.twiny()
color = 'tab:red'
#line plot creation
ax2.set_ylabel('5 days SMA', fontsize=16)
ax2 = sns.lineplot(x='DateRep', y='Deaths-5-days_SMA', data = USA_df.reset_index()[:40], color=color)
ax2.tick_params(axis='y', color=color)
#show plot
plt.show()


Peaks = covid_19_df.groupby(by='Countries and territories').max().sort_values(by='Cases',ascending=False)['Cases']
Peaks = Peaks.reset_index()



countries_temperature.columns = [1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0]

def Get_Country_Temperature(ISO3,Month):
    return countries_temperature.loc[ISO3][Month]

    
#covid_19_df
#countries_iso
#countries_population
#countries_temperature
#countries_bed

#ML_data PREDICTION (GLOBAL)

covid_19_df_ml = covid_19_df.reset_index()[['DateRep','Cases','Deaths','Alpha-3 code','Population']]
#covid_19_df_ml

countries_bed.columns = ['Alpha-3 code', 'INDICATOR', 'SUBJECT', 'MEASURE', 'FREQUENCY', 'TIME','Value']
covid_19_df_ml = pd.merge(covid_19_df_ml,countries_bed,on='Alpha-3 code',how='right')[['DateRep', 'Cases', 'Deaths', 'Alpha-3 code','Population', 'Value']]
covid_19_df_ml = covid_19_df_ml.dropna()

covid_19_df_ml['Temperature mavg'] = covid_19_df_ml.apply(lambda x: Get_Country_Temperature(x['Alpha-3 code'],x['DateRep'].month),axis=1)
covid_19_df_ml['DateRep'] = covid_19_df_ml['DateRep'].apply(lambda x: x.toordinal())

covid_19_df_ml = pd.get_dummies(covid_19_df_ml,columns=['Alpha-3 code'],drop_first=True)
covid_19_df_ml.columns

#Global grupping
#covid_19_df_ml_global = covid_19_df_ml.groupby(by='DateRep').agg({'Cases':'sum','Deaths':'sum','Population':'sum','Temperature mavg':'mean'})
#covid_19_df_ml_global = covid_19_df_ml_global.reset_index()

X=covid_19_df_ml[['DateRep', 'Population', 'Value', 'Temperature mavg',
        'Alpha-3 code_AUT', 'Alpha-3 code_BEL', 'Alpha-3 code_BRA',
       'Alpha-3 code_CAN', 'Alpha-3 code_CHE', 'Alpha-3 code_CHL',
       'Alpha-3 code_CHN', 'Alpha-3 code_COL', 'Alpha-3 code_CRI',
       'Alpha-3 code_DEU', 'Alpha-3 code_DNK', 'Alpha-3 code_ESP',
       'Alpha-3 code_EST', 'Alpha-3 code_FIN', 'Alpha-3 code_FRA',
       'Alpha-3 code_GBR', 'Alpha-3 code_GRC', 'Alpha-3 code_HUN',
       'Alpha-3 code_IDN', 'Alpha-3 code_IND', 'Alpha-3 code_IRL',
       'Alpha-3 code_ISL', 'Alpha-3 code_ISR', 'Alpha-3 code_ITA',
       'Alpha-3 code_JPN', 'Alpha-3 code_KOR', 'Alpha-3 code_LTU',
       'Alpha-3 code_LUX', 'Alpha-3 code_LVA', 'Alpha-3 code_MEX',
       'Alpha-3 code_NLD', 'Alpha-3 code_NOR', 'Alpha-3 code_NZL',
       'Alpha-3 code_POL', 'Alpha-3 code_PRT', 'Alpha-3 code_RUS',
       'Alpha-3 code_SVK', 'Alpha-3 code_SVN', 'Alpha-3 code_SWE',
       'Alpha-3 code_TUR', 'Alpha-3 code_USA', 'Alpha-3 code_ZAF']]
y=covid_19_df_ml[['Cases','Deaths']]


#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)