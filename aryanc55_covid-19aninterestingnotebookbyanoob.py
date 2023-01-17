
!pip install pyramid
!pip install pyramid-arima
!pip install colour
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.colors as mcolors
import pandas as pd 
import random
import math
import time
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error

import operator 
plt.style.use('seaborn')
import geopandas as gpd
import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff
%matplotlib inline 
pd.plotting.register_matplotlib_converters()
import seaborn as sns

import matplotlib.ticker as ticker
import matplotlib.animation as animation
import emoji
from IPython.display import HTML
import warnings
warnings.filterwarnings('ignore')

from datetime import datetime
from scipy import stats
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot
from matplotlib.pylab import rcParams
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
#rcParams['figure.figsize'] = 30, 10

import pyramid as pm
from pyramid.arima import auto_arima
from scipy.optimize import curve_fit


print("Setup Complete")

conf_df = pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')
death_df = pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv')
reco_df = pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv')
conf_df.head(2)

#extracting all the dates
dates = conf_df.columns[4:]

#making new dataframes for confiremed ,death and recovred cases
conf_df_long = conf_df.melt(id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'], 
                            value_vars=dates, var_name='Date', value_name='Confirmed')

deaths_df_long = death_df.melt(id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'], 
                            value_vars=dates, var_name='Date', value_name='Deaths')

recv_df_long = reco_df.melt(id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'], 
                            value_vars=dates, var_name='Date', value_name='Recovered')

#full table combinig all confirmed,recovered and deaths 
full_table = pd.concat([conf_df_long, deaths_df_long['Deaths'], recv_df_long['Recovered']], 
                       axis=1, sort=False)

full_table.head()
cases = ['Confirmed', 'Deaths', 'Recovered', 'Active']

#adding active cases to the dataset

full_table['Active'] = full_table['Confirmed'] - full_table['Deaths'] - full_table['Recovered']

#Cleaning 
full_table = full_table[full_table['Province/State'].str.contains(',')!=True]

full_table['Country/Region'] = full_table['Country/Region'].replace('Korea, South', 'South Korea')
full_table['Country/Region'] = full_table['Country/Region'].replace('Mainland China', 'China')



#full_table.to_csv('covid_19_full_updatedNew.csv', index=False)
#full_table = pd.read_csv('covid_19_full_updatedNew.csv')
#current_date,according to dataset
today = dates[-1]
#full latest keeps today's data for all countries
full_latest = full_table[full_table['Date'] == today].reset_index()
full_latest.head(2)
full_latest_grouped = full_table.groupby('Country/Region')['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()
temp_f = full_latest_grouped.sort_values(by='Confirmed', ascending=False)
temp_f = temp_f.reset_index(drop=True)
temp_f.head(10).style.background_gradient(cmap='Reds')
full_latest_grouped = full_table.groupby('Country/Region')['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()
temp_g = full_latest_grouped.sort_values(by='Recovered', ascending=False)
temp_g = temp_g.reset_index(drop=True)
temp_g.head(10).style.background_gradient(cmap='Greens')
#getting all the dates
cols = conf_df.keys()
conf = conf_df.loc[:, cols[4]:cols[-1]]
death = death_df.loc[:, cols[4]:cols[-1]]
reco = reco_df.loc[:, cols[4]:cols[-1]]
conf.head(2)
dates = conf.keys()
world_cases = []
total_deaths = [] 
mortality_rate = []
recovery_rate = [] 
total_recovered = [] 
total_active = [] 

india_cases_conf = []
india_cases_reco = []
india_cases_death = []
india_cases_active = []

j=0
for i in dates:
    
    confirmed_sum = conf[i].sum()
    death_sum = death[i].sum()
    recovered_sum = reco[i].sum()
    
    # confirmed, deaths, recovered, and active
    world_cases.append(confirmed_sum)
    total_deaths.append(death_sum)
    total_recovered.append(recovered_sum)
    total_active.append(confirmed_sum-death_sum-recovered_sum)
    
    # calculate rates
    mortality_rate.append(death_sum/confirmed_sum)
    recovery_rate.append(recovered_sum/confirmed_sum)

    # case studies 
    #china_cases.append(conf_df[conf_df['Country/Region']=='China'][i].sum())
    #italy_cases.append(conf_df[conf_df['Country/Region']=='Italy'][i].sum())
#     india_cases_conf.append(conf_df[conf_df['Country/Region']=='India'][i].sum())
#     india_cases_reco.append(reco_df[reco_df['Country/Region']=='India'][i].sum())
#     india_cases_death.append(death_df[death_df['Country/Region']=='India'][i].sum())
#     india_cases_active.append(india_cases_conf[j] - india_cases_reco[j] - india_cases_death[j])
    
    j = j+1
# india_cases_conf_df = pd.DataFrame(index=dates,data=india_cases_conf, columns=['conf_cases'])
# india_cases_reco_df = pd.DataFrame(index=dates,data=india_cases_reco, columns=['reco_cases'])
# india_cases_death_df = pd.DataFrame(index=dates,data=india_cases_death, columns=['death_cases'])
# india_cases_active_df = pd.DataFrame(index=dates,data=india_cases_active, columns=['activ_cases'])
    
# india_full = pd.concat([india_cases_conf_df,india_cases_reco_df,india_cases_death_df,india_cases_active_df], axis=1)
# india_full.head(3)
#total confirmed cases in the world
world_df = pd.DataFrame(index=dates,data=world_cases, columns=['conf_cases'])
world_df['daily_new_cases'] = world_df.diff()
world_df['daily_new_cases'][0] = world_df['conf_cases'][0]
world_df['daily_new_cases'] =world_df['daily_new_cases'].astype('int64')
world_df.head(2)
total_death_df = pd.DataFrame(index=dates,data=total_deaths, columns=['total_death'])
total_death_df['new_death'] = total_death_df.diff()
total_death_df['new_death'][0] = total_death_df['total_death'][0] 
total_death_df['new_death'] = total_death_df['new_death'].astype('int64')
total_reco_df = pd.DataFrame(index=dates,data=total_recovered, columns=['total_recovered'])
total_reco_df['new_reco'] = total_reco_df.diff()
total_reco_df['new_reco'][0] = total_reco_df['total_recovered'][0] 
total_reco_df['new_reco'] = total_reco_df['new_reco'].astype('int64')
total_reco_df.head(3)

##this is tricky but true as it is daily new active cases being found both way it give same ans
total_active_df = pd.DataFrame(index=dates,data=total_active, columns=['total_active'])

#total_active_df['new_active111'] = world_df['daily_new_cases'] - total_death_df['new_death'] - total_reco_df['new_reco']
total_active_df['new_active'] = total_active_df['total_active'].diff()
total_active_df['new_active'][0] = total_active_df['total_active'][0] 
total_active_df['new_active'] = total_active_df['new_active'].astype('int64')

total_active_df=total_active_df.clip(lower =0)
world_full = pd.concat([world_df,total_reco_df,total_death_df,total_active_df], axis=1)
world_full.head(3)
import datetime
days_since_1_22 = np.array([i for i in range(len(dates))]).reshape(-1, 1)
world_cases = np.array(world_cases).reshape(-1, 1)
total_deaths = np.array(total_deaths).reshape(-1, 1)
total_recovered = np.array(total_recovered).reshape(-1, 1)
#Dates manipulation for Future forcasting
days_in_future = 10
future_forcast = np.array([i for i in range(len(dates)+days_in_future)]).reshape(-1, 1)
adjusted_dates = future_forcast[:-10]
start = '1/22/2020'
start_date = datetime.datetime.strptime(start, '%m/%d/%Y')
future_forcast_dates = []
for i in range(len(future_forcast)):
    future_forcast_dates.append((start_date + datetime.timedelta(days=i)).strftime('%m/%d/%Y'))
sns.set(context='notebook',
    style='darkgrid',
    palette='deep',
    font='sans-serif',
    font_scale=2,
    color_codes=True,
    rc=None)

plt.figure(figsize=(20, 10))
plt.title('Number of Total CONFIRMED COVID-19 Cases Over Time', size=30)
plt.xlabel('Days Since 1/22/2020', size=30)
plt.ylabel('Number of Cases', size=30)
sns.lineplot(data=world_full.reset_index()['conf_cases']);
#sns.regplot(x=world_full.index ,y = world_full.reset_index()['world_cases']);
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()


sns.set(context='notebook',
    style='darkgrid',
    palette='deep',
    font='sans-serif',
    font_scale=2,
    color_codes=True,
    rc=None)

plt.figure(figsize=(20, 10))
plt.title('Daily New CONFIRMED COVID-19 Cases Over Time', size=30)
plt.xlabel('Days Since 1/22/2020', size=30)
plt.ylabel('Number of Cases', size=30)
sns.lineplot(data=world_full.reset_index()['daily_new_cases']);
#sns.regplot(x=world_full.index ,y = world_full.reset_index()['world_cases']);
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()
plt.figure(figsize=(20, 12))
sns.lineplot(data=world_full.reset_index()['total_active'])

plt.title('# of Coronavirus Active Cases Over Time', size=30)
plt.xlabel('Days Since 1/22/2020', size=30)
plt.ylabel('# of Active Cases', size=30)
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()
plt.figure(figsize=(20, 12))
sns.lineplot(data=world_full.reset_index()['new_active'])

plt.title('Daily New Active Cases Over Time', size=30)
plt.xlabel('Days Since 1/22/2020', size=30)
plt.ylabel('# of Active Cases', size=30)
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()
plt.figure(figsize=(20, 12))
sns.lineplot(data=world_full.reset_index()['total_death'])
plt.title('# of Coronavirus Deaths Over Time', size=30)
plt.xlabel('Days Since 1/22/2020', size=30)
plt.ylabel('# of Deaths', size=30)
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()
plt.figure(figsize=(20, 12))
sns.lineplot(data=world_full.reset_index()['new_death'])
plt.title('Daily Deaths Over Time', size=30)
plt.xlabel('Days Since 1/22/2020', size=30)
plt.ylabel('# of Deaths', size=30)
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()
mean_mortality_rate = np.mean(mortality_rate)
plt.figure(figsize=(20, 12))
plt.plot(adjusted_dates, mortality_rate, color='orange')
plt.axhline(y = mean_mortality_rate,linestyle='--', color='black')
plt.title('Mortality Rate of Coronavirus Over Time', size=30)
plt.legend(['mortality rate', 'y='+str(mean_mortality_rate)], prop={'size': 20})
plt.xlabel('Days Since 1/22/2020', size=30)
plt.ylabel('Mortality Rate', size=30)
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()
plt.figure(figsize=(20, 12))
sns.lineplot(data=world_full.reset_index()['total_recovered'])


plt.title('# of Coronavirus Cases Recovered Over Time', size=30)
plt.xlabel('Days Since 1/22/2020', size=30)
plt.ylabel('# of Cases', size=30)
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()
plt.figure(figsize=(20, 12))

sns.lineplot(data=world_full.reset_index()['new_reco'])

plt.title('Daily Recovery over Time', size=30)
plt.xlabel('Days Since 1/22/2020', size=30)
plt.ylabel('# of Cases', size=30)
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()
mean_recovery_rate = np.mean(recovery_rate)
plt.figure(figsize=(20, 12))
plt.plot(adjusted_dates, recovery_rate, color='blue')
plt.axhline(y = mean_recovery_rate,linestyle='--', color='black')
plt.title('Recovery Rate of Coronavirus Over Time', size=30)
plt.legend(['recovery rate', 'y='+str(mean_recovery_rate)], prop={'size': 20})
plt.xlabel('Days Since 1/22/2020', size=30)
plt.ylabel('Recovery Rate', size=30)
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()
plt.figure(figsize=(20, 12))
sns.lineplot(data=world_full.reset_index()['conf_cases'] );
sns.lineplot(data=world_full.reset_index()['total_recovered'])
sns.lineplot(data=world_full.reset_index()['total_death'])
sns.lineplot(data=world_full.reset_index()['total_active'])


plt.legend(['confirmed', 'recoveries','deaths','active'], loc='best', fontsize=20)
plt.title('# of Coronavirus Cases in World', size=30)
plt.xlabel('Days Since 1/22/2020', size=30)
plt.ylabel('# of Cases', size=30)
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()
#india_full.head(2)
# plt.figure(figsize=(20, 12))
# sns.lineplot(data=india_full.reset_index()['conf_cases'])
# sns.lineplot(data=india_full.reset_index()['reco_cases'])
# sns.lineplot(data=india_full.reset_index()['death_cases'])
# sns.lineplot(data=india_full.reset_index()['activ_cases'])


# plt.legend(['confirmed', 'recoveries','deaths','active'], loc='best', fontsize=20)
# plt.title('# of Coronavirus Cases in India', size=30)
# plt.xlabel('Days Since 1/22/2020', size=30)
# plt.ylabel('# of Cases', size=30)
# plt.xticks(size=20)
# plt.yticks(size=20)
# plt.show()
f, ax = plt.subplots(figsize=(20, 12))

# Load the example car crash dataset
sns.set_color_codes("deep")
sns.barplot(x =world_full.reset_index().index,y=world_full.reset_index()['conf_cases'] , label = 'Confirmed',color ='b');

sns.set_color_codes("muted")
sns.barplot(x =world_full.reset_index().index,y=world_full.reset_index()['total_recovered'] , label = 'Recovered',color ='r');


sns.set_color_codes("muted")
sns.barplot(x =world_full.reset_index().index,y=world_full.reset_index()['total_death'] , label = 'Deaths',color ='black');
# Add a legend and informative axis label
#plt.legend(['confirmed', 'recoveries','deaths'], loc='best', fontsize=20)
plt.xlabel('Days Since 1/22/2020', size=30)
plt.ylabel('# of Cases', size=30)

#ax.legend(ncol=1, loc="upper right", frameon=True)
#ax.set(xlim=(0, 60) ,ylabel="no of people",
#       xlabel="# of cases ")
sns.despine(left=True, bottom=True)
plt.xticks(size=10)
plt.yticks(size=20)
plt.show()

plt.figure(figsize=(20, 12))
sns.lineplot(data=world_full.reset_index()['daily_new_cases'] ,markers=True );
sns.lineplot(data=world_full.reset_index()['new_reco'])
sns.lineplot(data=world_full.reset_index()['new_death'])
sns.lineplot(data=world_full.reset_index()['new_active'])


plt.legend(['Daily new confirmed', 'Daily new recoveries','dailty new deaths','daily new active'], loc='best', fontsize=20)
plt.title('# of Coronavirus Cases in World', size=30)
plt.xlabel('Days Since 1/22/2020', size=30)
plt.ylabel('# of Cases', size=30)
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()
#current date data
latest_confirmed = conf_df[dates[-1]]
latest_deaths = death_df[dates[-1]]
latest_recoveries = reco_df[dates[-1]]
#all unique country list 
unique_countries =  list(conf_df['Country/Region'].unique())
country_confirmed_cases = []
no_cases = []
for i in unique_countries:
    cases = latest_confirmed[conf_df['Country/Region']==i].sum()
    if cases > 0:
        country_confirmed_cases.append(cases)
    else:
        no_cases.append(i)
        
for i in no_cases:
    unique_countries.remove(i)
    
unique_countries = [k for k, v in sorted(zip(unique_countries, country_confirmed_cases), key=operator.itemgetter(1), reverse=True)]
for i in range(len(unique_countries)):
    country_confirmed_cases[i] = latest_confirmed[conf_df['Country/Region']==unique_countries[i]].sum()
# Only show 10 countries with the most confirmed cases, the rest are grouped into the other category
visual_unique_countries = [] 
visual_confirmed_cases = []
others = np.sum(country_confirmed_cases[10:])

for i in range(len(country_confirmed_cases[:10])):
    visual_unique_countries.append(unique_countries[i])
    visual_confirmed_cases.append(country_confirmed_cases[i])
    
visual_unique_countries.append('Others')
visual_confirmed_cases.append(others)
c = random.choices(list(mcolors.CSS4_COLORS.values()),k = len(unique_countries))
plt.figure(figsize=(20,20))
plt.title('Covid-19 Confirmed Cases per Country', size=20)
plt.pie(visual_confirmed_cases, colors=c)
plt.legend(visual_unique_countries, loc='best', fontsize=15)
plt.show()

formated_gdf = full_table.groupby(['Date', 'Country/Region'])['Confirmed', 'Deaths', 'Recovered'].sum()
full_latest.head(3)

formated_gdf = full_table.groupby(['Date', 'Country/Region'])['Confirmed', 'Deaths', 'Recovered'].max()
formated_gdf = formated_gdf.reset_index()
formated_gdf['Date'] = pd.to_datetime(formated_gdf['Date'])
formated_gdf['Date'] = formated_gdf['Date'].dt.strftime('%m/%d/%Y')
formated_gdf['size'] = formated_gdf['Confirmed'].pow(0.3)

fig = px.scatter_geo(formated_gdf, locations="Country/Region", locationmode='country names', 
                     color="Confirmed", size='size', hover_name="Country/Region", 
                     range_color= [0, max(formated_gdf['Confirmed'])+2], 
                     projection="natural earth", animation_frame="Date", 
                     title='Global Spread over time')
fig.update(layout_coloraxis_showscale=False)
fig.show()

HTML('''<div class="flourish-embed flourish-bar-chart-race" data-src="visualisation/1651020"><script src="https://public.flourish.studio/resources/embed.js"></script></div>''')
adj_dates = full_table.Date.unique()
df = full_table.copy()

def genratepseudorandomcolors():
    import random
    from colour import Color
    random_i = random.randint(0,10000000)
    random_j = random.randint(0,10000000)
    c = Color(random_i,random_j)
    hex_number = c.hex_l
    return hex_number

colours = []
for i in range(len(df['Country/Region'].unique())) :
    colours.append(genratepseudorandomcolors())
colorsset = dict(zip(
    [i for i in df['Country/Region'].unique()],
    colours
))

start_date ='1/22/20'
current_date = start_date



def draw_barchart_conf(current_date):
    
    #creating dataframe for plotting 
    #In each loop it contains top 10 coutires having most number of confirmed cases
    dff = df[df['Date'].eq(current_date)].groupby('Country/Region').sum().sort_values(by='Confirmed', ascending=False).head(10)
    dff = dff.reset_index()
    dff = dff[::-1]
    
    
    
    #Now, let's plot a basic bar chart. We start by creating a figure and an axes.
    #Then, we use `ax.barh(x, y)` to draw horizontal barchart.

    
    ax.clear()
    ax.barh(dff['Country/Region'], dff['Confirmed'], color=[colorsset[x] for x in dff['Country/Region']])
    dx = dff['Confirmed'].max() / 200
    
    
    #Next, let's add text,color,labels

    for i, (value, name) in enumerate(zip(dff['Confirmed'], dff['Country/Region'])):
        
        ax.text(value-dx, i,     name,           size=14, weight=600, ha='right', va='bottom')
        ax.text(value+dx, i,     f'{value:,.0f}',  size=14, ha='left',  va='center')
    
    
    ax.text(1, 0.4, current_date, transform=ax.transAxes, color='#777777', size=46, ha='right', weight=800)
    ax.text(0, 1.06, 'Population (thousands)', transform=ax.transAxes, size=12, color='#777777')
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
    ax.xaxis.set_ticks_position('top')
    ax.tick_params(axis='x', colors='#777777', labelsize=12)
    ax.set_yticks([])
    ax.margins(0, 0.01)
    ax.grid(which='major', axis='x', linestyle='-')
    ax.set_axisbelow(True)
    
    #adding other textual infomations
    ax.text(0, 1.15, f'Most number of confirmed cases in the world from 1/22/20{emoji.emojize(":worried_face:")}',
            transform=ax.transAxes, size=24, weight=600, ha='left', va='top')
    ax.text(1, 0, 'by @aryanc55', transform=ax.transAxes, color='#777777', ha='right',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='white'))
    plt.box(False)
    
#draw_barchart_conf(current_date=start_date)
fig, ax = plt.subplots(figsize=(15, 8))
animator = animation.FuncAnimation(fig, draw_barchart_conf, frames=adj_dates)
HTML(animator.to_jshtml())
# or use animator.to_html5_video() or animator.save() 

def draw_barchart_deaths(current_date):
    #dff = df[df['Date'].eq(current_year)].sort_values(by='value', ascending=True).tail(10)
    dff = df[df['Date'].eq(current_date)].groupby('Country/Region').sum().sort_values(by='Deaths', ascending=False).head(10)
    dff = dff.reset_index()
    dff = dff[::-1]

    ax.clear()
    ax.barh(dff['Country/Region'], dff['Deaths'], color=[colorsset[x] for x in dff['Country/Region']])
    dx = dff['Deaths'].max() / 200
    
    for i, (value, name) in enumerate(zip(dff['Deaths'], dff['Country/Region'])):
        
        ax.text(value-dx, i,     name,           size=14, weight=600, ha='right', va='bottom')
        #ax.text(value-dx, i-.25, colorsset[name], size=10, color='#444444', ha='right', va='baseline')
        ax.text(value+dx, i,     f'{value:,.0f}',  size=14, ha='left',  va='center')
    
    ax.text(1, 0.4, current_date, transform=ax.transAxes, color='#777777', size=46, ha='right', weight=800)
    ax.text(0, 1.06, 'Population (thousands)', transform=ax.transAxes, size=12, color='#777777')
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
    ax.xaxis.set_ticks_position('top')
    ax.tick_params(axis='x', colors='#777777', labelsize=12)
    ax.set_yticks([])
    ax.margins(0, 0.01)
    ax.grid(which='major', axis='x', linestyle='-')
    ax.set_axisbelow(True)
    ax.text(0, 1.15, f'Most number of Death cases in the world from 1/22/20{emoji.emojize(":pensive_face:")}',
            transform=ax.transAxes, size=24, weight=600, ha='left', va='top')
    ax.text(1, 0, 'by @aryanc55', transform=ax.transAxes, color='#777777', ha='right',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='white'))
    plt.box(False)
    
#draw_barchart_conf(current_date=start_date)

fig, ax = plt.subplots(figsize=(15, 8))
animator = animation.FuncAnimation(fig, draw_barchart_deaths, frames=adj_dates)
HTML(animator.to_jshtml())

def draw_barchart_Recovered(current_date):
    #dff = df[df['Date'].eq(current_year)].sort_values(by='value', ascending=True).tail(10)
    dff = df[df['Date'].eq(current_date)].groupby('Country/Region').sum().sort_values(by='Recovered', ascending=False).head(10)
    dff = dff.reset_index()
    dff = dff[::-1]

    ax.clear()
    ax.barh(dff['Country/Region'], dff['Recovered'], color=[colorsset[x] for x in dff['Country/Region']])
    dx = dff['Recovered'].max() / 200
    
    for i, (value, name) in enumerate(zip(dff['Recovered'], dff['Country/Region'])):
        
        ax.text(value-dx, i,     name,           size=14, weight=600, ha='right', va='bottom')
        #ax.text(value-dx, i-.25, colorsset[name], size=10, color='#444444', ha='right', va='baseline')
        ax.text(value+dx, i,     f'{value:,.0f}',  size=14, ha='left',  va='center')
    
    ax.text(1, 0.4, current_date, transform=ax.transAxes, color='#777777', size=46, ha='right', weight=800)
    ax.text(0, 1.06, 'Population (thousands)', transform=ax.transAxes, size=12, color='#777777')
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
    ax.xaxis.set_ticks_position('top')
    ax.tick_params(axis='x', colors='#777777', labelsize=12)
    ax.set_yticks([])
    ax.margins(0, 0.01)
    ax.grid(which='major', axis='x', linestyle='-')
    ax.set_axisbelow(True)
    ax.text(0, 1.15, f'Most number of Recovered cases in the world from 1/22/20 {emoji.emojize(":smiling_face:")}',
            transform=ax.transAxes, size=24, weight=600, ha='left', va='top')
    ax.text(1, 0, 'by @aryanc55', transform=ax.transAxes, color='#777777', ha='right',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='white'))
    plt.box(False)
    
#draw_barchart_deaths(current_date=start_date)
fig, ax = plt.subplots(figsize=(15, 8))
animator = animation.FuncAnimation(fig, draw_barchart_Recovered, frames=adj_dates)
HTML(animator.to_jshtml())

X_train_confirmed, X_test_confirmed, y_train_confirmed, y_test_confirmed = train_test_split(days_since_1_22, world_cases, test_size=0.15, shuffle=False) 
# use this to find the optimal parameters for SVR
# c = [0.01, 0.1, 1]
# gamma = [0.01, 0.1, 1]
# epsilon = [0.01, 0.1, 1]
# shrinking = [True, False]
# degree = [3, 4, 5]

# svm_grid = {'C': c, 'gamma' : gamma, 'epsilon': epsilon, 'shrinking' : shrinking, 'degree': degree}

# svm = SVR(kernel='poly')
# svm_search = RandomizedSearchCV(svm, svm_grid, scoring='neg_mean_squared_error', cv=3, return_train_score=True, n_jobs=-1, n_iter=30, verbose=1)
# svm_search.fit(X_train_confirmed, y_train_confirmed)
# svm_confirmed = svm_search.best_estimator_
svm_confirmed = SVR(shrinking=True, kernel='poly',gamma=0.01, epsilon=1,degree=5, C=0.1)
svm_confirmed.fit(X_train_confirmed, y_train_confirmed)
svm_pred = svm_confirmed.predict(future_forcast)
# check against testing data
svm_test_pred = svm_confirmed.predict(X_test_confirmed)
plt.plot(svm_test_pred)
plt.plot(y_test_confirmed)
print('MAE:', mean_absolute_error(svm_test_pred, y_test_confirmed))
print('MSE:',mean_squared_error(svm_test_pred, y_test_confirmed))


plt.figure(figsize=(20, 12))
plt.plot(adjusted_dates, world_cases)
plt.plot(future_forcast, svm_pred, linestyle='dashed', color='purple')
plt.title('# of Coronavirus Cases Over Time', size=30)
plt.xlabel('Days Since 1/22/2020', size=30)
plt.ylabel('# of Cases', size=30)
plt.legend(['Confirmed Cases', 'SVM predictions'], prop={'size': 20})
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()

# Future predictions using SVM 
print('SVM future predictions:')
list(zip(future_forcast_dates[-10:], np.round(svm_pred[-10:])))

# Also creating Polynomial Features
poly = PolynomialFeatures(degree=5)
poly_X_train_confirmed = poly.fit_transform(X_train_confirmed)
poly_X_test_confirmed = poly.fit_transform(X_test_confirmed)
poly_future_forcast = poly.fit_transform(future_forcast)

linear_model = LinearRegression(normalize=True, fit_intercept=False)
linear_model.fit(poly_X_train_confirmed, y_train_confirmed)
test_linear_pred = linear_model.predict(poly_X_test_confirmed)
linear_pred = linear_model.predict(poly_future_forcast)

print('MAE:', mean_absolute_error(test_linear_pred, y_test_confirmed))
print('MSE:',mean_squared_error(test_linear_pred, y_test_confirmed))

plt.plot(test_linear_pred)
plt.plot(y_test_confirmed)
# Future predictions using Polynomial Regression 
linear_pred = linear_pred.reshape(1,-1)[0]
print('Polynomial regression future predictions:')
list(zip(future_forcast_dates[-10:], np.round(linear_pred[-10:])))


plt.figure(figsize=(20, 12))
plt.plot(adjusted_dates, world_cases)
plt.plot(future_forcast, linear_pred, linestyle='dashed', color='orange')
plt.title('# of Coronavirus Cases Over Time', size=30)
plt.xlabel('Days Since 1/22/2020', size=30)
plt.ylabel('# of Cases', size=30)
plt.legend(['Confirmed Cases', 'Polynomial Regression Predictions'], prop={'size': 20})
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()

pm.__version__
#world_cases

#import pyramid as pm
#from pyramid.arima import auto_arima
tsarima = auto_arima(world_cases,start_p=1, d=2, start_q=1, max_p=5, max_d=2, max_q=5, 
 start_P=1, D=None, start_Q=1, max_P=2, max_D=1, max_Q=2, 
 max_order=10, m=2, seasonal=True, stationary=False, information_criterion='aic', alpha=0.05, 
 test='kpss', seasonal_test='ch', stepwise=True,n_jobs=-1, start_params=None, trend='c',
 method=None, transparams=True, solver='lbfgs',maxiter=50, disp=0, callback=None, 
 offset_test_args=None, seasonal_test_args=None, suppress_warnings=True, 
 error_action='warn', trace=False, random=False,random_state=None,n_fits=10, 
return_valid_fits=False, out_of_sample_size=0, scoring='mse')

tsarima.fit(world_cases)
arima_pred = tsarima.predict(n_periods=10)
arima_pred = pd.DataFrame(arima_pred,index = future_forcast_dates[-10:],columns=['Prediction'])
plt.figure(figsize=(20, 12))
plt.plot(world_df["conf_cases"])
plt.plot(arima_pred, linestyle='dashed', color='purple')
plt.title('# of Coronavirus Cases Over Time', size=30)
plt.xlabel('Days Since 1/22/2020', size=30)
plt.ylabel('# of Cases', size=30)
plt.legend(['Confirmed Cases', 'ARIMA Predictions'], prop={'size': 20})
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()

# Future predictions using ARIMA
print('ARIMA future predictions:')
arima_pred

world_full.head()
data = world_full.copy()
#country_data['GrowthFactor'] = growth_factor(country_data['Confirmed'])

# we will want x_data to be the number of days since first confirmed and the y_data to be the confirmed data. This will be the data we use to fit a logistic curve
x_data = range(len(data.index))
y_data = data['conf_cases']

def log_curve(x, k, x_0, ymax):
    return ymax / (1 + np.exp(-k*(x-x_0)))

# Fit the curve
popt, pcov = curve_fit(log_curve, x_data, y_data, bounds=([0,0,0],np.inf), maxfev=5000)
estimated_k, estimated_x_0, ymax= popt


# Plot the fitted curve
k = estimated_k
x_0 = estimated_x_0
y_fitted = log_curve(x_data, k, x_0, ymax)
print(k, x_0, ymax)
#print(y_fitted)
y_data.tail()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x_data, y_fitted, '--', label='fitted')
ax.plot(x_data, y_data, 'o', label='Confirmed Data')

