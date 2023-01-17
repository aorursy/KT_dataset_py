#!pip install pycountry_convert 

#!pip install folium

#!pip install calmap
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from matplotlib import ticker 

# import pycountry_convert as pc

import folium

import branca

from datetime import datetime, timedelta,date

from scipy.interpolate import make_interp_spline, BSpline

import plotly.express as px

import json, requests

#import calmap

import seaborn as sns



from keras.layers import Input, Dense, Activation, LeakyReLU

from keras import models

from keras.optimizers import RMSprop, Adam

plt.style.use('fivethirtyeight')



import warnings

warnings.filterwarnings('ignore')



%matplotlib inline
latest_date = datetime.today()- timedelta(days=2)

latest_date = latest_date.strftime('%m/%d/%y')[1:]



df_cases = pd.read_csv('https://usafactsstatic.blob.core.windows.net/public/data/covid-19/covid_confirmed_usafacts.csv')[['countyFIPS', 'County Name', 'State', latest_date]]

df_cases = df_cases.rename(columns={'countyFIPS': 'county_fips',

                                                  latest_date: 'confirmed'}).set_index('county_fips')



df_deaths = pd.read_csv('https://usafactsstatic.blob.core.windows.net/public/data/covid-19/covid_deaths_usafacts.csv')[['countyFIPS', latest_date]]

df_deaths = df_deaths.rename(columns={'countyFIPS': 'county_fips',

                                                  latest_date: 'deaths'}).set_index('county_fips')





df_pop = pd.read_csv('https://usafactsstatic.blob.core.windows.net/public/data/covid-19/covid_county_population_usafacts.csv')[['countyFIPS', 'population']]

df_pop = df_pop.rename(columns={'countyFIPS': 'county_fips'}).set_index('county_fips')



df = df_cases.join(df_deaths)

df = df.join(df_pop)



df = df[df.index > 999]

df = df[df.population>0]



del df_cases, df_deaths, df_pop



df['mortality'] = df['deaths']/ df['confirmed']

df['mortality'] = df['mortality'].fillna(0)



df['deaths_per_million'] = df['deaths'] * 1000000/ df['population']

df['cases_per_million'] = df['confirmed'] * 1000000/ df['population']



df['likely_infected_high'] = np.round(df['confirmed'] * 80/ df['population'], 2)

df['likely_infected_high'] = np.clip(df['likely_infected_high'], 0, 1)

df['likely_infected_low'] = np.round(df['confirmed'] * 28/ df['population'], 2)

df['likely_infected_low'] = np.clip(df['likely_infected_low'], 0, 1)



df['county_state'] = df['County Name'] + ', ' + df['State']

print('Number of counties: ' + str(df.index.nunique()))
df_county_stats = pd.read_csv('/kaggle/input/uncover/UNCOVER/county_health_rankings/county_health_rankings/us-county-health-rankings-2020.csv')[['fips',

                                                                                            'segregation_index',

                                                                                            'percent_black',

                                                                                            'median_household_income',

                                                                                            'percent_adults_with_obesity',

                                                                                            'percent_smokers',

                                                                                            'percent_adults_with_diabetes',                                           

                                                                                            'percent_with_access_to_exercise_opportunities',

                                                                                            'percent_some_college',

                                                                                            'percent_unemployed',

                                                                                            'percent_children_in_poverty',

                                                                                            'percent_female',

                                                                                            'percent_rural',

                                                                                            'percent_non_hispanic_white',

                                                                                            'food_environment_index',

                                                                                            'percent_food_insecure',

                                                                                            'income_ratio',

                                                                                            '80th_percentile_income',

                                                                                            'percent_excessive_drinking',

                                                                                            'average_number_of_physically_unhealthy_days',

                                                                                            'percent_uninsured',

                                                                                            'high_school_graduation_rate',

                                                                                            'percent_single_parent_households',

                                                                                            'social_association_rate',

                                                                                            'infant_mortality_rate',

                                                                                            'age_adjusted_death_rate',                                             

                                                                                             ]]

df_county_stats = df_county_stats.rename(columns={'fips': 'county_fips',

                                                  'segregation_index': 'segregation_level',

                                                  }).set_index('county_fips')



df = df.join(df_county_stats)



df_county_stats = pd.read_csv('/kaggle/input/county-ranking-data/county_ranking.csv')[['fipscode',

                                                                                            'v052_rawvalue',

                                                                                            'v053_rawvalue',

                                                                                            'v044_rawvalue',

                                                                                            'v147_rawvalue',

                                                                                            'v002_cilow',

                                                                                            'v136_other_data_2']]



df_county_stats = df_county_stats.rename(columns={'fipscode': 'county_fips',

                                                  'v052_rawvalue': 'percent_below_18',

                                                  'v053_rawvalue': 'percent_above_65',

                                                  'v044_rawvalue': 'income_inequality',

                                                  'v147_rawvalue': 'life_expectancy',

                                                  'v002_cilow': 'poor_fair_health',

                                                  'v136_other_data_2': 'over_crowding'

                                                  }).set_index('county_fips')





df = df.join(df_county_stats)

df = df.reset_index()

df['county_fips'] = df['county_fips'].astype(str).str.rjust(5,'0')

#df.head()
#gender

plt.style.use('dark_background')

plt.figure(figsize=(25,8))

val = [500, 2000, 8000]

colors = ["magenta","yellow","cyan"]

col_name = 'percent_female'





for i in range(3):

    plt.subplot(1, 3, i+1)

    sns.regplot(df[df.confirmed>val[i]][col_name], df[df.confirmed>val[i]].mortality,lowess=True, color=colors[i])

    plt.title('Confirmed Cases > ' + str(val[i]),fontsize=20)

    plt.xlabel(col_name)

    plt.ylabel('mortality', fontsize=20)

    plt.grid(False)

    if i%3 !=0:

            plt.ylabel(' ')
#rural

col_name = 'percent_rural'

plt.figure(figsize=(25,8))





for i in range(3):

    plt.subplot(1, 3, i+1)

    sns.regplot(df[df.confirmed>val[i]][col_name], df[df.confirmed>val[i]].mortality,lowess=True, color=colors[i])

    plt.title('Confirmed Cases > ' + str(val[i]), fontsize=20)

    plt.xlabel(col_name)

    plt.ylabel('mortality', fontsize=20)

    plt.grid(False)

    if i%3 !=0:

            plt.ylabel(' ')
#age above 65

col_name = 'percent_above_65'

plt.figure(figsize=(25,8))





for i in range(3):

    plt.subplot(1, 3, i+1)

    sns.regplot(df[df.confirmed>val[i]][col_name], df[df.confirmed>val[i]].mortality,lowess=True, color=colors[i])

    plt.title('Confirmed Cases > ' + str(val[i]), fontsize=20)

    plt.xlabel(col_name)

    plt.ylabel('mortality', fontsize=20)

    plt.grid(False)

    if i%3 !=0:

            plt.ylabel(' ')
#race

plt.figure(figsize=(20,4))

col_name = 'percent_non_hispanic_white'





for i in range(3):

    plt.subplot(1, 3, i+1)

    sns.regplot(df[df.confirmed>val[i]][col_name], df[df.confirmed>val[i]].mortality,lowess=True, color=colors[i])

    plt.title('Confirmed Cases > ' + str(val[i]))

    plt.xlabel(col_name)

    plt.ylabel('mortality', fontsize=20)

    plt.grid(False)

    if i%3 !=0:

            plt.ylabel(' ')
col_name = 'percent_single_parent_households'

plt.figure(figsize=(25,8))





for i in range(3):

    plt.subplot(1, 3, i+1)

    sns.regplot(df[df.confirmed>val[i]][col_name], df[df.confirmed>val[i]].mortality,lowess=True, color=colors[i])

    plt.title('Confirmed Cases > ' + str(val[i]))

    plt.xlabel(col_name)

    plt.ylabel('mortality', fontsize=20)

    plt.grid(False)

    if i%3 !=0:

            plt.ylabel(' ')
#income

col_name = 'income_ratio'

plt.figure(figsize=(25,8))





for i in range(3):

    plt.subplot(1, 3, i+1)

    sns.regplot(df[df.confirmed>val[i]][col_name], df[df.confirmed>val[i]].mortality,lowess=True, color=colors[i])

    plt.title('Confirmed Cases > ' + str(val[i]))

    plt.xlabel(col_name)

    plt.ylabel('mortality', fontsize=20)

    plt.grid(False)

    if i%3 !=0:

            plt.ylabel(' ')
#unemployed

col_name = 'percent_unemployed'

plt.figure(figsize=(25,8))





for i in range(3):

    plt.subplot(1, 3, i+1)

    sns.regplot(df[df.confirmed>val[i]][col_name], df[df.confirmed>val[i]].mortality,lowess=True, color=colors[i])

    plt.title('Confirmed Cases > ' + str(val[i]))

    plt.xlabel(col_name)

    plt.grid(False)

    plt.ylabel('mortality', fontsize=20)

    if i%3 !=0:

            plt.ylabel(' ')
#uninsured

col_name = 'percent_uninsured'

plt.figure(figsize=(25,8))





for i in range(3):

    plt.subplot(1, 3, i+1)

    sns.regplot(df[df.confirmed>val[i]][col_name], df[df.confirmed>val[i]].mortality,lowess=True, color=colors[i])

    plt.title('Confirmed Cases > ' + str(val[i]))

    plt.xlabel(col_name)

    plt.grid(False)

    plt.ylabel('mortality', fontsize=20)

    if i%3 !=0:

            plt.ylabel(' ')
#segregation

col_name = 'segregation_level'

plt.figure(figsize=(25,8))





for i in range(3):

    plt.subplot(1, 3, i+1)

    sns.regplot(df[df.confirmed>val[i]][col_name], df[df.confirmed>val[i]].mortality,lowess=True, color=colors[i])

    plt.title('Confirmed Cases > ' + str(val[i]))

    plt.xlabel(col_name)

    plt.grid(False)

    plt.ylabel('mortality', fontsize=20)

    if i%3 !=0:

            plt.ylabel(' ')
#food-environment

col_name = 'food_environment_index'

plt.figure(figsize=(25,8))





for i in range(3):

    plt.subplot(1, 3, i+1)

    sns.regplot(df[df.confirmed>val[i]][col_name], df[df.confirmed>val[i]].mortality,lowess=True, color=colors[i])

    plt.title('Confirmed Cases > ' + str(val[i]))

    plt.xlabel(col_name)

    plt.grid(False)

    plt.ylabel('mortality', fontsize=20)

    if i%3 !=0:

            plt.ylabel(' ')
#food-insecure

col_name = 'percent_food_insecure'

plt.figure(figsize=(25,8))





for i in range(3):

    plt.subplot(1, 3, i+1)

    sns.regplot(df[df.confirmed>val[i]][col_name], df[df.confirmed>val[i]].mortality,lowess=True, color=colors[i])

    plt.title('Confirmed Cases > ' + str(val[i]))

    plt.xlabel(col_name)

    plt.grid(False)

    plt.ylabel('mortality', fontsize=20)

    if i%3 !=0:

            plt.ylabel(' ')
#smokers

col_name = 'percent_smokers'

plt.figure(figsize=(25,8))





for i in range(3):

    plt.subplot(1, 3, i+1)

    sns.regplot(df[df.confirmed>val[i]][col_name], df[df.confirmed>val[i]].mortality,lowess=True, color=colors[i])

    plt.title('Confirmed Cases > ' + str(val[i]))

    plt.xlabel(col_name)

    plt.grid(False)

    plt.ylabel('mortality', fontsize=20)

    if i%3 !=0:

            plt.ylabel(' ')
#alcohol

col_name = 'percent_excessive_drinking'

plt.figure(figsize=(25,8))





for i in range(3):

    plt.subplot(1, 3, i+1)

    sns.regplot(df[df.confirmed>val[i]][col_name], df[df.confirmed>val[i]].mortality,lowess=True, color=colors[i])

    plt.title('Confirmed Cases > ' + str(val[i]))

    plt.xlabel(col_name)

    plt.grid(False)

    plt.ylabel('mortality', fontsize=20)

    if i%3 !=0:

            plt.ylabel(' ')
#diabetes

col_name = 'percent_adults_with_diabetes'

plt.figure(figsize=(25,8))





for i in range(3):

    plt.subplot(1, 3, i+1)

    sns.regplot(df[df.confirmed>val[i]][col_name], df[df.confirmed>val[i]].mortality,lowess=True, color=colors[i])

    plt.title('Confirmed Cases > ' + str(val[i]))

    plt.xlabel(col_name)

    plt.grid(False)

    plt.ylabel('mortality', fontsize=20)

    if i%3 !=0:

            plt.ylabel(' ')
#life expectancy

col_name = 'life_expectancy'

plt.figure(figsize=(25,8))





for i in range(3):

    plt.subplot(1, 3, i+1)

    sns.regplot(df[df.confirmed>val[i]][col_name], df[df.confirmed>val[i]].mortality,lowess=True, color=colors[i])

    plt.title('Confirmed Cases > ' + str(val[i]))

    plt.xlabel(col_name)

    plt.grid(False)

    plt.ylabel('mortality', fontsize=20)

    if i%3 !=0:

            plt.ylabel(' ')
#imr

col_name = 'infant_mortality_rate'

plt.figure(figsize=(25,8))





for i in range(3):

    plt.subplot(1, 3, i+1)

    sns.regplot(df[df.confirmed>val[i]][col_name], df[df.confirmed>val[i]].mortality,lowess=True, color=colors[i])

    plt.title('Confirmed Cases > ' + str(val[i]))

    plt.xlabel(col_name)

    plt.grid(False)

    plt.ylabel('mortality', fontsize=20)

    if i%3 !=0:

            plt.ylabel(' ')
#soc association

col_name = 'social_association_rate'

plt.figure(figsize=(25,8))





for i in range(3):

    plt.subplot(1, 3, i+1)

    sns.regplot(df[df.confirmed>val[i]][col_name], df[df.confirmed>val[i]].mortality,lowess=True, color=colors[i])

    plt.title('Confirmed Cases > ' + str(val[i]))

    plt.xlabel(col_name)

    plt.grid(False)

    plt.ylabel('mortality', fontsize=20)

    if i%3 !=0:

            plt.ylabel(' ')
df_confirmed = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')

df_deaths = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')



# Depricated

# df_recovered = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Recovered.csv')

df_covid19 = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/web-data/data/cases_country.csv")

df_table = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/web-data/data/cases_time.csv",parse_dates=['Last_Update'])
date_usa = datetime.strptime(df_confirmed.columns[-1],'%m/%d/%y').strftime("%m-%d-%Y")

df_temp = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/"+date_usa+".csv")

df_usa = df_temp.loc[df_temp["Country_Region"]== "US"]

df_usa = df_usa.rename(columns={"Admin2":"County"})
f = plt.figure(figsize=(15,10))

f.add_subplot(111)



plt.axes(axisbelow=True)

plt.barh(df_usa.groupby(["Province_State"]).sum().sort_values('Confirmed')["Confirmed"].index[-10:],df_usa.groupby(["Province_State"]).sum().sort_values('Confirmed')["Confirmed"].values[-10:],color="cyan")

plt.tick_params(size=5,labelsize = 13)

plt.xlabel("Confirmed Cases",fontsize=18)

plt.title("Top 10 States: USA (Confirmed Cases)",fontsize=20)

plt.grid(False)

plt.savefig('Top 10 States_USA (Confirmed Cases).png')
f = plt.figure(figsize=(15,10))

f.add_subplot(111)



plt.axes(axisbelow=True)

plt.barh(df_usa.groupby(["Province_State"]).sum().sort_values('Deaths')["Deaths"].index[-10:],df_usa.groupby(["Province_State"]).sum().sort_values('Deaths')["Deaths"].values[-10:],color="cyan")

plt.tick_params(size=5,labelsize = 13)

plt.xlabel("Deaths",fontsize=18)

plt.title("Top 10 States: USA (Deaths)",fontsize=20)

plt.grid(False)

plt.savefig('Top 10 States_USA (COVID Deaths).png')
f = plt.figure(figsize=(15,10))

f.add_subplot(111)



plt.axes(axisbelow=True)

plt.barh(df_usa.groupby(["County"]).sum().sort_values('Deaths')["Deaths"].index[-10:],df_usa.groupby(["County"]).sum().sort_values('Deaths')["Deaths"].values[-10:],color="cyan")

plt.tick_params(size=5,labelsize = 13)

plt.xlabel("Deaths",fontsize=18)

plt.title("Top 10 County: USA (Deaths)",fontsize=20)

plt.grid(False)

plt.savefig('Top 10 County_USA (COVID Deaths).png')
f = plt.figure(figsize=(15,10))

f.add_subplot(111)



plt.axes(axisbelow=True)

plt.barh(df_usa.groupby(["County"]).sum().sort_values('Confirmed')["Confirmed"].index[-10:],df_usa.groupby(["County"]).sum().sort_values('Confirmed')["Confirmed"].values[-10:],color="cyan")

plt.tick_params(size=5,labelsize = 13)

plt.xlabel("Confirmed Cases",fontsize=18)

plt.title("Top 10 Counties: USA (Confirmed Cases)",fontsize=20)

plt.grid(False)

plt.savefig('Top 10 Counties_USA (Confirmed Cases).png')
df_usa = df_usa.replace(np.nan, 0, regex=True)

usa = folium.Map(location=[37, -102], zoom_start=4,max_zoom=9,min_zoom=4, zoom_control=False, scrollWheelZoom= False, dragging=False)

for i in np.int32(np.asarray(df_usa[df_usa['Confirmed'] > 0].index)):

    folium.Circle(

        location=[df_usa.loc[i]['Lat'], df_usa.loc[i]['Long_']],

        tooltip = "<h5 style='text-align:center;font-weight: bold'>"+df_usa.loc[i]['Province_State']+"</h5>"+

                    "<div style='text-align:center;'>"+str(np.nan_to_num(df_usa.loc[i]['County']))+"</div>"+

                    "<hr style='margin:10px;'>"+

                    "<ul style='color: #444;list-style-type:circle;align-item:left;padding-left:20px;padding-right:20px'>"+

        "<li>Confirmed: "+str(df_usa.loc[i]['Confirmed'])+"</li>"+

        "<li>Active:   "+str(df_usa.loc[i]['Active'])+"</li>"+

        "<li>Recovered:   "+str(df_usa.loc[i]['Recovered'])+"</li>"+       

        "<li>Deaths:   "+str(df_usa.loc[i]['Deaths'])+"</li>"+

        "<li>Mortality Rate:   "+str(np.round(df_usa.loc[i]['Deaths']/(df_usa.loc[i]['Confirmed']+1)*100,2))+"</li>"+

        "</ul>"

        ,

        radius=int((np.log2(df_usa.loc[i]['Confirmed']+1))*6000),

        color='#ff6600',

        fill_color='#ff8534',                                  

        fill=True).add_to(usa)



usa
#US daily cases

df_confirmed['country'] = df_confirmed['Country/Region']

df_confirmed_report = df_confirmed.copy()



df_confirmed_report.loc[df_confirmed_report['country'] != "US", "country"] = "Outside United States"

df_confirmed_report = df_confirmed_report.groupby("country").sum().drop(["Lat","Long"],axis =1)

df_confirmed_report.loc["Total"] = df_confirmed_report.sum()

df_confirmed_newcases = df_confirmed_report.groupby(level =0).diff(axis =1)

df_confirmed_newcases = df_confirmed_newcases.replace(np.nan, 0, regex=True) 

f = plt.figure(figsize=(20,10))

plt.grid(False)

ax1 = f.add_subplot(111)

ax1.bar(df_confirmed_report[df_confirmed_report.index == "US"].columns,df_confirmed_newcases[df_confirmed_newcases.index == "US"].values[0], label = "US (New)",color='limegreen')

ax1.bar(df_confirmed_report[df_confirmed_report.index == "Outside United States"].columns,df_confirmed_newcases[df_confirmed_newcases.index == "Outside United States"].values[0],bottom=df_confirmed_newcases[df_confirmed_newcases.index == "US"].values[0],label = "Outside United States (New)",color='cyan')



# Labels

ax1.set_xlabel("Dates",fontsize=17)

ax1.set_ylabel("New Cases Reported",fontsize =17)



ax1.tick_params(size=10,labelsize=15)

ax1.set_xticks(np.arange(0.5, len(df_confirmed_report.columns), 6))

ax1.set_xticklabels([datetime.strptime(date,'%m/%d/%y').strftime("%d %b") for date in df_confirmed_report.columns][::6],fontsize=15)

l = np.arange(0, df_confirmed_report.max(axis = 1)[2]/10+10000, 5000)

ax1.set_yticks(l[::int(len(l)/5)])

# ax1.spines['bottom'].set_position('zero')



ax2 = ax1.twinx()

marker_style = dict(linewidth=6, linestyle='--',markersize=25, markerfacecolor='#ffffff')



ax2.plot(df_confirmed_report[df_confirmed_report.index == "Total"].columns ,df_confirmed_report[df_confirmed_report.index == "Total"].values[0],**marker_style,label = "World Total (Cumulative)",color="red",clip_on=False)

ax2.plot(df_confirmed_report[df_confirmed_report.index == "US"].columns ,df_confirmed_report[df_confirmed_report.index == "US"].values[0],**marker_style,label = "US (Cumulative)",color="magenta",clip_on=False)

ax2.plot(df_confirmed_report[df_confirmed_report.index == "Outside United States"].columns ,df_confirmed_report[df_confirmed_report.index == "Outside United States"].values[0],**marker_style,label ="Outside United States (Cumulative)",color="yellow",clip_on=False)

ax2.bar([0],[0])



# Label

ax2.tick_params(labelsize=15)

ax2.set_ylabel("Cumulative (Million)",fontsize =17)

ax2.set_xticks(np.arange(0.5, len(df_confirmed_report.columns), 6))

ax2.set_xticklabels([datetime.strptime(date,'%m/%d/%y').strftime("%d %b") for date in df_confirmed_report.columns][::6])

l = np.arange(0, df_confirmed_report.max(axis = 1)[2]+100000, 100000)

ax2.set_yticks(l[::int(len(l)/5)])



f.tight_layout()

f.legend(loc = "upper left", bbox_to_anchor=(0.1,0.95), fontsize = 22)

plt.title("COVID-19 Confirmed Cases: United States and the Rest of the World",fontsize = 25)

plt.savefig('United States vs Rest of the world.png')

plt.show()