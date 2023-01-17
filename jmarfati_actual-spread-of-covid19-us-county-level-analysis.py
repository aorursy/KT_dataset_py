import numpy as np 

import pandas as pd 



import os

from urllib.request import urlopen

import json



import plotly.express as px       

import plotly.offline as py       

import plotly.graph_objects as go 

from plotly.subplots import make_subplots



import seaborn as sns

import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')



from datetime import datetime, timedelta

import warnings

warnings.filterwarnings("ignore")



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        

        print(os.path.join(dirname, filename))
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

                                                                                            'percent_with_access_to_exercise_opportunities',

                                                                                            'percent_some_college',

                                                                                            'percent_unemployed',

                                                                                            'percent_children_in_poverty',

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

df.head()
!pip install chart_studio

!pip install plotly-geo
plt.figure(figsize=(10,5))



sns.distplot(df.likely_infected_high, hist=True, kde=False, color = 'red', 

             hist_kws={'edgecolor':'black', 'linewidth':1},

             kde_kws={'linewidth': 2})



print('Summary Statistic of Percetnage of Population Likely Infected across counties: \n')

print(df.likely_infected_high.describe())



plt.xlim(0, 1)

plt.title('Distribution of county population likely infected')

plt.xlabel('Percentage of population likely infected')



plt.show()
def show_values_on_bars(axs, h_v="v", space=0.4, text_size=10):

    def _show_on_single_plot(ax):

        if h_v == "v":

            for p in ax.patches:

                _x = p.get_x() + p.get_width() / 2

                _y = p.get_y() + p.get_height()

                value = p.get_height()

                ax.text(_x, _y, value, ha="center", size=text_size) 

        elif h_v == "h":

            for p in ax.patches:

                _x = p.get_x() + p.get_width() + float(space)

                _y = p.get_y() + p.get_height()- float(0.2)

                value = p.get_width() 

                value = "{:.1%}".format(value)

                ax.text(_x, _y, value, ha="left", size=text_size)



    if isinstance(axs, np.ndarray):

        for idx, ax in np.ndenumerate(axs):

            _show_on_single_plot(ax)

    else:

        _show_on_single_plot(axs)



plt.figure(figsize=(10,10))

g=sns.barplot(x='likely_infected_low', y='county_state',data=df.sort_values(['likely_infected_low'], ascending=False).head(20), color="lightgreen")

show_values_on_bars(g, "h", space=0.01, text_size=10)

plt.xlim(0, 1.1)

plt.xlabel("Percentage of population infected")

plt.ylabel("County, State")

plt.title("Likely spread of Virus if spread is 28 fold")
import plotly.express as px

with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:

    counties = json.load(response)



fig = px.choropleth(df, geojson=counties, locations='county_fips', color='likely_infected_low',

                           color_continuous_scale="Reds",

                           range_color=(0, 0.2),

                           scope="usa",

                           title="Percentage of population likely already infected if spread is 28 folds",

                           hover_name= "county_state",

                           hover_data=["confirmed", "deaths"],

                           labels={'likely_infected_low': '% Likely Infected',

                                   'confirmed': 'Confirmed Cases ',

                                   'deaths': 'Deaths '}

                          )

fig.update_layout(margin={"r":0,"t":30,"l":0,"b":0})

fig.layout.template = None

config = dict({'scrollZoom': False})

    

fig.show(config=config)

fig.write_html("1.html")
fig = px.choropleth(df, geojson=counties, locations='county_fips', color='likely_infected_high',

                           color_continuous_scale="Reds",

                           range_color=(0, 0.2),

                           scope="usa",

                           title="Percentage of population likely already infected if spread is 80 folds",

                           hover_name= "county_state",

                           hover_data=["confirmed", "deaths"],

                           labels={'likely_infected_high': '% Likely Infected',

                                   'confirmed': 'Confirmed Cases ',

                                   'deaths': 'Deaths '}

                          )

fig.update_layout(margin={"r":0,"t":30,"l":0,"b":0})

fig.layout.template = None

config = dict({'scrollZoom': False})

    

fig.show(config=config)

fig.write_html("2.html")
plt.figure(figsize=(10,5))

df_temp = df[df.confirmed>500]



sns.distplot(df_temp.mortality, hist=True, kde=False, color = 'green', 

             hist_kws={'edgecolor':'black', 'linewidth':1},

             kde_kws={'linewidth': 2})

plt.title('Distribution of Mortality Rate')

plt.xlabel('Covid Mortality Rate')
plt.figure(figsize=(20,8))



plt.subplot(1, 2, 1)

g=sns.barplot(x='mortality', y='county_state',data=df[df.confirmed>500].sort_values(['mortality'], ascending=False).head(10), color="red")

show_values_on_bars(g, "h", space=0.002, text_size=20)

plt.xlim(0, 0.15)

plt.xlabel("Covid Mortality Rate", size=20)

plt.ylabel(" ", size=20)

plt.yticks(size=15) 



plt.title("Counties with highest Covid Mortality", size=25)



plt.subplot(1, 2, 2)

g=sns.barplot(x='mortality', y='county_state',data=df[df.confirmed>500].sort_values(['mortality'], ascending=True).head(10), color="blue")

show_values_on_bars(g, "h", space=0.002, text_size=20)

plt.xlim(0, 0.05)

plt.xlabel("Covid Mortality Rate", size=20)

plt.ylabel(" ")

plt.yticks(size=15) 

plt.title("Counties with lowest Covid Mortality", size=25)

plt.tight_layout()

df['temp'] = - 1

df.loc[df.confirmed>100, 'temp'] = df['mortality']



fig = px.choropleth(df, geojson=counties, locations='county_fips', color='temp',

                           color_continuous_scale="Reds",

                           range_color=(0, 0.06),

                           hover_name= "county_state",

                           scope="usa",

                           labels={'temp':'Mortality Rate '}

                          )

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

fig.layout.template = None

config = dict({'scrollZoom': False})



fig.show(config=config)
plt.figure(figsize=(20,5))

plt.subplot(1, 2, 1)

sns.regplot(df_temp.percent_above_65, df_temp.mortality)



plt.subplot(1, 2, 2)

sns.barplot(pd.qcut(df_temp.percent_above_65, 4), df_temp.mortality)
plt.figure(figsize=(20,5))

plt.subplot(1, 2, 1)

sns.regplot(df_temp.percent_below_18, df_temp.mortality)



plt.subplot(1, 2, 2)

sns.barplot(pd.qcut(df_temp.percent_below_18, 4), df_temp.mortality)
df['dependency_ratio'] = df.percent_above_65 / df.percent_below_18



plt.figure(figsize=(20,5))

plt.subplot(1, 2, 1)

sns.regplot(df[df['confirmed']>500].dependency_ratio, df_temp.mortality)



plt.subplot(1, 2, 2)

sns.barplot(pd.qcut(df[df['confirmed']>500].dependency_ratio, 4), df_temp.mortality)
plt.figure(figsize=(20,5))

val = [500, 1000, 8000]

colors = ['red', 'blue', 'green']



for i in range(3):

    plt.subplot(1, 3, i+1)

    sns.regplot(df[df.confirmed>val[i]].segregation_level, df[df.confirmed>val[i]].mortality,lowess=True, color=colors[i])

    plt.title('Confirmed Cases > ' + str(val[i]))

    plt.xlabel('Segregation Level')

    if i>=1:

        plt.ylabel(' ')
import statsmodels.api as sm



x_vars = ['population', 'percent_black', 'median_household_income',

       'percent_adults_with_obesity', 'percent_smokers',

       'percent_with_access_to_exercise_opportunities', 'percent_some_college',

       'percent_unemployed', 'percent_children_in_poverty', 'percent_below_18',

       'percent_above_65', 'income_inequality', 'life_expectancy', 'over_crowding',

       'dependency_ratio', 'poor_fair_health'

       ]



y = df[df.confirmed>500].mortality

x = df[df.confirmed>500][x_vars]



x = sm.add_constant(x)

ols = sm.OLS(y, x).fit()

print('R2: ', ols.rsquared)
df['predicted_mortality'] = ols.predict(sm.add_constant(df[x_vars]))



fig = px.choropleth(df[df.confirmed<=500], geojson=counties, locations='county_fips', color='predicted_mortality',

                           color_continuous_scale="Reds",

                           range_color=(0, 0.06),

                           scope="usa",

                           hover_name= "county_state",

                           labels={'predicted_mortality':' Predicted Mortality Rate'}

                          )

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

fig.layout.template = None

config = dict({'scrollZoom': False})



fig.show(config=config)