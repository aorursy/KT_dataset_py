# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# from scipy.stats import binom



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
%matplotlib notebook
from scipy.stats import binom



def get_risk(df1, df2, states, counties, bias, max_group_size):

    census_df = df1

    us_df = df2



    def get_population(county, state):

        return census_df[(census_df.county == county)&(census_df.state == state)]['population'].values[0]

    

    county_population_sizes = []

    for i in range(len(states)):

        county_population_sizes.append(get_population(counties[i], states[i]))



    county_dfs = []



    for i in range(len(states)):

        temp = us_df[(us_df.state == states[i]) & (us_df.county == counties[i])].copy()

        county_dfs.append(temp)



    last_fourteen_days = []

    for county in county_dfs:

        last_fourteen_days.append(county.tail(14).copy())



    prob_arrs = []

    for i in range(len(states)):

        total_cases = abs(last_fourteen_days[i].iloc[-1,:].cases - last_fourteen_days[i].iloc[0,:].cases)

        # this is the line we need to change to account for under reporting for covid cases

        infected = total_cases * int(bias)

        pi = infected/county_population_sizes[i]

        group_size = range(max_group_size+1)

        prob_arrs.append((1-binom.pmf(0, group_size, pi)) * 100)



    new_dfs = []

    for i in range(len(states)):

        df = pd.DataFrame({'Risk': prob_arrs[i]})

        df['State/County'] = states[i] + '-' + counties[i]

        new_dfs.append(df)



    risk_df = pd.concat(new_dfs)

    risk_df2 = risk_df.reset_index()

    risk_df2.columns = ['Group_Size' if x=='index' else x for x in risk_df2.columns]



    return risk_df2
covid_df = pd.read_csv('https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv')

covid_df = covid_df[(covid_df.county !='Unknown') & (covid_df.state != 'Puerto Rico') & (covid_df.state != 'Virgin Islands')].copy()

census_df = pd.read_csv('https://raw.githubusercontent.com/dirtylittledirtbike/census_data/master/census_formatted3.csv')
# counties = ['Cook', 'Harris', 'Cook']

# states = ['Illinois', 'Texas', 'Florida', 'Minnesota']

# county_population_sizes = [5132480, 4767540, 2.7*10**6, 2.7*10**6]



# max_group_size = 100



max_group_size = 100

counties = ['Cook', 'Harris', 'Cook']

states = ['Illinois', 'Texas', 'Georgia']

bias = 10



risk_df = get_risk(census_df, covid_df, states, counties, bias, max_group_size)
risk_df
import plotly

plotly.offline.init_notebook_mode(connected=True)
from plotly import express as px



fig = px.line(risk_df, x="Group_Size", y="Risk",\

              color='State/County', width=800, height=700, title="Current Covid Risk % by Group Size")

fig.show()