import pandas as pd

import numpy as np

import plotly.express as px
race_data = pd.read_csv('/kaggle/input/covid-tracking-project-racial-data-tracker/Race Data Entry - CRDT.csv').fillna('0')

todays_date = race_data['Date'].max() # Find the most recent date in the dataset

race_data = race_data[race_data.Date.isin([todays_date])] # Limit to latest data onlyf



race_data_simple = race_data[['Cases_Total', 'Cases_White',

       'Cases_Black', 'Cases_LatinX', 'Cases_Asian', 'Cases_AIAN',

       'Cases_NHPI', 'Cases_Multiracial', 'Cases_Other',

       'Cases_Unknown', 'Cases_Ethnicity_Hispanic',

       'Cases_Ethnicity_NonHispanic', 'Cases_Ethnicity_Unknown']].astype(float)

race_data_simple = race_data_simple.div(race_data_simple.Cases_Total,axis=0)*100 # Calculate Percent of Total

race_data_simple = race_data_simple.fillna('0')

race_data_simple['State']= race_data['State']

race_data = race_data_simple

#race_data.to_csv('/kaggle/working/black_covid_data.csv',index=False)
fig = px.choropleth(race_data, 

                    locations="State", 

                    color="Cases_Black", 

                    locationmode = 'USA-states', 

                    hover_name="Cases_Black",

                    range_color=[10,70],

                    scope="usa",

                    title='Percentage of Total COVID-19 Cases by Black Communities')

fig.show()
fig = px.bar(race_data.sort_values('State', ascending=True), 

             x="State", 

             y="Cases_Black",

             title='Percentage of Total COVID-19 Cases by Black Communities (alphabetical)')

fig.show()

df = pd.read_csv('/kaggle/input/percent-black-population-for-every-state-in-usa/percent_black_over_time.csv')

df = df[1:]

cols_to_check = ['1790', '1800', '1810', '1820', '1830', '1840',

       '1850', '1860', '1870', '1880', '1890', '1900', '1910', '1920',

       '1930', '1940', '1950', '1960', '1970', '1980', '1990', '2000',

       '2010', '2018']

df[cols_to_check] = df[cols_to_check].replace({'%':''}, regex=True)

df = df.transpose()

new_header = df.iloc[0]

df = df[1:].fillna(0)

df.columns = new_header

df = df.astype(float)

df = df[-1:]

df = df.transpose()

df.columns = ['Percent Black']

df['State/Territory'] = df.index

df.to_csv('/kaggle/working/percent_black_state_by_state.csv',index=False) # save to notebook output
plot = px.bar(df, x=df.index, y="Percent Black", hover_name="Percent Black",title='Percentage of Population in USA that Identifies as Black (in 2018)') 

plot
df = pd.read_csv('/kaggle/input/percent-black-population-for-every-state-in-usa/percent_black_in_2018.csv')

fig = px.choropleth(df, 

                    locations="State Code", 

                    color="Percent Black", 

                    locationmode = 'USA-states', 

                    hover_name="State",

                    range_color=[0,30],scope="usa",

                    title='Percent of Population that Identifies as Black (2018)')

fig.show()
race_data['Cases_Black'] = race_data['Cases_Black'].astype(float)

fig = px.bar(race_data.sort_values('Cases_Black',ascending=False), 

             x="State", 

             y="Cases_Black",

             title='Percentage of Total COVID-19 Cases by Black Communities (sorted)')

fig.show()

race_data['Percent_Black'] = df['Percent Black']

race_data_simple = race_data[['Cases_Black','Percent_Black']].astype(float)

relative_to_population = race_data_simple['Cases_Black'].div(race_data_simple.Percent_Black,axis=0) # Calculate Percent of Total

relative_to_population = relative_to_population.fillna('0')

race_data['Black Cases Relative to Population'] = relative_to_population

race_data['State Code'] = df['State Code']

race_data['State'] = df['State']

race_data.to_csv('/kaggle/working/covid_cases_by_race.csv',index=False) # save to notebook output
race_data['Black Cases Relative to Population'] = race_data['Black Cases Relative to Population'].astype(float)

fig = px.bar(race_data.sort_values('Black Cases Relative to Population',ascending=False), 

             x="State", 

             y="Black Cases Relative to Population",

             title='Percent of COVID-19 Cases vs Percent of Population (sorted)')

fig.show()

fig = px.choropleth(race_data, 

                    locations="State Code", 

                    color="Black Cases Relative to Population", 

                    locationmode = 'USA-states', 

                    hover_name="State",

                    range_color=[0,3],scope="usa",

                    title='Percent of COVID-19 Cases vs Percent of Population')

fig.show()