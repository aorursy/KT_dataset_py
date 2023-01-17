import numpy as np

import pandas as pd

import plotly.express as px
wikipedia_2010_census = {'white': ['63.8'], 

        'black': ['18.6'], 

        'hispanic': ['10.5'],

        'asian': ['5.6'],

        'other': ['5.6'],

        'american_indian': ['2.0']}

wikipedia_2010_census_df = pd.DataFrame(wikipedia_2010_census, columns = ['white','black','hispanic','asian','other','american_indian']).transpose()

wikipedia_2010_census_df.columns = ['Percentage of Population']



fig = px.bar(wikipedia_2010_census_df, 

             x=wikipedia_2010_census_df.index, 

             y=wikipedia_2010_census_df['Percentage of Population'],

             title='Demographics of the City of Minneapolis (2020 Census: Wikipedia)')

fig.show()
police_stops_by_race = pd.read_csv('/kaggle/input/minneapolis-police-stops-and-police-violence/police_stop_data.csv',low_memory=False)

police_stops_by_race = police_stops_by_race['race'].value_counts()

police_stops_by_race_df = pd.DataFrame(police_stops_by_race)

police_stops_by_race_df.columns = ['Count of Incidents']



fig = px.bar(police_stops_by_race_df, 

             x=police_stops_by_race_df.index, 

             y=police_stops_by_race_df['Count of Incidents'],

             title='Minneapolis Police Stops by Race')

fig.show()
police_violence_by_race = pd.read_csv('/kaggle/input/minneapolis-police-stops-and-police-violence/police_use_of_force.csv',low_memory=False)

police_violence_by_race = police_violence_by_race['Race'].value_counts()

police_violence_by_race_df = pd.DataFrame(police_violence_by_race)

police_violence_by_race_df.columns = ['Count of Incidents']



fig = px.bar(police_violence_by_race_df, 

             x=police_violence_by_race_df.index, 

             y=police_violence_by_race_df['Count of Incidents'],

             title='Minneapolis Police Violence by Race')

fig.show()
black_over_white_stops = round(float(police_stops_by_race_df[0:1].values/police_stops_by_race_df[1:2].values),2)

black_over_white_violence = round(float(police_violence_by_race_df[0:1].values/police_violence_by_race_df[1:2].values),2)

print('These data suggest that African-Americans in Minneapolis are')

print(black_over_white_stops, 'times more often the subject of police stops')

print('and',black_over_white_violence, 'times more often the subject of ')

print('police violence as compared to citizens that identify as white --')

print('despite being less frequently represented in the overall population.\n')

black_percent_of_population = round(float(wikipedia_2010_census_df[1:2].values),2)

black_percent_of_incidents = round(float(police_stops_by_race_df[0:1].values*100/police_stops_by_race_df.sum().values),2)

black_percent_of_incidents_with_force = round(float(police_violence_by_race_df[0:1].values*100/police_violence_by_race_df.sum().values),2)

print(black_percent_of_population,'% of the population of Minneapolis identifies as Black,')

print(black_percent_of_incidents,'% of people stopped by police in Minneapolis are Black (~2x the % population), and')

print(black_percent_of_incidents_with_force,'% of people involved in police use of force incidents in Minneapolis are Black (~3.3x the % population).')
