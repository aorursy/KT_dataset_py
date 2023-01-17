import numpy as np

import pandas as pd

import plotly.express as px

from datetime import date



def load_b4bl_data(file_name):

    df = pd.read_csv(file_name)[13:]

    df.columns = ['State', 

                  'Data Source',

                  'Total positive cases',

                  'Total deaths',

                  'Percent Black Cases',

                  'Percent Black Deaths',

                  'Percent Black Population',

                  'Updated', 

                  'Notes']

    return df

df = load_b4bl_data('/kaggle/input/covid19-cases-and-deaths-by-race/covid-19-data-by-race.csv')

df = df[23:] # Drop V1 data, retain only V2 data

df['Percent of Population'] = df['Percent Black Population'].astype(float)

df = df.sort_values('Percent of Population',ascending=False)[:]

df.head()
fig = px.bar(df, 

             x="State", 

             y="Percent Black Population",

             title='Percent of Population Identifying as Black')

fig.show()



fig = px.bar(df, 

             x="State", 

             y="Percent Black Deaths",

             title='Percent of COVID-19 Deaths in Black Communities')

fig.show()
todays_date = str(date.today())

print('Last run on '+todays_date)

df.to_csv('/kaggle/working/covid19-cases-and-deaths-by-race.csv')