import pandas as pd

import numpy as np

import plotly.express as px

df = pd.read_csv('//kaggle/input/colorado-covid19-cases-by-county/fbae539746324ca69ff34f086286845b_0.csv')

todays_date = df['Date_Data_Last_Updated'][0]

df = df[['FULL_', 'County_Pos_Cases']].sort_values('County_Pos_Cases',ascending=False)[0:15]

px.bar(df,x="FULL_",y="County_Pos_Cases",title='COVID-19 Infections in Colorado by County - '+todays_date).show()
