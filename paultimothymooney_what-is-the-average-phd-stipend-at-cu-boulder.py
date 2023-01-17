import numpy as np

import pandas as pd

import plotly.express as px



def return_avg_result(dataframe, university, department, year, measurement):

    new_df = pd.DataFrame()

    dataframe = dataframe[dataframe['Academic Year'].isin([year])]

    dataframe = dataframe[dataframe['University'].isin([university])]

    smaller_dataframe = dataframe[dataframe['Department'].isin([department])]

    new_df.loc[university+' All Departments'+' '+YEAR,'mean'] = dataframe.loc[:,measurement].mean(axis=0)

    new_df.loc[university+' All Departments'+' '+YEAR,'std'] = dataframe.loc[:,measurement].std(axis=0)

    new_df.loc[university+' All Departments'+' '+YEAR,'count'] = dataframe.loc[:,measurement].shape[0]

    new_df.loc[university+' '+department+' Department'+' '+YEAR,'mean'] = smaller_dataframe.loc[:,measurement].mean(axis=0)

    new_df.loc[university+' '+department+' Department'+' '+YEAR,'std'] = smaller_dataframe.loc[:,measurement].std(axis=0)

    new_df.loc[university+' '+department+' Department'+' '+YEAR,'count'] = smaller_dataframe.loc[:,measurement].shape[0]

    #print(measurement+' at '+university+' in '+year+':\n')

    return new_df



def return_popular_universities(dataframe,number_of_values):

    popular_universities = pd.DataFrame(dataframe['University'].value_counts()[1:number_of_values])

    #print('Number of Records Per University (Top '+str(number_of_values)+'):\n')

    return popular_universities



def return_popular_departments(dataframe,university,number_of_values):

    dataframe = dataframe[dataframe['University'].isin([university])]

    popular_departments = pd.DataFrame(dataframe['Department'].value_counts()[0:number_of_values])

    #print('Number of Records Per Department at '+UNIVERSITY+' (Top '+str(number_of_values)+'): \n')

    return popular_departments
PHD_STIPENDS = pd.read_csv('/kaggle/input/phd-stipends/csv') # load the data

PHD_STIPENDS['Overall Pay'].replace(regex=True,inplace=True,to_replace=r'\D',value=r'') # remove $ sign from column

PHD_STIPENDS['Overall Pay'] = PHD_STIPENDS['Overall Pay'].astype(float).fillna(0) # convert column to float

PHD_STIPENDS[['University','Department','Overall Pay','LW Ratio']].head(10) # preview the data
df = return_popular_universities(PHD_STIPENDS,number_of_values = 100)



df.reset_index(level=0, inplace=True)

df.columns=['University','Number of Records']

fig = px.bar(df, x='University', y="Number of Records",title='Number of Records Per University (Top 10)')

fig.update(layout=dict(xaxis_title='University',yaxis_title='Number of Records',legend_orientation="h",showlegend=True))

fig.update_yaxes(range=[0,140])

fig.show()
UNIVERSITY = 'University of Colorado - Boulder (UCB)'

df = return_popular_departments(PHD_STIPENDS, UNIVERSITY,number_of_values = 10)



df.reset_index(level=0, inplace=True)

df.columns=['Department','Number of Records']

fig = px.bar(df, x='Department', y="Number of Records",title='Number of Records Per Department at '+UNIVERSITY+' (Top 10)')

fig.update(layout=dict(xaxis_title='Department',yaxis_title='Number of Records',legend_orientation="h",showlegend=True))

fig.update_yaxes(range=[0,8])

fig.show()
UNIVERSITY = 'University of Colorado - Boulder (UCB)'

YEAR = '2019-2020'

MEASUREMENT = 'Overall Pay'

DEPARTMENT = 'Sociology'

df = return_avg_result(PHD_STIPENDS, UNIVERSITY, DEPARTMENT, YEAR, MEASUREMENT)



df.reset_index(level=0, inplace=True)

df.columns=['Cohort','Avg','Std','n']

fig = px.bar(df, x='Cohort', y="Avg",error_y="Std",title='Average Overall Pay at '+UNIVERSITY)

fig.update(layout=dict(xaxis_title='Cohort',yaxis_title='Average Overall Pay',legend_orientation="h",showlegend=True))

fig.update_yaxes(range=[0,35000])

fig.show()