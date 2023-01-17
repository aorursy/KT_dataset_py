import numpy as np 

import pandas as pd 

import re

import warnings



warnings.filterwarnings('ignore')



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
population_df = pd.read_csv("/kaggle/input/covid19-in-india/population_india_census2011.csv", index_col=0) 

statewise_testing_df = pd.read_csv("/kaggle/input/covid19-in-india/StatewiseTestingDetails.csv", index_col=0)
population_df.head()
statewise_testing_df.head()
statewise_testing_df.reset_index(inplace=True)



statewise_testing_df['Date'] = pd.to_datetime(statewise_testing_df['Date'], format="%Y-%m-%d")



statewise_testing_df['Date'].min(), statewise_testing_df['Date'].max()
population_df = population_df[['State / Union Territory','Density']]

statewise_testing_df = statewise_testing_df[['Date','TotalSamples','State','Positive']]
print(population_df['State / Union Territory'].nunique())

pop_states = set(population_df['State / Union Territory'].unique())
print(statewise_testing_df['State'].nunique())

statewise_testing_states = set(statewise_testing_df['State'].unique())
pop_states - statewise_testing_states
statewise_testing_states - pop_states
population_df.loc[population_df['State / Union Territory'].str.contains('ngana')]
population_df.loc[population_df['State / Union Territory'].str.contains('ngana'),'State / Union Territory'] = "Telangana"
statewise_features = statewise_testing_df.reset_index().merge(population_df, \

                                    how='inner', \

                                   left_on='State', \

                                   right_on='State / Union Territory')

statewise_features = statewise_features.drop(["State / Union Territory",'index'], axis=1)
statewise_features[statewise_features['State']=='West Bengal'].head()
statewise_features['Density'] = statewise_features['Density'].apply(lambda density: re.sub(",", "",density))

statewise_features['pop_density'] = statewise_features['Density'].str.extract("(\d+)").astype(float)

statewise_features.drop("Density", axis=1, inplace=True)
statewise_features.head()
statewise_features.info()
statewise_daily_df = None

for state in statewise_features['State'].unique():

    covid_data_state = statewise_features[statewise_features['State']==state]

    covid_data_state['previous_day'] = covid_data_state['Positive'].shift(1)

    covid_data_state['new_cases'] = covid_data_state['Positive'] - covid_data_state['previous_day']



    covid_data_state['previous_day'] = covid_data_state['TotalSamples'].shift(1)

    covid_data_state['samples_tested'] = covid_data_state['TotalSamples'] - covid_data_state['previous_day']



    covid_data_state = covid_data_state.drop('previous_day',axis=1)

    statewise_daily_df = pd.concat([statewise_daily_df, covid_data_state], axis=0)

    

statewise_daily_df.set_index('Date', inplace=True)

statewise_daily_df.head()
statewise_daily_df.drop(['TotalSamples','Positive'], axis=1).corr()
statewise_daily_df.dropna(inplace=True)

statewise_daily_df.drop(['TotalSamples','Positive'], axis=1).to_csv("./statewise_features.csv")