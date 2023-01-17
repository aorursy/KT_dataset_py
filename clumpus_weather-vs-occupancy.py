#Importing libraries

import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

import os



#Importing the weather data and the occupants per hour

weather_log = pd.read_csv('../input/weather-log/weather_log.csv')

occupant_log = pd.read_csv('../input/occupant-log/occupant_log.csv')
#Dropping the placeholder columns created by each dataset

weather_log = weather_log.drop(columns = ['time', 'Unnamed: 0'])

occupant_log = occupant_log.drop(columns = {'Unnamed: 0'})



#Ensuring both 'realtime' columns are a datetime type, as that's what we're merging these two sets on

weather_log['realtime'] =  pd.to_datetime(weather_log['realtime'])

occupant_log['realtime'] =  pd.to_datetime(occupant_log['realtime'])
weather_log.head()
occupant_log.head()
#Merging both datasets together

df_combined = pd.merge(weather_log, occupant_log, on='realtime', how='left')

df_combined.set_index('realtime')
df_combined = df_combined.drop(columns = {'windGust'})
#Plotting heatmap to see correlations with Occupants

fig, ax = plt.subplots(figsize=(10,10))

sns.heatmap(df_combined.corr(), cmap='Blues')
#Setting up targets, features, and splitting up the data

targets = df_combined['Occupants']

features = df_combined.drop(columns = {'Occupants', 'realtime', 'precipType', 'icon', 'summary', 'Date'})



X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)
#Determining prediction score with Random Forest

forest = RandomForestRegressor(n_estimators=200,max_depth=6,random_state=0)

forest.fit(X_train, y_train)

y_predict = forest.predict(X_test)

forest_score = (forest.score(X_test, y_test))*100

forest_score