''' Import libraries and utilities used in the analysis '''
import numpy as np
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
from matplotlib import cm as cm
from pandas.tools.plotting import parallel_coordinates

% matplotlib inline
''' Create connection '''
cnx = sqlite3.connect('../input/database.sqlite')
df_attributes = pd.read_sql_query('SELECT * FROM Player_Attributes',cnx)
df_players = pd.read_sql_query('SELECT * FROM Player',cnx)

''' Display first 5 rows in player's bio dataframe '''
df_players.head()
''' Display first 5 rows in player's attributes dataframe '''
df_attributes.head()
df_players = df_players[['player_api_id', 'player_name', 'birthday']]

df = pd.merge(df_players, df_attributes, how='inner', on=['player_api_id'], copy=False)
df.head()
df['birthday'] = pd.to_datetime(df['birthday'])
df['date'] = pd.to_datetime(df['date'])
df['age_in_years'] = df['date'].subtract(df['birthday']).divide(np.timedelta64(365, 'D')).round(decimals=0)
df.head()


df.columns
'''Select only the attributes of interest'''
cols_of_interest = ['player_name', 'age_in_years', 'overall_rating', 'ball_control', 
                    'short_passing', 'acceleration', 'sprint_speed']
df = df[cols_of_interest]
df.head(10)
(rows_before, cols_before) = df.shape
print("There are ", rows_before, " records in the initial dataframe.")
df = df.dropna()
(rows_after, cols_after) = df.shape
print("There are ", rows_after, " records with no NaN values in the cleaned dataframe.")
print("A total of ",rows_before - rows_after, "records were eliminated.")
df.describe().round(2)

df = df[df['age_in_years'] >= 16]
df = df[df['age_in_years'] <= 40]
df.shape
df.describe().round(2)
age_hist = df['age_in_years']
plt.hist(age_hist, 10, normed=False, facecolor='blue')
plt.xlabel("Age, (in years)")
plt.ylabel("Ratings Count")
plt.title("Age at time of rating")
plt.grid(True)
plt.show()

df.corr().round(3)
def correlation_matrix(df):

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap('jet', 10)
    cax = ax1.imshow(df.corr(), interpolation="nearest", cmap=cmap)
    ax1.grid(True)
    plt.title('Soccer Players\' Attribute Correlation')
    labels=['age','rating','control','passing', 'acceleration','speed']
    ax1.set_xticks(np.arange(len(labels)))
    ax1.set_yticks(np.arange(len(labels)))
    ax1.set_xticklabels(labels,fontsize=8)
    ax1.set_yticklabels(labels,fontsize=8)
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    fig.colorbar(cax, ticks=[-0.2,0.0,0.2,0.4,.6,.8,1.0])
    plt.show()

correlation_matrix(df)
average_ball_control = df[['age_in_years','ball_control']].groupby('age_in_years').mean()
average_short_passing = df[['age_in_years','short_passing']].groupby('age_in_years').mean()
average_speed = df[['age_in_years','sprint_speed']].groupby('age_in_years').mean()
average_acceleration = df[['age_in_years','acceleration']].groupby('age_in_years').mean()
average_overall_rating = df[['age_in_years','overall_rating']].groupby('age_in_years').mean()
age_series = df['age_in_years']
age_series = age_series.sort_values().unique()

attribute1 = plt.scatter(age_series, average_ball_control, label='Control', marker='o', c='magenta')
attribute2 = plt.scatter(age_series, average_short_passing, label='Passing', marker='^', c='cyan')
attribute3 = plt.scatter(age_series, average_speed, label='Speed', marker='s', c='lime')
attribute4 = plt.scatter(age_series, average_acceleration, label='Acceleration', marker='o', c='red')
attribute5 = plt.scatter(age_series, average_overall_rating, label='Rating', marker='s', c='blue')
plt.xlabel("Age, (in years)", fontsize = 10)
plt.ylabel("Attribute Average", fontsize = 10)
plt.title("Attribute Average as a Function of Age", fontsize = 12)

plt.grid(True)
plt.xlim(15,41)
plt.ylim(1,75)
plt.legend(handles=[attribute1, attribute2, attribute3, attribute4, attribute5], loc=4, fontsize=10)
plt.show()
