import sqlite3

import numpy as np

import numpy.polynomial.polynomial as poly

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from math import sqrt
conn = sqlite3.connect('database.sqlite')

cursor = conn.cursor()

pa = pd.read_sql_query("SELECT * FROM Player_Attributes",conn)

ta = pd.read_sql_query("SELECT * FROM Team_attributes", conn)

te = pd.read_sql_query("SELECT * FROM Team", conn)

pl = pd.read_sql_query("SELECT * FROM Player",conn)
players = pa.merge(pl,on='player_fifa_api_id', how='inner')

average_players = players.groupby('player_name').mean()

average_players = average_players.sort_values(['overall_rating'],ascending=False)
# This is the variable used to compare player performances

target = ['overall_rating']



# choose attributes to determine correlation with the overall_rating of the player;

# Omitting all categorical features

chooseFeatures = ['potential','crossing','finishing','heading_accuracy',

                  'short_passing','volleys','dribbling','curve',

                  'free_kick_accuracy','long_passing','ball_control',

                  'acceleration','sprint_speed','agility','reactions',

                  'balance','shot_power','jumping','stamina',

                'strength','long_shots','aggression','interceptions',

                  'positioning','vision','penalties','marking',

                  'standing_tackle','sliding_tackle']
# Find corelation between overall_rating and individual attributes

# to identify the top features affecting player performance



# initialize Series

sorted_coeff = pd.Series([])

sorted_coeff_avg = pd.Series([])



# find corelation coefficients in for loop

for f in chooseFeatures:

    

    coeff = players['overall_rating'].corr(players[f])

    coeff_avg = average_players['overall_rating'].corr(average_players[f])

    

    sorted_coeff = sorted_coeff.append(pd.Series([coeff],[f]))

    sorted_coeff_avg = sorted_coeff_avg.append(pd.Series([coeff_avg],[f]))

    

sorted_coeff = sorted_coeff.sort_values(axis=0,ascending=False) # sort values

sorted_coeff_avg = sorted_coeff_avg.sort_values(axis=0,ascending=False)



# create data frame

relations = pd.DataFrame(sorted_coeff_avg,columns=['corelation coeff avg'])

relations['corelation coeff'] = sorted_coeff



# sort data frame and print

relations.sort_values(by=['corelation coeff avg', 'corelation coeff'],ascending=False)
%matplotlib inline



x1 = players[chooseFeatures]

y1 = players[target]



# sample data to reduce size of dataframe

sample_frequency = 200 # choose every nth data point

x_sampled = x1[x1.index % sample_frequency == 0]

y_sampled = y1[y1.index % sample_frequency == 0]



# calculate indices of dropped rows in x_sampled

na_free = x_sampled.dropna() # df free of NaN rows

only_na = x_sampled[~x_sampled.index.isin(na_free.index)] # df WITH NaN rows

# https://stackoverflow.com/questions/34296292/pandas-dropna-store-dropped-rows



x_sampled = na_free # set x_sampled to df with non NaN rows



# drop NaN rows from y_sampled with NaN rows in x_sampled.

# This is needed in order to perform analysis with both dfs 

# which requires both df to have the same indices    

y_sampled = y_sampled.drop(y_sampled.index[y_sampled.index.isin(only_na.index)])
fig, axs = plt.subplots(2,3,figsize=(15,10))

axs = axs.ravel() 

# https://stackoverflow.com/questions/17210646/..

# ..python-subplot-within-a-loop-first-panel-appears-in-wrong-position

for i,f in zip(range(6),sorted_coeff_avg[:6].index):

    axs[i].scatter(x_sampled[f],y_sampled)

    coefs = poly.polyfit(x_sampled[f],y_sampled,1)

    coefs = coefs.ravel() 

    # https://stackoverflow.com/questions/13730468/from-nd-to-1d-arrays

    ffit = poly.Polynomial(coefs)

    axs[i].plot(x_sampled[f],ffit(x_sampled[f]),color='red')

    axs[i].set_xlabel(f)

    axs[i].set_ylabel(y_sampled.columns[0])
fig, axs = plt.subplots(2,3,figsize=(15,10))

axs = axs.ravel() 

# https://stackoverflow.com/questions/17210646/..

#..python-subplot-within-a-loop-first-panel-appears-in-wrong-position

for i,f in zip(range(6),sorted_coeff[:6].index):

    axs[i].scatter(x_sampled[f],y_sampled)

    coefs = poly.polyfit(x_sampled[f],y_sampled,1)

    coefs = coefs.ravel()

    # https://stackoverflow.com/questions/13730468/from-nd-to-1d-arrays

    ffit = poly.Polynomial(coefs)

    axs[i].plot(x_sampled[f],ffit(x_sampled[f]),color='red')

    axs[i].set_xlabel(f)

    axs[i].set_ylabel(y_sampled.columns[0])
# create new dataframe with features having the top corelation to overall_rating



# choose set of attributes from relations, 

# which is a sorted dataframe with corelation coefficients in descending order

# therefore only the first 5 attributes are chosen and written to X

X = average_players[relations.index[0:6]]

X.index = average_players.index



Y = average_players['overall_rating']

Y.index = average_players.index



attributes = X.columns # store the chosen attributes in a varibale for later use



X = X.dropna()

Y = Y.dropna()
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=324)

regressor = LinearRegression()

regressor.fit(x_train,y_train)

y_prediction = regressor.predict(x_test)

y_prediction

RMSE = sqrt(mean_squared_error(y_true = y_test, y_pred = y_prediction))

RMSE
fig, ax = plt.subplots(3,2,figsize=(15,10))

ax = ax.ravel()

for i in range(0,6):

    fracs = X.iloc[0].sum()

    ax[i].pie(X.iloc[i],labels=X.columns,autopct=abs_val,shadow=True,startangle=90)

    ax[i].axis('equal')

    ax[i].set(title=X.iloc[i].name)