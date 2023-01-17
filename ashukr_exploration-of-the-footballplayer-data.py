import sqlite3 as sq

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

from sklearn.preprocessing import scale
#connecting to the sqlite

conn = sq.connect('../input/database.sqlite')

#loading the sqlite data to a data frame

# parameter1 the sql query parameter2 the connection to the sql

# in the sql query "select (name of the column) from (name of the table)

data = pd.read_sql_query("SELECT * FROM Player_Attributes",conn)
data.head()
# number of the columns in the df

len(data.columns)
data.columns
#getting the descriptions

data.describe()
# changing the rows to columns and the columns to row

data.describe().transpose()
len(data['id'])
#finding the null values 

data.isnull().any()
data.isnull().any().any()
data.isnull().sum()

#it presents the ersult as per the columns
initial_rows = data.shape[0]
data_cln=data.dropna(axis=0)
#the number of rows droped

data_cln.shape[0]-initial_rows
data_cln.isnull().any().any()
data_cln.head()
#accesing the required features

data_cln[:10][['penalties','overall_rating']]
data_cln['penalties'].corr(data_cln['overall_rating'])
features = ['acceleration','curve','free_kick_accuracy','ball_control','shot_power','stamina']
a=[]

for f in features:

    related=data_cln['overall_rating'].corr(data_cln[f])

    a.append(related)

    print("%s: %f" %(f,related))
#finding the maximum of a list

max(a)
max(a)
cols = ['potential',  'crossing', 'finishing', 'heading_accuracy',

       'short_passing', 'volleys', 'dribbling', 'curve', 'free_kick_accuracy',

       'long_passing', 'ball_control', 'acceleration', 'sprint_speed',

       'agility', 'reactions', 'balance', 'shot_power', 'jumping', 'stamina',

       'strength', 'long_shots', 'aggression', 'interceptions', 'positioning',

       'vision', 'penalties', 'marking', 'standing_tackle', 'sliding_tackle',

       'gk_diving', 'gk_handling', 'gk_kicking', 'gk_positioning',

       'gk_reflexes']
#creating a list of correlations

correlations = [data_cln['overall_rating'].corr(data_cln[f]) for f in cols]
len(correlations),len(cols)
def plot_dataframe(df,y_label):

    color = 'coral'

    fig = plt.gcf()

    fig.set_size_inches(20,12)

    plt.ylabel(y_label)

    ax = df.correlation.plot(linewidth=3.3, color=color)

    ax.set_xticks(df.index)

    ax.set_xticklabels(df.attributes, rotation=75); #Notice the ; (remove it and see what happens !)

    plt.show()
df2 = pd.DataFrame({'attributes': cols, 'correlation': correlations}) 
# let's plot above dataframe using the function we created      

plot_dataframe(df2, 'Player\'s Overall Rating')
df2.correlation.plot(linewidth=3.3, color='coral')
select5features = ['gk_kicking', 'potential', 'marking', 'interceptions', 'standing_tackle']

features=data_cln[select5features].copy(deep=True)
features.head()
data = scale(features)
num_cluster = 4

model = KMeans(init='k-means++',n_clusters=num_cluster,n_init=20)
model.fit(data)
model.labels_