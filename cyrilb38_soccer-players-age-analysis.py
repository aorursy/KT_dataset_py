# Import Libraries

import sqlite3

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.feature_selection import SelectFromModel

from sklearn.metrics import accuracy_score



# I use the xkcd theme for charts design, perfect for an informal analysis like this one 

# (go check the xkcd website for those who don't know this gem)

plt.xkcd()

sns.set()

sns.set_style('whitegrid') # I always prefer to have a grid to help read charts



%matplotlib inline
# Get the 2 tables on players in the database

cnx = sqlite3.connect('../input/database.sqlite')

df_attributes = pd.read_sql_query("SELECT * FROM Player_Attributes", cnx)

df_names = pd.read_sql_query("SELECT * FROM Player", cnx)
# Merge datasets in a single table

df = (pd.merge(df_attributes, df_names, how = "left", on="player_api_id")

      .dropna())



# Remove non relevant columns

df = df.drop(['id_x','id_y','player_fifa_api_id_x','player_fifa_api_id_y'],axis = 1)



# Convert Dates columns to date format in pandas

df["birthday"] = pd.to_datetime(df["birthday"])

df["date"] = pd.to_datetime(df["date"])

df["date_year"] = df["date"].apply(lambda x : x.year)



# Calculate approximate age in years

df["age"] = (df["date"].apply(lambda x : x.year) - df["birthday"].apply(lambda x : x.year)

            .astype(float))
# Number of records per year

df[["date_year", "date"]].groupby("date_year").count()
# Calculate the number of players per age on 2015

df_age_2015 = (df[df["date_year"] == 2015] # Filter to keep only 2015 data 

              .groupby("player_api_id")

              .agg({"age" : "max"}) # if a player has more than 2 records for 2015, keep only the latest

              .reset_index()

              .groupby("age")

              .count()

              .rename(columns ={"player_api_id" : "number of players"}))



# Calculate proportion and cumulative proportion per age

df_age_2015["players prop"] = df_age_2015["number of players"] / df_age_2015["number of players"].sum()

df_age_2015["players cum prop"] = df_age_2015["players prop"].cumsum()
# Plot the 2 distributions

fig = plt.figure()

ax1 = fig.add_subplot(2,1,1)

ax2 = fig.add_subplot(2,1,2)

fig.set_size_inches(12, 8)



ax1.plot(df_age_2015[["players prop"]])



ax2.plot(df_age_2015[["players cum prop"]], color = "orange")

ax2.set_ylabel("cumsum")



plt.subplots_adjust(hspace=0)



mean_age = interq_range =(df[df["date_year"] == 2015]

               .groupby("player_api_id")

               .agg({"age" : "max"})

               .rename(columns = {"age" : "mean"})

               .mean())["mean"]

print("the mean age for 2015 is {0:.2f} years".format(mean_age))
# Calculate quantile, min and max

interq_range =(df[df["date_year"] == 2015]

               .groupby("player_api_id")

               .agg({"age" : "max"})

               .rename(columns = {"age" : "age quantile"})

               .quantile([0,0.25,0.5, 0.75, 1]))

interq_range
# Take a look at players proportion for 29 and 30 years old

df_age_2015.loc[[29,30]]
# Calculate the interquartile range over years

iqr_evolution = (df.groupby(["date_year", "player_api_id"])

    .agg({"age":"max"})

    .reset_index()

    .groupby("date_year")

    .quantile([0.25,0.5, 0.75])

    .drop("player_api_id", axis = 1)

    .reset_index()

    .rename(columns = {"level_1" : "quartile"})

    .pivot_table("age", index = "date_year", columns = "quartile"))



# Plot the evolution on a line chart

plot_iqr_year = iqr_evolution.plot(figsize=(12,8))

plot_iqr_year.set_title("Evolution of Inter-Quartile Range per year")

plot_iqr_year.set_xlabel("Year")

plot_iqr_year.set_ylabel("Age")
# Remove non numeric features like preferred_foot. We can use this quick and dirty method as age and our ID 

# column (player_api_id) are also numeric.

df = df.select_dtypes([np.number])
# Filter data : keep only players with stats from 20 to 30 years old and with more than 10 data points

players_to_keep = (df.groupby("player_api_id")

                  .agg({"age" : ["min", "max", "count"]}))

players_to_keep = players_to_keep[(players_to_keep["age","count"] <= 20) &

                   (players_to_keep["age", "max"] >= 30) &

                   (players_to_keep["age", "count"] >= 10)].index

df = df[(df["player_api_id"].isin(players_to_keep)) &

       (df["age"] > 19) & (df["age"] < 35)]
# Norm the data : as we are interested to know when a player is better, 

# we will norm based on the max value of each feature. 

not_to_norm = ["age", "player_api_id", "date_year"] # not every column is relevant to norm !

for dim in df:

    if dim not in not_to_norm :

        df["normed_" + dim] = df.groupby("player_api_id")[dim].apply(lambda x : x / x.max())
rating_age = df.groupby("age")["normed_overall_rating"].mean()

rating_age.plot(figsize=(12,8))
# Create a dataframe that group attribute per age

kpi_age = df.groupby("age").mean()[['normed_overall_rating', 'normed_potential',

       'normed_crossing', 'normed_finishing', 'normed_heading_accuracy',

       'normed_short_passing', 'normed_volleys', 'normed_dribbling',

       'normed_curve', 'normed_free_kick_accuracy', 'normed_long_passing',

       'normed_ball_control', 'normed_acceleration', 'normed_sprint_speed',

       'normed_agility', 'normed_reactions', 'normed_balance',

       'normed_shot_power', 'normed_jumping', 'normed_stamina',

       'normed_strength', 'normed_long_shots', 'normed_aggression',

       'normed_interceptions', 'normed_positioning', 'normed_vision',

       'normed_penalties', 'normed_marking', 'normed_standing_tackle',

       'normed_sliding_tackle']].reset_index()



# Melt the attributes (in columns) in a single column : this will ease the creation of the plots

kpi_age_melt = kpi_age.melt("age", var_name='variables', value_name='vals')



# Plot the evolution of each feature over time

g = sns.FacetGrid(kpi_age_melt, col="variables", col_wrap=4)

g = g.map(sns.pointplot, "age", "vals", ci=None, markers = "", order = range(20,35))
corr_kpi = kpi_age.corr()

corr_kpi = (corr_kpi[["age"]]

           .sort_values("age", ascending = False)

           .drop("age", axis = 0))



fig, ax = plt.subplots()

fig.set_size_inches(12, 8)



sns.barplot(x = "age", y = corr_kpi.index, data = corr_kpi)
features_to_consider = ['normed_crossing', 'normed_finishing', 'normed_heading_accuracy',

       'normed_short_passing', 'normed_volleys', 'normed_dribbling',

       'normed_curve', 'normed_free_kick_accuracy', 'normed_long_passing',

       'normed_ball_control', 'normed_acceleration', 'normed_sprint_speed',

       'normed_agility', 'normed_reactions', 'normed_balance',

       'normed_shot_power', 'normed_jumping', 'normed_stamina',

       'normed_strength', 'normed_long_shots', 'normed_aggression',

       'normed_interceptions', 'normed_positioning', 'normed_vision',

       'normed_penalties', 'normed_marking', 'normed_standing_tackle',

       'normed_sliding_tackle'] # Note that we removed normed_potential as it is rather a rating than a feature



X = df[features_to_consider]

y = df["normed_overall_rating"].apply(lambda x: int(x))



# Create a random forest classifier

clf = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)



# Train the classifier

clf.fit(X, y)
# Store the name and importance of each feature in a dataframe

df_feature_importance = (pd.DataFrame(clf.feature_importances_, 

                                     index = features_to_consider, 

                                     columns = ["importance"])

                        .sort_values(by = "importance", ascending = False))



fig, ax = plt.subplots()

fig.set_size_inches(12, 8)



sns.barplot(x = "importance", y = df_feature_importance.index, data = df_feature_importance)
# Perform an inner join between our correlation coefficient dataframe and the one with feature importance

# Note that we perform an inner join to get rid of "normed_potential" in the correlation coef df which is not

# relevant in this context

df_trend_importance = (pd.merge(corr_kpi, 

                               df_feature_importance, 

                               left_index = True, 

                               right_index = True, 

                               how = "inner")

                      .sort_values(by = "importance", ascending = True))



fig, ax = plt.subplots()

fig.set_size_inches(12, 8)



# Calculate the rounded importance of each feature. this will be used to determined the height of

# the bar chart

h = df_trend_importance["importance"].apply(lambda x : int (x *100)).tolist()



# Calculate y position on the chart

y_pos = [0]

for i in range(1, len(h)):

    y_pos.append(y_pos[i - 1] + h[i] + 1)

    

# Calculate the RGBA colors for each feature : 

# negative correlation with age will be red and positive correlation will be green

col = []

for j in range(len(df_trend_importance["age"])):

    if df_trend_importance["age"][j] > 0:

        col.append((0,df_trend_importance["age"][j],0,df_trend_importance["age"][j]))

    else :

        col.append((abs(df_trend_importance["age"][j]),0,0,abs(df_trend_importance["age"][j])))



# Plot the chart

g = plt.barh(y_pos,

             height = h,

             width = df_trend_importance["age"],

            color = col)

g = plt.yticks(y_pos, df_trend_importance.index)