# Supress Warnings

import warnings

warnings.filterwarnings('ignore')



#import libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import os, sys

pd.set_option('display.max_columns', 500)

pd.set_option('display.max_rows', 500)

plt.rcParams["figure.figsize"] = [20,15]
import os

path = '/kaggle/input'

for dirname, _, filenames in os.walk(path):

    Data = []

    rows = 0

    for filename in filenames:

        p = os.path.join(dirname, filename)

        df = pd.read_csv(p)

        rows += df.shape[0]

    

        Data.append(df)

    

    

for df in Data:

    df.columns = ['Inning', 'Over', 'Batting Team', 'Batsman', 'Non-Striker', 'Baller', 'Runs', 'Extra']

    

combine_data = pd.concat(Data).reset_index(drop=True)
combine_data.shape
combine_data.head()
combine_data.isnull().sum()
combine_data.describe()
Four = []

six = []

for x in combine_data['Runs']:

    if x == 4:

        Four.append(1)

        six.append(0)

    elif x == 6:

        Four.append(0)

        six.append(1)

    else:

        Four.append(0)

        six.append(0)

combine_data["Four"] = Four

combine_data['Six'] = six
combine_data.head()
batsman_group = combine_data.groupby("Batsman").sum()

batsman_group = batsman_group.sort_values(by='Runs', ascending=False).drop("Inning", axis=1).reset_index()

batsman_group.head()
Top_100_batsman = batsman_group.head(100)
top_runs_plot = sns.barplot(x = "Batsman", y = "Runs", data=Top_100_batsman)

top_runs_plot.set_xticklabels(top_runs_plot.get_xticklabels(), rotation=90)

plt.title("Top 100 Batsman")

plt.show()
top_runs_plot = sns.barplot(x = "Batsman", y = "Six", data=Top_100_batsman)

top_runs_plot.set_xticklabels(top_runs_plot.get_xticklabels(), rotation=90)

plt.title("Top 100 Batsman")

plt.show()
Ball_Bat_Stats = pd.pivot_table(combine_data, values="Runs", index="Baller", columns="Batsman", aggfunc=np.sum)

Ball_Bat_Stats.shape
Na_col = (Ball_Bat_Stats.isnull().sum()/Ball_Bat_Stats.shape[0])*100

column_to_drop = list(Na_col[Na_col.values >= 70].index)

Ball_Bat_Stats.drop(labels = column_to_drop,axis =1,inplace=True)        

Ball_Bat_Stats.info()
Na_Rows = ((Ball_Bat_Stats.isnull().sum(axis=1))/Ball_Bat_Stats.shape[1])*100

rows_to_drop = list(Na_Rows[Na_Rows.values >= 10].index)

Ball_Bat_Stats.drop(labels = rows_to_drop, axis=0, inplace=True)        
Ball_Bat_Stats.fillna(0, inplace=True)

Ball_Bat_Stats
sns.heatmap(Ball_Bat_Stats, cmap="YlGnBu")

plt.show()
Baller_group = combine_data.groupby("Baller").sum()

Baller_group = Baller_group.sort_values(by="Runs", ascending=False)

Baller_group_Top_50 = Baller_group.head(50).reset_index()

Baller_group_Top_50.head()
Top_ballers_plot = sns.barplot(x = "Baller", y = "Runs", data=Baller_group_Top_50)

Top_ballers_plot.set_xticklabels(Top_ballers_plot.get_xticklabels(), rotation=90)

plt.show()
combine_data["Boundries"] = combine_data["Four"]+combine_data["Six"]

combine_data.head()
Ball_Bat_boundries_stat = pd.pivot_table(combine_data, values="Boundries", index="Baller", columns="Batsman", aggfunc=np.sum)

Na_cols = Ball_Bat_boundries_stat.isnull().sum()/ Ball_Bat_boundries_stat.shape[0]*100

columns_to_drop = list(Na_cols[Na_cols.values >= 70].index)

Ball_Bat_boundries_stat.drop(labels = columns_to_drop, axis=1, inplace=True)

Ball_Bat_boundries_stat.shape
Na_Row = Ball_Bat_boundries_stat.isnull().sum(axis=1)/Ball_Bat_boundries_stat.shape[1]*100

rows_to_drop =list(Na_Row[Na_Row.values >= 10].index)

Ball_Bat_boundries_stat.drop(labels = rows_to_drop, axis=0, inplace=True)

Ball_Bat_boundries_stat.fillna(0, inplace=True)

Ball_Bat_boundries_stat.head()
sns.heatmap(Ball_Bat_boundries_stat, cmap="Blues")

plt.show()
Top_batsman_sixes = combine_data.groupby("Batsman").sum()

Top_batsman_sixes = Top_batsman_sixes.sort_values(by= "Six", ascending=False)

Top_batsman_sixes = Top_batsman_sixes.head(50).reset_index()

Top_batsman_sixes.head()
import squarify 

squarify.plot(sizes=Top_batsman_sixes["Six"], label=Top_batsman_sixes["Batsman"], alpha=0.6 )