import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib as plt
df = pd.read_csv("../input/cricinfo-statsguru-data/ODI Player Innings Stats - All Teams.csv")
df.head(5)
df.info()
df_batting = df[['Innings Player','Innings Runs Scored Num','Innings Minutes Batted','Innings Batted Flag','Innings Balls Faced','Innings Boundary Fours','Innings Boundary Sixes','Innings Batting Strike Rate']]
df_batting.head(5)
df_batting.isnull().sum()
df_batting = df_batting[~df_batting['Innings Runs Scored Num'].isin(['-'])]

df_batting = df_batting[~df_batting['Innings Minutes Batted'].isin(['-'])]

df_batting = df_batting[~df_batting['Innings Batted Flag'].isin(['-'])]

df_batting = df_batting[~df_batting['Innings Balls Faced'].isin(['-'])]

df_batting = df_batting[~df_batting['Innings Boundary Fours'].isin(['-'])]

df_batting = df_batting[~df_batting['Innings Boundary Sixes'].isin(['-'])]

df_batting = df_batting[~df_batting['Innings Batting Strike Rate'].isin(['-'])]
df_batting = df_batting.dropna()
df_batting.columns.values
df_batting['Innings Runs Scored Num'] = df_batting['Innings Runs Scored Num'].astype(int)

df_batting['Innings Minutes Batted'] = df_batting['Innings Minutes Batted'].astype(int)

df_batting['Innings Balls Faced'] = df_batting['Innings Balls Faced'].astype(int)

df_batting['Innings Boundary Fours'] = df_batting['Innings Boundary Fours'].astype(int)

df_batting['Innings Boundary Sixes'] = df_batting['Innings Boundary Sixes'].astype(int)

df_batting['Innings Batting Strike Rate'] = df_batting['Innings Batting Strike Rate'].astype(float)
df_compare = df_batting[(df_batting['Innings Player'] == 'RG Sharma') | (df_batting['Innings Player'] == 'SR Tendulkar') | (df_batting['Innings Player'] == 'V Kohli')]
df_compare
df_compare.info()
df_compare.isnull().sum()
import seaborn as sns

df_compare = df_compare.drop('Innings Batted Flag', axis=1)

sns.heatmap(df_compare.corr(), annot=True)
b = df_compare.drop(['Innings Balls Faced','Innings Minutes Batted'],axis=1)

sns.pairplot(b,hue='Innings Player')