# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



df_matches = pd.read_csv('../input/matches.csv')

df_deliveries= pd.read_csv('../input/deliveries.csv')

print(df_deliveries.head())
df_matches.rename(columns={'id':'match_id'},inplace=True)



df_deliveries= df_deliveries.merge(df_matches,on=['match_id'])



df_batsman= df_deliveries.groupby(['batsman','match_id','season'])['batsman_runs'].sum().reset_index()



#print(df_batsman.head(50))

df_deliveries.columns



balls_faced=df_deliveries[(df_deliveries['wide_runs']==0)&(df_deliveries['noball_runs']==0)].groupby(['batsman','match_id','season'])['total_runs'].count().reset_index()

balls_faced.rename(columns={'total_runs':'total_balls'},inplace=True)



df_batsman= df_batsman.merge(balls_faced,on=['match_id','batsman','season'])

#print(balls_faced.head())

#print(df_batsman.head())



sixes= df_deliveries[df_deliveries['batsman_runs']==6].groupby(['batsman','match_id','season'])['batsman_runs'].count().reset_index()

fours= df_deliveries[df_deliveries['batsman_runs']==4].groupby(['batsman','match_id','season'])['batsman_runs'].count().reset_index()



sixes.rename(columns={'batsman_runs':'sixes'},inplace=True)

fours.rename(columns={'batsman_runs':'fours'},inplace=True)

df_batsman= df_batsman.merge(sixes,on=['match_id','batsman','season'])

df_batsman= df_batsman.merge(fours,on=['match_id','batsman','season'])



print(df_batsman.head())





#runs in each over



df_over= df_deliveries.groupby(['over','match_id','season'])['total_runs'].sum().reset_index()



#sns.pointplot(x=df_over[df_over['season']=='2008']['over'], y=df_over[df_over['season']=='2008']['total_runs'])



df_over_mean= df_over.groupby(['over','season'])['total_runs'].mean().reset_index()



sns.pointplot(df_over_mean[df_over_mean['season']==2008]['over'],df_over_mean[df_over_mean['season']==2008]['total_runs'],color='red',marker=2007

             )

sns.pointplot(df_over_mean[df_over_mean['season']==2009]['over'],df_over_mean[df_over_mean['season']==2009]['total_runs'],color='green')


