# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df_results = pd.read_csv('../input/results.csv') # import barrier trial data from results.csv
df_barrier = pd.read_csv('../input/barrier.csv') # import barrier trial data from barrier.csv
df_comments = pd.read_csv('../input/comments.csv') # import comments data from comments.csv
df_trackwork = pd.read_csv('../input/trackwork.csv') # import track work data from trackwork.csv
df_horseinfo = pd.read_csv('../input/horse_info.csv') # import horse information data from horseinfo.csv

# feature engineering example 

# create a new dataframe for the features
df_features = df_results[['date', 'raceno', 'horseno', 'row']]
df_features.assign(horse_win_times_career="", jockey_win_times_career="", trainer_win_times_career="", 
                   horse_pla_times_career="", jockey_pla_times_career="", trainer_pla_times_career="")

# Calculate features for each race.
list_results_dates = (sorted(set(df_results['date']))) # list of race dates.
for race_date in list_results_dates:

    for raceno in range(1, 12):
        df_race = df_results.query("date==" + "'" + race_date + "'" + " and raceno==" + str(raceno)).sort_values(by=['row'])
        if len(df_race)>0:
            for i in range(len(df_race)):
                index = list(df_race['Unnamed: 0'])[i]
                horse = list(df_race['horse'])[i]
                jockey = list(df_race['jockey'])[i]
                trainer = list(df_race['trainer'])[i]

                #career total win times for the horse
                df_features.set_value(index, 'horse_win_times_career', len(df_results.query("horse==" + '"' + horse + '"' + " and row==1 and date<"+ '"' + race_date + '"')))
                
                #career total win times for the jockey
                df_features.set_value(index, 'jockey_win_times_career', len(df_results.query("jockey==" + '"' + jockey + '"' + " and row==1 and date<"+ '"' + race_date + '"')))
                
                #career total win times for the trainer
                df_features.set_value(index, 'trainer_win_times_career', len(df_results.query("trainer==" + '"' + trainer + '"' + " and row==1 and date<"+ '"' + race_date + '"')))
                
                #career total pla times for the horse
                df_features.set_value(index, 'horse_pla_times_career', len(df_results.query("horse==" + '"' + horse + '"' + " and row<=3 and date<"+ '"' + race_date + '"')))
                
                #career total pla times for the jockey
                df_features.set_value(index, 'jockey_pla_times_career', len(df_results.query("jockey==" + '"' + jockey + '"' + " and row<=3 and date<"+ '"' + race_date + '"')))
                
                #career total pla times for the trainer
                df_features.set_value(index, 'trainer_pla_times_career', len(df_results.query("trainer==" + '"' + trainer + '"' + " and row<=3 and date<"+ '"' + race_date + '"')))
                
df_features.to_csv('features.csv')
