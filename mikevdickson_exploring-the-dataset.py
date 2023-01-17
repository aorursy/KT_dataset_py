# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import tree, linear_model

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/data.csv')

data.set_index('shot_id', inplace=True)
data["action_type"] = data["action_type"].astype('object')
data["combined_shot_type"] = data["combined_shot_type"].astype('category')
data["game_event_id"] = data["game_event_id"].astype('category')
data["game_id"] = data["game_id"].astype('category')
data["period"] = data["period"].astype('object')
data["playoffs"] = data["playoffs"].astype('category')
data["season"] = data["season"].astype('category')
data["shot_made_flag"] = data["shot_made_flag"].astype('category')
data["shot_type"] = data["shot_type"].astype('category')
data["team_id"] = data["team_id"].astype('category')
data["opponent"] = data["opponent"].astype('category')
print(data.dtypes)
print(data.shape)
print(data.head(2))
data.describe(include=['number'])
data.describe(include=['object', 'category'])
data_cl = data.copy() # create a copy of data frame
target = data_cl['shot_made_flag'].copy()

# Remove some columns
data_cl.drop('team_id', axis=1, inplace=True) # Always one number
data_cl.drop('lat', axis=1, inplace=True) # Correlated with loc_x
data_cl.drop('lon', axis=1, inplace=True) # Correlated with loc_y
#data_cl.drop('game_id', axis=1, inplace=True) # Independent
data_cl.drop('game_event_id', axis=1, inplace=True) # Independent
data_cl.drop('team_name', axis=1, inplace=True) # Always LA Lakers
#data_cl.drop('shot_made_flag', axis=1, inplace=True)

#New Features
data_cl["period_minutes"] = 12 #Regular periods are 12 minutes long
data_cl.loc[data_cl["period"].astype('int')>4,"period_minutes"] = 5 #Overtime periods are only 5 minutes long
data_cl["game_time_elapsed"] = (data_cl["period"].astype('int')-1)*12*60+(data_cl["period_minutes"]-1-data_cl["minutes_remaining"])*60+(59-data["seconds_remaining"])
#data_cl["game_time_elapsed"].describe()

## Matchup - (away/home)
data_cl['home_play'] = data_cl['matchup'].str.contains('vs').astype('int')
data_cl.drop('matchup', axis=1, inplace=True)

# Game date
data_cl['game_date'] = pd.to_datetime(data_cl['game_date'])
data_cl['game_year'] = data_cl['game_date'].dt.year
data_cl['game_month'] = data_cl['game_date'].dt.month
data_cl.drop('game_date', axis=1, inplace=True)

# Opponent dynasty
data_cl["dynasty"] = data_cl["opponent"].astype('str')+data_cl["season"].astype('str')
#data_cl["dynasty"] = data["dynasty"].astype('category')

# Separate dataset for validation
unknown_mask = data['shot_made_flag'].isnull()
data_submit = data_cl[unknown_mask]

# Separate dataset for training
train = data_cl[~unknown_mask].copy()
train["shot_made_flag"] = train["shot_made_flag"].astype('int')
#trainX = data_cl[~unknown_mask].copy()
#trainY = target[~unknown_mask].copy()

test = data_cl[unknown_mask].copy()
#testX = data_cl[unknown_mask].copy()
#testY = target[unknown_mask].copy()
accuracyByX = train.groupby("shot_distance")["shot_made_flag"].mean()

bins = np.arange(0,accuracyByX.shape[0],1)
barWidth = (bins[1]-bins[0])
plt.figure(figsize=(16,8))
plt.subplot(1,1,1)
plt.bar(bins,accuracyByX,align='edge',width=barWidth)
plt.ylabel('FG Percentage')
plt.xlabel('Distance')
plt.title('Kobe Bryant Field Goal Percentage by Distance from Basket')
accuracyByX = train.groupby("opponent",sort=True)["shot_made_flag"].mean().sort_values(ascending=False)
bins = np.arange(0,accuracyByX.shape[0],1)
barWidth = (bins[1]-bins[0])
plt.figure(figsize=(16,8))
plt.subplot(1,1,1)
plt.bar(bins,accuracyByX,align='center',width=barWidth)
plt.xticks(bins,accuracyByX.index)
plt.ylabel('FG Percentage')
plt.xlabel('Opponent')
plt.title('Kobe Bryant Field Goal Percentage by Opponent')
accuracyByX = train.groupby("dynasty",sort=True)["shot_made_flag"].mean()
bins = np.arange(0,accuracyByX.shape[0],1)
barWidth = 1 #[0.25*i**2 for i in bins]#
plt.figure(figsize=(16,150))
plt.subplot(1,1,1)
plt.barh(bins-barWidth/2.0,accuracyByX,height=barWidth)
plt.yticks(bins,accuracyByX.index)
plt.ylabel('FG Percentage')
plt.xlabel('Opponent')
plt.title('Kobe Bryant Field Goal Percentage by Dynasty')
timeBins = np.arange(0,train["game_time_elapsed"].max(),60)
train["game_time_elapsed_bin"] = pd.cut(train["game_time_elapsed"],timeBins)
accuracyByX = train.groupby("game_time_elapsed_bin",sort=True)["shot_made_flag"].mean()
bins = np.arange(0,accuracyByX.shape[0],1)
barWidth = 1
plt.figure(figsize=(16,8))
plt.subplot(1,1,1)
plt.bar(bins,accuracyByX,width=barWidth)
plt.xticks(bins[::4],timeBins[::4]/60)
plt.xlabel('Time')
plt.ylabel('FG Percentage')
plt.title('Kobe Bryant Field Goal Percentage by Game Time')
train = train.drop("game_time_elapsed_bin",axis=1)
accuracyByX = train.groupby("season",sort=True)["shot_made_flag"].mean()
bins = np.arange(0,accuracyByX.shape[0],1)
barWidth = 1
plt.figure(figsize=(16,8))
plt.subplot(1,1,1)
plt.barh(bins,accuracyByX,height=barWidth,align="center")
plt.yticks(bins[:],accuracyByX.index[:])
plt.ylabel('Season')
plt.xlabel('FG Percentage')
plt.title('Kobe Bryant Field Goal Percentage by Season')
train = train.sort_values(["game_id","game_time_elapsed"])
train["seconds_between_shots"] = train["game_time_elapsed"]-train["game_time_elapsed"].shift(1)
mask = train["seconds_between_shots"]<0
train["seconds_between_shots"] = train["seconds_between_shots"].where(train["seconds_between_shots"]>0,other=10000)
train[["game_time_elapsed","seconds_between_shots"]]

timeBins = np.arange(0,200,5)
train["seconds_between_shots_bin"] = pd.cut(train["seconds_between_shots"],timeBins)
accuracyByX = train.groupby("seconds_between_shots_bin",sort=True)["shot_made_flag"].mean()
bins = np.arange(0,accuracyByX.shape[0],1)
barWidth = 1
plt.figure(figsize=(16,8))
plt.subplot(1,1,1)
plt.bar(bins,accuracyByX,width=barWidth)
plt.xticks(bins[:],timeBins[:])
plt.xlabel('Seconds since last shot')
plt.ylabel('FG Percentage')
plt.title('Kobe Bryant Field Goal Percentage by Seconds Since Last Shot')
train = train.drop("seconds_between_shots_bin",axis=1)
train = train.sort_values(["game_id","game_time_elapsed"])
train["seconds_to_next_shot"] = train["game_time_elapsed"].shift(-1)-train["game_time_elapsed"]
mask = train["seconds_to_next_shot"]<0
train["seconds_to_next_shot"] = train["seconds_to_next_shot"].where(train["seconds_between_shots"]>0,other=10000)
train[["game_time_elapsed","seconds_to_next_shot"]]

timeBins = np.arange(0,200,5)
train["seconds_between_shots_bin"] = pd.cut(train["seconds_to_next_shot"],timeBins)
accuracyByX = train.groupby("seconds_between_shots_bin",sort=True)["shot_made_flag"].mean()
bins = np.arange(0,accuracyByX.shape[0],1)
barWidth = 1
plt.figure(figsize=(16,8))
plt.subplot(1,1,1)
plt.bar(bins,accuracyByX,width=barWidth)
plt.xticks(bins[:],timeBins[:])
plt.xlabel('Seconds to next (future) shot')
plt.ylabel('FG Percentage')
plt.title('Kobe Bryant Field Goal Percentage by Seconds to Next (Future) Shot')
train = train.drop("seconds_between_shots_bin",axis=1)
timeBins = np.arange(0,200,5)
train["seconds_between_shots_bin"] = pd.cut(train["seconds_between_shots"],timeBins)

madeByX = train.groupby("seconds_between_shots_bin",sort=True)["shot_made_flag"].sum()
missedByX = train.groupby("seconds_between_shots_bin",sort=True)["shot_made_flag"].count()-madeByX

bins = np.arange(0,accuracyByX.shape[0],1)
barWidth = 1
plt.figure(figsize=(16,8))
plt.subplot(1,1,1)
plt.bar(bins,madeByX,width=barWidth,label='Made FG',color='green')
plt.bar(bins,missedByX,width=barWidth,bottom=madeByX,label='Missed FG',color='red')
plt.xticks(bins[:],timeBins[:])
plt.xlabel('Seconds since last shot')
plt.ylabel('FG Attempts')
plt.legend(loc='best', prop={'size':'large'})
plt.title('Kobe Bryant Field Goal Attempts by Seconds Since Last Shot')
train = train.drop("seconds_between_shots_bin",axis=1)
timeBins = np.arange(0,200,5)
train["seconds_between_shots_bin"] = pd.cut(train["seconds_between_shots"],timeBins)
distanceByX = train.groupby("seconds_between_shots_bin",sort=True)["shot_distance"].mean()
distanceByX_median = train.groupby("seconds_between_shots_bin",sort=True)["shot_distance"].median()

bins = np.arange(0,accuracyByX.shape[0],1)
barWidth = 1
plt.figure(figsize=(16,8))
plt.subplot(1,1,1)
plt.bar(bins,distanceByX,width=barWidth)
plt.plot(bins,distanceByX_median,'g-o')
plt.xticks(bins[:],timeBins[:])
plt.xlabel('Seconds since last shot')
plt.ylabel('Shot distance')
plt.title('Kobe Bryant Shot Distance by Seconds Since Last Shot')
train = train.drop("seconds_between_shots_bin",axis=1)
train = train.sort_values(["game_id","game_time_elapsed"])
train["seconds_between_shots"] = train["game_time_elapsed"]-train["game_time_elapsed"].shift(1)
mask = (train["seconds_between_shots"]<0) | (train["game_time_elapsed"]<((48-43)*60))
train["seconds_between_shots"] = train["seconds_between_shots"].where(mask,other=10000)

timeBins = np.arange(0,200,5)
train["seconds_between_shots_bin"] = pd.cut(train["seconds_between_shots"],timeBins)
accuracyByX = train.groupby("seconds_between_shots_bin",sort=True)["shot_made_flag"].mean()
bins = np.arange(0,accuracyByX.shape[0],1)
barWidth = 1
plt.figure(figsize=(16,8))
plt.subplot(1,1,1)
plt.bar(bins,accuracyByX,width=barWidth)
plt.xticks(bins[:],timeBins[:])
plt.xlabel('Seconds since last shot')
plt.ylabel('FG Percentage')
plt.title('Kobe Bryant Field Goal Percentage by Seconds Since Last Shot (first 43 minutes of game)')
train = train.drop("seconds_between_shots_bin",axis=1)
