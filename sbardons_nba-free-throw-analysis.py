# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib as mlp

import matplotlib.pyplot as plt

%matplotlib inline

sns.set()

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))     ##???????



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/free_throws.csv')

df.head()
# df.shape

# #(537577, 12) rows and columns

# df.describe()

# #used for general stat

# df.isnull()  #no empty value
#Attempted free throws

d1=df.groupby(by=["season", "playoffs"])['shot_made'].count()

#Succesfull free throws

df.groupby(["season", "playoffs"])['shot_made'].sum()



#unstack for plotting purposes

t1=df.groupby(["season", "playoffs"])['shot_made'].count().unstack()



# this has to be divided by the number of games for each season to get an average

ngames=df.groupby(["season", "playoffs"])['game_id'].nunique().unstack()



average_for_each_season = t1/ngames
#Plot average throws for each season

average_for_each_season.plot(marker='o', figsize=(18,7), xticks=range(10),color=['b','r'], rot=90)

plt.xlabel('season')

plt.ylabel('count')

plt.legend(loc=2, prop={'size': 20})

plt.title('Average number of free throws per period ', fontsize=25)

plt.show()
#Not all the throws were successfull, only those corresponding to shot_made=1



successful_throws=df.groupby(["season", "playoffs"])['shot_made'].sum().unstack()

averaged_successful= successful_throws/ngames
#Plot together

f, (ax1) = plt.subplots(figsize=(18,18))

first=average_for_each_season.plot(ax=ax1, marker='o', figsize=(15,8), xticks=range(10), color=['b','r'], rot=90)

second=averaged_successful.plot(ax=ax1, marker='o', linestyle='--', figsize=(15,8), xticks=range(10), color=['b','r'], rot=90)

ax1.set_title('Average number of free throws per period. Attempted vs Successful)', size=25)

legend=plt.legend((' playoffs attempted','regular attempted','playoffs successful','regular successful'), loc=6, prop={'size': 15})

ax1.add_artist(legend)

plt.show()

(average_for_each_season.mean()-averaged_successful.mean())*100/average_for_each_season.mean()

# On average we have 24% of missed throws for both regular season and playoffs
#Who made the most throws

players=df.groupby(["player"])

players['shot_made'].count().sort_values(ascending=False)[:10]
#Who made the most successful throws

players_success=df.groupby(["player"])

players_success['shot_made'].sum().sort_values(ascending=False)[:10]
#Let's build a dataframe that show the most successfull players based on the percentage of successful shots
df_count=pd.DataFrame(players['shot_made'].count())

df_count=df_count.rename(columns={"shot_made": "attempt"})

df_sum=pd.DataFrame(players_success['shot_made'].sum())

df_sum=df_sum.rename(columns={"shot_made": "success"})

df_pct=pd.DataFrame((players['shot_made'].sum()/players_success['shot_made'].count())*100)

df_pct=df_pct.rename(columns={"shot_made": "pct_success"})
df_pct.head()
#Merge datasets



df_merge=pd.concat([df_count, df_sum, df_pct], axis=1)

df_merged=df_merge.sort_values(by="pct_success", ascending=False)

df_merged.head()    # probabilistically the most successful players are the one with less throws
#Distribution of successful attempts

df_merged['pct_success'].hist(bins=80, figsize=(10,7))



plt.title("Distribution of Shooting Percentages", fontsize=15)

plt.xlabel("Percentage of successfull throws")

plt.show()
df_merge.head()

#Players with more than 900 throws

more_than900_attempts=df_merge[df_merge.attempt>900]

sorted_success=more_than900_attempts.sort_values(['pct_success'],ascending=False)

sorted_success.head()     
# Plot of most successful players

plt.subplots(figsize=(18,3))

sorted_success.head(20)['pct_success'].plot('bar', color='m')

plt.xlabel('Occupation')

plt.ylabel('Count')

plt.title('User by Occupation')

plt.show()
df['Consec_id'] = df['time'] == df.shift(1)['time']   

#false is first throw, true is second throw

df.groupby(["Consec_id"])['shot_made'].sum()

df.groupby(["Consec_id"])['shot_made'].count()



df_first_second=df.groupby(["Consec_id"])['shot_made'].sum()/df.groupby(["Consec_id"])['shot_made'].count()*100



indexNamesArr = df_first_second.index.values # get a list of all the column names 

indexNamesArr[0] = 'First'  #rename them

indexNamesArr[1] = 'Second'



df_first_second.plot('bar',color=['#cc0066','#606060'])  

plt.xlabel('Throws')

plt.ylabel('Pct of success')

plt.title('Percentage of success first and second throw')

plt.show()

#second throws are in general more successful than first ones
df.groupby(["period"])['shot_made'].count().plot()

plt.xlabel('period')

plt.ylabel('Count of throws')

plt.title('Throws per period')

plt.show()
##### this reflect the fact that period 5,6,7,8 last only 5 minute. There are more free throws as the game progresses 

# from period 1 to 4 as the stakes are higher
#analysis by time

test=df.groupby(["period","time"])['shot_made'].count().unstack()

#remove period 5,6,7,8

test = test.drop([5.0, 6.0, 7.0, 8.0], axis=0)

#transpose dataframe

test2=test.transpose()

test3=test2

test3.index = pd.to_datetime(test2.index)

test3=test3.sort_index() #otherwise 10.00 comes after 0:59

test3=test3.reset_index()





#see https://stackoverflow.com/questions/20618804/how-to-smooth-a-curve-in-the-right-way

def smooth(y, box_pts):

    box = np.ones(box_pts)/box_pts

    y_smooth = np.convolve(y, box, mode='same')

    return y_smooth



plt.subplots(figsize=(16,8))

plt.plot(list(range(534)),smooth(test3[1.0],10), color='blue', lw=2) 

plt.plot(list(range(534)),smooth(test3[2.0],10), color='green', lw=2) 

plt.plot(list(range(534)),smooth(test3[3.0],10), color='red', lw=2) 

plt.plot(list(range(534)), smooth(test3[4.0],10), color='yellow', lw=2)   

labels=('first', 'second','third','fourth')

plt.legend(labels, fontsize=15)

plt.title('Number of throws for each period by time', fontsize=20)

plt.xlabel('time', fontsize=15)

plt.ylabel('count', fontsize=15)

plt.show()
# I don't understand why at time zero there are such a high number of throws (more investigation needed)

# As the game progresses, under pressure, more fouls are committed and free throws awarded at the end of the period

# (between 200 and 300). The drop around 350 corresponds to overtime which does not happen at each game.

# The fourth period, when the stakes are higher, is when more free throws are given.