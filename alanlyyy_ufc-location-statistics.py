# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/ufcdata/data.csv')

df2 = pd.read_csv('/kaggle/input/ufcdata/raw_total_fight_data.csv', sep=";")

df2.head()
df['location'].value_counts().head()
mask = df['title_bout'] == True

Location_stats = df['location'].loc[mask].value_counts()

Location_stats.head(25)
Location_stats.head(25).plot.bar(y='locations', title="Popular Places for Title Bouts",figsize=(15,15))
#total matches

location_sum = df2['location'].value_counts().sum()



#how many locations with less than 50 matches

less_mask = df2['location'].value_counts() < 50

other_loc_freq = sum(df2['location'].value_counts() < 50)



#sum of 'Other' locations

other_loc = df2['location'].value_counts().loc[less_mask]

sum_of_other =sum(df2['location'].value_counts().loc[less_mask])



print(other_loc_freq, "'Other' locations have less than 50 matches", " with a total of ", sum_of_other, 'matches.' )



#get the frequency counts of all the other refs

mask = (df2['location'].value_counts() > 50)

loc_freq = df2['location'].value_counts().loc[mask]

loc_freq.set_value("Other",sum_of_other)



#remove las vegas from dataset

mask2 = (df2['location'].value_counts() < 1000) & (df2['location'].value_counts() > 50)

loc_freq2 = df2['location'].value_counts().loc[mask2]

loc_freq2.set_value("Other",sum_of_other)



#looking strictly at the 'Other' locations with 30 or matches

mask3 = (df2['location'].value_counts() < 50) & (df2['location'].value_counts() > 30)

loc_freq3 = df2['location'].value_counts().loc[mask3]



#create subplot showing number of fights per location

import matplotlib.pyplot as plt



fig, axes = plt.subplots(nrows=1, ncols=3,figsize=(15,15))



loc_freq.plot.bar(y='locations', title="With Las Vegas", ax=axes[0])

loc_freq2.plot.bar(y='locations', title="Without Las Vegas",ax=axes[1])

loc_freq3.plot.bar(y='locations', title="Other Locations",ax=axes[2])
mask = (df['title_bout'] == True) & (df['Winner'] == "Blue")

df['location'].loc[mask].value_counts()
start_date = "2018-01-01"

stop_date = "2019-01-01"



mask = (df['title_bout'] == True) & (df["date"] > start_date) & (df["date"] < stop_date) & (df["location"] == "Las Vegas, Nevada, USA")

Location_stats = df['location'].loc[mask].value_counts()

Location_stats.head()
df2['Referee'].value_counts().head(10)
parameters = df2["win_by"].unique()

print(parameters)



mask = (df2['win_by'] == parameters[0]) | (df2['win_by'] == parameters[1]) | (df2['win_by'] == parameters[3]) | (df2['win_by'] == parameters[6]) | (df2['win_by'] == parameters[8])



df2['Referee'].loc[mask].value_counts().head()
#total matches

total_officiated_matches = df2['Referee'].value_counts().sum()



#how many total matches officiated by refs with less than 100 matches officiated

other_ref_freq = sum(df2['Referee'].value_counts() < 100)

print(other_ref_freq, "matches officiated by 'Other' refs")



#get the frequency counts of all the other refs

mask = (df2['Referee'].value_counts() > 100)

ref_freq = df2['Referee'].value_counts().loc[mask]

ref_freq.set_value("Other",other_ref_freq)

ref_freq



plot = ref_freq.plot.pie(y='Referee', figsize=(5, 5))