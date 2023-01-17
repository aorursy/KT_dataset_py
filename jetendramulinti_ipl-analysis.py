# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#reading the files
ipl = pd.read_csv("../input/matches.csv")
#printing the head() of matches data
ipl.head()
#printing the info(),describe(),shape of the Matches data
print(ipl.info())
print(ipl.describe())
print(ipl.shape)
#Printing Null values
print("null values of matches data \n",+ipl.isnull().sum())
# Total no of matches in the ipl 
no_of_matches = ipl.iloc[:,0].max()
no_of_matches
# Total no of seasons in the ipl 
no_of_seasons = len(ipl.iloc[:,1].unique())
no_of_seasons
matches_won =ipl["winner"].value_counts()
matches_won
ipl[:] = ipl[:].replace("Rising Pune Supergiants","Rising Pune Supergiant")
runs = ipl.groupby("winner").win_by_runs.idxmax()
runs
runs.plot(kind="barh",figsize=(18,13))
plt.ylabel("teams")
plt.xlabel("won by maximum_runs")
plt.show()
player_of_series = ipl.groupby(["season","winner"]).player_of_match.value_counts().idxmax()
player_of_series
mini_player_of_series = ipl.groupby(["season","winner"]).player_of_match.value_counts().idxmin()
mini_player_of_series
#Convertin to excel 
ipl.to_csv("ipl2.csv")
ipl1 = pd.read_csv("../input/deliveries.csv")
ipl1[:] = ipl1[:].replace("Rising Pune Supergiants","Rising Pune Supergiant")
ipl1.to_csv("ipl1.csv")
