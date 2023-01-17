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
dataset=pd.read_csv("/kaggle/input/indian-premier-league-match-analysis/matches.csv")

dataset.columns

features=['season','city','team1','team2','toss_winner','winner','venue','player_of_match']

dataset=dataset[features]

dataset.head()

#toss anaysis per city
#dataset.isnull().sum()

#overall 10 values with null 7 in city and 3 in winner so dropping the columns

dataset.dropna(inplace=True)

#dont replace with dataset as it becomes a none object
dataset.isnull().sum()#so we have zero null values

#moving forward to analyze the toss
dataset['toss_analysis']="No"

dataset.head()
dataset['toss_analysis'].loc[dataset.winner==dataset.toss_winner]="Yes"

dataset.toss_analysis.value_counts()

#its a completely balanced dataset 
dataset.groupby(['city','toss_analysis'])['toss_analysis'].count().sort_values(ascending=False)
dataset.groupby(['city','toss_analysis'])['toss_analysis'].count().sort_values(ascending=False)
#Mumbai is a balanced place 

#in bangalore everyone who wins the toss mostly wins the match

#toss winner has never beoome a match winner in hyderabad//some key analysis on toss by cities

dataset.head()
dataset.groupby(['season','player_of_match'])['player_of_match'].count().sort_values(ascending=False)





var1.head(5)



#from this we can follow that gayle has recieved the ever most player_of_matches in season 2011

#would appreciate if i could recieve some help on plotting

dataset.groupby(['season','player_of_match'])['player_of_match'].count().sort_values(ascending=False).head(4).plot.line()