#There are two data files 1.matches & 2.deliveries

#we will analyse the data and answer few questions such as 

#1.number of teams?

#2.number of matches?

#3.which team won by maximum Runs?

#4.which team won by maximum wickets?

#5.which IPL team is more successful?

#we will first load the libraries which are important for analysis

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt #visualization

import seaborn as sns # visualization
#now we will read the csv data set



IPL = pd.read_csv('../input/matches.csv')

IPL

#we will perform some basic operation on the dataframe

#1. head() which gives the first 5 rows of the dataframe

IPL.head()

#2. If we want to read only first two rows of dataframe the we can use

IPL.head(2)
#2.we will use tail() which gives last fir rows of the dataframe

IPL.tail()

# if we want to see only last two rows then we can write as 

IPL.tail(2)
# now we will get the number of rows and columns in the dataset

IPL.shape
#now we will get the summary statistics 

IPL.describe()



#Nan here is null value which can be replaced
#now we will get the variables data types from the data set

IPL.info()
#now we will solve the actual questions stated above at the top.

#1. we will find the number of teams 

IPL['team1'].unique()



#in this dataset we have two teams column which have played with other once. so we can use IPL['team'].unique as well.
#2. we will find the number of matches from the dataset?

IPL['id'].max()
#3. which team won by maximum runs?

#for this we will iloc operation which takes the index value and returns the row and we will use the max function to get this on the id column.

IPL.iloc[IPL['win_by_runs'].idxmax()]['winner']
#4. now we will find which team won by maximum wickets?

#we can use the same above code of #3 and just replace win_by_runs with the wickets column.

IPL.iloc[IPL['win_by_wickets'].idxmax()]['winner']
#5. now we will find which Ipl team is most successfull. we will answer it by visualization

#sns.countplot(y='winner', data = matches)

#plt.show

data = IPL.winner.value_counts()

sns.barplot(y = data.index, x = data, orient='h')