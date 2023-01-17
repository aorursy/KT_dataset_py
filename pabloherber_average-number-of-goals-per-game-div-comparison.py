# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import matplotlib 

import matplotlib.pyplot as plt

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/FMEL_Dataset.csv")

print (df.isnull().any())

print (df["localGoals"].describe())

print (df["visitorGoals"].describe())
#We need a new column on the dataset, which collects the total number

#of goals from each match

df["total_goals"] = df["localGoals"] + df["visitorGoals"]



#Let's take a dataframe per division

df_div1 = df[df["division"]==1]

df_div2 = df[df["division"]==2]

print ("Total games in division 1:",str(len(df_div1)))

print ("Total games in division 2:",str(len(df_div2)))
#We would need a pair of dicts (one per division), which contains the total number of matches played by season

games_per_year_1 = df_div1[["season","total_goals"]].groupby("season", axis = 0).count().to_dict()

games_per_year_2 = df_div2[["season","total_goals"]].groupby("season", axis = 0).count().to_dict()



#Now, we need the total sum of goals by season, that will be cointained in this pair of dicts:

total_goals_1 = df_div1[["season","total_goals"]].groupby("season", axis = 0).sum().to_dict()

total_goals_2 = df_div2[["season","total_goals"]].groupby("season", axis = 0).sum().to_dict()



# We would need this last pair of dicts as dataframes, so we can merge them easily:

total_goals_1_df = df_div1[["season","total_goals"]].groupby("season", axis = 0).sum().reset_index()

total_goals_2_df = df_div2[["season","total_goals"]].groupby("season", axis = 0).sum().reset_index()
def av_goals (division, season):

	if division == 1:

		return float(total_goals_1["total_goals"][season])/games_per_year_1["total_goals"][season]

	else:

		return float(total_goals_2["total_goals"][season])/games_per_year_2["total_goals"][season]



#Mapping seasons, we will get the average number of goals (per game)

#I created a new column for each dataframe

total_goals_1_df["average_goals_div1"] = total_goals_1_df["season"].map(lambda x: av_goals(1,x))

total_goals_2_df["average_goals_div2"] = total_goals_2_df["season"].map(lambda x: av_goals(2,x))
#Getting our dataframe

df_average = pd.concat([total_goals_1_df["season"],total_goals_1_df["average_goals_div1"],total_goals_2_df["average_goals_div2"]], axis = 1)



#Plotting it

df_average.plot(x = df_average["season"])

plt.title ("Average number of goals per game, by season")

plt.show()
df_average[["average_goals_div1", "average_goals_div2"]].corr()