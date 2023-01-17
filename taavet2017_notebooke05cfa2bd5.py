import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import sys



df = pd.read_csv("../input/Video_Games_Sales_as_at_22_Dec_2016.csv")



#from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))
df.set_index('Global_Sales').head(50)
df.Year_of_Release.plot.hist(bins=40, rwidth=0.6);
df.Genre.value_counts()
df.Platform.value_counts()
df.Critic_Score.plot.hist(bins = 100, grid = True);
df[~df.Critic_Score.isnull()].groupby("Platform").aggregate({"Global_Sales": ["sum", "mean"],

                                  "Critic_Score": ["mean", "median"]})
(df[["Name", "Global_Sales","Genre", "Platform"]]

 .sort_values("Global_Sales",ascending = False).head(20))
(df[["Name", "EU_Sales","Genre", "Platform"]]

 .sort_values("EU_Sales", ascending = False).head(20))
(df[["Name","NA_Sales","Genre","Platform"]]

.sort_values("NA_Sales",ascending=False).head(20))
(df[["Name","JP_Sales","Genre","Platform"]]

.sort_values("JP_Sales",ascending =False).head(20))
df.plot.scatter("Critic_Score", "NA_Sales", alpha = 0.5);
df.plot.scatter("Critic_Score", "EU_Sales", alpha = 0.5);
df.plot.scatter("Critic_Score", "JP_Sales", alpha = 0.5);