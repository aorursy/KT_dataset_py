import numpy as np

import pandas as pd 



%matplotlib inline

pd.set_option('display.max_rows', 20)
df = pd.read_csv("../input/Video_Games_Sales_as_at_22_Dec_2016.csv")
df
(df[["Name", "Publisher", "Critic_Score", "Year_of_Release"]]

 .sort_values("Year_of_Release", ascending=False))
df.Critic_Score.plot.hist(bins=11, grid=False, rwidth=0.95);
df.plot.scatter("Critic_Score", "Global_Sales", alpha=0.2)
df.groupby("Year_of_Release").aggregate({"Critic_Score": ["sum", "mean", "median"],

                                      "Critic_Count" : ["median", "sum"]})