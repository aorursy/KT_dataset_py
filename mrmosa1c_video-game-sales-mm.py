import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

%matplotlib inline



df = pd.read_csv("../input/Video_Games_Sales_as_at_22_Dec_2016.csv")
df.plot.scatter("Critic_Score", "Global_Sales", alpha=0.5);
df.plot.scatter("NA_Sales", "EU_Sales", alpha=0.5);
df[~df.Critic_Score.isnull()].groupby("Platform").aggregate({"Global_Sales": ["sum", "mean"],

                                  "Critic_Score": ["mean", "median"]})
df.Year_of_Release.plot.hist(bins=8, rwidth=0.9);