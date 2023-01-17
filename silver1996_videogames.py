# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
"""

UURIMISKÜSIMUS: Kriitikute skoori suurendes suurenevad müüginumbrid? Aastate suurenedes suurenevad müüginumbrid?

"""
import numpy as np

import pandas as pd 



%matplotlib inline

pd.set_option('display.max_rows', 20)
df=pd.read_csv("../input/Video_Games_Sales_as_at_22_Dec_2016.csv")
sales=df["NA_Sales"]+df["EU_Sales"]+df["JP_Sales"]+df["Other_Sales"]+df["Global_Sales"]

uus=pd.DataFrame({"Name":df["Name"],"Critic_Score":df["Critic_Score"], "User_Score":df["User_Score"],"Sold":sales,"Year_of_Release":df["Year_of_Release"]})
uus=uus[["Name","Year_of_Release","Critic_Score","User_Score","Sold"]]

uus
uus[["Name","Year_of_Release","Critic_Score","User_Score","Sold"]].sort_values("Critic_Score",ascending=False)
uus.groupby("Critic_Score").aggregate({"Sold": ["sum", "mean", "median"],})
uus.Critic_Score.plot.hist(rwidth=0.95, color="lime")
uus.plot.scatter("Sold","Critic_Score", alpha=0.3, color="aqua")
uus.groupby("Year_of_Release").aggregate({"Sold": ["sum", "mean", "median"],})
uus.Year_of_Release.plot.hist(bins=40,rwidth=0.85, color="red")
uus.plot.scatter("Sold","Year_of_Release", alpha=0.3, color="orange")