%matplotlib inline



import pandas as pd

import seaborn as sns



import matplotlib.pyplot as plt



from __future__ import division
df = pd.read_csv("../input/celebrity_deaths_4.csv", encoding="latin1")
len(df[(df.death_month=="December") & (df.death_year==2006)])
df[df.name=="James Brown"]
wikipedia_2006 = [184, 200, 192, 210, 237, 214, 262, 234, 235, 230, 229, 210]

wikipedia_2012 = [535, 428, 476, 418, 417, 409, 407, 416, 406, 418, 462, 520]

wikipedia_2016 = [671, 555, 639, 542, 536, 531, 544, 521, 502, 516, 488, 599]
df_comparison = {"Data Set": [len(df[df.death_year==2006]), len(df[df.death_year==2012]), len(df[df.death_year==2016])],

                 "Wikipedia": [sum(wikipedia_2006), sum(wikipedia_2012), sum(wikipedia_2016)]}

df_comparison = pd.DataFrame(df_comparison, index=[2006, 2012, 2016])

df_comparison
df_comparison.plot(kind="bar", title="Number of Celebrities that died")
df_comparison["Data Set"] / df_comparison["Wikipedia"]
month_numerical = {"January": 1, "February": 2, "March": 3, "April": 4,

                   "May": 5, "June": 6, "July": 7, "August": 8, 

                   "September": 9, "October": 10, "November": 11, "December": 12}



df["death_month"] = df.death_month.map(month_numerical)
data_set = {2006: [], 2012: [], 2016: []}



for year in data_set.keys():

    for month in range(1,13):

        deaths = len(df[(df.death_year==year) & (df.death_month==month)])

        data_set[year].append(deaths)
comparison_month = {"2006": pd.Series(data_set[2006]) / pd.Series(wikipedia_2006), 

                    "2012": pd.Series(data_set[2012]) / pd.Series(wikipedia_2012),

                    "2016": pd.Series(data_set[2016]) / pd.Series(wikipedia_2016)}



comparison_month = pd.DataFrame(comparison_month)

comparison_month.index = range(1,13)

comparison_month
comparison_month.plot(marker="o", xticks=range(1,13), figsize=(12,6), legend="reverse")



plt.title("Proportion of dead celebrities that are in the data set compared to the Wikipedia article")

plt.xlabel("Month")

plt.ylabel("Proportion")