# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
import matplotlib.pyplot as plt

# Any results you write to the current directory are saved as output.
winter=pd.read_csv("../input/winter.csv")
winter.head()
summer=pd.read_csv("../input/summer.csv")
summer.head()
#checking for null values for summer olympics
summer.isnull().any()
#checking for null values for winter olympics
winter.isnull().any()
#data for year wise  and city wise olympics for summer
summer.groupby("Year").City.first()
#data for year wise  and city wise olympics for summer
winter.groupby("Year").City.first()
#Countries with most medals in summer olympics from 1st to till date olympics (Top 10 teams)
summer.Country.value_counts().head(10).plot(kind="bar");
#Countries with most medals in winter olympics from 1st to till date olympics (Top 10 teams)
winter.Country.value_counts().head(10).plot(kind="bar");
#overall status of medal winners in summer and winter olympics combined (top 10)
combined_olympics=pd.concat([summer,winter])
combined_olympics.Country.value_counts().head(10).plot(kind="bar")
#listing top 10 teams
top_medal=summer.Country.value_counts()
top_summer=top_medal.reset_index()
top_summer=top_summer.rename(columns={"index":"Team"})
top_s=list(top_summer.Team.head(10))
top_s

#Top teams to win gold, silver, bronze for summer olympics till date top 10
medal_top=summer[["Year","Country","Medal"]]
medal_top.groupby(["Country","Medal"]).Year.count().reset_index().pivot("Country","Medal","Year").fillna(0).sort_values("Gold",ascending=False).head(10).plot(kind="bar");

#Top teams to win gold, silver, bronze for winter olympics till date (top 10)
medal_top=winter[["Year","Country","Medal"]]
medal_top.groupby(["Country","Medal"]).Year.count().reset_index().pivot("Country","Medal","Year").fillna(0).sort_values("Gold",ascending=False).head(10).plot(kind="bar");
#Top teams to win gold, silver, bronze for summer and  winter olympics till date (top 10)
medal_wise=combined_olympics[["Year","Country","Medal"]]
medal_wise.groupby(["Country","Medal"]).Year.count().reset_index().pivot("Country","Medal","Year").fillna(0).sort_values("Gold",ascending=False).head(10).plot(kind="bar")
#athlete who has won maximum olympics gold till date in summer olympics top(10)
summer.Athlete.value_counts().head(10)
#athlete who has won maximum olympics gold till date in winter olympics top(10)
winter.Athlete.value_counts().head(10)
#Year wise male and female participation summer olympics
summer.groupby(["Year","Gender"]).Medal.count().reset_index().pivot("Year","Gender","Medal").fillna(0).plot();
ax=plt.gcf()
ax.set_size_inches(12,8)


#Maximum number of participants discipline wise ( summer top 10)
summer.Discipline.value_counts().head(10).plot(kind="bar")
ax=plt.gcf()
ax.set_size_inches(10,8)

#Maximum number of participants discipline wise ( winter top 10)
winter.Discipline.value_counts().head(10).plot(kind="bar")
ax=plt.gcf()
ax.set_size_inches(10,8)

