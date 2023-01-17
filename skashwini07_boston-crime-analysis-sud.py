# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns
boston_crime = pd.read_csv("../input/crime.csv", encoding = "ISO-8859-1",low_memory = False)
#year-wise crime

fig = plt.figure(figsize=(8,4))

ax = fig.add_subplot(1,1,1)

sns.countplot(x="YEAR",data=boston_crime)

ax.set_title("Year wise crime frequency")

plt.tight_layout()

plt.show()
#Most crimes occurred in which day?



day_crime= boston_crime.groupby(["DAY_OF_WEEK"])["OFFENSE_CODE"].count().reset_index()

fig = plt.figure(figsize=(12,8))

ax = fig.add_subplot(111)

p=sns.lineplot(x=day_crime.iloc[:,0], y=day_crime.iloc[:,1], data=day_crime)

p.set_ylabel("No. of Crimes Occurred")

p.set_xlabel("Any Day Of the Week")

plt.tight_layout()

plt.show()
#District-wise shooting occurred

shootingOccurred = boston_crime[boston_crime["SHOOTING"] == "Y"].groupby("DISTRICT").agg("SHOOTING").count().reset_index().sort_values("SHOOTING", ascending=False)

shootingOccurred

fig = plt.figure(figsize=(12,8))

ax = fig.add_subplot(111)

p = sns.barplot(x=shootingOccurred.DISTRICT, y=shootingOccurred.SHOOTING, data=shootingOccurred)

plt.tight_layout()

plt.show()
#Based on type of offense

fig = plt.figure(figsize=(6,6))

ax = fig.add_subplot(111)

d = boston_crime["OFFENSE_CODE_GROUP"].value_counts().head(5).reset_index()

p = sns.barplot(x=d.iloc[:,1], y=d.iloc[:,0], data=d, palette="winter")

p.set_xlabel("No. of crimes occurred")

p.set_ylabel("Nature of offense")

plt.tight_layout()

plt.show()
##Distribution of crimes by Hour

fig = plt.figure(figsize=(8,6))

ax = fig.add_subplot(1,1,1)

sns.countplot(x="HOUR", data = boston_crime)

ax.set_title("Hour wise crime frequency")

plt.tight_layout()

plt.show()

#Crime rates are at a minimum at 4 and 5 am in the morning

#Crime rates peak at between 4-6 pm
#YEARWISE breakup of Crimes by District



sns.catplot(x="DISTRICT",       # Variable whose distribution (count) is of interest

            hue="MONTH",      # Show distribution, pos or -ve split-wise

            col="YEAR",       # Create two-charts/facets, gender-wise

            data=boston_crime,

            kind="count")

#Crime rates are consistent in the districts B2, C11 & D4 across the 4 years
def get_SeasonName(x):

    y = (x%12+3)//3

    if(y==4):

        return "Spring"

    elif(y==3):

        return "Winter"

    elif(y==2):

        return "Autumn"

    else:

        return "Summer"

    



boston_crime["SEASON"] = boston_crime["MONTH"].apply(lambda x: get_SeasonName(x))
#No. of crimes occurred in each season

df = boston_crime.groupby(["YEAR","SEASON"])["OFFENSE_CODE"].count().reset_index()

df

fig = plt.figure(figsize=(8,8))

ax = fig.add_subplot(111)

p = sns.barplot(x="SEASON", y="OFFENSE_CODE", hue="YEAR", data=df)#, palette="spring")

p.set_ylabel("crime frequency")

p.set_xlabel("Season")

plt.tight_layout()

plt.show()
import calendar

boston_crime["MONTH_NAME"] = boston_crime["MONTH"].apply(lambda x : calendar.month_abbr[x])



df = boston_crime.groupby(["YEAR","MONTH_NAME","DAY_OF_WEEK"])["OFFENSE_CODE"].count().reset_index()

df

fig = plt.figure(figsize=(12,12))

ax = fig.add_subplot(111)

p = sns.scatterplot(x="MONTH_NAME", y="OFFENSE_CODE", hue="DAY_OF_WEEK", data=df)

p.set_ylabel("No. of Crimes Occurred")

p.set_xlabel("Months")

plt.tight_layout()

plt.show()


#Top 5 most crime occurred street

df = boston_crime["STREET"].value_counts().head(5).reset_index()

fig = plt.figure(figsize=(6,6))

ax = fig.add_subplot(111)

p = sns.barplot(x=df.iloc[:,0], y=df["STREET"], data=df)#, palette="spring")

p.set_ylabel("No. of Crimes Occurred")

p.set_xlabel("Street Name")

p.set_xticklabels(p.get_xticklabels(), rotation=90)

plt.tight_layout()

plt.show()



#No. of hours spent on each Offence by district

df = boston_crime.groupby(["DISTRICT", "OFFENSE_CODE_GROUP"])["HOUR"].sum().reset_index().sort_values("HOUR", ascending=False)

df

fig = plt.figure(figsize=(12,12))

ax = fig.add_subplot(111)

p = sns.scatterplot(x="HOUR", y="OFFENSE_CODE_GROUP", hue="DISTRICT", data=df, palette="summer")

p.set_ylabel("No. of Crimes Occurred")

p.set_xlabel("Hours")

plt.tight_layout()

plt.show()