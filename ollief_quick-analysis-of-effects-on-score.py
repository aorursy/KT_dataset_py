import numpy as np

import pandas as pd

import seaborn as sns

%matplotlib inline

import datetime

import re



df = pd.read_csv("../input/Hotel_Reviews.csv")

df.head()
sns.distplot(df["Reviewer_Score"],kde=False,bins=15)
# If there are more than 100 instances of the country

countries = df["Reviewer_Nationality"].value_counts()[df["Reviewer_Nationality"].value_counts() > 100]

g = df.groupby("Reviewer_Nationality").mean()

g.loc[countries.index.tolist()]["Reviewer_Score"].sort_values(ascending=False)[:10].plot(kind="bar",ylim=(8,9),title="Top Reviewing Countries of Origin")
g.loc[countries.index.tolist()]["Reviewer_Score"].sort_values()[:10].plot(kind="bar",ylim=(7,8),title="Bottom Reviewing Countries of Origin")
def country_ident(st):

    last = st.split()[-1]

    if last == "Kingdom": return "United Kingdom"

    else: return last

    

df["Hotel_Country"] = df["Hotel_Address"].apply(country_ident)

df.groupby("Hotel_Country").mean()["Reviewer_Score"].sort_values(ascending=False)
sns.boxplot(data=df,y="Reviewer_Score",x="Hotel_Country",showfliers=False)
df["Review_Date"] = df["Review_Date"].apply(lambda date: datetime.datetime.strptime(date, '%m/%d/%Y').strftime('%Y-%m-%d'))

df["Review_Date_Month"] = df["Review_Date"].apply(lambda x: x[5:7])

df[["Review_Date","Reviewer_Score"]].groupby("Review_Date").mean().plot(figsize=(15,7))
sns.boxplot(y="Reviewer_Score",x="Review_Date_Month",data=df,showfliers=False)
g = df.groupby(["Hotel_Name","Hotel_Country"]).mean().sort_values("Average_Score",ascending=False)

g["Average_Score"].head(20)
def splitString(string):

    array = string.split(" ', ' ")

    array[0] = array[0][3:]

    array[-1] = array[-1][:-3]

    if not 'trip' in array[0]:

        array.insert(0,None)

    try:

        return float(array[3].split()[1])

    except:

        return None



df["Nights"] = df["Tags"].apply(splitString)

sns.jointplot(data=df,y="Reviewer_Score",x="Nights",kind="reg")