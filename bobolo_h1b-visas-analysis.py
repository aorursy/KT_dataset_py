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
import matplotlib.pyplot as plt

import matplotlib

import seaborn as sns

import warnings

warnings.filterwarnings("ignore")
df=pd.read_csv("../input/h1b_kaggle.csv", index_col=0)
df=df.dropna() 

df["CASE_STATUS"].unique()
df.CASE_STATUS.value_counts()
cases=["CERTIFIED-WITHDRAWN", "WITHDRAWN", "CERTIFIED", "DENIED"]
df["YEAR"]=df["YEAR"].astype(str).str[:4]
df = df.reset_index(drop=True)

df.head(3)
df2=df[["YEAR","CASE_STATUS"]]

df2=df2[df2["CASE_STATUS"].isin(cases)]

df2.head(3)
ax=sns.factorplot("YEAR", data=df2, hue="CASE_STATUS", kind="count", palette="PRGn",size=8,aspect=2.)

sns.set(style="white", color_codes=True)
df=df[df.CASE_STATUS == "CERTIFIED"]
city=[]

state=[]

for x in df["WORKSITE"]:

    city.append(x.split(",")[0])  # city.append(x.split(",")[0].replace(" ", "~"))

    state.append(x.split(",")[1][1:])
df["CITY"]=city

df["STATE"]=state
df.head(3)
df["STATE"].value_counts()[0:5]
states=["CALIFORNIA","TEXAS","NEW YORK","NEW JERSEY","ILLINOIS"]
df3=df[["STATE", "YEAR"]]

df3=df3[df3["STATE"].isin(states)]

ax=sns.factorplot("STATE", data=df3, hue="YEAR", kind="count", palette="PRGn",size=8,aspect=2.)

sns.set(style="white", color_codes=True)
df["CITY"].value_counts()[0:5]
cities=["NEW YORK", "HOUSTON", "SAN FRANCISCO", "ATLANTA", "CHICAGO"]

df4=df[["CITY", "YEAR"]]

df4=df4[df4["CITY"].isin(cities)]

ax=sns.factorplot("CITY", data=df4, hue="YEAR", kind="count", palette="PRGn",size=8,aspect=2.)

sns.set(style="white", color_codes=True)
df["JOB_TITLE"].value_counts()[0:5,]
jobs=["SOFTWARE ENGINEER","PROGRAMMER ANALYST","SYSTEMS ANALYST","COMPUTER PROGRAMMER","SOFTWARE DEVELOPER"]
df5=df[df["STATE"].isin(states)]

df5=df5[df5["JOB_TITLE"].isin(jobs)]

ax=sns.factorplot("STATE", data=df5, hue="JOB_TITLE", kind="count", palette="PRGn",size=8,aspect=2.)

sns.set(style="white", color_codes=True)
df6=df[df["JOB_TITLE"].isin(jobs)&(df.PREVAILING_WAGE<250000.0)&(df.PREVAILING_WAGE>10000.0)]

ax=sns.factorplot("JOB_TITLE", "PREVAILING_WAGE", "YEAR",df6, kind="box",                        

                   palette="PRGn",size=8,aspect=2.)

ax.set(xlabel="Job Title")

ax.set(ylabel="Wage USD")

sns.set(style="white", color_codes=True)
df7=df[df["STATE"].isin(states)&(df.PREVAILING_WAGE<500000.0)&(df.PREVAILING_WAGE>10000.0)]

ax=sns.factorplot("STATE", "PREVAILING_WAGE", "YEAR",df7, kind="box",                        

                   palette="PRGn",size=8,aspect=2.)

ax.set(xlabel="State")

ax.set(ylabel="Wage USD")

sns.set(style="white", color_codes=True)
df8=df[df["JOB_TITLE"].isin(jobs)&(df.PREVAILING_WAGE<250000.0)&(df.PREVAILING_WAGE>10000.0)]

g = sns.FacetGrid(df8, row="JOB_TITLE", aspect=4, size=5, margin_titles=False)

g.map(sns.kdeplot, "PREVAILING_WAGE", shade=True, color="g")

for ax in g.axes.flat:

    ax.yaxis.set_visible(False)

sns.despine(left=True)

g.fig.subplots_adjust(hspace=0.1)

g.set(xlim=(10000, 150000))

sns.set(style="white", color_codes=True)
df9=df[df["STATE"].isin(states)&(df.PREVAILING_WAGE<500000.0)&(df.PREVAILING_WAGE>10000.0)]

g = sns.FacetGrid(df9, row="STATE", aspect=4, size=5, margin_titles=False)

g.map(sns.kdeplot, "PREVAILING_WAGE", shade=True, color="g")

for ax in g.axes.flat:

    ax.yaxis.set_visible(False)

sns.despine(left=True)

g.fig.subplots_adjust(hspace=0.1)

g.set(xlim=(10000, 150000))

sns.set(style="white", color_codes=True)