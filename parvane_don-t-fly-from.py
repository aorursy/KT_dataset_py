import pandas as ps

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec

import matplotlib

from matplotlib import cm

import pylab

import string

import numpy as np
fileR = ps.read_csv("../input/3-Airplane_Crashes_Since_1908.txt", sep=",")

print(fileR.head())
matplotlib.rcParams['figure.figsize'] = (10,5)

ops = fileR["Operator"].value_counts()[:20]

ops.plot(kind="bar",legend="Operator",color ="g",fontsize=10, title="Operators with Highest Crashes")
types = fileR["Type"].value_counts()[:20]

types.plot(kind="bar",legend="Types",color ="g", fontsize=10,title="Types with Highest Crashes")
fileR['Date'] = ps.to_datetime(fileR['Date'])

fileR['year'] = fileR['Date'].dt.year

fileR['month'] = fileR['Date'].dt.month

fileR['day'] = fileR['Date'].dt.day

sub_years    = [1900,1910,1920,1930,1940,1950,1960,1970,1980,1990,2000,2010]

years_legend = list(string.ascii_letters[:len(sub_years)])

fileR["year_group"] = ""

for i in range(0,(len(sub_years)-1)):

   fileR.loc[(sub_years[i+1]>fileR["year"]) & (fileR["year"] >= sub_years[i]) , ["year_group"]] = years_legend[i]
matplotlib.rcParams['figure.figsize'] = (10,5)

fileR[["Fatalities","year_group"]].groupby("year_group").count().plot(kind="bar",fontsize=14,legend=True,color ="g", title="Fatalities based on decades"

                                                               )
labels = ["1900-1910","1910-1920","1920-1930","1930-1940","1940-1950","1950-1960","1960-1970","1970-1980","1980-1990","1990-2000","2000-2010"]

sizes  = fileR[["Fatalities","year_group"]].groupby("year_group").sum()

explode = (0, 0, 0, 0, 0, 0, 0, 0.1, 0.1, 0,0)  

colors = cm.Set1(np.arange(20)/30.)

plt.pie(sizes, explode=explode, labels=labels, colors=colors,

        autopct='%1.1f%%', shadow=True, startangle=45)

plt.axis('equal')

plt.show()
subfile2 = (fileR[["Aboard","Fatalities","year","Operator","Type"]].groupby("Operator").sum())

subfile2["survived"] = subfile2["Aboard"]- subfile2["Fatalities"]

subfile2["percentageSurvived"] = subfile2["survived"]/subfile2["Aboard"]

subfile3= subfile2[subfile2["year"]>max(fileR["year"])] # Exclude the records with one observation
highSurvive = subfile3.sort_values(by="percentageSurvived", ascending=False)[:20]# sorting the values 

highSurvive
highSurvive["percentageSurvived"].plot(kind='bar', color="g",fontsize=14, title="Operators with high percentage of survivers")


subfile = (fileR[["Aboard","Fatalities","year"]].groupby("year").sum())

subfile["survived"] = subfile["Aboard"]- subfile["Fatalities"]

pylab.plot(subfile["Aboard"],label="Aboard")

pylab.plot(subfile["Fatalities"],label="Fatalities")

pylab.plot(subfile["survived"],label="Survived")

pylab.legend(loc='upper left')
test = fileR["Location"]

loc = test.str.split(",")

countries=[]

for i in range(0,len(loc)):

    if(not(type(loc.iloc[i])==float)):

        dim = len(loc[i])

        countries.append(loc[i][dim-1])

    else:

        countries.append("")

fileR["countries"] = countries
countrySub = fileR.groupby("countries").sum()

dangerousCountries = countrySub.sort_values("Fatalities",ascending=False)

dangerousCountries["Fatalities"][:20].plot(kind = "bar", color = "g", fontsize=14, title="Highest fatalities based on the location")