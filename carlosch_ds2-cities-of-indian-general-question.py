# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as mpl

import seaborn as sb

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

print ("key questiom:")

print("1.-Which are cities in each states?")

print("2.-How many people on average are literate in each state by gender?")

print("3.-How stable are the states with respect to their populations of cities?")

print("4.-How many literates are graduate within each states?")

print("5.-Which is the most/least literates state?")

print("6.-Which is the most/least graduates state?")

print("7.-Which is the most child rate state on average ? and the least child rate state on average?")

print("8.-Which do city have most child rate? most graduates?  most literates?")

print("9.-Which is the most population state?")

print("10.-is there men are graduater than women?")

print ("try it!!!!" )

print ("Post_data : sorry fo my english i'm a new with it! , and i enjoy my work")

print("if you have a improved note , tell me please!!!")





# Any results you write to the current directory are saved as output.
Data = pd.read_csv("../input/cities_r2.csv") # take data 

Data.info()

for x in Data.columns: # see the integrity of data: do those have a NaN??

    if Data[x].isnull().values.any():

        print( x,' has a value NaN.')

else:

    print ("all are cleaned apparently")
Data.head(5) # see the structure of datas
DataN = Data.groupby("state_name").filter( lambda x : len(x) > 4 ) # select states with size > 4

Data = DataN

print( Data.groupby("state_name").size() )
#NameState = { x:[] for x in DataN["state_name"].unique() } # first version

from collections import defaultdict

NameState = defaultdict(list)

for x in Data.itertuples():

    NameState[ x.state_name ].append( x.name_of_city)

print( NameState["TAMIL NADU"]) # put the name of state you want
Q2 = Data.groupby("state_name").agg({"literates_male": np.mean , "literates_female": np.mean }).sort(["literates_male"], ascending = True )

Q2.plot(kind = "barh")
DataN = Data[[ "state_name", "population_total"] ].groupby("state_name").plot(kind="box")
DataN = Data[["state_name", "literates_total","total_graduates"]].groupby("state_name").agg({"literates_total":np.sum , "total_graduates":np.sum})

DataN["graduates_by_literates_rates"] = DataN["total_graduates"]/DataN["literates_total"]

DataN[["graduates_by_literates_rates"]].sort(["graduates_by_literates_rates"], ascending= True).plot(kind="barh")
DataN = Data[["state_name", "population_total", "literates_total"]].groupby("state_name").agg({"population_total":np.sum,"literates_total":np.sum})

DataN["literates_rate"] = DataN["literates_total"]/DataN["population_total"]

DataN[["literates_rate"]].sort(["literates_rate"], ascending = True ).plot(kind = "barh")
DataN = Data[["state_name", "population_total", "total_graduates"]].groupby("state_name").agg({"population_total":np.sum,"total_graduates":np.sum})

DataN["graduates_rate"] = DataN["total_graduates"]/DataN["population_total"]

#DataN[["graduates_rate"]].sort(["graduates_rate"], ascending = True ).plot(kind="barh")
#DataN = Data[["state_name","0-6_population_total"]].groupby("state_name").agg({ "0-6_population_total" : np.mean })

#DataN.sort_values(by="0-6_population_total", ascending = True).plot(kind = "barh")
Data1 = pd.read_csv("../input/cities_r2.csv")

Data1 = Data1[["name_of_city","literates_total","total_graduates","0-6_population_total"]].set_index("name_of_city")

Data0 = Data1[["0-6_population_total"]].sort_values(by = "0-6_population_total", ascending= True).nlargest(10,"0-6_population_total")

Data0.plot(kind= "barh")
Data0 = Data1[["total_graduates"]].sort_values(by="total_graduates",ascending = True ).nlargest(10,"total_graduates")

#Data0.plot(kind="barh")
Data0 = Data1[["literates_total"]].sort_values(by="literates_total",ascending = True ).nlargest(10,"literates_total")

Data0.plot(kind = "barh")
Data0 = Data[["state_name","population_total"]].groupby("state_name").agg({"population_total":np.sum})

Data0.sort_values(by="population_total").plot(kind = "barh")
print( Data[["male_graduates","female_graduates"]].sum(axis = 0 ) )