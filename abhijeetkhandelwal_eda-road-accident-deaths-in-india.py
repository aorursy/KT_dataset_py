# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# States are the individual states where death happened.

# Cause is the reason for road fatalities (Other is for the reason where details not available)

# Male and female breakup of the data

# Lat and Long are the latitude and longitude of the incident where death happended on exploration I found out this is state location.

dataframe = pd.read_excel('../input/datafile.xls')

dataframe.head(5)
# we have 6300 unique values with no "NULL" value Yaay!

dataframe.info()
# Describing data statistically

dataframe.describe()
# Finding count of unique causes and unique states

print (dataframe.describe(include=['O']))
#Death of people state wise from 2001-2012

# From graph we can infer maximum death happened in Andhra Pradesh

fig = plt.figure(figsize=(60,100))

df = dataframe.groupby(['States']).agg({'Total':sum}).head(10)

df.plot(kind="barh", fontsize = 8)

plt.xlabel('Number of death by states', fontsize = 6)

plt.show ()
# Finding out fatalities over the year. 2012 has the highest recoreded death.

plt.bar(dataframe["Year"],dataframe["Total"])

plt.show()
#Plotting causes for the death and as per data, Lories and trucks are major reason for the death and 2 wheeler is second major reason

fig = plt.figure(figsize=(40,40))

df_vehicle = dataframe.groupby(['CAUSE']).agg({'Total':sum})

df_vehicle.plot(kind="barh", fontsize = 8)

plt.xlabel('Deaths Cause', fontsize = 8)

plt.show ()
#Breaking down deaths on the basis of gender over the year. Majority death was of Male.

new_dataframe = dataframe[["Year","Male","Female"]]

var = new_dataframe.groupby(['Year']).sum()

var.plot(kind='bar',stacked=True,grid=False)

plt.xlabel('Deaths', fontsize = 8)

plt.ylabel('Number of deaths gender wise', fontsize=8)

plt.title("Stacked Death Genderwise over the year")

plt.show()
#Maximum deaths occured in Andhrapradesh, Lets explore causes of road deaths in Andhrapradesh

# Fatalities in road accident in andhra is same as nationally. But the death due to 3 wheeler and others are more in comparision to india.

fig = plt.figure(figsize=(40,40))

data_frame_andhra = dataframe[dataframe["States"] == "ANDHRA PRADESH"]

df_Andhdra = data_frame_andhra.groupby(['CAUSE']).agg({'Total':sum})

df_Andhdra.plot(kind="barh", fontsize = 8)

#plt.grid(b=True, which='both', color='Black',linestyle='-')

plt.xlabel('Deaths Cause in Andhrapradesh', fontsize = 8)

plt.show ()
#Finding out if any correlation exist

dataframe.corr()
# Now lets break down more and try to find out death causes for female and male seperatly

new_dataframe_female = dataframe[["Female","CAUSE"]]

var = new_dataframe_female.groupby(['CAUSE']).agg({'Female':sum})

var.plot(kind='bar',grid=False)

plt.xlabel('Deaths', fontsize = 8)

plt.ylabel('Number of deaths of female with cause', fontsize=8)

plt.title("Stacked Death Female over the year")

plt.show()
new_dataframe_male = dataframe[["Male","CAUSE"]]

var = new_dataframe_male.groupby(['CAUSE']).agg({'Male':sum})

var.plot(kind='bar',grid=False)

plt.xlabel('Deaths', fontsize = 8)

plt.ylabel('Number of deaths of Male with cause', fontsize=8)

plt.title("Stacked Death Male over the year")

plt.show()