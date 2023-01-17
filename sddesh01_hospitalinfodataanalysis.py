# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

import pandas as pd

print(os.listdir("../input"))



df=pd.read_csv("../input/HospInfo.csv")

#print(df.columns)

df.shape

df.head(5)

df.describe(include="all")



#number of hospitals in each state

df1=df.groupby(['State'])['Hospital Name'].count()

df1





df.describe(exclude=[np.number])

df.describe(include=[np.number])



df1=df.groupby('State')[['Provider ID']].count()

#df2.plot.pie(subplots=True,figsize=(5,5))

df1.plot.bar(figsize=(20,10))

#plt.figure(figsize=(20,10))





#Hospital types pie chart

df2=df.groupby('Mortality national comparison')[['Provider ID']].count()

df2.plot.pie(subplots=True,figsize=(5,5))



#hospital pwnership type

df2=df.groupby('Hospital Ownership')[['Provider ID']].count()

#df2.plot.pie(subplots=True,figsize=(5,5))

plt=df2.plot.bar()

df.head()



#pivot table gives total number of hospitals aggreagated by ratings in each state. Gives a statewise comparison of which state has highest number of best rated hispitals

df3=df.pivot_table(index='State',columns='Hospital overall rating',aggfunc={'Provider ID':'count'})

df3



#df4=df[df['Emergency Services'] == "True"]

#df[df['nationality'] == "USA"]

df3=df.pivot_table(index='State',columns='Timeliness of care national comparison',aggfunc={'Provider ID':'count'})

df3

df.head()



df.columns = [c.replace(' ', '_') for c in df.columns]

df.describe(include=[np.number])

#df.Mortality_national_comparison[df.Mortality_national_comparison == 'Not Available'] = 0

#df.Mortality_national_comparison[df.Mortality_national_comparison == 'Below the national average'] = 1

#df.Mortality_national_comparison[df.Mortality_national_comparison == 'Same as the national average'] = 2

#df.Mortality_national_comparison[df.Mortality_national_comparison == 'Above the national average'] = 3

#df["Mortality_national_comparison"] = pd.to_numeric(df["Mortality_national_comparison"])

#df.describe(include=[np.number])

#df.head()



#df4=df.pivot_table(index='State',columns='Timeliness of care national comparison',aggfunc={'Provider ID':'count'})

#df5=df.groupby('Hospital Ownership','Emergency Services')[['Provider ID']].count()

df.groupby(['State','Emergency_Services'])['Provider_ID'].count().unstack().plot.bar(figsize=(20,10))

print(df.columns)