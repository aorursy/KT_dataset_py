# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns  # visualization tool



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/leagueoflegends/matchinfo.csv")
data.info()
# correlation map

f,ax = plt.subplots(figsize=(18, 18))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
data.head(8)  #first 5 rows

data.tail(8) # last 5 rows
data.columns  #features
# Line plot

data.gamelength.plot(kind='line',color='b',label='Game Length',linewidth =1,alpha = 0.5,grid=True,linestyle = "-") 

data.Year.plot(color='g',label = 'Year',linewidth = 3,grid=True,linestyle= ":")

plt.legend(loc="center")  # puts label into center

plt.xlabel("Game Length")

plt.ylabel("Year")

plt.title("Line Plot")

plt.show()
# Scatter plot

plt.scatter(data.Year,data.rResult,color="red",alpha=0.7)

plt.show()
# Histogram

data.Year.plot(kind ='hist',bins=50,figsize = (10,10))

plt.show()
x = (data['blueTeamTag'] == 'TSM') & (data['gamelength']<30)

data[x]
# This is also same with previous code line.

data[np.logical_and(data['blueTeamTag'] == 'TSM' ,data['gamelength']<30)]
# Lets classify game length whether game is under or over threshould value. 

t_value = data.gamelength.mean() #t_value =threshould value

print(t_value)

data["Status"] = ["over" if each > t_value else "under " for each in data.gamelength] # new feature

data.loc[:20,["Status","gamelength"]]

data = pd.read_csv("../input/leagueoflegends/matchinfo.csv")
data.shape # number of rows and columns in table
# Exploratory Data Analysis

print(data['blueTeamTag'].value_counts(dropna=False))

# As you can see SKT is 195 times on the blue side.
# For example mean of game length is 37 minutes

data.describe() 
# Visual Exploratory Data Analysis

# We'll use box plot

data.boxplot(column='gamelength',by ='rResult')
data_melting = data.head(10)

data_melting
# Lets melt

# id_vars = what we do not wish to melt

# value_vars = what we want to melt

melted = pd.melt(frame=data_melting,id_vars='redJungle',value_vars ='redJungleChamp')

melted
# Pivoting Data = Reverse of melting

# melted.pivot(index ='redJungle',columns ='variable',values ='value',)

# Concatenating Data

data1 = data.head()

data2 = data.tail()

data_conc_row = pd.concat([data1,data2],axis=0,ignore_index=True) # axis = 0 adds dataframes horizontally 

data_conc_row
data1 = data['redADC'].head(10)

data2 = data['redADCChamp'].head(10)

data_conc_col = pd.concat([data1,data2],axis=1) # axis = 1 adds dataframes vertically

data_conc_col
data.dtypes
# lets convert int to object

# we will use astype() function

data["gamelength"] = data["gamelength"].astype("object")

data.dtypes

# gamelength's type is now object.
data.info()

# As you can see there are 7620 entries. However blueTeamTag has 7582 non-null object like redTop so they have 38 null objects.

# Let's drop missing values

data1=data # also we will use data to fill missing value so I assign it to data1 variable

data1['blueMiddle'].dropna(inplace=True)
assert  data['blueMiddle'].notnull().all() # returns nothing because we drop missing values

# returns nothing's means we wrote correct code.
assert 1==2 # return error because it is false