# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

# -*- coding: utf-8 -*-



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns # visualization tool



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls","../input"]).decode("utf8"))

import os

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

data = pd.read_csv("../input/cards.csv",encoding='iso-8859-1')
data.columns
data.head()

# data.head(10) -> Show us 10 rows from the top
data.tail()

# data.tail(10) -> Show us 10 rows from the bottom
data.info()
data.describe()
data.isnull().head()
# To check if is there any null value in column or not

data.isnull().any()
# To get info about the total number of values in each oolumn

data.isnull().sum()
# data.playerClass

data["playerClass"]
data.loc[2:10]  # shows all column values between 2 and 10 rows

data.loc[2:10,"name"] # shows only name row values between 2 and 10 rows

data.loc[:,"name"] # shows all rows for name rows values

data.loc[2:10,["name", "type"]] # shows name and type row values between 2 and 10 rows

data.loc[2:10,:"type"] # shows the columns up to type columns between 2 and 10 rows
filter = data.health > 50

data[filter]
first_filter = data.health > 50

second_filter = data.attack > 20

data[first_filter | second_filter].loc[:,["name","health","attack"]][:8]
first_filter = data.health > 50

second_filter = data.health < 100

data[first_filter & second_filter]
first_filter = data.health > 50

second_filter = data.attack < 5

third_filter = data.type == "MINION"

data[first_filter & second_filter & third_filter]
dictionary = {"Name": ["Naruto","Sasuke","Sakura","Kakashi","Orochimaru","Kabuto","Itachi"],

              "Age": [16,15,17,33,38,40,32]}



temporary_data = pd.DataFrame(dictionary)

temporary_data
temporary_data.loc[0,["Age"]] = 18

temporary_data.loc[3,["Name"]] = "Mitsuki"

temporary_data

temporary_data.loc[4:5,["Age"]] = 20

temporary_data
temporary_data["Age Value"] = ["Old" if each> 30 else "Young" for each in temporary_data.Age]

temporary_data
temporary_data.rename(columns = {"Name":"name","Age":"age"}, inplace=True) 

# inplace -> Whether to return a new DataFrame

temporary_data
temporary_data.columns = [each.lower().split()[0] +"_"+ each.lower().split()[1]

                     if (len(each.split()))> 1 else each.lower()

                     for each in temporary_data.columns]



temporary_data
temporary_data.drop(["age_value"], axis=1) 

# axis - {0 or ‘index’, 1 or ‘columns’}, default 0

# axis -> Whether to drop labels from the index (0 or ‘index’) or columns (1 or ‘columns’).

# temporary_data.drop(["age_value","name"], axis=1) # to delete more columns

temporary_data.drop([0])

# temporary_data.drop([0,3,5])

temporary_data
temporary_data.drop(["age_value"], axis=1, inplace=True)

temporary_data
minion_data = data[data.type=="MINION"]

minion_data.head()
minion_data.reset_index(inplace=True)

minion_data.head()
minion_data = minion_data.drop(["index"], axis=1)

minion_data.head()
minion_data.info()
minion_data.notnull().any()
minion_data = minion_data.drop(columns=["durability"])

minion_data.head()
minion_data.iloc[0:4]
# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line

minion_data.iloc[0:100].attack.plot(kind = 'line', color = 'g',label = 'Attack',linewidth=1,alpha = 0.7,grid = True,linestyle = '--')

minion_data.iloc[0:100].health.plot(color = 'r',label = 'Health',linewidth=1, alpha = 0.7,grid = True,linestyle = '-.')

plt.legend(loc='upper right')     # legend = puts label into plot

plt.xlabel('x axis')              # label = name of label

plt.ylabel('y axis')

plt.title('Line Plot')            # title = title of plot

plt.show()
# x = attack, y = helath

minion_data.iloc[0:200].plot(kind='scatter', x='attack', y='health',alpha = 0.5,color = 'red')

plt.xlabel('Attack')                                # label = name of label

plt.ylabel('Health')

plt.title('Attack Health Scatter Plot')            # title = title of plot
# bins = number of bar in figure

minion_data.attack.plot(kind = 'hist',bins = 50,figsize = (12,12))

plt.show()
minion_data.corr()
# annot, Indicates whether numbers should appear 

# linewidths, Indicates the thickness of the lines between the tabs.

# fmt, Indicates how many digits we will write after zero (decimal number).

# figsize, Indicates the size of the area to be displayed on the screen like 18x18 



f,ax = plt.subplots(figsize=(10, 10))

sns.heatmap(minion_data.corr(), annot=True, linewidths=0.5, fmt= '.2f',ax=ax)

plt.show()