# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

import seaborn as sns
data = pd.read_csv("../input/fifa19/data.csv")
data
data.columns
data.corr()
# Heatmap of player's attributes correlation

fig,axes = plt.subplots(figsize = (30,30))

sns.heatmap(data.corr(),annot = True,linewidth = 0.5,axes=axes)

plt.show()
data_best = data.head(n=21)

data_best
data_best["Age"].mean()
data_best["Overall"].mean()
data
# Scatter Plot

# x = value  y = overall

# kind = kind of plot,x = given value of x axis,y = given value of y axis,alpha = opacity,color = color



data.plot(kind = "scatter",x = "Value",y = "Overall",alpha = 0.5,color = "g",figsize = (20,20))

plt.xlabel("Value")

plt.ylabel("Overall")

plt.title("Value - Overall Scatter Plot")

plt.show()
data_young = data[(data["Age"] >= 18) & (data["Age"] <= 20)]
data_young
data_young1 = data_young[(data_young["Overall"] >= 75)]
data_young1
player_french = data_young1[data_young1["Nationality"] == "France"]

player_french
player_german = data_young1[data_young1["Nationality"] == "Germany"]

player_german
player_spanish = data_young1[data_young1["Nationality"] == "Spain"]

player_spanish
player_english = data_young1[data_young1["Nationality"] == "England"]

player_english
# Before the cleaning data,first I want to delete unnecessary columns from DataFrame

# Let's see the columns of DataFrame

data.columns
data = data.drop(["Photo","Flag","Club Logo","Real Face","Joined","Loaned From",'LS', 'ST', 'RS', 'LW', 'LF', 'CF', 'RF', 'RW',

       'LAM', 'CAM', 'RAM', 'LM', 'LCM', 'CM', 'RCM', 'RM', 'LWB', 'LDM',

       'CDM', 'RDM', 'RWB', 'LB', 'LCB', 'CB', 'RCB', 'RB'],axis = 1)
data
data.info()
data_eda = data.describe()

data_eda
data20 = data.head(20)

data20
# Melting data



data_melted1 = pd.melt(frame = data20,id_vars = "Name",value_vars = ["Club","Overall","Nationality","Age","Value"])

data_melted1
# Pivoting data



data_pivot1 = data_melted1.pivot(index = "Name",columns = "variable",values = "value")

data_pivot1
# This time,we melt some other columns : Preferred Foot,Position,Jersey Number,Height,Weight,Release Clause



data_melted2 = pd.melt(frame = data20,id_vars = "Name",value_vars = ["Preferred Foot","Position","Jersey Number","Height","Weight","Release Clause"])

data_melted2
data_pivot2 = data_melted2.pivot(index = "Name",columns = "variable",values = "value")

data_pivot2
# Concatenating dataframes



data_new20 = pd.concat([data_pivot1,data_pivot2],axis = 1)

data_new20
data_new20.boxplot(column = "Overall", by = "Preferred Foot",figsize = (15,15))

plt.show()