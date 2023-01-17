# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Read the dataset

df = pd.read_csv("../input/military-expenditure-of-countries-19602019/Military Expenditure.csv")
# Select the rows that contain data on Turkey

df_turkey = df[df.Name == "Turkey"]
# Select columns that contain year data

df_turkey_years = df_turkey.iloc[:,4:63]
# Melt the dataframe so that each year has its own row

df_turkey_years_rows = df_turkey_years.melt()
# Rename the columns

df_turkey_years_rows.rename(columns = {"variable":"year", "value":"Turkish expenditure"}, inplace = True)

print(df_turkey_years_rows.head())
# Did the same stuff as Turkey but for Greece

df_greece = df[df.Name == "Greece"]

df_greece_years = df_greece.iloc[:, 4:63]

df_greece_years_rows = df_greece_years.melt()

df_greece_years_rows.rename(columns = {"variable":"year", "value":"Greek expenditure"}, inplace = True)

print(df_greece_years_rows.head())
# Merged the Greek and Turkish dataframes

turkey_greece = pd.merge(df_turkey_years_rows, df_greece_years_rows, how = "left")

print(turkey_greece.head())

print(turkey_greece.describe())
# Set the titles for x and y axes, and also the graph

plt.title("Greek and Turkish military expenditure in the last 58 years")

plt.xlabel("Years")

plt.ylabel("Expenditure (in US Dollars)")



# Set the x-axis as year and the y-axis as expenditures

plt.plot(turkey_greece["year"], turkey_greece["Turkish expenditure"], color = "red")

plt.plot(turkey_greece["year"], turkey_greece["Greek expenditure"], color = "blue")



# Set custom labels for the x and y axes

ax = plt.subplot()

ax.set_xticks([-2, 0, 10, 20, 30, 40, 50, 58, 60])

ax.set_xticklabels(["", "1960", "1970", "1980", "1990", "2000", "2010", "2018", ""])

ax.set_yticks([0, 3.397263e+09, 6.788288e+09, 1.064135e+10, 1.896711e+10, 1.996711e+10])

ax.set_yticklabels(["0", "3.3b", "6.7b", "10.6b", "18.9b", ""])



# Leave a little bit of empty space between the graph itself and the outlines

plt.axis([-2, 60, 0 ,1.996711e+10])



# Create a legend

plt.legend(["Turkey", "Greece"])



# Show the graph

plt.show()