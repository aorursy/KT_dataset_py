# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.

data = pd.read_csv("../input/StudentsPerformance.csv")
name = ["Gorulmeyenler", "Eastern Turkey"]

author = ["Roy Jacobsen", "Sevan Nisanyan"]

pages = [179,195]

ListLabel = ["name","author","pages"]

ListColumn = [name,author,pages]

zipp = dict(zip(ListLabel,ListColumn))

df = pd.DataFrame(zipp)

df
df["type"] = ["Novel","Travel"] #add new column

df["year"] = 0 #broadcasting

df

#before plotting examine data set.

data.info()
#change and organize column names.

data.columns = [each.split()[0]+"_"+each.split()[1] if (len(each.split()) > 1) else each for each in data.columns]

data.rename(columns = {"race/ethnicity" : "race_ethnicity"}, inplace = True)

data.columns
#line plot

data2 = data.loc[:,["writing_score","math_score","reading_score"]]

data2.plot()

plt.show()
#subplots

data2.plot(subplots = True)

plt.show()
#scatter plot

data2.plot(kind = "scatter", x = "reading_score", y = "writing_score")

plt.xlabel("Reading Score")

plt.ylabel("Writing Score")

plt.title("Correlation Between Writing and Reading Scores")

plt.show()
#histogram

data2.plot(kind = "hist", y = "math_score", alpha = 0.7, color = "r", range = (0,45), density = False)

data2.plot(kind = "hist", y = "math_score", alpha = 0.7, color = "r", density = False,cumulative = True)

plt.show()
time_list = ["1994-09-07","1986-01-06"]

print(time_list)

print(type(time_list[1]))

time_list = pd.to_datetime(time_list)

print(time_list)

print(type(time_list))

import warnings

warnings.filterwarnings("ignore")



data1 = data.head()

date_list = ["2000-01-10","2000-01-02","2000-03-11","2000-04-05","2001-07-08"]

date_list = pd.to_datetime(date_list)

data1["date"] = date_list

data1 = data1.set_index("date")

data1
print(data1.loc["2000-01-10"])
print(data1.loc["2000-01-01" : "2000-03-03"])
data1.resample("A").mean()
data1.resample("M").mean()
data1.resample("M").first().interpolate("linear")
data1.resample("M").mean().interpolate("linear")