# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from subprocess import check_output

from wordcloud import WordCloud, STOPWORDS



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os



# Any results you write to the current directory are saved as output.
df_startups = pd.read_csv("../input/startups/startup_funding.csv",index_col=0)
print(df_startups.shape)

print(df_startups.nunique())

print(df_startups.info())
df_startups.head()
df_startups.shape
print("Data discription")

print(df_startups.isnull().sum())



Investors = df_startups.InvestorsName.value_counts()



print("Number of Investors")

print(Investors[:20])
plt.figure(figsize = (12,5))

plt.title("Major Investors")

g=sns.barplot(x=Investors.index[:10], y=Investors.values[:10])



locs, labels = plt.xticks()

plt.setp(labels, rotation =45)

g.set_xlabel("Investor", fontsize=17)

g.set_ylabel("Count", fontsize=17)

plt.show()
df_startups["CityLocation"]= df_startups["CityLocation"].replace("Gurugram", "Gurgaon")

#Formatting of data.
City = df_startups.CityLocation.value_counts()

print ("Name of the city")

print (City[:10])



plt.figure(figsize = (12, 5))

plt.title ("Major Cities")

sns.barplot(x=City.index[:10], y=City.values[:10])

           

plt.xticks(rotation = 45)

plt.xlabel("City", fontsize = 20)

plt.ylabel("Count", fontsize=20)

plt.show()
vertical = df_startups.IndustryVertical.value_counts()

print("Name of the Industry Veritcal")

print(vertical[:20])