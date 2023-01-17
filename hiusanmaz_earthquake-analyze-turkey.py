# This Python 3 environment comes with many helpful analytics libraries installed



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

earthquake_df= pd.read_csv(("../input/earthquake.csv"))
e_df=earthquake_df.copy()
# Columns

e_df.columns
# Info about dataset

e_df.info()
# 20 Observation from dataset

e_df.sample(20)
# Earthquake Frequency by countries..
sns.set_palette(sns.dark_palette('blue',15))

e_df['country'].value_counts().plot(kind='bar',figsize=(30,30),fontsize = 10)

plt.xlabel("Country",fontsize=20,color="Black")

plt.ylabel("Frequency",fontsize=20,color="Black")

plt.show()



# Earthquake Frequency for cities of Turkey.
sns.set_palette('pastel',15)

e_df['city'].value_counts().plot(kind = "bar", figsize=(30,30),fontsize = 20)

plt.xlabel("City",fontsize=20,color="Black")

plt.ylabel("Frequency",fontsize=20,color="Black")

plt.show()

# Plotting Richter's of Earthquakes.

sns.set_palette('hot',15)

e_df['city'].value_counts().plot(kind='bar',figsize=(30,30),fontsize = 10)

plt.xlabel("City",fontsize=20,color="Black")

plt.ylabel("Rihcter",fontsize=20,color="Black")

plt.show()





sns.set_palette('pastel',15)

e_df['city'].value_counts().plot(kind = "bar", figsize=(30,30),fontsize = 20)

plt.xlabel("City",fontsize=20,color="Black")

plt.ylabel("Frequency",fontsize=20,color="Black")

plt.show()

# Creating Correlation Heatmap
plt.figure(figsize=(20,20))

sns.heatmap(e_df.corr(), annot = True, fmt= ".1f", linewidths = .3,cmap='coolwarm_r')

plt.show()
e_df.city.isin(["istanbul","mugla","van","canakkale","izmir","kutahya","manisa","denizli"])
small_scale = e_df[e_df.city.isin(["mugla","van","canakkale","izmir","kutahya","manisa","denizli","istanbul"]) & (( e_df.richter <= 5.0) & (e_df.richter >=2.5)) & (e_df.depth > 30)]
small_scale.head()
sns.relplot(x="city", y="richter", kind="scatter",data=small_scale.sample(n=25),hue="date");

plt.xlabel("City",fontsize=20,color="Black")

plt.ylabel("Richter",fontsize=20,color="Black")

plt.title("Some of the Small-Scaled Earthquakes with dates")

plt.show()

moderate = e_df[e_df.city.isin(["sakarya","kocaeli","mugla","van","izmir","kutahya","denizli","istanbul"]) & ( e_df.richter >= 5.5) & (e_df.depth > 15)]
moderate
sns.relplot(x="city", y="richter", kind="scatter",data=moderate,hue="date");

plt.xlabel("City",fontsize=20,color="Black")

plt.ylabel("Richter",fontsize=20,color="Black")

plt.title("Some of the Moderate Earthquakes with dates")

plt.show()

large_scaled = e_df[e_df.city.isin(["kocaeli","sakarya","mugla","van","izmir","kutahya","denizli","istanbul"]) & ( e_df.richter >= 5.5) & (e_df.depth < 12)]
large_scaled
sns.relplot(x="city", y="richter", kind="scatter",data=large_scaled,hue="date");

plt.xlabel("City",fontsize=20,color="Black")

plt.ylabel("Richter",fontsize=20,color="Black")

plt.title("Some of the Large-Scaled Earthquakes with dates")

plt.show()
