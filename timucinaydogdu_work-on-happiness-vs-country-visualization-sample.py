# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

"""

Hi.. Data science sample in this kernel.



Work on Vs Code with Anaconda. And shared github.



And i try to use all imported library.



Try to make english but i know good level turkish language. 



Contact at from https://github.com/timucinaydogdu or tim.aydogdu@gmail.com



Thanks for sperate time...



"""

import numpy as np      

import pandas as pd 

import matplotlib.pyplot as plt 

import seaborn as sns   



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.





data = pd.read_csv("../input/HappinesRecords.csv")   #Maybe requare change in kaggle

print(data.info())
#HeatMap Sample. 



f,ax=plt.subplots(figsize=(12,10))

sns.heatmap(data.corr(),annot=True,linewidths=.5,fmt=".1f",ax=ax)

plt.show()
#Bar Plot Sample



plt.bar(data.Happiness_Score,data.Country)

plt.title("Happines at countryies")

plt.xlabel("Happines Score")

plt.ylabel("Countryies")

plt.show()
#Scatter Plot Sample



plt.scatter(data.Family,data.Economy_GDP_per_Capita,color="red")

plt.legend() 

plt.xlabel("Family")

plt.ylabel("Economy")

plt.title("Family vs Economy")

plt.show()
#Line Plot



data.Economy_GDP_per_Capita.plot(kind="line",color ="g",label="Economy_GDP_per_Capita",linewidth=1,alpha=0.5,grid=True,linestyle=":")

data.TrustGovernmentCorruption.plot(color="r",label="TrustGovernmentCorruption",linewidth=1,alpha=0.5,grid=True,linestyle="-.")

plt.legend()

plt.xlabel("Economy GDP per Capital")

plt.ylabel("Trust GovernmentC orruption")

plt.title("Economy vs Trust Government")

plt.show()