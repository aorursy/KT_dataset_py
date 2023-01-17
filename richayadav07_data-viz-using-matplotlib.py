import pandas as pd

import numpy as np

import matplotlib

import matplotlib.pyplot as plt

%matplotlib inline
data = pd.read_csv("../input/10_Property_stolen_and_recovered.csv")
data.dtypes
plt.plot([1,4,5,6], [1,8,9,16])

plt.axis([0, 7, 0, 18])

plt.show()
line = data[data['Area_Name']=='Delhi'].pivot_table(index='Year',values='Cases_Property_Stolen')
plt.plot(line)

plt.xlabel('Year')

plt.ylabel('Number of Property Stolen Cases in Delhi')

plt.show()
x = data[data.loc[:,'Sub_Group_Name']=='1. Dacoity']
bar = x.pivot_table(index='Area_Name',values='Cases_Property_Stolen',aggfunc=np.sum)
index=bar[5:10].index
plt.figure(figsize=(9,5))

plt.bar(index,bar.Cases_Property_Stolen[5:10],width=0.5)

plt.ylabel('Number of property stolen')

plt.xlabel('States in India')

plt.title('Dacoity from 2001- 2010')

plt.show()
plt.figure(figsize=(10,6))

plt.pie(bar.Cases_Property_Stolen[5:10],labels=index,autopct='%1.1f%%',

        shadow=True, startangle=90)

plt.axis('equal')

plt.title("DACOITY CASES RECORDED(2000 - 2010)")

plt.show()
scatter
plt.figure(figsize=(10,5))

colors = np.random.rand(35)

scatter = data.pivot_table(index='Area_Name',values=['Cases_Property_Stolen','Value_of_Property_Stolen','Value_of_Property_Recovered'],aggfunc=np.sum)

plt.scatter(scatter.Cases_Property_Stolen,scatter.Value_of_Property_Stolen,c=colors)

plt.xlabel('Number of Property Stolen cases')

plt.ylabel('Value of Property Stolen')

plt.show()
plt.figure(figsize=(16,9))

plt.hist(data.Cases_Property_Stolen,bins=35)

plt.xlabel("Bins bases on quantity of property stealth")

plt.ylabel("Count")

plt.axis([0,50000,0,3000])

plt.show()
plt.figure(figsize=(8,10))

plt.boxplot(data.Cases_Property_Recovered)

plt.ylabel("Count")

plt.show()