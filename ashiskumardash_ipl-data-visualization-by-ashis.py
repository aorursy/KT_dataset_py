import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
data = pd.read_excel("../input/DIM_MATCH.xlsx")

data.head()

data.describe()
data.columns.tolist()
# Number of IPL data
ipl_seasons = data['Season_Year'].unique()
ipl_seasons = np.sort(ipl_seasons)
print(ipl_seasons)
print(data.groupby('Season_Year')['ManOfMach'].mode())
info = data.ManOfMach.value_counts()[0:10]

fig,ax = plt.subplots(figsize = (15,6))

ax.bar(info.keys(),info)

plt.title("Man of the Match")
plt.xlabel("Player Name")
plt.ylabel("No of times")

plt.show()




    
data.isnull().sum()

data=data.dropna(axis=0,how="any")

data.isnull().sum()
data.count()
data.head()

info = data.Season_Year.value_counts()

fig,ax = plt.subplots(figsize = (12,6))

ax.bar(info.keys(),info, color="c", alpha=.3, width=.5)

plt.xlabel("Year")
plt.ylabel("Matches Played")

plt.show()


info = data.Venue_Name.value_counts()

fig,ax = plt.subplots(figsize = (15,6))

ax.bar(info.keys(),info)

ax.set_xticklabels(info.keys())
plt.tight_layout()

fig.autofmt_xdate()

plt.show()
data.head()

info = data.match_winner.value_counts()    

fig,ax = plt.subplots()

ax = plt.pie(info,labels=info.keys(), autopct = '%1.1f%%', radius=2)


plt.show()