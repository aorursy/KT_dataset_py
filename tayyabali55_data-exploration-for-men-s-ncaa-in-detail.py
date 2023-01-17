

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# for visualizations

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
MTeams = pd.read_csv("/kaggle/input/march-madness-analytics-2020/2020DataFiles/2020-Mens-Data/MDataFiles_Stage1/MTeams.csv")
MTeams.columns
MTeams.describe()
MTeams.head()
MSeasons = pd.read_csv("/kaggle/input/march-madness-analytics-2020/2020DataFiles/2020-Mens-Data/MDataFiles_Stage1/MSeasons.csv")
MSeasons.columns
MSeasons.head()
MTourneySeed = pd.read_csv("/kaggle/input/march-madness-analytics-2020/2020DataFiles/2020-Mens-Data/MDataFiles_Stage1/MNCAATourneySeeds.csv")
MTourneySeed.columns
MTourneySeed.shape
MTourneySeed.head()
MTourneySeed['Seed'] = MTourneySeed['Seed'].astype(str)

MTourneySeed['Seed'].unique()
sns.set(font_scale=1.4)
plt.figure(figsize=(25,10))

plt.xticks( rotation=85)



sns.countplot(x="Seed",data=MTourneySeed,order=MTourneySeed['Seed'].value_counts().sort_values(ascending=False).index)

sns.set_style("whitegrid")





plt.title("Seed Counts") 

plt.xlabel("Seeds") 

plt.ylabel("Counts") 

plt.show()


plt.figure(figsize=(20,30))

plt.xticks( rotation=85)



sns.scatterplot(x=MTourneySeed['Season'],y=MTourneySeed['Seed'],hue=MTourneySeed['Season'],s=150)

sns.set_style("whitegrid")





plt.title("Teams by Seed") 

plt.ylabel("Teams") 

plt.xlabel("Seed") 

plt.show()
MTS_MTeams = pd.merge(MTourneySeed,

                 MTeams[['TeamID','TeamName']],

                 on='TeamID')

MTS_MTeams


plt.figure(figsize=(20,100))

plt.xticks( rotation=85)



sns.scatterplot(x=MTS_MTeams['Season'],y=MTS_MTeams['TeamName'],hue=MTS_MTeams['Season'],s=150)

sns.set_style("whitegrid")





plt.title("Teams by Each Season") 

plt.ylabel("Teams") 

plt.xlabel("Seasons") 

plt.show()
plt.figure(figsize=(20,80))

plt.xticks( rotation=0)



sns.countplot(y="TeamName",data=MTS_MTeams,order=MTS_MTeams['TeamName'].value_counts().sort_values(ascending=False).index)

sns.set_style("whitegrid")





plt.title("Seed Counts") 

plt.xlabel("Seeds") 

plt.ylabel("Counts") 

plt.show()