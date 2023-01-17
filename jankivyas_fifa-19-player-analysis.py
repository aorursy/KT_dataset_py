# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
df = pd.read_csv("../input/data.csv")

# Any results you write to the current directory are saved as output.
df.head()
#check for duplicates
df.duplicated().sum()
#check for nulls
df.isnull().values.any()
data_1=df[['ID','Name','Age','Nationality','Overall','Potential','Preferred Foot','International Reputation','Skill Moves','Jersey Number',]]
# Columns that contain `null` values
for i in list(data_1.columns):
    if data_1[i].isnull().values.any():
        print(i, end=",")
# replacing NaN values with unkowns and defaults
data_1["Jersey Number"].fillna("Unknown", inplace=True)
data_1["International Reputation"].fillna("0", inplace=True)
data_1["Preferred Foot"].fillna("Unknown", inplace=True)
data_1["Skill Moves"].fillna("0", inplace=True)
data_1.head()
data_1.isnull().values.any()
data_1.Age.value_counts()
#there are very few players with age >40, hence removing them
data_1 = data_1[data_1.Age<=40]
data_1.head()
#plotting number of players against age
AgeCount = pd.value_counts(data_1['Age'].values, sort=True)
AgeCount.plot.bar()
#Lets see player's international reputation by their age
plt.plot(df['Age'], df['International Reputation'], 'bo')
plt.ylabel('International Reputation')
plt.xlabel('Age')
plt.show()
#To view clearer distribution let us calculate mean of international reputation grouped by age
import matplotlib.pyplot as plt
age = df.sort_values("Age")['Age'].unique()
reputation = df.groupby(by="Age")["International Reputation"].mean().values
plt.title("Age vs International Reputation")
plt.xlabel("Age")
plt.ylabel("International Reputation")
plt.plot(age, reputation)
plt.show()

#skills vs international reputation
overall_skill_reputation = df.groupby(by="International Reputation")["Overall"].mean()
potential_skill_reputation = df.groupby(by="International Reputation")["Potential"].mean()
plt.plot(overall_skill_reputation, marker='o', c='r', label='Overall Skillpoint')
plt.plot(potential_skill_reputation, marker='x', c='b', label='Potential Skillpoint')
plt.title('Overall, Potential vs Reputation')
plt.xlabel('Reputation')
plt.ylabel('Skill point')
plt.legend(loc='lower right')
plt.show()