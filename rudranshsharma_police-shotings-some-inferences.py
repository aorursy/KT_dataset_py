# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import seaborn as sns

data = pd.read_csv("/kaggle/input/data-police-shootings/fatal-police-shootings-data.csv", index_col = "id")

data
data.index = list(range(len(data.index)))

data.index
data.info()
data.isnull().apply(pd.value_counts)
missing = [(col_name, ((data[col_name].isnull().sum())/data[col_name].value_counts().sum())*100) for col_name in data.columns if data[col_name].isnull().any() == True]

missing
data.dropna(inplace = True)
data["month"] = pd.to_datetime(data["date"]).dt.month

data["year"] = pd.to_datetime(data["date"]).dt.year

data.drop(columns = "date", inplace = True)
data.city.unique()

city_crime = data.city.value_counts()

city_high_crime = [(city, city_crime[city]) for city in city_crime.index if city_crime[city] > 30]

city_high_crime

state_crime = data.groupby("state")["name"].count()

state_high_crime = [(state, state_crime[state]) for state in state_crime.sort_values(ascending = False).index if state_crime[state] > 50]

state_high_crime

state_only_high_crime = [state_high_crime[i][0] for i in range(len(state_high_crime))]
race = data["race"].value_counts()

fig, ax = plt.subplots(figsize = (10, 5))

sns.barplot( x = race.index, y = race )

plt.xlabel("Race")

plt.ylabel("Number of Shootings");



fig, a = plt.subplots(1, 2,figsize = (20, 5))

sns.countplot(data = data, x = "race", hue = "flee", ax = a[0])

a[0].set_xlabel("race vs fleeing")

a[1] = sns.countplot(data  = data, x = "race", hue = "threat_level", ax = a[1])

a[1].set_xlabel("race vs threat level")


figm, ax = plt.subplots(1, 2, figsize = (20, 5))

sns.countplot(data = data, x ="race", hue = "gender", ax = ax[0])

sns.countplot(data = data, x = "race", hue = "manner_of_death", ax = ax[1])

fig, ax =plt.subplots(1, 2, figsize = (20, 5))

threat = data[["threat_level", "signs_of_mental_illness"]]

sns.countplot(data = threat, x = "threat_level", hue = "signs_of_mental_illness", ax = ax[0])

sns.countplot(data = data[["manner_of_death", "signs_of_mental_illness"]], x = "manner_of_death", hue = "signs_of_mental_illness")
plt.subplots(figsize = (10, 5))

sns.distplot( data["age"])
fig, ax = plt.subplots(1, 2, figsize = (20, 5))

sns.countplot(data  = data, x = "month",  ax = ax[0])

sns.countplot(data = data, x = "year",  ax = ax[1])
fig, ax = plt.subplots(1, 2, figsize = (20, 5), sharey = True)

sns.countplot(data  = data, x = "month",hue = "race" , ax = ax[0])

sns.countplot(data = data, x = "year",hue = "race",  ax = ax[1])
plt.subplots(figsize = (20, 10))

sns.lineplot(data = state_crime)
data_high_state_crime = data[data.state.isin(state_only_high_crime)]

fig, ax = plt.subplots(1, 2, figsize = (20, 5), sharey = True)

sns.countplot(data = data_high_state_crime, x = "race", ax = ax[0])

sns.countplot(data = data_high_state_crime, x = "race", hue= "threat_level", ax = ax[1])