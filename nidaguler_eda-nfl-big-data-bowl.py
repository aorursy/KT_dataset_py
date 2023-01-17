import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# import warnings

import warnings

# ignore warnings

warnings.filterwarnings("ignore")



import os
# read csv (comma separated value) into data

data = pd.read_csv('../input/nfl-big-data-bowl-2020/train.csv')

print(plt.style.available) # look at available plot styles

plt.style.use('ggplot')
# to see features and target variable

data.head()
nullvalues = data.loc[:, data.isnull().any()].isnull().sum().sort_values(ascending=False)



print(nullvalues)
data=data.dropna()
data.head()
data.drop(["GameId","PlayId"],axis=1,inplace=True)
#show columns

for i,col in enumerate(data.columns):

    print(i+1,". column is ",col)
data.nunique()
f,ax = plt.subplots(figsize=(12,10))

sns.heatmap(data.corr(),annot=True, linewidths=.1, fmt='.0%', ax=ax)



plt.show()
dic = {"S":data["S"],"Dis":data["Dis"],"JerseyNumber":data["JerseyNumber"],"PlayerWeight":data["PlayerWeight"],

      "HomeScoreBeforePlay":data["HomeScoreBeforePlay"],"Quarter":data["Quarter"],"Team":data["Team"],

      "Season":data["Season"],"PlayDirection":data["PlayDirection"], "VisitorScoreBeforePlay": data["VisitorScoreBeforePlay"]}

data_new = pd.DataFrame(dic)

data_new=pd.get_dummies(data_new)

data_new.head()
data_new.nunique()
data_new.shape
sns.barplot(x=data_new['Quarter'].value_counts().index,y=data_new['Quarter'].value_counts().values)

plt.title('Quarter other rate')

plt.ylabel('Rates')

plt.legend(loc=0)

plt.show()
sns.barplot(x=data_new['Season'].value_counts().index,y=data_new['Season'].value_counts().values)

plt.title('Season other rate')

plt.ylabel('Rates')

plt.legend(loc=0)

plt.show()
plt.figure(figsize=(10,5))

sns.barplot(x = "Team_away", y = "PlayDirection_right", hue = "Quarter", data = data_new)

plt.xticks(rotation=45)

plt.show()
plt.figure(figsize=(10,5))

sns.barplot(x = "Team_away", y = "PlayDirection_left", hue = "Quarter", data = data_new)

plt.xticks(rotation=45)

plt.show()
plt.figure(figsize=(10,5))

sns.barplot(x = "Team_home", y = "PlayDirection_left", hue = "Quarter", data = data_new)

plt.xticks(rotation=45)

plt.show()
plt.figure(figsize=(10,5))

sns.barplot(x = "Team_home", y = "PlayDirection_right", hue = "Quarter", data = data_new)

plt.xticks(rotation=45)

plt.show()
plt.figure(figsize=(15,7))

sns.lmplot(x='S',y='Dis',data=data_new)

plt.xlabel('S')

plt.ylabel('Dis')

plt.title('S vs Dis')

plt.show()
plt.figure(figsize=(15,7))

sns.lmplot(x='JerseyNumber',y='PlayerWeight',data=data_new)

plt.xlabel('JerseyNumber')

plt.ylabel('PlayerWeight')

plt.title('JerseyNumber vs PlayerWeight')

plt.show()
data_new["class"]=(data_new.Team_away>data_new.Team_home) & (data_new.Team_home<data_new.Team_away)

data_new.head()