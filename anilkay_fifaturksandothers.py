# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data=pd.read_csv("/kaggle/input/fifa19/data.csv")

data.head()
data.shape
turks=data[data["Nationality"]=="Turkey"]

len(turks)
turks[0:100]
maxoverall=turks["Overall"].max()

turks[turks["Overall"]==maxoverall]
maxpotential=turks["Potential"].max()

turks[turks["Potential"]==maxpotential]
minoverall=turks["Overall"].min()

turks[turks["Overall"]==minoverall]
minpot=turks["Potential"].min()

turks[turks["Potential"]==minpot]
turks.sort_values(by='Age', ascending=False)[0:20]
turks.sort_values(by='Marking', ascending=False)[0:10]
data.sort_values(by="Overall",ascending=False)[0:10]
data.sort_values(by="Overall",ascending=True)[0:10]
data.sort_values(by="Potential",ascending=True)[0:10]
data.sort_values(by="StandingTackle",ascending=False)[0:10]
data.sort_values(by="GKReflexes",ascending=False)[0:10]
data.sort_values(by="GKPositioning",ascending=False)[0:10]
%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

plt.figure(figsize=(30,20))

betters=data[data["Overall"]>85]

sns.countplot(data=betters,x="Nationality")
data.columns
plt.figure(figsize=(30,20))

betters=data[data["Overall"]>85]

sns.countplot(data=data,x="Position")
plt.figure(figsize=(30,20))

sns.countplot(data=betters,x="Position")
plt.figure(figsize=(25,25))

sns.scatterplot(data=data,x="LongShots",y="Marking",hue="Position")
sns.scatterplot(data=data,x="Jumping",y="Aggression",hue="Nationality")