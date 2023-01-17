# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



import matplotlib.pyplot as plt

import seaborn as sns

import math

import matplotlib.ticker as ticker
ronaldo_data = pd.read_csv("/kaggle/input/ronaldo-20022020-league-career-data/Ronaldo_Statistics_Github")
ronaldo_data.head()
ronaldo_data.index = ronaldo_data.index + 1

ronaldo_data = ronaldo_data.drop(columns = "Unnamed: 0")
ronaldo_data

# Now it looks better
# Correlation between his age and the number of goals:

plt.plot(ronaldo_data["Age"], ronaldo_data["Goals"])

plt.xticks(ronaldo_data["Age"])

plt.grid()

plt.show()
plt.plot(ronaldo_data["Age"], ronaldo_data["Goals per 90min"])

plt.xticks(ronaldo_data["Age"])

plt.grid()

plt.show()

# Similar to the upper graph

# It should be scary for the rival to know that Ronaldo will score at least 1 goal in the match 
grouped_data = ronaldo_data.groupby(['Club']).sum()

grouped_data = grouped_data.drop(columns = "Age")
grouped_data
clubs = [club for club in grouped_data.index]



plt.bar(clubs,ronaldo_data.groupby(['Club']).sum()['Goals'])

plt.xticks(clubs)

plt.ylabel('Goals')

plt.xlabel('Clubs')

plt.show()
plt.bar(clubs,ronaldo_data.groupby(['Club']).sum()['Assists'])

plt.xticks(clubs)

plt.ylabel('Assists')

plt.xlabel('Clubs')

plt.show()
m, b = np.polyfit(ronaldo_data["Age"], ronaldo_data["Goals"], 1)
plt.xticks(ronaldo_data["Age"])

plt.plot(ronaldo_data["Age"], ronaldo_data["Goals"], 'o')

plt.plot(ronaldo_data["Age"], m*ronaldo_data["Age"] + b)
m, b = np.polyfit(ronaldo_data["Age"], ronaldo_data["Assists"], 1)

plt.xticks(ronaldo_data["Age"])

plt.plot(ronaldo_data["Age"], ronaldo_data["Assists"], 'o')

plt.plot(ronaldo_data["Age"], m*ronaldo_data["Age"] + b)
sns.jointplot(x="Goals", y="Assists", data=ronaldo_data,

                  kind="reg", truncate = False,

                  color="m", height=7)

# It can be seen that the season when he scored the highest number of his goals, his number of assists is the highest of his career