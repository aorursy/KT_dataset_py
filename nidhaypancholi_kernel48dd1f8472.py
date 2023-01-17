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
df=pd.read_csv('/kaggle/input/la-liga-dataset/LaLiga_dataset.csv')

df.head()
df.set_index('season',inplace=True)
twenty_sixteen=df[888:]
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

twenty_sixteen
plt.figure(figsize=(16,4))

sns.barplot(y='points',x='club',data=twenty_sixteen)
plt.figure(figsize=(16,4))

sns.barplot(y='goals_scored',x='club',data=twenty_sixteen)
plt.figure(figsize=(10,6))

sns.barplot(x=twenty_sixteen['points']/twenty_sixteen['goals_scored'],y=twenty_sixteen['club'])
plt.figure(figsize=(20,6))

sns.barplot(y=twenty_sixteen['goals_conceded'],x=twenty_sixteen['club'])
plt.figure(figsize=(10,6))

sns.barplot(x=twenty_sixteen['points']/twenty_sixteen['goals_scored'],y=twenty_sixteen['club'])
plt.figure(figsize=(10,6))

sns.barplot(x=twenty_sixteen['goals_conceded']/twenty_sixteen['points'],y=twenty_sixteen['club'])
plt.figure(figsize=(10,6))

sns.barplot(x=twenty_sixteen['goals_scored']/twenty_sixteen['goals_conceded'],y=twenty_sixteen['club'])
plt.figure(figsize=(10,5))

sns.scatterplot(x='home_win',y='away_win',data=twenty_sixteen[-6:],label='TOP 6')

sns.scatterplot(x='home_win',y='away_win',data=twenty_sixteen[4:13],label='Rest of the teams')

sns.scatterplot(x='home_win',y='away_win',data=twenty_sixteen[:3],label='Relegated Teams')

plt.legend()
plt.figure(figsize=(6,5))

sns.scatterplot(y='goals_scored',x='goals_conceded',data=twenty_sixteen[-6:],label='TOP 6')

sns.scatterplot(y='goals_scored',x='goals_conceded',data=twenty_sixteen[4:13],label='Rest of the teams')

sns.scatterplot(y='goals_scored',x='goals_conceded',data=twenty_sixteen[:3],label='Relegated Teams')

plt.legend()
h=df.groupby('club')['points'].agg(['sum','max','min','mean'])

from collections import Counter
s=Counter(df['club'])
u=[]

for x in h.index:

    u.append(s[x])

h['appearances']=u


h.columns=['Total Points','Max Points Earned in a Season','Min Points Earned in a Season','Average points earnt in a season','Number of times played in a la liga season']
h