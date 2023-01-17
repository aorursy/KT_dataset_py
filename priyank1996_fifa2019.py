# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df= pd.read_csv("../input/data.csv")

df.head()
arr = []

for a in df['Age']:

    if a<20:

        arr.append('Below 20')

    elif a>20 and a<25:

        arr.append('20-25')

    elif a>25 and a<30:

        arr.append('25-30')

    elif a>30 and a<35:

        arr.append('30-35')

    else:

        arr.append('Above 35')

arr = pd.Series(arr)

df['Age_Grp'] = arr

df[['Name','Age','Age_Grp']].head()
import seaborn as sns

import matplotlib.pyplot as plt

plt.xticks(rotation = 0)

sns.countplot(x = 'Age_Grp', data = df)
v_con = {'€': 1, 'K': 1000, 'M':1000000}

def con(val):    

    unit = val[-1]

    if unit == '0':

        return 0

    number = float(val[1:-1])

    if unit in v_con:

        val = v_con[unit]*number

    return val

value = []

for v in df['Value']:

    value.append(con(v))

df['value'] = pd.Series(value)

df[['Name','Value','value']].head()
sns.scatterplot(df.Overall, df.value, palette='RdBu_r')
grp_age = df.groupby('Age_Grp', as_index=False)

df_age = grp_age['value'].mean()

sns.barplot(x = 'Age_Grp', y = 'value', data = df_age, palette='rainbow')
grp_age1 = df.groupby('Age', as_index=False)

df_a = grp_age1.value.mean()

sns.lineplot(df.Age, df.value)
sns.set(style = 'whitegrid')

grp_age1 = df.groupby('Age', as_index=False)

df_a = grp_age1.Overall.mean()

sns.lineplot(df.Age, df.Overall)
sns.set(style = 'whitegrid')

grp_age1 = df.groupby('Age', as_index=False)

df_a = grp_age1['Stamina','HeadingAccuracy', 'ShotPower','Acceleration', 'Crossing', 'Potential'].mean()

sns.lineplot(df_a.Age, df_a.Stamina)

sns.lineplot(df_a.Age, df_a.HeadingAccuracy)

sns.lineplot(df_a.Age, df_a.ShotPower)

sns.lineplot(df_a.Age, df_a.Acceleration)

sns.lineplot(df_a.Age, df_a.Crossing)

plt.legend(['Stamina','HeadingAccuracy', 'ShotPower','Acceleration', 'Crossing'])
sns.lineplot(df_a.Age, df_a.Potential)
x = df[['Overall', 'value', 'Composure', 'Penalties', 'Stamina', 'Balance']].corr(method = 'spearman')

sns.heatmap(x, linewidths=0.1, annot= True)
arr = []

df.Position.fillna('F', inplace =True)

def con_pos(pos):

    for p in pos:

        if p == 'M':

            return 'MID'

    for p in pos:

        if p == 'B':

            return 'DEF'

    for p in pos:

        if p == 'G':

            return 'GK'

    else:

        return 'FWD'

        

for pos in df['Position']:

    arr.append(con_pos(pos))

df['position'] = pd.Series(arr)

df[['Name','Position','position']].head()
sns.countplot(x = 'position', data = df, palette='YlGn_r')
sns.barplot(x = 'position', y = 'value', data = df, palette='YlGnBu_r')
grp_pos = df.groupby('position', as_index=False)

df_pos = grp_pos.mean()

df_pos.drop(['ID', 'Unnamed: 0','Jersey Number'], axis= 1)

m = df_pos.melt('position', value_vars = ['Crossing', 'Composure', 'Marking', 'SlidingTackle','Stamina'])

sns.barplot(m.position,m.value, hue = m.variable, palette='rainbow')

plt.legend(loc = 'lower left')
df_club = df.groupby('Club', as_index=False)['value'].mean()

df_club.sort_values(by= ['value'],ascending=False, inplace =True)

df_club.head()
plt.xticks(rotation  = 90)

sns.barplot(x ='Club', y= 'value', data = df_club.head(15), palette='rainbow')
sns.set(style="whitegrid")

f,ax = plt.subplots(figsize = (9,6))

plt.xticks(rotation = 90)

ax =sns.boxplot(x = 'Club', y = 'Overall', data = df.head(80),ax = ax )
labels = ['Stamina', 'Finishing', 'Crossing', 'HeadingAccuracy', 'Dribbling', 'Penalties','Acceleration']

stat_m = df.loc[0,labels].values

stat_r = df.loc[1,labels].values

angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)

stat_m = np.concatenate((stat_m, [stat_m[0]]))

stat_r = np.concatenate((stat_r, [stat_r[0]]))

angles = np.concatenate((angles,[angles[0]]))
fig = plt.figure()

ax = fig.add_subplot(111,polar = True)

ax.plot(angles, stat_m, 'o-', label= 'Messi')

ax.plot(angles, stat_r, 'o-', label = 'Ronaldo')

ax.set_thetagrids(angles*180/np.pi, labels)

plt.legend(['Messi','Ronaldo'], loc = 'lower left')