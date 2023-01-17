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
df = pd.read_csv('/kaggle/input/covid19-mexico-clean-order-by-states/Covid_19_Mexico_Clean_Complete.csv')
def edaFromData(dfA, allEDA=False, desc='Exploratory Data Analysis'):

    print(desc)

    print(f'\nShape:\n{dfA.shape}')

    print(f'\nIs Null:\n{dfA.isnull().mean().sort_values(ascending=False)}')

    dup = dfA.duplicated()

    print(f'\nDuplicated: \n{dfA[dup].shape}\n')

    try:

        print(dfA[dfA.duplicated(keep=False)].sample(4))

    except:

        pass

    if allEDA:  # here you put yours prefered analysis that detail more your dataset

        

        print(f'\nDTypes - Numerics')

        print(dfA.describe(include=[np.number]))

        print(f'\nDTypes - Categoricals')

        print(dfA.describe(include=['object']))

        

        #print(df.loc[:, df.dtypes=='object'].columns)

        print(f'\nHead:\n{dfA.head()}')

        print(f'\nSamples:\n{dfA.sample(5)}')

        print(f'\nTail:\n{dfA.tail()}')

edaFromData(df)
df.sample(3)
totConf = df.Confirmed.sum()

totDeaths = df.Deaths.sum()

from IPython.display import HTML

line1 = f""

line2 = f"<div><font size=5>{totConf} Cases Confirmed</font></div>"

line3 = f"<div></div>"

line4 = f"<div><font size=5>{totDeaths} Deaths</font></div>"

line5 = f"<div></div>"

HTML(line1+line2+line3+line4+line5) 
df.State = df.State.str.lower()

df.Municipality = df.Municipality.str.lower()
states = df.State.unique().tolist()

municipality = df.Municipality.unique().tolist()
df['StateId'] = df['State'].apply(lambda x: states.index(x))

df['MunicipalityId'] = df['Municipality'].apply(lambda x: municipality.index(x))
df.sample()
grpState = df.groupby('StateId').sum()
deathState = grpState['Deaths'].sort_values(ascending=False)

confirmedState = grpState['Confirmed'].sort_values(ascending=False)

recoveredState = grpState['Recovered'].sort_values(ascending=False)

activeState = grpState['Active'].sort_values(ascending=False)
liState = []

[liState.append(states[x]) for x in confirmedState.index]
import seaborn as sns

import matplotlib.pyplot as plt

sns.set(style="whitegrid")

# Two subplots

fig, ax1 = plt.subplots(figsize=(18,7))

sns.barplot(x=liState, y=confirmedState.values, palette="rocket", ax=ax1).set_title('Confirmed')

plt.xticks(rotation=80)
liState = []

[liState.append(states[x]) for x in deathState.index]

fig, ax1 = plt.subplots(figsize=(15,5))

sns.barplot(x=liState, y=deathState.values, palette="rocket", ax=ax1).set_title('Deaths')

plt.xticks(rotation=80)
sumDays = df[['Deaths', 'Confirmed', 'Date']].groupby(['Date']).sum()

last30Days = sumDays.tail(30)



fig, ax1 = plt.subplots(figsize=(25,10))

sns.lineplot(x=last30Days.index, y=last30Days.Confirmed, ax=ax1, color='darkblue').set_title('Dash Mexico (last 30 days)')

sns.lineplot(x=last30Days.index, y=last30Days.Deaths, ax=ax1, color='red')

plt.xticks(rotation=90)

plt.grid(axis='x')

sns.despine(left=True, bottom=True)
g = sns.FacetGrid(df,  col="State", col_wrap=1, height=4)

g.map(sns.pointplot, "Date", "Confirmed", color=".3", ci=None)

plt.xticks(rotation=90)