import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from scipy.stats import stats



%matplotlib inline

import warnings

warnings.filterwarnings("ignore")



import os

print(os.listdir("../input"))
pd.set_option('display.max_columns',None)
games = pd.read_csv('../input/videogamesales/vgsales.csv')
games.head()
games.shape
games.info()
games.isnull().sum()
games.nunique()
games[(games.Year.isnull()) & (games.Publisher.isnull())].head()
games[games.Publisher.isnull()]['Genre'].value_counts()
games[(games.Genre == 'Action') & (games.Publisher.isnull())]
games[games.Publisher == 'Unknown'].shape[0]
(games[(games.Publisher == 'Unknown') | (games.Publisher.isnull())]['Publisher'].count()/games.shape[0])*100
games.Publisher.fillna('Unknown',inplace=True)
games.Publisher.isnull().sum()
games.head()
games[games.Platform == 'Wii']['Year'].mode()[0]
games.Genre.nunique()
med=pd.DataFrame(games.groupby(by=["Platform","Genre"])["Year"].median().reset_index())

med
bnm=pd.DataFrame(games[games[["Platform","Genre","Year"]]["Year"].isna()][["Platform","Genre","Year"]]).reset_index()

bnm
for i in range(0,len(med)):

    for j in range(0,len(bnm)):

        if (med["Platform"][i]== bnm["Platform"][j]) & (bnm["Genre"][j]  ==  med["Genre"][i]) :

            bnm["Year"][j]=med["Year"][i]
for i in range(0,len(games)):

    for j in range(0,len(bnm)):

        if (games["Platform"][i]== bnm["Platform"][j]) & (bnm["Genre"][j]  ==  games["Genre"][i]) :

            games["Year"][i]=bnm["Year"][j]
games.isnull().sum()
games.head()
plt.figure(figsize=(15,10))

plt.xticks(rotation = 70, color='white', size=10)

sns.countplot(x='Year',data=games)

plt.show()
plt.figure(figsize=(15,10))

plt.xticks(rotation = 70, color='white', size=10)

sns.countplot(x='Genre',data=games)

plt.show()
games.groupby(['Year'])['Global_Sales'].sum()
games.groupby('Platform')['Global_Sales'].sum().reset_index().sort_values('Global_Sales',ascending=False).head(1)
games.groupby(['Platform','Year'])['Global_Sales'].sum().reset_index().groupby(['Platform'])['Global_Sales','Year'].max()