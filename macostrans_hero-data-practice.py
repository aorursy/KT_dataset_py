# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
hero = pd.read_csv("../input/heroes_information.csv",index_col=0)
hero.head()
#hero.shape
#hero['Skin color'].unique()
#hero.Alignment.unique()
#hero['Hair color'].unique()
hero.Race.unique()
hero['Gender'].replace(
    to_replace=['-'],
    value=np.nan,
    inplace=True
)
#hero.Gender.unique()
hero.Gender.isnull().sum()
hero['Skin color'].replace(
    to_replace=['-'],
    value=np.nan,
    inplace=True
)
#hero.Gender.unique()
hero['Skin color'].isnull().sum()
hero['Alignment'].replace(
    to_replace=['-'],
    value=np.nan,
    inplace=True
)
hero.Alignment.isnull().sum()
hero['Hair color'].replace(
    to_replace=['-'],
    value=np.nan,
    inplace=True
)
hero['Hair color'].isnull().sum()
#hero['Eye color'].unique()
hero['Eye color'].replace(
    to_replace=['-'],
    value=np.nan,
    inplace=True
)
hero['Eye color'].isnull().sum()
hero.Race.unique()
hero['Race'].replace(
    to_replace=['-'],
    value=np.nan,
    inplace=True
)
hero.Height.unique()
import missingno as msno
msno.matrix(hero)
#hero.groupby(['Gender'])['Height'].mean()
#hero.groupby(['Gender'])['Weight'].mean()
hero.groupby(['Gender'])['Alignment'].value_counts()
hero.groupby(['Publisher'])['Gender'].value_counts().sort_values(ascending= False).head()
hero[hero['Weight']<100][hero['Weight']>0][hero['Height']>0][hero['Height']<400].plot.hexbin(x='Weight', y = 'Height',  gridsize= 15)
