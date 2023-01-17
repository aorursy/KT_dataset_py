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
#Importing data

loldata=pd.read_csv('/kaggle/input/leagueoflegends/LeagueofLegends.csv')
#Selecting games played in LCK

loldata['League'].value_counts()

lckdata=loldata.copy().loc[loldata['League']=='LCK']

lckdata=lckdata.reset_index(drop=True)
#Importing visualization library

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

print("Setup Complete!")
#Plotting swarmplot between each season and the game time played

sns.swarmplot(x=lckdata.Year, y=lckdata.gamelength)
#Making separate dataframe containing the year played and game length

lckgamelength=lckdata.loc[:,['Year', 'gamelength']]

print(lckgamelength.groupby('Year').gamelength.count())

print(lckgamelength.groupby('Year').gamelength.mean())