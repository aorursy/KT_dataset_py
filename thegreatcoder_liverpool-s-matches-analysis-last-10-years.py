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
df = pd.read_csv("/kaggle/input/english-premier-league-results-from-2010-to-2020/EPL Results from 2010 to 2020.csv")
df.head()
#I like Liverpool hence using them...

lpdf = df[df['HomeTeam']== 'Liverpool'] 
lpdf['FTR'].value_counts().plot(kind = 'bar')

# H - WON, A -LOST, D - DRAW
lpdf['HS'].value_counts()
lpdf['HS'].value_counts().plot(kind = 'bar')
lpdf['HS'].mean()
lpdf['HF'].value_counts()
lpdf['HF'].value_counts().plot(kind = 'bar')
lpdf['HF'].mean()
lddf = lpdf[lpdf['FTR']=='A'].reset_index()

lddf['AwayTeam'].value_counts()
df['FTR'].value_counts()
df['Referee'].value_counts()[:8].plot(kind = 'bar')
df['Date'].value_counts()
hwdf = df[df['FTR']=='H']
hwdf['HomeTeam'].value_counts()
hwdf['HomeTeam'].value_counts()[:8].plot(kind = 'bar')
awdf = df[df['FTR']=='A']
awdf['AwayTeam'].value_counts()
awdf['AwayTeam'].value_counts()[:8].plot(kind = 'bar')
lpdf['FTHG'].value_counts()
a = lpdf['FTHG'] - lpdf['FTAG']
a.value_counts()
a.value_counts().plot(kind = 'bar')