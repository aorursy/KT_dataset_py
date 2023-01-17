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
        
import seaborn as sns

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_excel('/kaggle/input/notre-dame-football-2019-game-log/Final Clean Game Log.xlsx',index_col=None)
# add home, away, and neutral site information to each game
df['HomeFieldAdvantage']=['A','H','A','H','H','H','A','H','A','H','H','A','N']

# add Off and Def Diff
df['PassRatio']=df['PassYds']/df['OppPassYds']
df['RushRatio']=df['RushYds']/df['OppRushYds']
df['OffRatio']=df['TotalYds']/df['OppTotalYds']

#correlations between offense and winning
sns.relplot(x="PassYds", y="PointDiff", hue='Result',data=df);

sns.relplot(x="RushYds", y="PointDiff", hue='Result',data=df);
sns.catplot(x="Week", y="PassRatio", hue="Result",
            kind="bar", data=df);

sns.catplot(x="Week", y="RushRatio", hue="Result",
            kind="bar", data=df);
sns.catplot(x="HomeFieldAdvantage", y="PassYds", hue="Result",
            kind="bar", data=df);
sns.boxplot(x="HomeFieldAdvantage", y="PassYds", hue='Result',data=df)
