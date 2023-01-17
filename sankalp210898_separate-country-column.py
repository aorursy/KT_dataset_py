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
df=pd.read_csv('/kaggle/input/icc-test-cricket-runs/ICC Test Batting Figures.csv',encoding= 'unicode_escape')
df
country=[]
player=[]
for i in range(len(df)):
    country.append(df['Player'][i].split('(')[1].replace(')','').replace('ICC/',''))
    player.append(df['Player'][i].split('(')[0])
country
df['Country']=country
df
df['Player']=player
df
df.columns
df1=df[['Player','Country', 'Span', 'Mat', 'Inn', 'NO', 'Runs', 'HS', 'Avg', '100', '50',
       '0', 'Player Profile']]
df1
