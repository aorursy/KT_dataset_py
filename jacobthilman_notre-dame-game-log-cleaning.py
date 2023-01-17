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
        
# These are the plotting modules adn libraries we'll use:
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# Command so that plots appear in the iPython Notebook
%matplotlib inline

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_excel('/kaggle/input/notre-dame-football-2019-game-log/Notre Dame Football 2019 GameLog.xls')
# split game result from score
df1=pd.DataFrame(df.Result.str.split("(",1).tolist(), columns = ['GameResult','Score'])
df1[['NDScore','OppScore']]=df1['Score'].str.split('-',expand=True)
df1['OppScore']=df1['OppScore'].str.rstrip(')')
df1.drop('Score',axis=1,inplace=True)
df1

# remove original result and score columns
df.drop('Result',axis=1,inplace=True)

#reinsert columns
df.insert(2,'Result',df1['GameResult'])
df.insert(3,'NDScore',df1['NDScore'])
df.insert(4,'OppScore',df1['OppScore'])

#convert results to numeric
df['NDScore']=pd.to_numeric(df['NDScore'])
df['OppScore']=pd.to_numeric(df['OppScore'])

# calculate point differential
df['PointDiff']=df['NDScore']-df['OppScore']

df.to_excel(r'CleanGameLog.xlsx', index=False)
#from IPython.display import FileLink
#FileLink('CleanGameLog.xlsx')
