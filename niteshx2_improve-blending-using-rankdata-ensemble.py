# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
os.listdir('../input/efficientnets/')

PATH = 'input/efficientnets'
dfs = []

i = 0

for df_loc in os.listdir('../input/efficientnets/'):

    print('../input/efficientnets/{}'.format(df_loc))

    df = pd.read_csv('../input/efficientnets/{}'.format(df_loc))

#     df.head()

    dfs.append(df)

# dfs    
from scipy.stats import rankdata
for i in range(4) :

    dfs[i]['target'] = rankdata(dfs[i]['target'], method='min')

# dfs[0]
dfs[0]['target'] = (dfs[0]['target'] + dfs[1]['target'] + dfs[2]['target'] + dfs[3]['target'])/4
dfs[0]


dfs[0]['target'] = rankdata(dfs[0]['target'], method='min')

dfs[0].to_csv('sol.csv' , index = False)