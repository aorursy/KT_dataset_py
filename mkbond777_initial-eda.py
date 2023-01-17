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
pd.set_option('display.max_columns', 500)
df_2018 = pd.read_csv("/kaggle/input/qs-world-university-rankings/2018-QS-World-University-Rankings.csv",engine='python',header=[0,1])
df_2019 = pd.read_csv("/kaggle/input/qs-world-university-rankings/2019-QS-World-University-Rankings.csv",engine='python')
df_2020 = pd.read_csv("/kaggle/input/qs-world-university-rankings/2020-QS-World-University-Rankings.csv",engine='python',header=[0,1])
df_2020.head()
cols = []
for column in df_2018:
    if column[0].startswith('Unnamed'):
        cols.append(column[1])
    elif column[1].startswith('Unnamed'):
        cols.append(column[0])
    else:
        cols.append(column[0]+'_'+column[1])
df_2018.columns = cols
cols = []
for column in df_2020:
    if column[0].startswith('Unnamed'):
        cols.append(column[1])
    elif column[1].startswith('Unnamed'):
        cols.append(column[0])
    else:
        cols.append(column[0]+'_'+column[1])
df_2020.columns = cols
df_2020.info()
df_2020[df_2020['Rank in 2020'].isnull()].head()
df_2020.replace('-', np.nan,inplace=True)
df_2020_only_ranking = df_2020[~df_2020['Rank in 2020'].isnull()]
df_2020_only_ranking.head()
df_2020_only_ranking.info()
df_2020_only_ranking.loc['Rank in 2020'] = df_2020_only_ranking['Rank in 2020'].str.strip().str.replace('=','')
# Last row contains some footer
df_2020_only_ranking.drop(df_2020_only_ranking.index[len(df_2020_only_ranking)-1], inplace=True)
df_2020_only_ranking['Country'].unique()
df_2020_only_ranking['Classification_SIZE'].unique()
df_2020_only_ranking['FOCUS'].isnull().sum()
df_2020_only_ranking[df_2020_only_ranking['RESEARCH INTENSITY'].isnull()]
df_2020_only_ranking.groupby('Country')['Institution Name'].count().nlargest(20)