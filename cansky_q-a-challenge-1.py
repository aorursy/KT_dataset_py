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
df = pd.read_csv('/kaggle/input/nutrition-facts/menu.csv')
df.head()
new_df = df.copy()
new_df['Item'] = new_df['Item'].str.lower()
new_df.head()
def finder(substrings):
    res = pd.DataFrame()
    for subs in substrings:
        s = new_df[new_df['Item'].str.contains(subs)]
        res = pd.concat([res,s])
    return res
finder(['grilled chicken'])
egg_df = new_df[new_df['Item'].str.contains('egg')]
egg_with_white = egg_df[egg_df['Item'].str.contains('white')]
egg_without_white = egg_df[~egg_df['Item'].str.contains('white')]
egg_with_white.head()
egg_without_white.head()
