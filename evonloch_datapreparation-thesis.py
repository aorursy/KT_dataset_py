# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df1 = pd.read_csv('../input/week2_v1.csv')

df2 = pd.read_csv('../input/week2_v2.csv')

df3 = pd.read_csv('../input/week2_v3.csv')



df = pd.merge(df1, df2, how = 'outer', on = 'data')

df = pd.merge(df, df3, how = 'outer', on = 'data')

#df[["title_x", "title_y", "title"]].fillna(0, inplace = True)

df.fillna(0, inplace = True)

df = df.drop(columns = ['name_x', 'name_y', 'name'])

df = df.replace(to_replace = ['নিশ্চিত নেতিবাচক','কিছুটা নেতিবাচক','নিরপেক্ষ','Skip ( বুঝতে পারছি না)', 'কিছুটা ইতিবাচক', 'নিশ্চিত ইতিবাচক'], value = [-2,-1,0,0,1,2])

df['value'] = df.title_x+df.title_y+df.title #adding a new column 

negative = df[df['value'] < 0]

positive = df[df['value'] > 0]

neutral  = df[df['value'] == 0]

df['title'].value_counts()
df['title_x'].value_counts()