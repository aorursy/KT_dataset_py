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
df = pd.read_csv('../input/athlete_events.csv')
df.head()
df.shape
df.columns
df['Age'][df['Year']==1996][df['Sex']=='M'].describe()
df['Age'][df['Year']==1996][df['Sex']=='F'].describe()
df[df['Sport']=='Gymnastics'][df['Sex']=='M'][df['Year']==2000].count()/df[df['Sex']=='M'][df['Year']==2000].count()
df['Sport'].unique()
df['Height'][df['Sex']=='F'][df['Year']==2000][df['Sport']=='Basketball'].describe()
df[:][df['Year']==2002][df['Weight']==123]

df['Year'][df['Name']=='Pawe Abratkiewicz'].unique()
df[df['Medal']=='Silver'][df['Team']=='Australia'][df['Year']==2000][df['Sport']=='Tennis']
df[df['Team']=='Switzerland'][df['Year']==2016].dropna().count()
df[df['Team']=='Serbia'][df['Year']==2016].dropna().count()
df[df['Year']==2014].groupby(by=df['Age']).count().sort_values(by='ID')
df[df['Season']=='Summer'][df['City']=='Lake Placid']
df[df['Season']=='Winter'][df['City']=='Sankt Moritz']
df['Sport'][df['Year']==1995]
len(df['Sport'][df['Year']==2016].unique())