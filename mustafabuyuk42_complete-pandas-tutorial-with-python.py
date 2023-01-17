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
df = pd.read_csv("../input/timesData.csv")
df.columns
df[["university_name","country"]][0:30]
df.iloc[99:114]
df.iloc[1,3]
for index,row in df.iterrows():

    print(index,row["university_name"])
df[df.country == 'Turkey']
df.loc[df["country"] == 'Turkey']

df.sort_values(['university_name','teaching'],ascending = True)
df.head()
df["total"] = df["research"]+df["citations"]+df["teaching"]
df.head()
df.drop(columns = ['total'])
df["total"]=df.iloc[:,3:6].sum(axis = 1)
df.head()
df[(df.country == 'Finland') & (df.teaching > 20)]
new_df = df.loc[(df["country"] == 'Turkey') & (df["teaching"]> 20) ] # filtering data 

new_df = new_df.reset_index(drop = True, inplace =False) # it will drop old index and create new index
new_df
df.loc[df["university_name"].str.contains('King')] # filtering data that contains 'of' string in the countries
df.loc[~df["country"].str.contains('of')] # filtering data that contains 'of' string in the countries
df.loc[df["country"].str.contains('Switzerland|Turkey',regex = True)]
df.loc[df["teaching"]> 50,['university_name','world_rank']] = ['test value','00000']
df
df = pd.read_csv("../input/timesData.csv")

df.head()
df.groupby(["country"]).mean().sort_values(['research'],ascending = [0])
df["count"] = 1

df.groupby(['country']).count()['count'].sort_values(ascending = False)
new_df =pd.DataFrame(columns = df.columns)

for df in pd.read_csv("../input/timesData.csv",chunksize =5):

    results = df.groupby(['country']).count()

    new_df = pd.concat([new_df,results])
new_df