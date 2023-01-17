# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output. 
!pwd

!ls
import pandas as pd

import numpy as np

import matplotlib as plt
df=pd.read_csv('/kaggle/input/peace-agreements-dataset/pax_20_02_2018_1_CSV.csv')

print(df.shape)

print('')

print(df.dtypes.value_counts())

df.sample(30)
df.ImSrc.value_counts()
df.columns.tolist() # using tolist to display all columns
df.iloc[:,0:20].sample()

# df[['Stage', 'StageSub', 'Part', 'ThrdPart', 'OthAgr', 'GCh', 'GChRhet', 'GChAntid', 'GChSubs', 'GChOth', 'GDis']]
df.info()
df.Con.unique()
df.Con.value_counts() #158 different countries
df.Contp.value_counts()
df.Reg.value_counts()
df.Status.value_counts()
df.Stage.value_counts()
df.StageSub.value_counts()
df[df.Con == 'Bosnia and Herzegovina/Yugoslavia (former)']
df.Con.isin(['China','Palestine','Iran','Sri Lanka','USA','America'])
df.isna().sum().tolist()
df.iloc[:,10:14]
df.index[12]
print(df.Dat.min())

print(df.Dat.max())
df.Dat.dtypes
df.Dat=pd.to_datetime(df.Dat)
df.Dat.dtypes
df.info()
df.dtypes.unique()
print(df.Dat.min())

print(df.Dat.max())
df['month']=df['Dat'].dt.year
df.month
df = df.rename(columns={'month' : 'year'})
# df.groupby(['year'])

# df.groupby(['Animal']).mean()

# df.year.sort_values().unique() # useless, we dont need the series YEAR sorted, we know 1990 is the min value and 2015 is max.

df.year.sort_values().value_counts() #sorting by value_counts

# df.year(sort=True).value_counts()
df.year.groupby(['year']).count()
df.groupby('year').year.value_counts() # only YEAR

# df.groupby('year').sum() # ALL columns
df.groupby(df["year"]).count().plot(kind="bar") # this is using the newly created YEAR column directly

# df.groupby(df["Dat"].dt.year).count().plot(kind="bar") # this uses the datetime column 'Dat' and extracts YEAR from it.
(df.groupby('year').year.value_counts()).plot(kind='bar')

plt.show()
df.plot.bar(x=df.year, rot=0)