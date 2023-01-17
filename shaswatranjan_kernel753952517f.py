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
df1=pd.read_csv("/kaggle/input/netflixdata/input/Movie_Customer1.csv")
df1
df2=pd.read_csv("/kaggle/input/netflixdata/input/Movie_Customer2.csv")
df2
df3=pd.read_csv("/kaggle/input/netflixdata/input/Movie_Customer3.csv")
df3
df4=pd.read_csv("/kaggle/input/netflixdata/input/Movie_Customer4.csv")
df4
e=pd.concat([df1,df2,df3,df4],axis=0,join="outer")
e
j=e['CustomerID'].unique()
j
j=e['CustomerID'].nunique()
j
f=e['Rating'].unique()
f
f=e['Rating'].nunique()
f
p=e['Movie'].nunique()
p
s=e.groupby('Rating')['Movie'].count()
s
import matplotlib.pyplot as plt
h=s.plot(kind = 'barh', legend = False, figsize = (15,10))
for i in range(1,6):
    h.text(s.iloc[i-1],i-1,'Rating {}:{} '.format(i, s.iloc[i-1]))
t=pd.DataFrame
d=e.groupby('Movie')['Rating'].mean()
d
d
i=pd.DataFrame()
d=e.groupby('CustomerID')['Rating'].mean()
d

g=pd.read_csv("/kaggle/input/netflixdata/movie.csv")
g
g=g.drop(columns=['Column4','Column5'])

g=g.rename(columns={'Column1':'MovieID'})
g

f=e['Movie'].mean()
f
h=pd.DataFrame(columns=['Movie_rank'])
for i in range(0,17770):
   h= h.append({"Movie_rank":d.iloc[i]},ignore_index=True)
h
g['Movie_rating']=h
g
