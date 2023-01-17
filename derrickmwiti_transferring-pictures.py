# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 

%matplotlib inline

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/akosombo.csv')

df.describe()
df.dtypes
pd.isnull(df).any()
#Removing the hours part of the eta.

df['eta'] = df.eta.map(lambda x: x.split(':')[-1])

df['percentage'] = df['percentage'].apply(lambda x: x.split('%')[0])

df['percentage'] = df['percentage'].astype(float)

df['size'] = df['size'].map(lambda x: x.split('KB')[0])

df['size'] = df['size'].astype(float)

speed = df['speed']

for sp in speed:

    ext = sp[-4:]

    if ext == 'MB/s':

        sp = sp.split('MB/s')[0] 

        sp= float(sp)

        sp = sp*1024 

    elif ext =='KB/s':

        sp = sp.split('KB/s')[0]

        sp=float(sp)

    df['speed'] = sp
#I am sure the baove code can be written in a better way. I am open to suggestions

df.head()

df['eta'] = df['eta'].astype(float)

df.describe()
df[df['size']==9010]
df[df['eta']==4]
sns.jointplot(data=df, x='speed',y='size')
sns.jointplot(data=df, x='eta',y='size')
total = df['eta'].sum()

mins = total/60

mins


sns.distplot(df['eta'],bins=10)
#Size distribution 

sns.distplot(df['size'],bins=30)