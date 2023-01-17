# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib as plt

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('../input/avengers/avengers.csv',encoding='latin-1')
df = pd.DataFrame(data)
df.info()
df.head(10)
df = df.drop(labels = ['URL','Death1', 'Death2', 'Death3', 'Death4', 'Death5', 'Return1', 'Return2', 'Return3', 'Return4', 'Return5','Notes'], axis = 1)





df.head()
df.info()
df = df.drop(labels = ['Probationary Introl', 'Full/Reserve Avengers Intro'], axis = 1)
df.info()
df.tail(10)
df = df.drop(['Current?'], axis = 1)
df.info()
df.head()
data_new = data.head()
fmelt = pd.melt(frame = data_new, id_vars = 'Name/Alias', value_vars = ['Gender'])

fmelt
data1 = df.head()

data2 = df.tail()



conc_data_row = pd.concat([data1, data2], axis = 0, ignore_index = True)

conc_data_row
q = df['Appearances'] > 3000  

df[q]

c = df['Appearances'].max()

c

#Peter Benjamin Parker --> best Appearances
w = df['Appearances'].min()

w
v = df['Appearances'] == 2

df[v]
wcharacters = df[np.logical_and(df['Gender'] == 'FEMALE', df['Appearances'].max())]

wcharacters
wcharacters.max()

#Best Appearances | W.characters | : Ororo Munro
df.describe()
df.dtypes
df['Honorary'] = df['Honorary'].astype('category')
df.dtypes
df["Honorary"].value_counts(dropna = False)
df["Gender"].value_counts(dropna = False)
df["Name/Alias"].dropna(inplace = True)
assert df["Name/Alias"].notnull().all()
df['Name/Alias'].fillna('empty', inplace = True) 
assert df['Name/Alias'].fillna('empty').all()
df.head(25)