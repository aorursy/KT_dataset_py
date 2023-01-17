# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
index = pd.date_range('1/1/2000', periods=8)

index


s = pd.Series(np.random.randn(5), index=['a', 'b', 'c', 'd', 'e'])

s
df = pd.DataFrame(np.random.randn(8, 3), index=index, columns=['A', 'B', 'C'])

df
wp = pd.Panel(np.random.randn(2, 5, 4), items=['Item1', 'Item2'],

           major_axis=pd.date_range('1/1/2000', periods=5),

           minor_axis=['A', 'B', 'C', 'D'])

wp
long_series = pd.Series(np.random.randn(1000))

long_series.head()
long_series.tail(3)
df[:2]

df.columns = [x.lower() for x in df.columns]

df
s.values
df.values
wp.values
df = pd.DataFrame({'one' : pd.Series(np.random.randn(3), index=['a', 'b', 'c']),

                   'two' : pd.Series(np.random.randn(4), index=['a', 'b', 'c', 'd']),

                   'three' : pd.Series(np.random.randn(3), index=['b', 'c', 'd'])})

df
row = df.ix[1]

column = df['two']

df.sub(row, axis='columns')

df
df
for row_index, row in df.iterrows():

    print('%s\n%s' % (row_index, row))
df = pd.DataFrame({'id': [1,2,3,4], 'HomeTeam': ['A', 'A', 'A', 'B'], 'A' : [4,5,6,7], 'B' : [10,20,30,40],'C' : [100,50,30,60]}); 

df

df['mean'] = df.groupby('HomeTeam')['B'].rolling(2).mean()

df