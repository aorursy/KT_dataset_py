import numpy as np
a = np.arange(15).reshape(3,5)

a
a.shape
a.ndim
a.dtype.name
a.itemsize
a.size
type(a)
b = np.array([6,7,8])

b
type(b)
a = np.array([2,3,4])

a
a.dtype
b = np.array([1.2, 3.5, 5.1])

b.dtype
b = np.array([(1.5,2,3), (4,5,6)])

b
c = np.array( [ [1,2], [3,4] ], dtype = complex )

c
np.zeros( (3,4) )
np.ones( (2,3,4), dtype=np.int16 )                            # dtype can also be spec
np.empty( (2,3) )                                            # uninitialized, output
np.arange( 10,30,5 )
np.arange( 0,2,0.3 )
from numpy import pi
np.linspace( 0,2,9 )
x = np.linspace( 0, 2*pi, 100)

f = np.sin(x)

f
a = np.arange(6)

print(a)
b = np.arange(12).reshape(4,3)

print(b)
c = np.arange(24).reshape(2,3,4)

print(c)
print(np.arange(10000))
print(np.arange(10000).reshape(100,100))
a = np.array( [20,30,40,50] )

b = np.arange( 4 )

b
c = a - b

c
b**2
10*np.sin(a)
a<35
A = np.array( [[1,1],

              [0,1]] )

B = np.array( [[2,0],

              [3,4]] )

A*B
A@B
A.dot(B)
a = np.ones((2,3), dtype = int)

b = np.random.random((2,3))

a*=3

a
b += a

b

#a += b#
a = np.ones(3,dtype = np.int32 )

b = np.linspace(0,pi,3)

b.dtype.name
c = a+b

c
c.dtype
d = np.exp(c*1j)

d
d.dtype
a = np.random.random((2,3))

print(a)

print(a.sum())

print(a.min())

a.max()
b = np.arange(12).reshape(3,4)

print(b)

print("Sum:",b.sum(axis=0))

print("Min:",b.min(axis=1))

b.cumsum(axis=1)
B = np.arange(3)

print(B)

print("Exp:",np.exp(B))

print("Sqrt:",np.sqrt(B))

C = np.array([2., -1., 4.])

np.add(B,C)
a = np.arange(10)**3

print(a)

print("Index:",a[2])

print("Index:",a[2:5])

a[:6:2] = -1000

print(a)

a[: :-1]

for i in a:

    print(i**(1/3.))
def f(x,y):                                                 #Multidimensional Array

    return 10*x+y

b = np.fromfunction(f,(5,4),dtype=int)

print(b)

print(b[2,3])

print(b[0:5, 1])

print(b[: ,1])

print(b[1:3, : ])

print(b[-1])
c = np.array( [[[ 0, 1, 2],

               [ 10, 12, 13]],

               [[100,101,102],

               [110,112,113]]])

print(c.shape)

print(c[1,...])

c[...,2]
for row in b:                            #Iterating over multidimensional arrays

    print(row)

    
for element in b.flat:

    print(element)
a = np.floor(10*np.random.random((3,4)))

print(a)

a.shape
print(a.ravel())                           #returns the array, flattened

print(a.reshape(6,2))

print(a.T)

print(a.T.shape)

a.shape
print(a)

a.resize((2,6))

a
a.reshape(3,-1)
a = np.floor(10*np.random.random((2,2)))

print(a)

b = np.floor(10*np.random.random((2,2)))

print(b)

print(np.vstack((a,b)))

np.hstack((a,b))
from numpy import newaxis

np.column_stack((a,b))
a = np.array([4.,2.])

b = np.array([3.,8.])

print(np.column_stack((a,b)))

print(np.hstack((a,b)))

print(a[:,newaxis])

print(np.column_stack((a[:,newaxis], b[:,newaxis])))

np.hstack((a[:,newaxis], b[:,newaxis]))
np.r_[1:4,0,4]
a = np.floor(10*np.random.random((2,12)))

print(a)

print(np.hsplit(a,3))

np.hsplit(a,(3,4))
a = np.arange(12)

b = a

print(b is a)

b.shape = 3,4

a.shape
def f(x):

    print(id(x))

print(id(a))

f(a)
c = a.view()

print(c is a) 

print(c.base is a)

print(c.flags.owndata)

c.shape = 2,6

print(a.shape)

c[0,4] = 1234

a
s = a[ : , 1:3]

s[:] = 10

a
d = a.copy()

print(d is a) 

print(d.base is a)

d[0,0] = 9999

a
a = np.arange(int(1e8))

b = a[:100].copy()

del a
import numpy as np

import pandas as pd
s = pd.Series([1,3,4,np.nan,6,8])          #Creating Series

s
dates = pd.date_range('20130101', periods=6)

print(dates)

df = pd.DataFrame(np.random.randn(6,4), index=dates, columns=list('ABCD'))

df
df2 = pd.DataFrame({'A': 1.,

                    'B': pd.Timestamp('20130102'),

                    'C': pd.Series(1, index=list(range(4)), dtype = 'float32'),

                    'D': np.array([3] * 4, dtype='int32'),

                    'E': pd.Categorical(["test", "train", "test", "train"]),

                    'F': 'foo'})

df2
df2.dtypes
df2.abs

df2.applymap

df2.D

df2.bool

df2.boxplot
print(df.head())

df.tail(3)
df.index

df.columns
df.to_numpy()
df2.to_numpy()
df.describe()
df.T
df.sort_index(axis=1, ascending=False)
df.sort_values(by='B')
df['A']
print(df[0:3])

df['20130102': '20130104']
df.loc[dates[0]]
df.loc[:, ['A', 'B']]
df.loc['20130102':'20130104', ['A', 'B']]
df.loc['20130102',['A','B']]
df.loc[dates[0], 'A']
df.at[dates[0],'A']
df.iloc[3]
df.iloc[3:5, 0:2]
df.iloc[[1, 2, 4], [0, 2]]
df.iloc[1:3, :]
df.iloc[:, 1:3]
df.iloc[1, 1]
df.iat[1, 1]
df[df['A']>0]
df[df>0]
df2 = df.copy()

df2['E'] = ['one', 'one', 'two', 'three', 'four', 'three']

print(df2)

df2[df2['E'].isin(['two', 'four'])]
s1 = pd.Series([1,2,3,4,5,6], index=pd.date_range('20130102', periods=6))

print(s1)

df['F'] = s1
df.at[dates[0], 'A'] = 0
df.iat[0, 1] = 0
df.loc[:, 'D'] = np.array([5] * len(df))
df
df2 = df.copy()

df2[df2 > 0] = -df2

df2
df1 = df.reindex(index=dates[0:4], columns=list(df.columns) + ['E'])

df1.loc[dates[0]:dates[1], 'E'] = 1

df1
df1.dropna(how='any')
df1.fillna(value=5)
pd.isna(df1)
print(df.mean())

df.mean(1)
s = pd.Series( [1, 3, 5, np.nan, 6, 8], index=dates).shift(2)

print(s)

df.sub(s, axis='index')
print(df.apply(np.cumsum))

df.apply(lambda x: x.max() - x.min())
s = pd.Series(np.random.randint(0, 7, size=10))

print(s)

s.value_counts()
s = pd.Series(['A', 'B', 'C', 'Aaba', 'Baca', np.nan, 'CABA', 'dog', 'cat'])

s.str.lower()
df=pd.DataFrame(np.random.randn(10, 4))

print(df)

#break intp pieces

pieces = [df[:3], df[3:7], df[7:]]

pd.concat(pieces)
left = pd.DataFrame({'key': ['foo', 'foo'], 'lval': [1,2]})

right = pd.DataFrame({'key': ['foo', 'foo'], 'rval': [4, 5]})

print(left)

print(right)

pd.merge(left, right, on='key')
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style="darkgrid")
df = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv")

df.head()

sns.relplot(x='Confirmed', y='Recovered', data=df)
sns.relplot(x='Confirmed', y='Recovered', hue='Deaths', data=df)
sns.relplot(x='Confirmed', y='Recovered', size='Deaths', data=df)
sns.relplot(x='Confirmed', y='Recovered', palette='ch:r=-.5,l=.75', data=df)
sns.relplot(x='Confirmed', y='Recovered', size='Deaths', sizes=(15, 200), data=df)
df = pd.DataFrame(dict(time=np.arange(500),

                       value=np.random.randn(500).cumsum()))

g = sns.relplot(x='time', y='value', kind='line', data=df)

g.fig.autofmt_xdate()
df = pd.DataFrame(np.random.randn(500, 2).cumsum(axis=0), columns=["x", "y"])

sns.relplot(x='x', y='y', sort=False, kind='line', data=df)
df = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv")

df.head()
sns.relplot(x="Confirmed", y="Recovered", kind="line", data=df)
sns.relplot(x="Confirmed", y="Recovered", ci=None, kind="line", data=df)
sns.relplot(x="Confirmed", y="Recovered", kind="line", ci="sd", data=df)
sns.relplot(x="Confirmed", y="Recovered", estimator=None, kind="line", data=df)
sns.relplot(x="Confirmed", y="Recovered", hue="Deaths", kind="line", data=df)
from matplotlib.colors import LogNorm

palette = sns.cubehelix_palette(light=.7, n_colors=6)

sns.relplot(x="Confirmed", y="Recovered", hue="Deaths",

           hue_norm=LogNorm(),

           kind="line", data=df)
df = pd.DataFrame(dict(time=pd.date_range("2017-1-1", periods=500),

                       value=np.random.randn(500).cumsum()))

g = sns.relplot(x="time", y="value", kind="line", data=df)

g.fig.autofmt_xdate()
df = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv")

df.head()
sns.relplot(x="Lat", y="Long", hue="3/14/20", col="1/27/20", data=df)
import seaborn as sns

import matplotlib.pyplot as plt

sns.set(style="ticks", color_codes=True)
sns.catplot(x="Lat", y="Long", data=df)
df = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv")

df.head()
sns.catplot(x="Confirmed", y="Recovered",jitter=False, data=df)
df = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv")

df.head()
df = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv")

df.head()
sns.catplot(data=df, orient="h", kind="box")
df = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv")

df.head()
import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from scipy import stats
sns.set(color_codes=True)
x = np.random.normal(size=100)

sns.distplot(x)
sns.distplot(x, kde=False, rug=True)
sns.distplot(x, bins=20, kde=False, rug=True)
sns.distplot(x, hist=False, rug=True)
x = np.random.normal(0, 1, size=30)

bandwidth = 1.06*x.std()*x.size**(-1/5.)

support = np.linspace(-4, 4, 200)

kernels = []

for x_i in x:

    kernel = stats.norm(x_i, bandwidth).pdf(support)

    kernels.append(kernel)

    plt.plot(support, kernel, color="r")

    

sns.rugplot(x, color=".2", linewidth=3)
from scipy.integrate import trapz

density = np.sum(kernels, axis=0)

density /= trapz(density, support)

plt.plot(support, density)