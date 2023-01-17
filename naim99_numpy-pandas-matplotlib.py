import numpy as np
x=[1,2,3.5,4.8]



a=np.array(x)

a
print (a*2)
x=np.arange(1,10)

x
x=np.arange(3.2)

y=np.arange(0.1,1.2,0.1)

print (x)

print (y)
# Numpy arrays



a=np.array([1,2,3,4])

b=np.array([[1,1],[2,2]])



print (a)

print (a.ndim)

print (b)

print (b.ndim)
np.shape(b)
np.shape(a)
print (b[0])

print (b[0][1])

print (a[0:3])
#generating random number

a=np.random.randn(6,4)

print (a)
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
#creating a dataframe

df=pd.DataFrame(np.random.rand(10,4),columns=['A','B','C','D'])

df
#creating a dataframe with dictionary

df2 = pd.DataFrame({'A' : [10,20],'B' : [20,100],'C' : [30,40],'D' : [40,50]})

df2
#looking at the variable types

df2.dtypes
#looking at the top few values, default 5

df.head()
#looking at the last few values

df.tail()
#to get the indexes

df.index
#to get the column names

df.columns
#to get the values

df.values
#to get quick statistics about the data

df.describe()
# for sorting the index

df.sort_index(axis=0, ascending=False)
#for sorting by a column

df.sort_values(by='A')
# selecting columns

df['A']
#selecting range of rows

df[0:2]
#selecting a particular row

df.loc[0]
#selecting a row and column

df.loc[0,['A']]
df.loc[:,['A']]
#using at instead for faster access , can't be used for a range

df.at[0,'A']
#selecting by posion 

df.iloc[0]
#selecting by condition 

df[df.A>0]
df2=df[df<.5]

df2
#drop all rows which have NaN

df2.dropna()
#replace NaN with other values

df2.fillna(value=1)
# taking mean

df.mean()

# mean in other axis

df.mean(1)
import matplotlib.pyplot as plt
k=[1.1,2.3,4.5,10.11]

plt.plot(k)

plt.ylabel("Y axis")

plt.xlabel("X axis")

plt.show()
plt.plot([1,2,3,4], [1,4,9,16], 'ro')

plt.axis([0, 5, 0, 16])

plt.show()
plt.plot(a,a**2,'ro',a,a**3,'b^')

plt.axis([0,5,0,5])

plt.show()
a