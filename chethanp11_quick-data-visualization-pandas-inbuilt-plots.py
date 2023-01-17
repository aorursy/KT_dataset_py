import numpy as np
import pandas as pd
import seaborn as sns
import os
%matplotlib inline

os.getcwd()
os.chdir("../input/")
df1 = pd.read_csv("df1", index_col = 0)
df1.head()

df2 = pd.read_csv("df2")
df2.head()
df1['A'].hist()
df1['A'].hist(bins=5)
df1['A'].plot(kind='hist')
df1['A'].plot.hist()
df2.plot.area()
df2.plot.bar(stacked=True)
df1.plot.line(x=df1.index, y='B')
df1.plot.scatter(x='A', y='B', c='C',cmap='coolwarm')
df2.plot.box()
df = pd.DataFrame(np.random.randn(1000,2), columns=['a','b'])
df.plot.hexbin('a', 'b', gridsize=25)
df['a'].plot.kde()
df['a'].plot.density()

df2.plot.kde()
