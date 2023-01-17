import numpy as np

import pandas as pd
s1 = pd.Series(range(0,4)) # -> 0, 1, 2, 3

s2 = pd.Series(range(1,5)) # -> 1, 2, 3, 4

s3 = s1 + s2 # -> 1, 3, 5, 7

s4 = pd.Series(['a', 'b']) * 3 # -> 'aaa','bbb'
# 获取Series索引

idx = s1.index

print(idx, type(idx), sep='\t')
# 索引的一些方法

a = idx.values # 得到一个numpy array

print(a, type(a), sep='\t')

l = idx.tolist() # 得到一个python list

print(l, type(l), sep='\t')
# 创建索引并使用

idx = pd.Index([1, 2, 3])

print(s1[idx])

print(idx.min(), idx.max(), sep='\t') # 获得最大和最小索引
df = pd.DataFrame(np.random.randint(0, 100, [10, 4]), columns=['A', 'B', 'C', 'D'])

print(type(df))

df
df.info() # 查看df的基本信息
df.head(3) # 得到前3行，参数省略则为5行
df.tail(3) # 得到后3行，参数省略则为5行
df.describe() # 得到统计信息
df.T # 对df转置
print(df.shape) # 获得df的大小形状

print(df.size) # 获得df的元素个数
print(type(df), type(df.values)) # 将df转成numpy array
df.columns # 获得列索引
# 索引改名 df.rename(columns={'old':'new'}, inplace=True)

df1 = df.rename(columns={'A':1,'B':'x'})

df1
df['C'] # 提取列为Series
df[['C']] # 提取列为DataFrame
df[['A', 'C', 'B']] # 提取多列，可以改变顺序
df[df.columns[[1, 2]]] # 利用列索引，通过列数提取。 

# 注意！对列的双括号索引得到的结果仍然为Index，和DataFram的双括号类似。如果用单括号，则会报错。
df.A # 用pyton特性提取列
df['new_col'] = range(len(df)) # 直接索引到新的列名即可

df.head()
df.index # 获得行索引
df1 = df.rename(index={1:'B', 2:20.5}) # 索引改名与列类似

df1
df.drop([3,5]) # 删除行
# 具体的：

df.loc[1, 'B'] # 可以

# df.loc[1, 2] # 报错

# df.iloc[1, 'B'] # 报错

df.iloc[1, 2] # 可以

df.ix[1, 'B'] # 可以

df.ix[1, 2] # 可以
df.ix[1:5, 'B':'D'] # 切片操作
df[1:2] # 简单提取行。注意！使用切片可以直接行提取， 如果不用切片，则会报错，因为此时会被误认为列提取
df['A'] # 简单提取列。