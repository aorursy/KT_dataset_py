import pandas as pd

import numpy as np



arr = np.array([10,20,30,40,50])



s = pd.Series(arr)



print(s)
s = pd.Series(np.random.randn(5), index=['a', 'b', 'c', 'd', 'e'])

print(s)
print(s.index)
import pandas as pd



data = {'a':10, 'b':20, 'c':30}

s = pd.Series(data)

print(s)
import pandas

dict = {'a': 0., 'b': 1., 'c': 2.}

print(dict)



s = pandas.Series(dict, index=['b', 'c', 'd', 'a'])

print(s)
s= pd.Series(5., index=['a', 'b', 'c', 'd', 'e'])

print(s)
import pandas as pd

import numpy as np



arr = ([10,20,30,40,50])

s = pd.Series(arr)



print(s)



# slicing or slice of index



print(s[1])

print(s[0:5])

print(s[::-1]) #Reverse the series

print([[2,4,5]])
import pandas as pd



data = {'a':10, 'b':20, 'c':30}

s = pd.Series(data)

print(s)



print(s['a']) #Accessing index value assign key 

print (s + s) #Addition of two series



print(s*2) #Multiplication of two series



print(np.exp(s)) #Exponential value 



s = pd.Series(np.random.randn(5), name='something')

print(s)
s2 = s.rename("different")

print(s2)
import pandas



listx = [10, 20, 30, 40, 50]



table = pandas.DataFrame(listx)



print (table)
import pandas

data_list = [{'a':10, 'b':20},{'a':20, 'b':30,'c':40}]

table = pandas.DataFrame(data_list, index = ['first','second'])

print(table)
d = {'one': pd.Series([1., 2., 3.], index=['a', 'b', 'c']),'two': pd.Series([1., 2., 3., 4.], index=['a', 'b', 'c', 'd'])}

df = pandas.DataFrame(d)

print(df)

print("---------------------------------------")



# Access Column 



df2 = pd.DataFrame(d, index=['d', 'b', 'a'])



print(df2)

print("---------------------------------------")

# Access index (Row) and column



df = pd.DataFrame(d, index=['d', 'b', 'a'], columns=['two', 'three'])

print(df)



#Note: The row and column labels can be accessed respectively by accessing the index and columns attributes.
import pandas as pd



data2 = [{'a': 1, 'b': 2}, {'a': 5, 'b': 10, 'c': 20}]

table = pd.DataFrame(data2)

print(table)
import pandas as pd



d = {'one': pd.Series([1, 2, 3], index=['a', 'b', 'c']),

     'two': pd.Series([1., 2., 3., 4.], index=['a', 'b', 'c', 'd'])}

table = pandas.DataFrame(d)



#Adding new column



table["three"] = pd.Series([1, 2, 3], index = ['a', 'b', 'c'])



print(table)



print("---------------------------------------------")

#Adding new column with boolen values



table['flag'] = table["three"]>2



print(table)
del table['two']

print(table)
three = table.pop('three')



print(three)



print(table)
table["foo"] = "bar"

print(table)
table.loc['a']
table.iloc[2]
print(table)

print("----------------------------------------")

row = pd.DataFrame([[1,'True'],[3,'False']], columns = ['one','flag'])

table1= table.append(row)

print(table1)
table1 = table1.drop('d')

print(table1)
df = pd.DataFrame(np.random.randn(10, 4), columns=['A', 'B', 'C', 'D'])



df2 = pd.DataFrame(np.random.randn(7, 3), columns=['A', 'B', 'C'])



print(df + df2)
print(df - df.iloc[0])
print(df[:5].T)
df = pandas.read_csv(path_to_file)
table.to_csv(path_to_file) # if the specified path doesn't exist, a file of the same is automatically created.
sheet = pandas.read_excel(path_to_file)
table.to_excel(path_to_file) # if the specified path doesn't exist, a file of the same is automatically created.
wp = pd.Panel(np.random.randn(2, 5, 4), items=['Item1', 'Item2'],

              major_axis=pd.date_range('1/1/2000', periods=5),

              minor_axis=['A', 'B', 'C', 'D'])

print(wp)



print("total dimension:", wp.ndim)
long_series = pd.Series(np.random.randn(1000))

print(long_series.head())
print(long_series.tail(3))
import pandas as pd

import numpy as np



df = pd.Series(np.arange(1,51))



print(df.ndim)
import pandas as pd

import numpy as np



df = pd.Series(np.arange(1,51))

print(df.axes)
import pandas as pd

import numpy as np



d = {'odd':np.arange(1,100,2),

     'even':np.arange(0,100,2)}



print(d['odd'])

print(d['even'])



df = pd.DataFrame(d)



print(df.sum())
import pandas as pd

import numpy as np



d = {'odd':np.arange(1,100,2),

     'even':np.arange(0,100,2)}



print(d['odd'])

print(d['even'])



df = pd.DataFrame(d)



print(df.std())
import pandas as pd

import numpy as np



d = {'odd':np.arange(1,100,2),

     'even':np.arange(0,100,2)}



# print(d['odd'])

# print(d['even'])



df = pd.DataFrame(d)



print(df.describe())
import pandas as pd

import numpy as np



df = pd.DataFrame(np.random.rand(5,4),

                 columns = ['col1', 'col2', 'col3', 'col4'])



# print(df)



for key, value in df.iteritems():

  print(key, value)
import pandas as pd

import numpy as np



df = pd.DataFrame(np.random.rand(5,4),

                 columns = ['col1', 'col2', 'col3', 'col4'])



# print(df)



for key, value in df.iterrows():

  print(key, value)
import pandas as pd

import numpy as np



df = pd.DataFrame(np.random.rand(5,4),

                 columns = ['col1', 'col2', 'col3', 'col4'])



# print(df)



for row in df.itertuples():

  print(row)
lst = [1, 2, 3, 1, 2, 3]



s = pd.Series([1, 2, 3, 10, 20, 30], lst)



print(s)



grouped = s.groupby(level=0)



print(grouped.first())

print(grouped.last())

print(grouped.sum())
import pandas as pd



world_cup = {'Team':['West Indies','West Indies','India', 'Australia', 'Pakistan', 'Sri Lanka', 'Australia','Australia','Australia', 'India', 'Australia'],

             'Rank':[7,7,2,1,6,4,1,1,1,2,1],

             'Year':[1975,1979,1983,1987,1992,1996,1999,2003,2007,2011,2015]}



df = pd.DataFrame(world_cup)

print(df)

import pandas as pd



world_cup = {'Team':['West Indies','West Indies','India', 'Australia', 'Pakistan', 'Sri Lanka', 'Australia','Australia','Australia', 'India', 'Australia'],

             'Rank':[7,7,2,1,6,4,1,1,1,2,1],

             'Year':[1975,1979,1983,1987,1992,1996,1999,2003,2007,2011,2015]}



df = pd.DataFrame(world_cup)

print(df.groupby('Team').groups)
import pandas as pd



world_cup = {'Team':['West Indies','West Indies','India', 'Australia', 'Pakistan', 'Sri Lanka', 'Australia','Australia','Australia', 'India', 'Australia'],

             'Rank':[7,7,2,1,6,4,1,1,1,2,1],

             'Year':[1975,1979,1983,1987,1992,1996,1999,2003,2007,2011,2015]}



df = pd.DataFrame(world_cup)

print(df.groupby(['Team','Rank']).groups)
import pandas as pd



world_cup = {'Team':['West Indies','West Indies','India', 'Australia', 'Pakistan', 'Sri Lanka', 'Australia','Australia','Australia', 'India', 'Australia'],

             'Rank':[7,7,2,1,6,4,1,1,1,2,1],

             'Year':[1975,1979,1983,1987,1992,1996,1999,2003,2007,2011,2015]}



df = pd.DataFrame(world_cup)

grouped = df.groupby('Team')



for name, group in grouped:

  print(name)
import pandas as pd



world_cup = {'Team':['West Indies','West Indies','India', 'Australia', 'Pakistan', 'Sri Lanka', 'Australia','Australia','Australia', 'India', 'Australia'],

             'Rank':[7,7,2,1,6,4,1,1,1,2,1],

             'Year':[1975,1979,1983,1987,1992,1996,1999,2003,2007,2011,2015]}



df = pd.DataFrame(world_cup)

gropued = df.groupby('Team')



print(grouped.get_group('India'))
import pandas as pd

import numpy as np



d = {'odd':np.arange(1,100,2), 'even':np.arange(0,100,2)}

print(d['odd'])

print(d['even'])



df = pd.DataFrame(d)

print(df.groupby('odd').groups)
left = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],

                      'A': ['A0', 'A1', 'A2', 'A3'],

                      'B': ['B0', 'B1', 'B2', 'B3']})

   



right = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],

                       'C': ['C0', 'C1', 'C2', 'C3'],

                       'D': ['D0', 'D1', 'D2', 'D3']})

  

result = pd.merge(left, right, on='key')

print(result)
import pandas as pd



campian_stats = {'Team':['India', 'Australia','West Indies', 'Pakistan', 'Sri Lanka'],

                  'Rank':[2,3,7,8,4],

                  'World_Champ_Year':[2011,2015,1979,1992,1996],

                  'Points':[874,787,753,673,855]}

match_stats = {'Team':['India', 'Australia','West Indies', 'Pakistan', 'Sri Lanka'],

               'World_Cup_Played':[11,10,11,9,8],

               'ODIs_Played':[733,988,712,679,662]}



df1 = pd.DataFrame(campian_stats)

df2 = pd.DataFrame(match_stats)



print(df1)

print(df2)



print('-------------------------------------------------------------------------------')

print(pd.merge(df1,df2, on = 'Team'))
left = pd.DataFrame({'key1': ['K0', 'K0', 'K1', 'K2'],

                     'key2': ['K0', 'K1', 'K0', 'K1'],

                      'A': ['A0', 'A1', 'A2', 'A3'],

                      'B': ['B0', 'B1', 'B2', 'B3']})

   



right = pd.DataFrame({'key1': ['K0', 'K1', 'K1', 'K2'],

                      'key2': ['K0', 'K0', 'K0', 'K0'],

                       'C': ['C0', 'C1', 'C2', 'C3'],

                       'D': ['D0', 'D1', 'D2', 'D3']})

   

result = pd.merge(left, right, how='left', on=['key1', 'key2'])

print(result)

import pandas as pd



world_campians = {'Team':['India', 'Australia','West Indies', 'Pakistan', 'Sri Lanka'],

                  'Rank':[2,3,7,8,4],

                  'Year':[2011,2015,1979,1992,1996],

                  'Points':[874,787,753,673,855]}

chokers = {'Team':['South Africa','New Zealand', 'Zimbambwe'],

                  'Rank':[1,5,9],

                  'Points':[895,764,656]}



df1 = pd.DataFrame(world_campians)

df2 = pd.DataFrame(chokers)



print(df1)

print(df2)



print('----------------------------------------------------------------')



result = pd.merge(df1,df2,on='Team',how = 'left')

print(result)
left = pd.DataFrame({'key1': ['K0', 'K0', 'K1', 'K2'],

                     'key2': ['K0', 'K1', 'K0', 'K1'],

                      'A': ['A0', 'A1', 'A2', 'A3'],

                      'B': ['B0', 'B1', 'B2', 'B3']})

   



right = pd.DataFrame({'key1': ['K0', 'K1', 'K1', 'K2'],

                      'key2': ['K0', 'K0', 'K0', 'K0'],

                       'C': ['C0', 'C1', 'C2', 'C3'],

                       'D': ['D0', 'D1', 'D2', 'D3']})

   

result = pd.merge(left, right, how='right', on=['key1', 'key2'])

print(result)
import pandas as pd



world_campians = {'Team':['India', 'Australia','West Indies', 'Pakistan', 'Sri Lanka'],

                  'Rank':[2,3,7,8,4],

                  'Year':[2011,2015,1979,1992,1996],

                  'Points':[874,787,753,673,855]}

chokers = {'Team':['South Africa','New Zealand', 'Zimbambwe'],

                  'Rank':[1,5,9],

                  'Points':[895,764,656]}



df1 = pd.DataFrame(world_campians)

df2 = pd.DataFrame(chokers)



print(df1)

print(df2)



print('----------------------------------------------------------------')



result = pd.merge(df1,df2,on='Team',how = 'right')

print(result)
left = pd.DataFrame({'key1': ['K0', 'K0', 'K1', 'K2'],

                     'key2': ['K0', 'K1', 'K0', 'K1'],

                      'A': ['A0', 'A1', 'A2', 'A3'],

                      'B': ['B0', 'B1', 'B2', 'B3']})

   



right = pd.DataFrame({'key1': ['K0', 'K1', 'K1', 'K2'],

                      'key2': ['K0', 'K0', 'K0', 'K0'],

                       'C': ['C0', 'C1', 'C2', 'C3'],

                       'D': ['D0', 'D1', 'D2', 'D3']})

   

result = pd.merge(left, right, how='outer', on=['key1', 'key2'])

print(result)
import pandas as pd



world_campians = {'Team':['India', 'Australia','West Indies', 'Pakistan', 'Sri Lanka'],

                  'Rank':[2,3,7,8,4],

                  'Year':[2011,2015,1979,1992,1996],

                  'Points':[874,787,753,673,855]}

chokers = {'Team':['South Africa','New Zealand', 'Zimbambwe'],

                  'Rank':[1,5,9],

                  'Points':[895,764,656]}



df1 = pd.DataFrame(world_campians)

df2 = pd.DataFrame(chokers)



print(df1)

print(df2)



print('----------------------------------------------------------------')



result = pd.merge(df1,df2,on='Team',how = 'outer')

print(result)
left = pd.DataFrame({'key1': ['K0', 'K0', 'K1', 'K2'],

                     'key2': ['K0', 'K1', 'K0', 'K1'],

                      'A': ['A0', 'A1', 'A2', 'A3'],

                      'B': ['B0', 'B1', 'B2', 'B3']})

   



right = pd.DataFrame({'key1': ['K0', 'K1', 'K1', 'K2'],

                      'key2': ['K0', 'K0', 'K0', 'K0'],

                       'C': ['C0', 'C1', 'C2', 'C3'],

                       'D': ['D0', 'D1', 'D2', 'D3']})

   

result = pd.merge(left, right, how='inner', on=['key1', 'key2'])

print(result)
import pandas as pd



world_campians = {'Team':['India', 'Australia','West Indies', 'Pakistan', 'Sri Lanka'],

                  'Rank':[2,3,7,8,4],

                  'Year':[2011,2015,1979,1992,1996],

                  'Points':[874,787,753,673,855]}

chokers = {'Team':['South Africa','New Zealand', 'Zimbambwe'],

                  'Rank':[1,5,9],

                  'Points':[895,764,656]}



df1 = pd.DataFrame(world_campians)

df2 = pd.DataFrame(chokers)



print(df1)

print(df2)



print('----------------------------------------------------------------')



result = pd.merge(df1,df2,on='Team',how = 'inner')

print(result)
import pandas as pd



df1 = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'],

                    'B': ['B0', 'B1', 'B2', 'B3'],

                    'C': ['C0', 'C1', 'C2', 'C3'],

                    'D': ['D0', 'D1', 'D2', 'D3']},

                     index=[0, 1, 2, 3])

df2 = pd.DataFrame({'A': ['A4', 'A5', 'A6', 'A7'],

                    'B': ['B4', 'B5', 'B6', 'B7'],

                    'C': ['C4', 'C5', 'C6', 'C7'],

                    'D': ['D4', 'D5', 'D6', 'D7']},

                     index=[4, 5, 6, 7])

df3 = pd.DataFrame({'A': ['A8', 'A9', 'A10', 'A11'],

                    'B': ['B8', 'B9', 'B10', 'B11'],

                    'C': ['C8', 'C9', 'C10', 'C11'],

                    'D': ['D8', 'D9', 'D10', 'D11']},

                     index=[8, 9, 10, 11])



frames = [df1, df2, df3]



result = pd.concat(frames)

print(result)
import pandas as pd



world_campians = {'Team':['India', 'Australia','West Indies', 'Pakistan', 'Sri Lanka'],

                  'Rank':[2,3,7,8,4],

                  'Year':[2011,2015,1979,1992,1996],

                  'Points':[874,787,753,673,855]}

chokers = {'Team':['South Africa','New Zealand', 'Zimbambwe'],

                  'Rank':[1,5,9],

                  'Points':[895,764,656]}



df1 = pd.DataFrame(world_campians)

df2 = pd.DataFrame(chokers)

print(pd.concat([df1,df2]))
df = pd.DataFrame({

    'one': pd.Series(np.random.randn(3), index=['a', 'b', 'c']),

    'two': pd.Series(np.random.randn(4), index=['a', 'b', 'c', 'd']),

    'three': pd.Series(np.random.randn(3), index=['b', 'c', 'd'])})



print(df)

print('----------------------------------------------------------------------')



print(df.mean(0))

print('----------------------------------------------------------------------')



print(df.mean(1))

print('----------------------------------------------------------------------')



print(df.sum(0, skipna=False))

print('----------------------------------------------------------------------')



print(df.sum(axis=1, skipna=True))

print('----------------------------------------------------------------------')



#Combined with the broadcasting / arithmetic behavior, one can describe various statistical procedures, 

#like standardization (rendering data zero mean and standard deviation 1), very concisely:



ts_stand = (df - df.mean()) / df.std()

print(ts_stand.std())



print('----------------------------------------------------------------------')



xs_stand = df.sub(df.mean(1), axis=0).div(df.std(1), axis=0)

print(xs_stand.std(1))
series = pd.Series(np.random.randn(1000))



series[::2] = np.nan



print(series.describe())
frame = pd.DataFrame(np.random.randn(1000, 5),

                    columns=['a', 'b', 'c', 'd', 'e'])

   

frame.iloc[::2] = np.nan



print(frame.describe())
s = pd.Series(['a', 'a', 'b', 'b', 'a', 'a', np.nan, 'c', 'd', 'a'])



print(s.describe())
s = pd.Series(np.random.randn(5), index=['a', 'b', 'c', 'd', 'e'])



print(s)



print('---------------------------------------------------------------')

print(s.reindex(['e', 'b', 'f', 'd']))
print(df)



print('---------------------------------------------------------------------------')

print(df.reindex(index=['c', 'f', 'b'], columns=['three', 'two', 'one']))
print(s)



print('-----------------------------------------------------------------')

print(s.rename(str.upper))
print(df)



print('----------------------------------------------------------------')



print(df.rename({'one': 'foo', 'two': 'bar'}, axis='columns'))