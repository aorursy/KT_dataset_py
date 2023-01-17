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
# Read csv file

df=pd.read_csv('../input/titanic/test.csv')

df.head(2)





# Read csv file with specified delimiter

df=pd.read_table('../input/titanic/test.csv', sep=',')

df.head(3)

# Without header

df=pd.read_csv('../input/titanic/test.csv', header=None, sep=',')

df.head(2)
# Self-defined header

names=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l','m']

df=pd.read_csv('../input/titanic/test.csv', header=None, sep=',', names=names)

df.head(2)
# use specified column as index

names=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k']

df=pd.read_csv('../input/titanic/test.csv', header=None, sep=',', names=names, index_col='b')

df.head(2)
# skip rows

df=pd.read_csv('../input/titanic/test.csv', skiprows=[0,1,3])

df.head(3)

# Reading a small piece of file

df=pd.read_csv('../input/titanic/test.csv', nrows=5)

df.head(10)
# Reading a small piece of file with chunker

from pandas import Series 

chunker=pd.read_csv('../input/titanic/test.csv', chunksize=1000)

tot=Series([])



for piece in chunker:

    tot=tot.add(piece['Age'].value_counts(), fill_value=0)

# tot=tot.order(ascending=False)



print(tot[:5])
# Writing data to csv

df=pd.read_csv('../input/titanic/test.csv')

df.to_csv('out.csv', sep='|')

pd.read_csv('out.csv', sep='|').head(2)

# write to console

import sys



df.head(3).to_csv(sys.stdout)
# Series has to_csv too



dates = pd.date_range('1/1/2000', periods=7)

ts = Series(np.arange(7), index=dates)

ts.to_csv(sys.stdout)
import json



# json to dict

obj = """ {"name": "Wes",

"places_lived": ["United States", "Spain", "Germany"], "pet": null,

"siblings": [{"name": "Scott", "age": 25, "pet": "Zuko"},

{"name": "Katie", "age": 33, "pet": "Cisco"}]

} """



result=json.loads(obj)

print(result['name'])

print(result['pet'])

print(result['siblings'][1]['pet'])

print(result)





obj = """ {"name": "Wes",

"places_lived": ["United States", "Spain", "Germany"], "pet": null,

"siblings": [{"name": "Scott", "age": 25, "pet": "Zuko"},

{"name": "Katie", "age": 33, "pet": "Cisco"}]

} """



result=json.loads(obj)

# dumps as string

asJson=json.dumps(result)

siblings=pd.DataFrame(result['siblings'], columns=['name', 'age'])



siblings
# !pip install urllib2

# from lxml.html import parse

# from urllib2 import urlopen

# parsed = parse(urlopen('http://finance.yahoo.com/q/op?s=AAPL+Options'))

# doc = parsed.getroot()

# links=doc.findall('.//a')

# links[15:20]

import pandas as pd

xls_file = pd.ExcelFile('../input/publisher-contrast/result.xlsx')

table = xls_file.parse('Sheet1')



table.head(10)
import sqlite3

import pandas as pd



query="""

CREATE TABLE test(

a varchar(20),

b INTERGER

);

"""

conn=sqlite3.connect(':memory:')

conn.execute(query)

conn.commit()



data=[("zhu",1),("jun",2)]

stmt="INSERT INTO test VALUES(?,?)"

conn.executemany(stmt, data)

conn.commit()



cursor = conn.execute('select * from test')

rows=cursor.fetchall()

print(rows)

pd.DataFrame(rows)



import pandas.io.sql as sql



sql.read_sql('select * from test', conn)



import pandas as pd

df1=pd.DataFrame({'key':['a','b','a','c','b'], 'data1': range(5)})

df2=pd.DataFrame({'key':['a','b','d'], 'data2': range(3)})



# Default inner join

pd.merge(df1, df2, on='key')



# change join manner

pd.merge(df1, df2, on='key', how='outer')

pd.merge(df1, df2, on='key', how='left')

pd.merge(df1, df2, on='key', how='right')



# multi key join

left = pd.DataFrame({'key1': ['foo', 'foo', 'bar'], 'key2': ['one', 'two', 'one'],

'lval': [1, 2, 3]})

right = pd.DataFrame({'key1': ['foo', 'foo', 'bar', 'bar'], 'key2': ['one', 'one', 'one', 'two'],

'rval': [4, 5, 6, 7]})

pd.merge(left, right, on=['key1', 'key2'], how='outer')



# add suffix for same name

pd.merge(left, right, on='key1', suffixes=('_left', '_right'))

# Merging on Index

import pandas as pd

left1=pd.DataFrame({'key': ['a', 'b', 'a', 'a', 'b', 'c'], 'value': range(6)})

right1=pd.DataFrame({'group_val': [3.5, 7]}, index=['a', 'b'])



pd.merge(left1,right1,left_on='key', right_index=True)

pd.merge(left1,right1,left_on='key', right_index=True,how='outer')
# Join  which defalut left join

import pandas as pd

left1=pd.DataFrame({'key': ['a', 'b', 'a', 'a', 'b', 'c'], 'value': range(6)})

right1=pd.DataFrame({'group_val': [3.5, 7]}, index=['a', 'b'])



left1.join(right1, on='key')



# join multiple, only support index



another1=pd.DataFrame([[1.0,2.0],[3.2,4.1],[5.3]], index=['a', 'c', 'b'],columns=['another1', 'another2'])

another2=pd.DataFrame([[1.1,2.2],[3.3,4.4],[5.5]], index=['c', 'a', 'b'],columns=['another3', 'another4'])



right1.join([another1, another2], how='outer')

# concat data

import pandas as pd

from pandas import Series

s1 = Series([0, 1], index=['a', 'b'])

s2 = Series([2, 3, 4], index=['b', 'd', 'e'])



print(pd.concat([s1,s2]))

print(pd.concat([s1,s2], axis=1))

print(pd.concat([s1,s2], axis=1, join='inner'))

# combine data: like coalesce

from pandas import DataFrame

import numpy as np

df1 = DataFrame({'a': [1., np.nan, 5., np.nan], 'b': [np.nan, 2., np.nan, 6.],

'c': range(2, 18, 4)})

df2 = DataFrame({'a': [5., 4., np.nan, 3., 7.], 'b': [np.nan, 3., 4., 6., 8.]})



df1.combine_first(df2)
# stack and unstack

import pandas as pd

import numpy as np

data=pd.DataFrame(np.arange(6).reshape((2, 3)),index=pd.Index(['Ohio', 'Colorado'], name='state'), 

               columns=pd.Index(['one', 'two', 'three'], name='number'))

print(data)

result=data.stack()

print(result)



print(result.unstack())

data.unstack()

from pandas import DataFrame



data=DataFrame({'k1':['one']*3+['two']*4,

               'k2': [1,1,2,3,3,4,4]})

print(data.duplicated())

print(data.drop_duplicates())

data['v1'] = range(7)

# dedup based on k1

data.drop_duplicates(['k1'])



data.drop_duplicates(['k1', 'k2'])
# Replaceing value

import pandas as pd

import numpy as np

data = pd.Series([1., -999., 2., -999., -1000., 3.])

data.replace(-999, np.nan)

data.replace([-999, -1000], [np.nan,0])
# Filting outlier

import numpy as np

import pandas as pd

from pandas import DataFrame

np.random.seed(12345)

data = DataFrame(np.random.randn(1000, 4))

data.describe()



col = data[3]



print(col[np.abs(col) > 3])



print(data[(np.abs(data) > 3).any(1)])