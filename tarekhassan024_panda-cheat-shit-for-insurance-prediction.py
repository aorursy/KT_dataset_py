# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas
import pandas as pd
import numpy
import numpy as np
import random as rn
import functools
import re

import warnings
warnings.filterwarnings('ignore')
import pandas
import pandas as pd
import numpy
import numpy as np
import random as rn
import functools
import re
print('Task 1:')  
print(pd.__version__)
print('Task 2:')
dtype = [('Col1','int32'), ('Col2','float32'), ('Col3','float32')]
values = numpy.zeros(20, dtype=dtype)
index = ['Row'+str(i) for i in range(1, len(values)+1)]

df = pandas.DataFrame(values, index=index)
print(df)

df = pandas.DataFrame(values)
print(df)
print('Task 3:')
df = pandas.read_csv('../input/datasets-for-pandas/data1.csv', sep=';', header=None)
print(df.iloc[:4]) # 0 - 4 = 5 values
print('Task 4:')
values = np.random.randint(2, 10, size=4)
print(values)
print('Task 5:')
df = pd.DataFrame(np.random.randint(0, 100, size=(3, 2)), columns=list('xy'))
print(df)
print('Task 6:')
df = pd.DataFrame(np.random.randint(0, 100, size=(2, 4)), columns=['A', 'B', 'C', 'D'])
print(df)
print('Task 7:')
values = np.random.randint(5, size=(2, 4))
print(values)
print(type(values))
print('Task 8:')
df = pd.DataFrame(np.random.randint(0, 100, size=(3, 5)), columns=['Toronto', 'Ottawa', 'Calgary', 'Montreal', 'Quebec'])
print(df)
print('Task 9:')  
dtype = [('One','int32'), ('Two','int32')]
values = np.zeros(3, dtype=dtype)
index = ['Row'+str(i) for i in range(1, 4)]

df = pandas.DataFrame(values, index=index)
print(df)
print('Task 10:')  
dtype = [('Science','int32'), ('Maths','int32')]
values = np.zeros(3, dtype=dtype)

#print(type(dtype))
#values = np.random.randint(5, size=(3, 2))
#print(values)
#index = ['Row'+str(i) for i in range(1, 4)]

df = pandas.DataFrame(values, index=index)
print(df)
print('Task 11:')  

csv = pd.read_csv('../input/datasets-for-pandas/uk-500.csv')
print(csv.head())
print('Task 12:')  
#df = df.from_csv(path, header, sep, index_col, parse_dates, encoding, tupleize_cols, infer_datetime_format)
df = pd.read_csv('../input/datasets-for-pandas/uk-500.csv')
print(df.head())
print('Task 13:') 
df = pandas.read_csv('../input/datasets-for-pandas/data1.csv', sep=',')
print(df.shape) 
#print(df[2:14])
print(df.iloc[0:4,0:2])
#print(df[df.columns[0]])
print('Task 14:') 
df = pandas.read_csv('/kaggle/input/datasets-for-pandas/abc.csv', sep=',', encoding = "utf-8")
print(df.shape)

print(df.iloc[::2, 0:3])    
print('Task 15:') 
df = pandas.read_csv('/kaggle/input/datasets-for-pandas/abc.csv', sep=',', encoding = "utf-8")
print(df.shape)
print(df) 
df['total'] = df.sum(axis=1)

print(df)
print('Task 16:') 
df = pandas.read_csv('/kaggle/input/datasets-for-pandas/abc.csv', sep=',', encoding = "utf-8")
print(df.shape)
print(df) 

df = df[df.science > 50]
print(df)
print('Task 17:') 
df = pandas.read_csv('/kaggle/input/datasets-for-pandas/abc.csv', sep=',', encoding = "utf-8")
print(df.shape)
print(df) 

df = df.query('science > 45')
print(df)
print('Task 18:') 
df = pandas.read_csv('/kaggle/input/datasets-for-pandas/abc.csv', sep=',', encoding = "utf-8", skiprows=[5])
print(df.shape)
print(df)
print('Task 19:') 
df = pandas.read_csv('/kaggle/input/datasets-for-pandas/abc.csv', sep=',', encoding = "utf-8", skiprows=[1, 5, 7])
print(df.shape)
#print(df) 

#df = df[df[[1]] > 45]
print(df)
print('Task 20:') 
df = pandas.read_csv('/kaggle/input/datasets-for-pandas/abc.csv', sep=',', encoding = "utf-8")
print(df.shape)
print(df) 

#df = df[int(df.columns[2]) > 45]
print(df)
print(type(df.columns[2]))
print('Task 21:') 
df = pandas.read_csv('/kaggle/input/datasets-for-pandas/abc.csv', sep=',', encoding = "utf-8", skiprows=[0])
print(df.shape)
print(df) 

#df = df[int(df.columns[2]) > 45]
#print(df)
print(df.columns[2])
print('Task 22:')
from io import StringIO

s = """
        1, 2
        3, 4
        5, 6
    """

df = pd.read_csv(StringIO(s), header=None)

print(df.shape)
print(df)
print('Task 23:') 
df = pandas.read_csv('/kaggle/input/datasets-for-pandas/abc.csv', sep=',', encoding = "utf-8")
print(df.shape)
df['sum'] = df.sum(axis=1)
df['max'] = df.max(axis=1)
df['min'] = df.min(axis=1)
df['average'] = df.mean(axis=1).astype(int)
print(df)
def apply_math_special(row):
    return (row.maths * 2 + row.language / 2 + row.history / 3 + row.science) / 4                

print('Task 24:') 
df = pandas.read_csv('/kaggle/input/datasets-for-pandas/abc.csv', sep=',', encoding = "utf-8")
print(df.shape)
df['sum'] = df.sum(axis=1)
df['max'] = df.max(axis=1)
df['min'] = df.min(axis=1)
df['average'] = df.mean(axis=1).astype(int)
df['math_special'] = df.apply(apply_math_special, axis=1).astype(int)
print(df)
def pass_one_subject(row):
    if(row.maths > 34):
        return 'Pass'
    if(row.language > 34 and row.science > 34):
        return 'Pass'
    
    return 'Fail'                

print('Task 25:') 
df = pandas.read_csv('/kaggle/input/datasets-for-pandas/abc.csv', sep=',', encoding = "utf-8")
print(df.shape)   

df['pass_one'] = df.apply(pass_one_subject, axis=1)
print(df)
print('Task 26:') 
df = pandas.read_csv('/kaggle/input/datasets-for-pandas/abc2.csv', sep=',', encoding = "utf-8")
print(df.shape)   
print(df)
df.fillna(df.mean(), inplace=True)

#df['pass_one'] = df.apply(pass_one_subject, axis=1)
print(df)
print('Task 27:')
df = pd.DataFrame(np.random.rand(10, 5))
df.iloc[0:3, 0:4] = np.nan # throw in some na values
print(df)
df.loc[:, 'test'] = df.iloc[:, 2:].sum(axis=1)
print(df)
print('Task 28:') 
df = pandas.read_csv('/kaggle/input/datasets-for-pandas/score.csv', sep=',', encoding = "ISO-8859-1")
print(df.shape) 
print('Task 29:') 
df = pd.DataFrame(np.random.rand(3,4), columns=list("ABCD"))
print(df.shape)   
print(df)
df.fillna(df.mean(), inplace=True)

print(df)
print('Task 30:')  
df = pandas.read_csv('/kaggle/input/datasets-for-pandas/data1.csv', sep=';') 
print(df[-4:])
print('Task 31:')
series1 = pd.Series([i / 100.0 for i in range(1,6)])
print(series1)
def CumRet(x,y):
    return x * (1 + y)
def Red(x):
    return functools.reduce(CumRet,x,1.0)
s2 = series1.expanding().apply(Red)
# s2 = series1.expanding().apply(Red, raw=True) # is not working
print(s2)
print('Task 32:')  
df = pandas.read_csv('/kaggle/input/datasets-for-pandas/data1.csv', sep=';') 
print(df[2:4])
print('Task 33:')  
df = pandas.read_csv('/kaggle/input/datasets-for-pandas/data1.csv', sep=';') 
print(df[-4:-1])
print('Task 34:')  
df = pandas.read_csv('/kaggle/input/datasets-for-pandas/data1.csv', sep=';') 
print(df.iloc[1:9])
print('Task 35:')  
df = pandas.read_csv('/kaggle/input/datasets-for-pandas/data1.csv', sep=';')
print('Task 36:')  
def xrange(x):
    return iter(range(x))

rnd_1  =  [ rn.randrange ( 1 , 20 )  for  x  in  xrange ( 1000 )] 
rnd_2  =  [ rn.randrange ( 1 , 20 )  for  x  in  xrange ( 1000 )] 
rnd_3  =  [ rn.randrange ( 1 , 20 )  for  x in  xrange ( 1000 )] 
date  =  pd . date_range ( '2012-4-10' ,  '2015-1-4' )
print(len(date))
data  =  pd . DataFrame ({ 'date' : date ,  'rnd_1' :  rnd_1 ,  'rnd_2' :  rnd_2 ,  'rnd_3' :  rnd_3 })

data.head()
print('Task 37:')
below_20 = data[data['rnd_1'] < 20]    
print(below_20)
print('Task 38:') 
def xrange(x):
    return iter(range(x))
rnd_1  =  [ rn.randrange ( 1 , 20 )  for  x  in  xrange ( 1000 )] 
rnd_2  =  [ rn.randrange ( 1 , 20 )  for  x  in  xrange ( 1000 )] 
rnd_3  =  [ rn.randrange ( 1 , 20 )  for  x in  xrange ( 1000 )] 
date  =  pd . date_range ( '2012-4-10' ,  '2015-1-4' )
print(len(date))
data  =  pd . DataFrame ({ 'date' : date ,  'rnd_1' :  rnd_1 ,  'rnd_2' :  rnd_2 ,  'rnd_3' :  rnd_3 })
below_20 = data[data['rnd_1'] < 20]
ten_to_20 = data[(data['rnd_1'] >= 5) & (data['rnd_1'] < 10)]
#print(ten_to_20)
print('Task 39:')      
date  =  pd . date_range ( '2018-08-01' ,  '2018-08-15' )
date_count = len(date)

def fill_rand(start, end, count):
    return [rn.randrange(1, 20 ) for x in xrange( count )]

rnd_1 = fill_rand(1, 20, date_count) 
rnd_2 = fill_rand(1, 20, date_count) 
rnd_3 = fill_rand(1, 20, date_count)
#print(len(date))
data  =  pd . DataFrame ({ 'date' : date ,  'rnd_1' :  rnd_1 ,  'rnd_2' :  rnd_2 ,  'rnd_3' :  rnd_3 })
#print(len(date))
ten_to_20 = data[(data['rnd_1'] >= 15) & (data['rnd_1'] < 20)]
print(ten_to_20)
print('Task 40:')      
date  =  pd . date_range ( '2018-08-01' ,  '2018-08-15' )
date_count = len(date)

def fill_rand(start, end, count):
    return [rn.randrange(1, 20 ) for x in xrange( count )]

rnd_1 = fill_rand(1, 20, date_count) 
rnd_2 = fill_rand(1, 20, date_count) 
rnd_3 = fill_rand(1, 20, date_count)

data  =  pd . DataFrame ({ 'date' : date ,  'rnd_1' :  rnd_1 ,  'rnd_2' :  rnd_2 ,  'rnd_3' :  rnd_3 })

ten_to_20 = data[(data['rnd_1'] >= 15) & (data['rnd_1'] < 33)]
print(ten_to_20)
print('Task 41:')  
date  =  pd . date_range ( '2018-08-01' ,  '2018-08-15' )
date_count = len(date)

def xrange(x):
    return iter(range(x))

def fill_rand(start, end, count):
    return [rn.randrange(1, 20 ) for x in xrange( count )]

rnd_1 = fill_rand(1, 20, date_count) 
rnd_2 = fill_rand(1, 20, date_count) 
rnd_3 = fill_rand(1, 20, date_count)

data  =  pd . DataFrame ({ 'date' : date ,  'rnd_1' :  rnd_1 ,  'rnd_2' :  rnd_2 ,  'rnd_3' :  rnd_3 })
filter_loc = data.loc[ 2 : 4 ,  [ 'rnd_2' ,  'date' ]]
print(filter_loc)
print('Task 42:')
date_date = data.set_index( 'date' ) 
print(date_date.head())
print('Task 43:') 
df = pd.DataFrame({
    'a' : [1,2,3,4], 
    'b' : [9,8,7,6],
    'c' : [11,12,13,14]
});
print(df) 

print('changing on one column')
# Change columns
df.loc[df.a >= 2,'b'] = 9
print(df)
print('Task 44:')  
print('changing on multipe columns')
df.loc[df.a > 2,['b', 'c']] = 45
print(df)
print('Task 45:')  
print(df)
df_mask = pd.DataFrame({
    'a' : [True] * 4, 
    'b' : [False] * 4,
    'c' : [True, False] * 2
})
print(df.where(df_mask,-1000))
print('Task 46:')
print(df)  
df['logic'] = np.where(df['a'] > 5, 'high', 'low')
print(df)
print('Task 47:')
marks_df = pd.DataFrame({
    'Language' : [60, 45, 78, 4], 
    'Math' : [90, 80, 23, 60],
    'Science' : [45, 90, 95, 20]
});
print(marks_df)
marks_df['language_grade'] = np.where(marks_df['Language'] >= 50, 'Pass', 'Fail')
marks_df['math_grade'] = np.where(marks_df['Math'] >= 50, 'Pass', 'Fail')
marks_df['science_grade'] = np.where(marks_df['Science'] >= 50, 'Pass', 'Fail')
print(marks_df)
print('Task 48:')  
marks_df = pd.DataFrame({
    'Language' : [60, 45, 78, 4], 
    'Math' : [90, 80, 23, 60],
    'Science' : [45, 90, 95, 20]
});
print(marks_df)
marks_df_passed_in_language = marks_df[marks_df.Language >=50 ]
print(marks_df_passed_in_language)
print('Task 49:')  
marks_df_passed_in_lang_math = marks_df[(marks_df.Language >=50) & (marks_df.Math >= 50)]
print(marks_df_passed_in_lang_math)
print('Task 50:')  
marks_df_passed_in_lang_and_sc = marks_df.loc[(marks_df.Language >=50) & (marks_df.Science >= 50)]
print(marks_df_passed_in_lang_and_sc)
print('Task 51:')
stars = {
    'age' : [31, 23, 65, 50],
    'movies' : [51, 23, 87, 200],
    'awards' : [42, 12, 4, 78]
    }
star_names = ['dhanush', 'simbu', 'kamal', 'vikram']
stars_df = pd.DataFrame(data=stars, index=[star_names])
print(stars_df)
print('Task 52:')  
print(stars_df.iloc[1:3])
print('Task 53:')  
numbers = pd.DataFrame({
        'one' : [10, 50, 80, 40],
        'two' : [2, 6, 56, 45]
    },
    index = [12, 14, 16, 18])
print(numbers)

print('label between 12 and 16')
print(numbers.loc[12:16])

print('index between 1 and 3')
print(numbers.iloc[1:3])
print('Task 54:') 
stars = {
    'age' : [31, 23, 65, 50],
    'movies' : [51, 23, 87, 200],
    'awards' : [42, 12, 4, 78]
    }
star_names = ['dhanush', 'simbu', 'kamal', 'vikram']
stars_df = pd.DataFrame(data=stars, index=[star_names])
numbers = pd.DataFrame({
        'one' : [10, 50, 80, 40],
        'two' : [2, 6, 56, 45]
    },
    index = [12, 14, 16, 18])
print(numbers)
print('Task 55:')

age_movies_25 = stars_df[(stars_df.movies > 25 ) & (stars_df.age > 25)]  
print(age_movies_25)
print('Task 56:')  
custom_stars = stars_df[stars_df.age.isin([31, 65])]
print(custom_stars)
print('Task 57:')  
print(numbers)
print(numbers[~( (numbers.one > 45) & (numbers.two < 50) )])
print('Task 58:')
def GrowUp(x):
    avg_weight =  sum(x[x['size'] == 'series1'].weight * 1.5)
    avg_weight += sum(x[x['size'] == 'M'].weight * 1.25)
    avg_weight += sum(x[x['size'] == 'L'].weight)
    avg_weight /= len(x)
    return pd.Series(['L',avg_weight,True], index=['size', 'weight', 'adult'])

animals_df = pd.DataFrame({'animal': 'cat dog cat fish dog cat cat'.split(),
                   'size': list('SSMMMLL'),
                   'weight': [8, 10, 11, 1, 20, 12, 12],
                   'adult' : [False] * 5 + [True] * 2})

gb = animals_df.groupby(['animal'])

expected_df = gb.apply(GrowUp)
print(expected_df)
print('Task 59:')
weights = animals_df.groupby(['weight']).get_group(20)  
print(weights)
print('Task 60:')
sides_df = pd.DataFrame({
    'a' : [1, 1, 2, 4],
    'b' : [2, 1, 3, 4]
    })  
print(sides_df)
source_cols = sides_df.columns
print(source_cols)
new_cols = [str(x)+"_side" for x in source_cols]
side_category = {
    1 : 'North',
    2 : 'East',
    3 : 'South', 
    4 : 'West'
    }
sides_df[new_cols] = sides_df[source_cols].applymap(side_category.get)
print(sides_df)
print('Task 61:')  
df = pd.DataFrame({'A' : [1, 1, 2, 2], 'B' : [1, -1, 1, 2]})
print(df)

gb = df.groupby('A')

def replace(g):
    mask = g < 0
    g.loc[mask] = g[~mask].mean()
    return g

gbt = gb.transform(replace)

print(gbt)
print('Task 62:') 
marks_df = pd.DataFrame({
    'Language' : [60, 45, 78, 4], 
    'Math' : [90, 80, 23, 60],
    'Science' : [45, 90, 95, 20]
});
print(marks_df)
marks_df_passed_in_lang_or_sc = marks_df.loc[(marks_df.Language >=50) | (marks_df.Science >= 50)]
print(marks_df_passed_in_lang_or_sc)
print('Task 63:')  
marks_df['passed_one_subject'] = 'Fail' 
marks_df.loc[(marks_df.Language >=50) , 'passed_one_subject'] = 'Pass'
print(marks_df)
print('Task 64:')  
df = pd.DataFrame({
    "a": np.random.randint(0, 100, size=(5,)), 
    "b": np.random.randint(0, 70, size=(5,))
})
print(df)
par = 65
print('with argsort')
df1 = df.loc[(df.a-par).abs().argsort()]
print(df1)

print(df.loc[(df.b-2).abs().argsort()])
print('Task 65:')  
stars = pd.DataFrame({
    "age": [17, 50, 24, 45, 65, 18], 
    "movies": [2, 3, 90, 45, 34, 2] 
})
print(stars.loc[(stars.age - 50).abs().argsort()])
print('Task 66:')  
print(stars.loc[(stars.age - 17).abs().argsort()])
print('Task 67:')
stars = pd.DataFrame({
    "age": [17, 50, 24, 45, 65, 18], 
    "movies": [22, 33, 90, 75, 34, 2] 
})  
print(stars)
print('Young and more movies acted')
young = stars.age < 30    
more_movies = stars.movies > 30
young_more = [young, more_movies]
young_more_Criteria = functools.reduce(lambda x, y : x & y, young_more)
print(stars[young_more_Criteria])
print('Task 68:')  
employees = pd.DataFrame({
    "age": [17, 50, 24, 45, 65, 18], 
    "salary": [75, 33, 90, 175, 134, 78],
    "grade" : [7, 8, 9, 2, 7, 8] 
})  
print(employees)
print('Young, Higher Salary, and Higher Position')
young = employees.age < 30
high_salary = employees.salary > 60
high_position = employees.grade > 6
young_salary_position = [young, high_salary, high_position]
young_salary_position_Criteria = functools.reduce(lambda x, y : x & y, young_salary_position)
print(employees[young_salary_position_Criteria])
print('Task 69:')  
employees = pd.DataFrame({
    "age": [17, 50, 24, 45, 65, 18], 
    "salary": [75, 33, 90, 175, 134, 78],
    "grade" : [7, 8, 9, 2, 7, 8] 
})  
print(employees)
employees.rename(columns={'age': 'User Age', 'salary': 'Salary 2018'}, inplace=True)
print(employees)
print('Task 70:')  
employees = pd.DataFrame({
    "age": [17, 50, 24, 45, 65, 18], 
    "salary": [75, 33, 90, 175, 134, 78],
    "grade" : [7, 8, 9, 2, 7, 8] 
})  
print(employees)
employees['group'] = pd.Series(np.random.randn(len(employees)))
print(employees)
print('Task 71:')  
employees = pd.DataFrame({
    "age": [17, 50, 24, 45, 65, 18], 
    "salary": [75, 33, 90, 175, 134, 78],
    "grade" : [7, 8, 9, 2, 7, 8] 
})  
print(employees)
employees['group'] = pd.Series(np.random.randn(len(employees)))
print(employees)
employees.drop(employees.columns[[0]], axis=1, inplace = True)
print(employees)
print('Task 72:')  
employees = pd.DataFrame({
    "age": [17, 50, 24, 45, 65, 18], 
    "salary": [75, 33, 90, 175, 134, 78],
    "grade" : [7, 8, 9, 2, 7, 8] 
})  
print(employees)
employees['group'] = pd.Series(np.random.randn(len(employees)))
print(employees)
employees.drop(employees.columns[[1, 2]], axis=1, inplace = True)
print(employees)
print('Task 73:')  
employees = pd.DataFrame({
    "age": [17, 50, 24, 45, 65, 18], 
    "salary": [75, 33, 90, 175, 134, 78],
    "grade" : [7, 8, 9, 2, 7, 8],
    "group" : [1, 1, 2, 2, 2, 1] 
    
})  
print(employees)
employees.drop(employees.columns[[0, len(employees.columns)-1]], axis=1, inplace = True)
print(employees)
print('Task 74:')  
employees = pd.DataFrame({
    "age": [17, 50, 24, 45, 65, 18], 
    "salary": [75, 33, 90, 175, 134, 78],
    "grade" : [7, 8, 9, 2, 7, 8],
    "group" : [1, 1, 2, 2, 2, 1] 
    
})  
print(employees)
group = employees.pop('group')
print(employees)
print(group)
print('Task 75:')  
# df = pd.DataFrame.from_items([('A', [1, 2, 3]), ('B', [4, 5, 6]), ('C', [7,8, 9])], orient='index', columns=['one', 'two', 'three'])
# print(df) # throwing error
print('Task 76:')
employees = pd.DataFrame({
    "age": [17, 50, 24, 45, 65, 18], 
    "salary": [75, 33, 90, 175, 134, 78],
    "grade" : [7, 8, 9, 2, 7, 8],
    "group" : [1, 1, 2, 2, 2, 1] 
    
})  
print(employees)  
employees_list1 = list(employees.columns.values) 
employees_list2 = employees.values.tolist()
#employees_list = list(employees)
print(employees_list1)
print(employees_list2)
print('Task 77:')
employees = pd.DataFrame({
    "age": [17, 50, 24, 45, 65, 18], 
    "salary": [75, 33, 90, 175, 134, 78],
    "grade" : [7, 8, 9, 2, 7, 8],
    "group" : [1, 1, 2, 2, 2, 1] 
    
})  
print(employees)  
employees_list2 = employees.values.tolist()
print(employees_list2)
print(type(employees_list2))
print(len(employees_list2))
print('Task 78:')
employees = pd.DataFrame({
    "age": [17, 50, 24, 45, 65, 18], 
    "salary": [75, 33, 90, 175, 134, 78],
    "grade" : [7, 8, 9, 2, 7, 8],
    "group" : [1, 1, 2, 2, 2, 1] 
    
})  
print(employees)  
employees_list2 = employees.values
print(employees_list2)
print(type(employees_list2))
print(employees_list2.shape)
print('Task 79:')
employees = pd.DataFrame({
    "age": [17, 50, 24, 45, 65, 18], 
    "salary": [75, 33, 90, 175, 134, 78],
    "grade" : [7, 8, 9, 2, 7, 8],
    "group" : [1, 1, 2, 2, 2, 1] 
    
})  
print(employees)  
employees_list2 = map(list, employees.values)
print(employees_list2)
print(type(employees_list2))
print('Task 80:')
employees = pd.DataFrame({
    "age": [17, 50, 24, 45, 65, 18], 
    "salary": [75, 33, 90, 175, 134, 78],
    "grade" : [7, 8, 9, 2, 7, 8],
    "group" : [1, 1, 2, 2, 2, 1] 
    
})  
print(employees)  
employees_list2 = list(map(list, employees.values))
print(employees_list2)
print(type(employees_list2))
print('Task 81:')  
users = pd.DataFrame({
    "id": [1, 1, 2, 2, 3, 3], 
    "city": ['Toronto', 'Montreal', 'Calgary', 'Montreal', 'Montreal', 'Ottawa'],
    "count" : [7, 8, 9, 2, 7, 8] 
    
}) 
print(users)
users.drop_duplicates('id', inplace=True, keep='last')
print(users)
print('Task 82:')  
users = pd.DataFrame({
    "id": [1, 1, 2, 2, 3, 3], 
    "city": ['Toronto', 'Montreal', 'Calgary', 'Montreal', 'Montreal', 'Ottawa'],
    "count" : [7, 8, 9, 2, 7, 8] 
    
}) 
print(users)
users1 = users[['id', 'city']]
print(users1)
print('Task 83:')  
users = pd.DataFrame({
    "id": [1, 1, 2, 2, 3, 3], 
    "city": ['Toronto', 'Montreal', 'Calgary', 'Montreal', 'Montreal', 'Ottawa'],
    "count" : [7, 8, 9, 2, 7, 8] 
}) 
print(users)
columns = ['id', 'count']
users1 = pd.DataFrame(users, columns=columns)
print(users1)
print('Task 84:')  
users = pd.DataFrame({
    "id": [1, 1, 2, 2, 3, 3], 
    "city": ['Toronto', 'Montreal', 'Calgary', 'Montreal', 'Montreal', 'Ottawa'],
    "count" : [7, 8, 9, 2, 7, 8] 
    
}) 
print(users)    
users1 = users.iloc[0:2, 1:3]
print(users1)
print('Task 85:')  
users = pd.DataFrame({
    "id": [1, 1, 2, 2, 3, 3], 
    "city": ['Toronto', 'Montreal', 'Calgary', 'Montreal', 'Montreal', 'Ottawa'],
    "count" : [7, 8, 9, 2, 7, 8] 
    
}) 
print(users)    
for index, row in users.iterrows():
    print(row['city'], "==>", row['count'])
print('Task 86:')  
users = pd.DataFrame({
    "id": [1, 1, 2, 2, 3, 3], 
    "city": ['Toronto', 'Montreal', 'Calgary', 'Montreal', 'Montreal', 'Ottawa'],
    "count" : [7, 8, 9, 2, 7, 8] 
    
}) 
print(users)    
for row in users.itertuples(index=True, name='Pandas'):
    print(getattr(row, 'city'))
    
for row in users.itertuples(index=True, name='Pandas'):
    print(row.count)
print('Task 87:')  
users = pd.DataFrame({
    "id": [1, 1, 2, 2, 3, 3], 
    "city": ['Toronto', 'Montreal', 'Calgary', 'Montreal', 'Montreal', 'Ottawa'],
    "count" : [7, 8, 9, 2, 7, 8] 
    
}) 
print(users)    
for i, row in users.iterrows():
    for j, col in row.iteritems():    
        print(col)
print('Task 88:')  
pointlist = [
                {'points': 50, 'time': '5:00', 'year': 2010}, 
                {'points': 25, 'time': '6:00', 'month': "february"}, 
                {'points':90, 'time': '9:00', 'month': 'january'}, 
                {'points_h1':20, 'month': 'june'}
            ]
print(pointlist)
pointDf = pd.DataFrame(pointlist)
print(pointDf)

pointDf1 = pd.DataFrame.from_dict(pointlist)
print(pointDf1)
print('Task 89:')
df = pd.DataFrame(np.random.randn(10,6))
# Make a few areas have NaN values
df.iloc[1:3,1] = np.nan
df.iloc[5,3] = np.nan
df.iloc[7:9,5] = np.nan
print(df)
df1 = df.isnull()
print(df1)
print('Task 90:')  
df = pd.DataFrame(np.random.randn(10,6))
# Make a few areas have NaN values
df.iloc[1:3,1] = np.nan
df.iloc[5,3] = np.nan
df.iloc[7:9,5] = np.nan
print(df)
print(df.isnull().sum())
print(df.isnull().sum(axis=1))
print(df.isnull().sum().tolist())
print('Task 91:')  
df = pd.DataFrame(np.random.randn(10,6))
# Make a few areas have NaN values
df.iloc[1:3,1] = np.nan
df.iloc[5,3] = np.nan
df.iloc[7:9,5] = np.nan
print(df)
print(df.isnull().sum(axis=1))
print('Task 92:')  
df = pd.DataFrame(np.random.randn(10,6))
# Make a few areas have NaN values
df.iloc[1:3,1] = np.nan
df.iloc[5,3] = np.nan
df.iloc[7:9,5] = np.nan
print(df)
print(df.isnull().sum().tolist())
print('Task 93:')  
users = pd.DataFrame({
    "id": [1, 1, 2, 2, 3, 3], 
    "city": ['Toronto', 'Montreal', 'Calgary', 'Montreal', 'Montreal', 'Ottawa'],
    "count" : [7, 8, 9, 2, 7, 8] 
    
}) 
print(users)

# below line throws error
# users1 = users.reindex_axis(['city', 'count', 'id'], axis=1)
# print(users1)

users2 = users.reindex(columns=['city', 'id', 'count'])
print(users2)
print('Task 94:')
numbers = pd.DataFrame({
    "id": [1, 2, 3, 4, 5, 6], 
    "number": [10, 20, 30, 30, 23, 12]
    
})  
print(numbers)
numbers.drop(numbers.index[[0, 3, 5]], inplace=True)
print(numbers)
print('Task 95:')  
numbers = pd.DataFrame({
    "id": [1, 2, 3, 4, 5, 6], 
    "number": [10, 20, 30, 30, 23, 12]
    
}, index=['one', 'two', 'three', 'four', 'five', 'six'])  
print(numbers)
numbers1 = numbers.drop(['two','six'])
print(numbers1)
numbers2 = numbers.drop('two')
print(numbers2)
print('Task 96:')
cats = animals_df.groupby(['animal']).get_group('cat')
print(cats)
print('Task 97:')  
x = numpy.array([
                    [ 1,  2,  3,  4,  5],
                    [ 6,  7,  8,  9, 10],
                    [11, 12, 13, 14, 15],
                    [16, 17, 18, 19, 20]
                ]
    )
print(x)
print(x[::2])
print('Task 98:')  
x = numpy.array([
                    [ 1,  2,  3,  4,  5],
                    [ 6,  7,  8,  9, 10],
                    [11, 12, 13, 14, 15],
                    [16, 17, 18, 19, 20]
                ]
    )
print(x)
print(x[:, 1::2])
print('Task 99:')  

x = numpy.array([
                    [ 1,  2,  3,  4,  5],
                    [ 6,  7,  8,  9, 10],
                    [11, 12, 13, 14, 15],
                    [16, 17, 18, 19, 20]
                ]
    )
print(x)
print(x[::2, 1::2])
print('Task 100:')  
users = pd.DataFrame({
    "id": [1, 1, 2, 2, 3, 3], 
    "city": ['Toronto', 'Montreal', 'Calgary', 'Montreal', 'Montreal', 'Ottawa'],
    "count" : [7, 8, 9, 2, 7, 8] 
    
}) 
print(users)
users.drop_duplicates('id', inplace=True)
print(users)
print('Task 101:')  
users = pd.DataFrame({
    "name": ['kevin', 'james', 'kumar', 'kevin', 'kevin', 'james'], 
    "city": ['Toronto', 'Montreal', 'Calgary', 'Montreal', 'Montreal', 'Ottawa'],
    "count" : [7, 8, 9, 2, 7, 8] 
    
}) 
print(users)
users.drop_duplicates('name', inplace=True, keep='last')
print(users)
users1 = users.drop_duplicates('name', keep=False)
print(users1)
print('Task 102:')
animals_df1 = animals_df.groupby('animal').apply(lambda x: x['size'][x['weight'].idxmax()])
print(animals_df1)
print('Task 103:')  
df = pd.DataFrame(np.random.randn(6,1), index=pd.date_range('2013-08-01', periods=6, freq='B'), columns=list('A'))
print(df)
df.loc[df.index[3], 'A'] = np.nan
print(df)
print('Task 104:')
df1 = df.reindex(df.index[::-1]).ffill()
print(df1)
print('Task 105:')
animals_df = pd.DataFrame({'animal': 'cat dog cat fish dog cat cat'.split(),
                   'size': list('SSMMMLL'),
                   'weight': [8, 10, 11, 1, 20, 12, 12],
                   'adult' : [False] * 5 + [True] * 2})
print(animals_df)
# 106. Change columns
print('Task 106:')

users = pd.DataFrame({
    "name": ['kevin', 'james', 'kumar', 'kevin', 'kevin', 'james'], 
    "city": ['Toronto', 'Montreal', 'Calgary', 'Montreal', 'Montreal', 'Ottawa']
}) 

print('Before changing columns : ')
print(users)

# change columns
users_new = users.rename({'name': 'first_name', 'city': 'current_city'}, axis = 1)

print('\nAfter changing columns : ')
print(users_new)
# 106. Find any matching value

print('Task 107:')

users = pd.DataFrame({
    "name": ['kevin', 'james', 'kumar', 'kevin', 'kevin', 'james'], 
    "city": ['Toronto', 'Montreal', 'Calgary', 'Montreal', 'Montreal', 'Ottawa']
}) 

print('Original Dataframe:')
print(users)

print('\nFinding `Montreal` in any cell :')
print(users[users.eq('Montreal').any(1)])
# 107. Match with isin function

print('Task 107:')

users = pd.DataFrame({
    "name": ['kevin', 'james', 'kumar', 'kevin', 'kevin', 'james'], 
    "city": ['Toronto', 'Montreal', 'Calgary', 'Montreal', 'Montreal', 'Ottawa']
}) 

print('Original Dataframe:')
print(users)

print('\nFinding `Montreal` in by using isin function :')

users.isin(['Montreal']).any()
# 108. Finding specific items by using `isin` function

print('Task 108:')

users = pd.DataFrame({
    "name": ['kevin', 'james', 'kumar', 'kevin', 'kevin', 'james'], 
    "city": ['Toronto', 'Montreal', 'Calgary', 'Montreal', 'Montreal', 'Ottawa']
}) 

print('Original Dataframe: ')
print(users)

print('\nFinding `Montreal` in using isin and stack them: ')
print(users[users.isin(['Montreal'])].stack())
# 109. Exclude specific matching

print('Task 109:')

users = pd.DataFrame({
    "name": ['kevin', 'james', 'kumar', 'kevin', 'kevin', 'james'], 
    "city": ['Toronto', 'Montreal', 'Calgary', 'Montreal', 'Montreal', 'Ottawa']
}) 

print('Original Dataframe: ')
print(users)

print('\nExcluding `Montreal` in using isin and stack them: ')
print(users[~users.isin(['Montreal'])].stack())
# 110. Apply a custom function on multiple columns

print('Task 110:')

amounts = pd.DataFrame({
    "CIBC": [200, 4200, 300, 300], 
    "TD": [1200, 800, 4000, 2000]
})

print('Original Dataframe: ')
print(amounts)

def get_total_amount(x):
    
    # if the amount is less than 500, skip it
    total_amount = 0
    
    if(x['CIBC'] > 499):
        total_amount += x['CIBC']
        
    if(x['TD'] > 499):
        total_amount += x['TD']
    
    return total_amount

amounts['Total'] = amounts.apply(get_total_amount, axis = 1)

print('Dataframe after applying the custom function: ')
print(amounts)
# 111. iterrows as tuples

print('Task 111:')

users = pd.DataFrame({
    "id": [1, 1, 2, 2, 3, 3], 
    "city": ['Toronto', 'Montreal', 'Calgary', 'Montreal', 'Montreal', 'Ottawa'],
    "count" : [7, 8, 9, 2, 7, 8] 
    
}) 

print(users)  

print('\nIterate rows as tuples:')
for row in users.itertuples():
    print(row)
# 112. Dataframe with NaN

print('Task 112:')

df = pd.DataFrame(np.nan, index = [0, 1, 2], columns = ['A', 'B', 'C'])

print(df)
# 113. Simple Dataframe with NaN

print('Task 113:')

df = pd.DataFrame([np.nan] * 5)

print(df)
# 114. Pandas and Date with Range

print('Task 114:')

import datetime as dt

pd.np.random.seed(0)

df = pd.DataFrame({
     "date" : [dt.date(2012, x, 1) for x in range(1, 11)]
})

print(df)
# 115. Pandas and Numpy Repeat

print('Task 115:')

df = pd.DataFrame({
     "entry" : np.repeat(3, 7) # Repeat the number for 7 times
})

print(df)
# 116. read_sql in Pandas

print('Task 116:')

import sqlite3 as sql

conn = sql.connect('/kaggle/input/rj-sample-datasets/sample.db')

demo_df = pd.read_sql('SELECT ID, NAME FROM DEMO', conn)

print(demo_df)

# Note: online sqlite https://sqliteonline.com/
# 117. Get a single value by iat

print('Task 117:')

df = pd.DataFrame(np.random.randint(0, 100, size = (7, 2)), columns = list('ab'))

print('\nOriginal Dataframe:')
print(df)

val = df.iat[3, 1]

print('\nGetting value at 3rd row and first col:')
print(val)
# 118. Get last n elements

print('Task 118:')

df = pd.DataFrame(np.random.randint(0, 100, size = (7, 2)), columns = list('ab'))

print('\nOriginal Dataframe:')
print(df)

tail_df = df.tail(3)

print('\nLast 3 rows:')
print(tail_df)
# 119. Add data one by one by using loc function

df = pd.DataFrame(columns = ['rows', 'cols', 'time taken'])

df.loc['Dataset1 - Dask'] = (1000, 20, 10)
df.loc['Dataset1 - Pandas'] = (1000, 20, 15)

df
!pip install country_converter
# 120. Convert country in Dataframe 

import country_converter as coco
import pandas as pd

df = pd.DataFrame({'code': ['IN', 'USA', 'BR', 'CAN']})

print('Before:')
print(df)

df['country'] = df.code.apply(lambda x: coco.convert(names = x, to = 'name_short', not_found = None))

print('\nAfter:')
print(df)