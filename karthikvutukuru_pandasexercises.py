# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
ser = pd.Series(['Apple', 'Orange', 'Plan', 'Python', 'Money'])

ser
vowels = ['a', 'e', 'i', 'o', 'u']



def count_vowels(x):

    k=0

    x=x.lower().strip()

    for i in x:

        if i in vowels:

            k=k+1

        if k>1:

            return True

            break

    return False
ser[ser.apply(count_vowels)]
emails = pd.Series(['buying books at amazom.com', 'rameses@egypt.com', 'matt@t.co', 'narendra@modi.com'])

pattern ='[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,4}'

emails
import re

[email for email in emails if len(re.findall(pattern,email)) > 0 ]
L = pd.Series(range(500))
def create_rows(ser, divide=4, stride=2):

    new_list= []

    i=0

    for k in range(len(ser)-4):

        new_list_to_append = ser[i:i+divide].values.tolist()

        if len(new_list_to_append) < 4:

            break

        else:

            new_list.append(new_list_to_append)

            i = i+2

    print(new_list)



%time create_rows(L, 4)
def gen_strides(ser, strides=5, window=5):

    n_strides = ((ser.size - window) // strides) +1

    return np.array([ser[s : s+ window] for s in np.arange(0, ser.size, strides)[:n_strides]])



%time gen_strides(L, 2 ,4)
intdf = pd.DataFrame(np.random.randint(-20, 50, 100).reshape(10,-1))

intdf.head()
arr = intdf[intdf>0].values.flatten()

arr_q = arr[~np.isnan(arr)]

arr_q
sizesq = int(np.floor(arr_q.shape[0]**0.5))

sizesq
top_indexes = np.argsort(arr_q)[::-1]

result = np.take(arr_q, sorted(top_indexes[:sizesq**2])).reshape(sizesq, -1)

result
flight_data = pd.read_csv('/kaggle/input/flights.csv')

flight_data.info()
flight_data.iloc[:10,:6]
meetup_data = pd.read_csv('/kaggle/input/meetup_groups.csv')

meetup_data.iloc[:,3]
flight_data[['DEP_DELAY', 'AIR_TIME']][:5]
d = {'DEP_DELAY': np.nanmean, 'AIR_TIME': np.nanmedian}



flight_data[['DEP_DELAY', 'AIR_TIME']].apply(lambda x, d: x.fillna(d[x.name](x)), args=(d, ))
even_series = pd.Series([x for x in range(100) if x%2 == 0])

even_series
odd_series = pd.Series([x for x in range(100) if x%2 != 0])

odd_series 
df1_mask = even_series == [x for x in even_series.values.tolist() if x not in odd_series.values.tolist()]

even_series[df1_mask]
even4_series = pd.Series([x for x in range(100) if x%4 == 0])

even4_series
union_series = pd.Series(np.union1d(even_series, odd_series))

intersection_series = pd.Series(np.intersect1d(even_series, odd_series))

union_series[~union_series.isin(intersection_series)]
ser = pd.Series(np.random.normal(10, 5, 25))

np.percentile(ser, [0,25,50,75, 100])
ser = pd.Series(np.take(list('abcdefgh'), np.random.randint(8, size=30)))

ser.value_counts()
np.random.RandomState(100)

ser = pd.Series(np.random.randint(1, 5, [12]))

ser
ser[~ser.isin(ser.value_counts().index[:2])] = 'Other'

ser

ser1 = pd.Series([10, 9, 6, 5, 3, 1, 12, 8, 13])

ser2 = pd.Series([1, 3, 10, 13])
[pd.Index(ser1).get_loc(i) for i in ser2]
ser = pd.Series(['how', 'to', 'kick', 'ass?'])

ser = pd.Series([x.title() for x in ser])

ser
word_count = [len(x) for x in ser]

word_count
ser = pd.Series(range(10000))



diffs = [ser.loc[i] - ser.loc[i+1] for i in range(0, len(ser)-1) ]

diffs
ser = pd.Series(['01 Jan 2010', '02-02-2011', '20120303', '2013/04/04', '2014-05-05', '2015-06-06T12:20'])

ser.describe()
from dateutil.parser import parse

ser.map(parse)
pd.to_datetime(ser)
ser = pd.Series(['01 Jan 2010', '02-02-2011', '20120303', '2013/04/04', '2014-05-05', '2015-06-06T12:20'])

ser = pd.to_datetime(ser)

#ser.dt.day.tolist()

#ser.dt.month.tolist()

#ser.dt.year.tolist()

#ser.dt.weekday_name.tolist()

#ser.dt.dayofyear.tolist()

ser.dt.week.tolist()
ser = pd.Series(['Jan 2010', 'Feb 2011', 'Mar 2012'])

ser = pd.Series(['04 ' +x for x in ser])

ser = pd.to_datetime(ser)

ser
p = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

q = pd.Series([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])



ed = (sum([(p.loc[i] - q.loc[i])**2 for i in range(len(p))]))**.5

ed
my_str = my_str = 'dbc deb abed gade'

from collections import Counter

c= Counter(my_str)

my_str.replace(' ', c.most_common()[-1][0])
timeser = pd.Series(np.random.randint(1, 10, 10), pd.date_range('2000-01-01', periods=10, freq='W-SAT'))

timeser.index
ser = pd.Series([1,10,3,np.nan], index=pd.to_datetime(['2000-01-01', '2000-01-03', '2000-01-06', '2000-01-08']))

ser.resample('D').ffill()
emp_df = pd.read_csv('/kaggle/input/employee.csv')

emp_df.head()
emp_df.loc[emp_df.BASE_SALARY == emp_df['BASE_SALARY'].max()]
row, col = np.where(emp_df.values==emp_df.BASE_SALARY.max())
emp_df.iloc[row[0], col[0]]
np.argmax(emp_df.isnull().sum())
emp_df['BASE_SALARY'] = emp_df['BASE_SALARY'].replace(to_replace=np.NaN, value=np.nanmean(emp_df['BASE_SALARY'].values))

emp_df['BASE_SALARY']
emp_df['BASE_SALARY'] = emp_df[['BASE_SALARY']].apply(lambda x: x.fillna(x.mean()))

emp_df['BASE_SALARY']
type(emp_df[['BASE_SALARY']])
df = pd.DataFrame(np.random.randint(10, 40, 60).reshape(-1, 4))

df
df.iloc[np.where(df.sum(axis=1)>100)[0][-2:],:]
df = pd.DataFrame(np.arange(25).reshape(5,-1), columns=list('abcde'))

df
pd.concat([pd.get_dummies(df['a'], columns=['a0','a1','a2','a3', 'a4']), df[list('bcde')]], axis=1)
df = pd.DataFrame(np.random.randint(1,100, 80).reshape(8, -1))

df['minbymaax'] = df.max(axis=1)/df.min(axis=1)

df
df['penultimate'] = df.apply(lambda x: x.sort_values().unique()[-2], axis=1)
x = [(x, np.argmax(df.iloc[x])) for x in range(df.shape[0])]

x

df = pd.DataFrame(np.random.randint(1,100, 100).reshape(10, -1))

df
for i in range(df.shape[0]):

    df.iat[i,i] = 0

    df.iat[9-i, i] = 0

df
df=pd.DataFrame([[1, 2, 8],[3, 4, 8], [5, 1, 8]], columns=['A', 'B', 'C'])

df
df[['A','B']].replace([1, 3, 2], [3, 6, 7])
df = pd.DataFrame(np.random.randint(1,100, 40).reshape(10, -1), columns=list('pqrs'), index=list('abcdefghij'))

df1 = df.copy()

df, df1
nearest_rows = []

nearest_distance = []



for i , row in df.iterrows():

    cur_row = row

    rest = df.drop(i)

    euclidean_distances = {}

    for j, contestant in rest.iterrows():

        euclidean_distances.update({j:round(np.linalg.norm(cur_row.values-contestant.values))})

    nearest_rows.append(max(euclidean_distances, key=euclidean_distances.get))

    nearest_distance.append(max(euclidean_distances.values()))

    

df['nearest_rows'] = nearest_rows

df['nearest_distance'] = nearest_distance



df
nearest_rows = []

nearest_distance = []



for i , cur_row in df1.iterrows():

    #cur_row = row

    #rest = df.drop(i)

    euclidean_distances = {}

    for j, contestant in df1.iterrows():

        euclidean_distances.update({j:round(np.linalg.norm(cur_row.values-contestant.values))})

    nearest_rows.append(max(euclidean_distances, key=euclidean_distances.get))

    nearest_distance.append(max(euclidean_distances.values()))

    

df1['nearest_rows'] = nearest_rows

df1['nearest_distance'] = nearest_distance



df1
sqDict = {x: x**2 for x in [1,2,3,4,5]} 

sqDict[5] = 0

sqDict
max(sqDict, key=sqDict.get)