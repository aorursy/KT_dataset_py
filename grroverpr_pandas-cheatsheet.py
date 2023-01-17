import pandas as pd
import numpy as np # necessity as pandas is built on np
from IPython.display import Image # to display images
Image('../input/images/pddf.jpg', width=700) 
Image('../input/images/series.png', width=500) # not pandas, just showing example series
# index will set the index for further reference
# data can be passed as list
s = pd.Series([3, -5, 7, 4], index=['a', 'b', 'c', 'd']) 
s
s[1:]
s['a'] 
s[:'c']
s.index
# series using dictionary

s2 = pd.Series({'a':3, 'b': -1, 'c': 12}); s2
s2['b']
Image('../input/images/df.png', width=500) 
data = {'Country': ['Belgium', 'India', 'Brazil'],
 'Capital': ['Brussels', 'New Delhi', 'Brasília'],
 'Population': [11190846, 1303171035, 207847528]}
df_sample = pd.DataFrame(data,
 columns=['Country', 'Capital', 'Population'])
df_sample
Image('../input/images/titanic.jpg', width=700) # not pandas, just showing example series
df = pd.read_csv('../input/titanic/train.csv') # read csv file
df.shape
df.head() # see top 5 rows of data
df.dtypes # see datatype of each variable
df.columns # column names
df.nunique() # unique value for each variable
df.info() # not null part is very useful to see how many nulls are there in data
df.iloc[0, 4] # 0 row, 4 column
df.iloc[1:4, 2:6] # indexes are maintained. Can reset_index() to start index from 0  
df.loc[1:2,'Name':"Age"] # here row indexes are numbers but column indexes are name of columns 
df.loc[2,['Name',"Age"]] # here row indexes are numbers. 
# select rows with either sex as female or Pclass as 1
df[(df.Sex == 'female') | (df.iloc[:,2] == 1) ].iloc[:3] # () are important
# first 3 rows of gives all columns which have all string values or all int > 1 values
df.loc[:,(df > 1).all()][:3] 
# first 3 rows of all columns which have all not null values
df.loc[:,(df.notnull().all() )][:3]
# first 3 rows of all columns which have atleast 1 null value
df.loc[:, df.isnull().any()][:3]
df[(df.iloc[:,2] == 1) & (df.Sex == 'female')].shape
# fraction of males with Age > 25, df.shape[0] -> number of rows 

sum((df.Age > 25) & (df.Sex == 'male'))/df.shape[0] 
# number of people who survived and were not in class 3

sum((df.Survived != 0) & (~(df.Pclass == 3)) ) 
# filter all rows which have Age > Passenger ID
df.query('Age > PassengerId')
# filter only sex and age columns (first 2 rows)

df.filter(items=['Age', 'Sex'])[:2]
# filter only 0 and 5 row index

df.filter(items=[0,5], axis=0)
# first 2 rows of column names ending with "ed" (think of past tense)

df.filter(like = 'ed', axis=1)[:2]
# Can use same thing as above using regex also

df.filter(regex='ed$', axis=1)[:2]
df[df.Pclass.isin([0,1])].head()
# setting 
df.set_index('Ticket')[:2]
# can set multiple columns as index also. Just pass them in list
# Setting Ticket and Name as index

df.set_index(['Ticket', 'Name'])[:2]
# can see what are values of index. 
# checking index of 1st row

df.set_index(['Ticket', 'Name']).index[0]
df_index = df.set_index(['Ticket', 'Name'])
df_index[:2]
df_index.reset_index()[:2]
df.rename(columns={'Name': 'Whats_name', 'Fare':'Price'})[:2]
# can use some mapper function also. default axis='index' (0)

df.rename(mapper=str.lower, axis='columns')[:2]
df.Sex.unique()
sum(df.PassengerId.duplicated()) # there are no duplicate passegerid. good thing to check
# can check duplicates in index also.
# useful if doubtful about duplicates in index doing bad things 

sum(df.index.duplicated())
# can help in getting unique combination of multiple columns
# unique() dosn't work in this case
df.loc[:,['Sex', 'Embarked']].drop_duplicates()
# group by sex then count. 
# returns count in each column. difference in some cases because of nulls in those columns
# can do iloc[:,0] to only get first column 

df.groupby(by = ['Sex']).count()
# can use multiple conditions
# group by sex and survived -> mean of age

df.groupby(by = ['Sex', 'Survived']).mean().loc[:,'Age']
# can group by indexes also by using levels= 
# useful when we have multindexes
# can use agg function with lambda func

df_index = df.set_index(['Sex', 'Pclass'])
df_index.groupby(level=[0,1]).agg({'Fare': lambda x: sum(x)/len(x), # this is also just mean actually
                                  'Age' : np.mean})
# shape of below code is same as original df

df_index.groupby(level=[0,1]).transform(lambda x: sum(x)/len(x)).head()
# how=any -> row with any column = NA

df.dropna(axis=0,  how='any').shape
# how=any -> row with all columns = NA

df.dropna(axis=0, how='all').shape
# drops column which have any row of NA

[set(df.columns) - set(df.dropna(axis=1, how='any').columns)]
# replace with mean of that column
# can put any specific value also
# would not work for columns with string type like Cabin

df.fillna(np.mean)[:1]
Image("../input/images/combine.png", width=500)
data1 = pd.DataFrame({'x1': list('abc'), 'x2': [11.432, 1.303, 99.906]})
data2 = pd.DataFrame({'x1': list('abd'), 'x3': [20.784,  np.NaN, 20.784]})
data1
data2
# inner join when both table have that key (like sql)

data1.merge(data2, how='inner', on='x1')
# outer joins on all keys in both df and creates NA

data1.merge(data2, how='outer', on='x1')
# if columns overlap, have to specify suffix as it makes for all

data1.join(data2, on='x1', how='left', lsuffix='L')
# join over axis=0, i.e rows combine 
# also adds all columns with na

pd.concat([data1, data2], axis=0)
pd.concat([data1, data2], axis=0, ignore_index=True)
data2.loc[3] = ['g', 500] # adding new row
data2
# join over axis=1, i.e columns combine 

pd.concat([data1, data2], axis=1)
pd.to_datetime('2018-2-19')
# gives datetimeindex format

pd.date_range('2018-4-18', periods=6, freq='d')
data1['date'] = pd.date_range('2018-4-18', periods=3, freq='d')
data1
data1.date
pd.DatetimeIndex(data1.date)
# index = new index, columns = new_columns, values = values to put

df.pivot(index='Sex', columns = 'PassengerId', values = 'Age')
df.stack()
list(df.Sex.iteritems())[:5]
list(df.iterrows())[0]
# function squares when type(x) = float, cubes when type(x) = int, return same when other

f = lambda x: x**2 if type(x) == float else x**3 if type(x) == int else x

# whole series is passed

df.Fare.apply(f)[:3]
# elements are passed

df.applymap(f)[:3]
# converts all rows into lower

df.Name.str.lower().head()
# converts all rows into upper 

df.Sex.str.upper().head()
# counts all the characters including spaces

df.Name.str.len().head()
# splits strings in each row over whitespaces ()
# expand=True : expand columns
# pat = regex to split on

df.Name.str.split(pat=',',expand=True).head().rename(columns={0:'First_Name', 1: 'Last_Name'})
# splits strings in each row over whitespaces ()
# expand=False : doesn't expand columns
# pat = regex to split on

df.Name.str.split(expand=False).head()
# replace Mr. with empty space

df.Name.str.replace('Mr.', '').head()
# get() is used to get particular row of split

df.Name.str.split().get(1)
df.Name[:10]
# Extract just last name

df.Name.str.extract('(?P<Last_Name>[a-zA-Z]+)', expand=True).head()