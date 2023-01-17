import pandas as pd

import numpy as np # necessity as pandas is built on np

import matplotlib.pyplot as plt

import seaborn as sns
data = {'Country': ['Belgium', 'India', 'Brazil'],

 'Capital': ['Brussels', 'New Delhi', 'Brasília'],

 'Population': [11190846, 1303171035, 207847528]}

df_sample = pd.DataFrame(data)

df_sample
df = pd.read_csv('../input/train.csv') # read csv file
df.shape
df.head() # see top 5 rows of data
df.dtypes # see datatype of each variable
df.columns # column names
df.Sex.unique()

df.nunique() # unique value for each variable

(df.nunique())["Sex"]

sum(df.PassengerId.duplicated()) # check if there is a duplicate: should be 0 if not.

df.loc[:,['Sex', 'Embarked']].drop_duplicates()
df.info() # not null part is very useful to see how many nulls are there in data
df.describe()

df.Sex.value_counts()

df.Fare.mean()
df.assign(FareSquared = df["Fare"]**2)[:2]

new_df = df.copy()

new_df.Sex = new_df.Sex.fillna("male")

new_df = new_df.assign(Male = (new_df.Sex == "male"))

new_df.Male = new_df.Male.astype(int)

# append for raws
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()



embarked_encoded = pd.DataFrame(data = ['Apple', 'Orange', 'Banana'], columns = ["names"])

embarked_encoded["cat"] = label_encoder.fit_transform(embarked_encoded.names)

embarked_encoded
df.iloc[1:4, 2:6] # indexes are maintained. Can reset_index() to start index from 0  
df.loc[1:2,'Name':"Age"] # here row indexes are numbers but column indexes are name of columns 
# select rows with either sex as female or Pclass as 1

df[(df.Sex == 'female') | (df.iloc[:,2] == 1) ].iloc[:3] # () are important
# first 3 rows of gives all columns which have all string values or all int > 1 values

df.loc[:,(df > 1).all()][:3]

# first 3 rows of all columns which have all not null values

df.loc[:,(df.notnull().all() )][:3]

# first 3 rows of all columns which have atleast 1 null value

df.loc[:, df.isnull().any()][:3]
# fraction of males with Age > 25, df.shape[0] -> number of rows 

sum((df.Age > 25) & (df.Sex == 'male'))/df.shape[0]

# number of people who survived and were not in class 3

sum((df.Survived != 0) & (~(df.Pclass == 3)) )
#isin

df[df.Pclass.isin([0,1])].head()
# filter all rows which have Age > Passenger ID

df.query('Age > PassengerId')



# filter only sex and age columns (first 2 rows)

df.filter(items=['Age', 'Sex'])[:2]



# filter only 0 and 5 row index

df.filter(items=[0,5], axis=0)



# first 2 rows of column names ending with "ed" (think of past tense)

df.filter(like = 'ed', axis=1)[:2]
df.sort_values(by=['Pclass', 'Age'], ascending=False,na_position='first')
# setting the index as the ticket column

df.set_index('Ticket')[:2]
# can set multiple columns as index also. Just pass them in list

# Setting Ticket and Name as index

df.set_index(['Ticket', 'Name'])[:2]
df_index = df.set_index(['Ticket', 'Name'])
df_index.reset_index()[:2]
df.rename(columns={'Name': 'Whats_name', 'Fare':'Price'})[:2]
# can use some mapper function also. default axis='index' (0)

df.rename(mapper=str.lower, axis='columns')[:2]
df.groupby(by = ['Sex', 'Survived']).mean() #.loc[:,'Age']
# can group by indexes also by using levels= 

# useful when we have multindexes

# can use agg function with lambda func



df_index = df.set_index(['Sex', 'Pclass'])

df_index.groupby(level=[0,1]).agg({'Fare': lambda x: sum(x)/len(x), # this is also just mean actually

                                  'Age' : np.mean})
df_index.groupby(level=[0,1]).transform(lambda x: sum(x)/len(x)).head()
df.groupby(['Pclass', 'Sex']).apply(lambda df: df.loc[df.Fare.idxmax()])
df.groupby(['Pclass', 'Sex']).agg([len, min, max]).Fare
df[pd.isnull(df.Age)]
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

df.Age.fillna("Unknown")
data1 = pd.DataFrame({'x1': list('abc'), 'x2': [11.432, 1.303, 99.906]})
data2 = pd.DataFrame({'x1': list('abd'), 'x3': [20.784,  np.NaN, 20.784]})
data1
data2
merged_inner = pd.merge(left=data1, right=data2, left_on='x1', right_on='x1')

merged_inner
merged_left = pd.merge(left=data1, right=data2, how='left', left_on='x1', right_on='x1')

merged_left
#with index as key

inner_concat = pd.concat([data1, data2], axis=1, join='inner')

inner_concat
# inner join when both table have that key (like sql)



data1.merge(data2, how='inner', on='x1')
# outer joins on all keys in both df and creates NA



data1.merge(data2, how='outer', on='x1')
# if columns overlap, have to specify suffix as it makes for all



data1.join(data2, on='x1', how='left', lsuffix='L')
# Stack the DataFrames on top each other

vertical_stack = pd.concat([data1, data2], axis=0)

vertical_stack
# Place the DataFrames side by side

horizontal_stack = pd.concat([data1, data2], axis=1)

horizontal_stack
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
# Stack: convert whole df into 1 long format

df.stack()
list(df.Sex.iteritems())[:5]
list(df.iterrows())[0]
# function squares when type(x) = float, cubes when type(x) = int, return same when other



f = lambda x: x**2 if type(x) == float else x**3 if type(x) == int else x

# whole series is passed



df.Fare.apply(f)[:3]

Fare_mean = df.Fare.mean()

def remean_Fare(row):

    row.Fare = row.Fare - Fare_mean

    return row

(df.apply(remean_Fare, axis='columns')).Fare.mean()
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
df[["Sex", "Survived", "Fare"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
g = sns.FacetGrid(df, col='Survived')

g.map(plt.hist, 'Age', bins=20)
grid = sns.FacetGrid(df, col='Survived', row='Pclass', size=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend();
grid = sns.FacetGrid(df, col='Pclass', hue='Survived')

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend();
df['AgeBand'] = pd.cut(df['Age'], 5)

df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)
df.loc[ df['Age'] <= 16, 'Age'] = 0

df.loc[(df['Age'] > 16) & (df['Age'] <= 32), 'Age'] = 1

df.loc[(df['Age'] > 32) & (df['Age'] <= 48), 'Age'] = 2

df.loc[(df['Age'] > 48) & (df['Age'] <= 64), 'Age'] = 3

df.loc[ df['Age'] > 64, 'Age']

df.head()