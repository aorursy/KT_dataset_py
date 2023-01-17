# import pandas and numpy

import pandas as pd

import numpy as np
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

data = '/kaggle/input/black-friday/train.csv'

df = pd.read_csv(data)
type(df)
df.shape
df.head()
df.info()
df.isnull().sum()
df.isna().sum()
df = df.fillna(method = 'pad')
df.isnull().sum()
df[['Product_Category_2', 'Product_Category_3']].head()
df = df.fillna(method = 'backfill')
df.isnull().sum()
#assert that there are no missing values in the dataframe

assert pd.notnull(df).all().all()

# make a copy of dataframe
df1 = df.copy()
# select first row of dataframe

df1.loc[0]
#select first five rows for a specific column

df1.loc[:,'Purchase'].head()
#select first row of dataframe

df1.iloc[0]
#select last row of dataframe

df1.iloc[-1]
# get index of first occurence of maximum Purchase value 

df1['Purchase'].idxmax()
# get the row with the maximum Purchase value 

df1.loc[df1['Purchase'].idxmax()]
# get value at 1st row and Purchase column pair

df1.at[1, 'Purchase']
# get value at 1st row and 11th column pair

df1.iat[1, 11]
# make a copy of dataframe df

df2 = df.copy()
df2.head()
# get the purchase amount with a given user_id and product_id

df2.loc[((df2['User_ID'] == 1000001) & (df2['Product_ID'] == 'P00069042')), 'Purchase']
values=[1000001,'P00069042','F',0-17,10,'A',2,0,3,6,14,8370]

df2_indexed=df2.isin(values)


df2_indexed.head(10)
row_mask = df2.isin(values).any(1)

df[row_mask]
df2_where=df2.where(df2 == 0)


(df2_where).head(10)
df2.query('(Product_Category_1 > Product_Category_2) & (Product_Category_2 > Product_Category_3)')
# let's create a new dataframe 

food = pd.DataFrame({'Place':['Home', 'Home', 'Hotel', 'Hotel'],
                   'Time': ['Lunch', 'Dinner', 'Lunch', 'Dinner'],
                   'Food':['Soup', 'Rice', 'Soup', 'Chapati'],
                   'Price($)':[10, 20, 30, 40]})

food
food_indexed1=food.set_index('Place')

food_indexed1
food_indexed2=food.set_index(['Place', 'Time'])

food_indexed2
food_indexed2.reset_index()
sales=pd.DataFrame([['books','online', 200, 50],['books','retail', 250, 75], 
                    ['toys','online', 100, 20],['toys','retail', 140, 30],
                    ['watches','online', 500, 100],['watches','retail', 600, 150],
                    ['computers','online', 1000, 200],['computers','retail', 1200, 300],
                    ['laptops','online', 1100, 400],['laptops','retail', 1400, 500],
                    ['smartphones','online', 600, 200],['smartphones','retail', 800, 250]],
                    columns=['Items', 'Mode', 'Price', 'Profit'])


sales
sales1=sales.set_index(['Items', 'Mode'])

sales1
# View index

sales1.index
# Swap the column  in multiple index

sales2=sales1.swaplevel('Mode', 'Items')

sales2
# sort the dataframe df2 by label

df2.sort_index()
df2.sort_values(by=['Product_Category_1'])
df3 = df.copy()

df3.dtypes
df3['Gender'].describe()
df3['Age'].describe()
df3['City_Category'].describe()
df3['Gender'].unique()
df3['Age'].unique()
df3['Gender'].value_counts()
df3['City_Category'].value_counts()
df3['Gender'].value_counts(ascending=True)
df3['City_Category'].value_counts(ascending=True)
df4=df.copy()

df4.max(0)
df4.describe()
df5=df.copy()


# view the covariance

df5.cov()
# view the correlation

df5.corr()
# view the top 25 rows of ranked dataframe

df5.rank(1).head(25)
df6=df.copy()


df6['Purchase'].aggregate(np.sum)
df6['Purchase'].aggregate([np.sum, np.mean])
df6[['Product_Category_1', 'Product_Category_2', 'Product_Category_3']].aggregate(np.mean)
df6[['Product_Category_1', 'Product_Category_2', 'Product_Category_3']].aggregate([np.sum, np.mean])
df6.aggregate({'Product_Category_1' : np.sum ,'Product_Category_2' : np.mean})
df8=df.copy()

df8.groupby('Gender')
# view groups of Gender column

df8.groupby('Gender').groups
# apply aggregation function sum with groupby

df8.groupby('Gender').sum()
# alternative way to apply aggregation function sum

df8.groupby('Gender').agg(np.sum)
# attribute access in python pandas

df8_grouped = df8.groupby('Gender')

print(df8_grouped.agg(np.size))
df8.groupby('Gender')['Purchase'].agg([np.sum, np.mean])
df9=df.copy()


score = lambda x: (x - x.mean()) / x.std()*10


print(df9.groupby('Gender')['Purchase'].transform(score).head(5))
df10=df.copy()


df10.groupby('Gender').filter(lambda x: len(x) > 4)
# let's create two dataframes

batsmen = pd.DataFrame({
   'id':[1,2,3,4,5],
   'Name': ['Rohit', 'Dhawan', 'Virat', 'Dhoni', 'Kedar'],
   'subject_id':['sub1','sub2','sub4','sub6','sub5']})

bowler = pd.DataFrame(
   {'id':[1,2,3,4,5],
   'Name': ['Kumar', 'Bumrah', 'Shami', 'Kuldeep', 'Chahal'],
   'subject_id':['sub2','sub4','sub3','sub6','sub5']})


print(batsmen)


print(bowler)
# merge two dataframes on a key

pd.merge(batsmen, bowler, on='id')
# merge two dataframes on multiple keys

pd.merge(batsmen, bowler, on=['id', 'subject_id'])
# left join

pd.merge(batsmen, bowler, on='subject_id', how='left')
# right join

pd.merge(batsmen, bowler, on='subject_id', how='right')
# outer join

pd.merge(batsmen, bowler, on='subject_id', how='outer')
# inner join

pd.merge(batsmen, bowler, on='subject_id', how='inner')
# let's create two dataframes

batsmen = pd.DataFrame({
   'id':[1,2,3,4,5],
   'Name': ['Rohit', 'Dhawan', 'Virat', 'Dhoni', 'Kedar'],
   'subject_id':['sub1','sub2','sub4','sub6','sub5']})

bowler = pd.DataFrame(
   {'id':[1,2,3,4,5],
   'Name': ['Kumar', 'Bumrah', 'Shami', 'Kuldeep', 'Chahal'],
   'subject_id':['sub2','sub4','sub3','sub6','sub5']})


print(batsmen)


print(bowler)
# concatenate the dataframes


team=[batsmen, bowler]

pd.concat(team)
# associate keys with the dataframes

pd.concat(team, keys=['x', 'y'])
pd.concat(team, keys=['x', 'y'], ignore_index=True)
pd.concat(team, axis=1)
batsmen.append(bowler)
df11=df.copy()

df11.columns
df12=(pd.melt(frame=df11, id_vars=['User_ID','Product_ID', 'Gender','Age','Occupation','City_Category',
                             'Marital_Status','Purchase'],                          
                    value_vars=['Product_Category_1','Product_Category_2','Product_Category_3'], 
                    var_name='Product_Category', value_name='Amount'))

df12.head(10)
df13=df12[['Product_Category', 'Amount']]

df14=df13.pivot(index=None, columns='Product_Category', values='Amount')

df14.head(25)
cols=pd.MultiIndex.from_tuples([('weight', 'kg'), ('weight', 'pounds')])

df15=pd.DataFrame([[75,165], [60, 132]],
                 index=['husband', 'wife'],
                 columns=cols)

df15
df16=df15.stack()

df16
df16.unstack()
# display maximum rows

pd.get_option("display.max_rows")
# display maximum columns

pd.get_option("display.max_columns")
# set maximum rows

pd.set_option("display.max_rows", 80)

pd.get_option("display.max_rows")
# set maximum columns

pd.set_option("display.max_columns", 30)

pd.get_option("display.max_columns")
# display maximum rows

pd.reset_option("display.max_rows")

pd.get_option("display.max_rows")
# display maximum columns

pd.reset_option("display.max_columns")

pd.get_option("display.max_columns")
# description of the display maximum rows parameter

pd.describe_option("display.max_rows")
# set the parameter value with option_context

with pd.option_context("display.max_rows",10):
   print(pd.get_option("display.max_rows"))
   print(pd.get_option("display.max_rows"))