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
# Creating Dataframe
df = pd.DataFrame({
    'A': ['foo', 'bar', 'foo', 'bar','foo', 'bar', 'foo', 'foo'],
    'B': ['one', 'one', 'two', 'three','two', 'two', 'one', 'three'],
    'C': np.random.randn(8),
    'D': np.random.randn(8)})
df
df.groupby("A")  # grouping df according to column "A"
df.groupby("A").sum()  #grouping according to A and summation of each kinds under A column
df.groupby(["A","B"])         # grouping df according to both columns "A" and "B"
df.groupby(["A","B"]).sum()   # summation
df2 = df.set_index(['A',"B"])   # setting index: "A" and "B" to be assigned as index
df2
df2.groupby(level=df2.index.names.difference(['B']))        # grouping according to index A  (difference B means: all indexes other than B)
df2.groupby(level=df2.index.names.difference(['B'])).sum()  # summation of data according to index "A"
# This will split dataframe on its indexes(rows)
df2.groupby(level="A").sum()
df2.groupby(level=["A","B"]).sum()
df2.groupby(level=0).sum()   # level 0 --> A and level 1 --> B

# Creating a series
lst = [1, 2, 3, 1, 2, 3]
s = pd.Series([1, 2, 3, 10, 20, 30],lst)  # set lst as index of s
s
# groupig according to index:
s.groupby(level=0).sum()  
# Grouping according to index and getting first group
s.groupby(level=0).first()
# Grouping according to index and getting last group
s.groupby(level=0).last()
df
df.groupby("B").sum()  # B colum is sorted during grouping
df.groupby("B", sort=False).sum()  # B column is not sorted during grouping
df
df.groupby("B").get_group("one")
df.groupby("B", sort=False).get_group("one")
df.groupby(["A","B"]).get_group(("bar","one"))

df
df.groupby("A").sum()
df.groupby("A").groups  # outputs a dictionary: keys are items of grouped column and values are indexes 
len(df.groupby("A").groups)
arrays = [['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'],
          ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']]


index = pd.MultiIndex.from_arrays(arrays, names=['first', 'second'])

df1 = pd.DataFrame({'A': [1, 1, 1, 1, 2, 2, 3, 3],
                   'B': np.arange(8)},
                  index=index)
df1
# grouping according to 2nd index and column A
df1.groupby([pd.Grouper(level=1), "A"]).sum()
df1.groupby([pd.Grouper(level="second"), "A"]).sum()
df1.groupby(["second","A"]).sum()
df1
df1.groupby("first").sum()
# Selecting column A of grouped dataframe 
df1.groupby("first")["A"].sum()
df1.groupby("first").A.sum()
df1
for each in df1.groupby("second"):
    print(each)
for name,group in df1.groupby("second"):
    print("grouped item in rows:",name)
    print(group)
    print("---------------")
df1.groupby(["second", "A"]).count()
for name,group in df1.groupby(["second", "A"]):
    print(name)
    print(group)
    print("")
df
# Grouping according to A and B columns and summation
df.groupby(["A","B"]).sum()
df.groupby(["A","B"]).sum().reset_index()
df.groupby(["A","B"], as_index=False).sum()
df.groupby(["A","B"], as_index=False).agg(np.sum)
df.groupby("A").agg(np.sum)
df
df.groupby("A").agg(np.size)  # means: there are 3 rows of bar, 5 rows of foo
df.groupby("A").size()
# for whole grouped object
df.groupby("A").agg([np.sum, np.mean, np.std, np.size])
# for a specific column from grouped object
df.groupby("A")["C"].agg([np.sum, np.mean, np.std, np.size])
# Name of columns can be changed after aggregation
df.groupby("A")["C"].agg([np.size, np.sum, np.std]).rename(columns={"size":"SIZE",
                                                                   "sum":"SUM",
                                                                   "std":"STD"})
animals = pd.DataFrame({'kind': ['cat', 'dog', 'cat', 'dog'],
                         'height': [9.1, 6.0, 9.5, 34.0],
                         'weight': [7.9, 7.5, 9.9, 198.0]})
animals
animals.groupby("kind").agg(min_height=pd.NamedAgg(column="height",aggfunc="min"),
                            min_weight=pd.NamedAgg(column="weight",aggfunc="min"),
                            average_weight=pd.NamedAgg(column="weight",aggfunc="mean"))

animals.groupby("kind").agg(min_height=("height", "min"),
                         min_weight=("weight", "min"),
                         average_weight=("weight", np.mean))
animals.groupby("kind").height.agg(min_height="min", max_height="max")
df
df.groupby("A").agg({"C":"sum", # or "C":"sum"      
                     "D":lambda x: np.std(x)})
# our example:
df
df.groupby("A").groups
    
df.groupby("A").mean()
df.groupby("A").transform(lambda x:x.mean())
df.groupby("A")["C"].groups
df.groupby("A")["C"].mean()
# creating dataframe
df_re = pd.DataFrame({'A': [1] * 10 + [5] * 10,
                      'B': np.arange(20)})
df_re
df_re.groupby("A").sum()
df_re.groupby('A').expanding().sum()
sf = pd.Series([1, 1, 2, 3, 3, 3])
sf
sf.groupby(sf).filter(lambda x: x.sum()>2)
dff = pd.DataFrame({'A': np.arange(8), 'B': list('aabbbbcc')})
dff
dff.groupby('B').filter(lambda x: len(x)>2)
dff.groupby('B').filter(lambda x: len(x) > 2, dropna=False)
dff['C'] = np.arange(8)
dff
dff.groupby('B').filter(lambda x: len(x['C']) > 2)