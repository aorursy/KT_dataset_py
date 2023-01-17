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
# Pandas is a newer package built on top of NumPy, and provides an efficient implementation of a DataFrame. DataFrames are essentially multidimen‐

# sional arrays with attached row and column labels, and often with heterogeneous types and/or missing data. 



# Once Pandas is installed, you can import it and check the version



import pandas

pandas.__version__



# Just as we generally import NumPy under the alias np, we will import Pandas under the alias pd:



import pandas as pd
# The Pandas Series Object



# A Pandas Series is a one-dimensional array of indexed data. It can be created from a list or array as follows



data=pd.Series([0.25,0.5,0.75,1])

data



# As we see in the preceding output, the Series wraps both a sequence of values and a sequence of indices, which we can access with the values and index attributes. The

# values are simply a familiar NumPy array

data.values
data.shape
data.index
# Like with a NumPy array, data can be accessed by the associated index via the familiar Python square-bracket notation



data[1]
data[3]
data[1:3]
# Series as generalized NumPy array

# From what we’ve seen so far, it may look like the Series object is basically inter‐changeable with a one-dimensional NumPy array. The essential difference is the pres‐

# ence of the index: while the NumPy array has an implicitly defined integer index used to access the values, the Pandas Series has an explicitly defined index associated with the values.



data1=pd.Series([0.75,.50,.75,1],index=['a','b','c','d'])

data1
data1['b']
# We can even use noncontiguous or nonsequential indices



data2=pd.Series([0.25,.5,0.75,1],index=[2,5,7,3])

data2
# Series as specialized dictionary



Population_dict={'California':123543,'Texas':87451,'Boston':986734,'Newyork':907856}



Population=pd.Series(Population_dict)



Population
# By default, a Series will be created where the index is drawn from the sorted keys.

# From here, typical dictionary-style item access can be performed



Population['California']
# Unlike a dictionary, though, the Series also supports array-style operations such as slicing



Population['California':'Texas']
pd.Series([2,4,7])
# data can be a scalar, which is repeated to fill the specified index



pd.Series(5, index=[100, 200, 300])
# data can be a dictionary, in which index defaults to the sorted dictionary keys:



pd.Series({2:'a', 1:'b', 3:'c'})
# In each case, the index can be explicitly set if a different result is preferred:



pd.Series({2:'a',4:'b',5:'c'},index=[3,2])
# DataFrame can be thought of either as a generalization of a NumPy array, or as a specialization of a Python dictionary. 



area_dict={'California': 12345,'Boston':6745,'newyork':9078,'newtown':23126}



area=pd.Series(area_dict)



area
# Now that we have this along with the population Series from before, we can use a dictionary to construct a single two-dimensional object containing this information:



states=pd.DataFrame({'population': Population,'area': area})

states



# Like the Series object, the DataFrame has an index attribute that gives access to the index labels

states.index
# Additionally, the DataFrame has a columns attribute, which is an Index object holding the column labels

states.columns



# Thus the DataFrame can be thought of as a generalization of a two-dimensional NumPy array, where both the rows and columns have a generalized index for access‐ing the data.
# DataFrame as specialized dictionary



# Similarly, we can also think of a DataFrame as a specialization of a dictionary. Where a dictionary maps a key to a value, a DataFrame maps a column name to a Series of

# column data. For example, asking for the 'area' attribute returns the Series object containing the areas we saw earlier



states['area']

# Notice the potential point of confusion here: in a two-dimensional NumPy array, data[0] will return the first row. For a DataFrame, data['col0'] will return the first column.



# Constructing DataFrame objects



# A Pandas DataFrame can be constructed in a variety of ways. Here we’ll give several examples.



# From a single Series object. A DataFrame is a collection of Series objects, and a singlecolumn DataFrame can be constructed from a single Series



pd.DataFrame(Population,columns=['Population'])

# From a list of dicts. Any list of dictionaries can be made into a DataFrame. We’ll use a simple list comprehension to create some data



data=[{'a':i,'b':2*i} for i in range(3)]

pd.DataFrame(data)
# Even if some keys in the dictionary are missing, Pandas will fill them in with NaN (i.e.,“not a number”) values



pd.DataFrame([{'a': 1, 'b': 2}, {'b': 3, 'c': 4}])
# From a dictionary of Series objects, As we saw before, a DataFrame can be constructed from a dictionary of Series objects as well



pd.DataFrame({'Population': Population,'area': area})





# From a two-dimensional NumPy array



# Given a two-dimensional array of data, we can create a DataFrame with any specified column and index names. If omitted, an integer index will be used for each



pd.DataFrame(np.random.rand(3,2),columns=['foo','bar'],index=['a','b','c'])

# From a NumPy structured array



# A Pandas DataFrame operates much like a structured array, and can be created directly from one



A=np.zeros(3,dtype=[('A','i8'),('B','f8')])



pd.DataFrame(A)
# The Pandas Index Object



ind=pd.Index([2,3,5,7,11])

ind
# Index as immutable array



# The Index object in many ways operates like an array. For example, we can use stan‐ dard Python indexing notation to retrieve values or slices



ind[1]
ind[::2]
# Index objects also have many of the attributes familiar from NumPy arrays:



print(ind.size,ind.shape,ind.ndim,ind.dtype)
# Data Indexing and Selection



import pandas as pd

data = pd.Series([0.25,0.50,0.75,1],index=['a','b','c','d'])

data
data['b']
# We can also use dictionary-like Python expressions and methods to examine the keys/indices and values

'a' in data
data.keys()
list(data.items())
# Series objects can even be modified with a dictionary-like syntax. Just as you can extend a dictionary by assigning to a new key, you can extend a Series by assigning to a new index value



data['e']=1.25

data
# Series as one-dimensional array



# A Series builds on this dictionary-like interface and provides array-style item selec‐ tion via the same basic mechanisms as NumPy arrays—that is, slices, masking, and

# fancy indexing. Examples of these are as follows



# # slicing by explicit index



data['a':'c']
# slicing by implicit integer index



data[0:2]
# # masking

data[(data>0.3) & (data<0.8)]
# # fancy indexing



data[['a', 'e']]
# Indexers: loc, iloc, and ix



data = pd.Series(['a', 'b', 'c'], index=[1, 3, 5])

data
# explicit index when indexing



data[1]
# implicit index when slicing

data[1:3]
# First, the loc attribute allows indexing and slicing that always references the explicit index



data.loc[1]
data.loc[1:3]
# The iloc attribute allows indexing and slicing that always references the implicit Python-style index:

data.iloc[1:3]
data.iloc[1]
# DataFrame as a dictionary



area=pd.Series({'California': 423967, 'Texas': 695662,'New York': 141297, 'Florida': 170312,'Illinois': 149995})

pop = pd.Series({'California': 38332521, 'Texas': 26448193,'New York': 19651127, 'Florida': 19552860,'Illinois': 12882135})



data=pd.DataFrame({'area':area,'pop':pop})

data



data['area']
data.area
data.area is data['area']
data.pop is data['pop']
# Like with the Series objects discussed earlier, this dictionary-style syntax can also be used to modify the object, in this case to add a new column:



data['density']=data['pop']/data['area']

data

# DataFrame as two-dimensional array



# As mentioned previously, we can also view the DataFrame as an enhanced twodimensional array. We can examine the raw underlying data array using the values attribute



data.values
# we can transpose the full DataFrame to swap rows and columns:

data.T

# When it comes to indexing of DataFrame objects, however, it is clear that the dictionary-style indexing of columns precludes our ability to simply treat it as a

# NumPy array. In particular, passing a single index to an array accesses a row



data.values[0]
data['area']
data.iloc[:3,:2]   # iloc= index location
data.loc[:'Illinois', :'pop']

# in the loc indexer we can combine masking and fancy indexing as in the following:



data.loc[data.density>100,['pop','density']]
# Any of these indexing conventions may also be used to set or modify values; this is done in the standard way that you might be accustomed to from working with NumPy



data.iloc[0,2]=90

data
# Additional indexing conventions



# while index‐ing refers to columns, slicing refers to rows:



data['Florida':'Illinois']
# Such slices can also refer to rows by number rather than by index:



data[1:3]
# direct masking operations are also interpreted row-wise rather than column-wise:



data[data.density>100]
# Ufuncs: Index Preservation

# Let’s start by defining a simple Series and DataFrame on which to demonstrate this:



import pandas as pd

import numpy as np



rng = np.random.RandomState(42)

ser = pd.Series(rng.randint(0, 10, 4))

ser



df = pd.DataFrame(rng.randint(0, 10, (3, 4)),columns=['A', 'B', 'C', 'D'])

df
# If we apply a NumPy ufunc on either of these objects, the result will be another Pandas object with the indices preserved



np.exp(ser)
# Or, for a slightly more complex calculation

np.sin(df*np.pi/4)
# UFuncs: Index Alignment



# For binary operations on two Series or DataFrame objects, Pandas will align indices in the process of performing the operation. This is very convenient when you are

# working with incomplete data, 


