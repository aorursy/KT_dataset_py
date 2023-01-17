import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.express  as px

import matplotlib.pyplot as plt

import plotly.graph_objects as go

import seaborn as sns



from matplotlib import cm

plt.style.use('ggplot')

pd.__version__
data = pd.Series([0.25, 0.5, 0.75, 1.0])

print(data)

print("-"*50)

# used to print the values

print(data.values)

print("-"*50)

# print particular value

print(data.values[0])

print("-"*50)

# slice with indexes

print(data[1:3])

print("-"*50)

# print index 

print(data.index)
data = pd.Series([0.25, 0.5, 0.75, 1.0],

index=['a', 'b', 'c', 'd'])

print(data)

print("-"*50)

print(data['a'])

print("-"*50)

print(data['a':'c'])
# convert a dict to pandas Series

age_dict = {'Ram': 17,

'Mathew': 20,

'Ali': 17,

'Raj': 19,

'Sharad': 15}

age = pd.Series(age_dict)

print(age)
# scalar 

data = pd.Series(2, index=['a','b','c'])

print(data)
# convert a dict to pandas Series

marks_dict = {'Ram': 90,

'Mathew': 80,

'Ali': 67,

'Raj': 49,

'Sharad': 35}

marks = pd.Series(marks_dict)

print(marks)
student_data = pd.DataFrame({'age': age, 'marks': marks})

print(student_data)

print("-"*50)

# print the index

print(student_data.index)

print("-"*50)

print(student_data.columns)
ind = pd.Index([2, 3, 5, 7, 11])

print(ind)
# if you try to modify the value , you can't . it will throw an exception here.

# ind[0] = 1

print("-"*50)

print(ind[0])

print("-----size-----shape-------ndim------dtype-----")

# it has also attributes that are similar to numpy array.

print(ind.size, ind.shape, ind.ndim, ind.dtype)
indA = pd.Index([1, 3, 5, 7, 9])

indB = pd.Index([2, 3, 5, 7, 11])
# intersection

print(indA & indB)

print("-"*50)

# union

print(indA | indB)



# symmetric

print("-"*50)

print(indA ^ indB)

data = pd.Series([0.25, 0.5, 0.75, 1.0],

index=['a', 'b', 'c', 'd'])

print(data)

print("-"*50)

print(data.keys())

print("-"*50)

print(data['a'])
print('a' in data)

print("-"*50)

print(list(data.items()))

print("-"*50)

data['e'] = 1.25

print(data)

print("-"*50)

# slicing by explicit index

print(data['a':'c'])

# slicing by implicit integer index

print("-"*50)

print(data[0:2])

print("-"*50)

# masking

print(data[(data > 0.3) & (data < 0.8)])

# fancy indexing

print("-"*50)

print(data[['a', 'e']])

df1 = pd.DataFrame({'lkey': ['foo', 'bar', 'baz', 'foo'],

                    'value': [1, 2, 3, 5]})

df2 = pd.DataFrame({'rkey': ['foo', 'bar', 'baz', 'foo'],

                    'value': [5, 6, 7, 8]})

df1
df2
df1.merge(df2, left_on='lkey', right_on='rkey')
netflix_data = pd.read_csv('/kaggle/input/netflix-shows/netflix_titles.csv')
# it will display the with rows , if not mentioned it will display only 5 rows.

netflix_data.head().style.set_properties(**{'background-color': '#161717','color': '#30c7e6','border-color': '#8b8c8c'})
# display the numerical values count, mean etc

netflix_data.describe()
I = netflix_data.dtypes[netflix_data.dtypes == 'object'].index
I
netflix_data[I].head().style.set_properties(**{'background-color': '#161717','color': '#30c7e6','border-color': '#8b8c8c'})
# displays count, unqiue, top and freq

netflix_data[I].describe()
netflix_data.loc[0]
netflix_data.iloc[0]
# finding out the null values

print(netflix_data.isnull().sum())

print("-"*50)

print(netflix_data.notnull().sum())
# this will drop the null values rows completly

netflix_data = netflix_data.dropna()

# netflix_data.dropna(axis='columns', how='all')
netflix_data['type'].unique()
temp_df = pd.DataFrame(netflix_data['type'].value_counts()).reset_index()



fig = go.Figure(data=[go.Pie(labels=temp_df['index'],

                             values=temp_df['type'],

                             hole=.7,

                             title = '% of Netflix by Type',

                             marker_colors = px.colors.sequential.Blues_r,

                            )

                     

                     ])

fig.update_layout(title='Netflix Shows')

fig.show()
netflix_data['rating']
ax = sns.countplot(x="rating", data=netflix_data)
ax = sns.countplot(x="type", data=netflix_data)
netflix_data.head()
# get movies with duration

movie_len = netflix_data[netflix_data['type'] == 'Movie']
# find movie duration 

ax = sns.countplot(x="release_year", data=movie_len)