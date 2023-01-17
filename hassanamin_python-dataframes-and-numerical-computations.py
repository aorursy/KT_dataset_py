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
import numpy as np

p = [[1, 0], [0, 1]]

q = [[1, 2], [3, 4]]

print("original matrix:")

print(p)

print(q)

result1 = np.dot(p, q)

print("Result of the said matrix multiplication:")

print(result1)

import pandas as pd

import numpy as np



exam_data  = {'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],

        'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],

        'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],

        'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']}

labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']



df = pd.DataFrame(exam_data , index=labels)

print(df)
import pandas as pd

import numpy as np

exam_data  = {'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],

        'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],

        'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],

        'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']}

labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']



df = pd.DataFrame(exam_data , index=labels)

print("Rows where score between 15 and 20 (inclusive):")

print(df[df['score'].between(15, 20)])

import pandas as pd

import numpy as np



exam_data  = {'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],

        'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],

        'attempts' : [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],

        'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']}

labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']



df = pd.DataFrame(exam_data , index=labels)

print("Number of attempts in the examination is greater than 2:")

print(df[df['attempts'] > 2])

import pandas as pd

import numpy as np

exam_data  = {'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],

        'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],

        'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],

        'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']}

labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']



df = pd.DataFrame(exam_data , index=labels)

print("\nOriginal data frame:")

print(df)

print("\nChange the score in row 'd' to 11.5:")

df.loc['d', 'score'] = 11.5

print(df)
import pandas as pd

import numpy as np

exam_data  = {'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],

        'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],

        'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],

        'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']}

labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']



df = pd.DataFrame(exam_data , index=labels)

print("\nSum of the examination attempts by the students:")

print(df['attempts'].sum())
import pandas as pd

import numpy as np

exam_data  = {'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],

        'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],

        'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],

        'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']}

labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']



df = pd.DataFrame(exam_data , index=labels)

print("\nMean score for each different student in data frame:")

print(df['score'].mean())
import pandas as pd

import numpy as np

exam_data  = {'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],

        'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],

        'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],

        'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']}

labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']

df = pd.DataFrame(exam_data , index=labels)

print("Orginal rows:")

print(df)

result_sort=df.sort_values(by=['name', 'score'], ascending=[False, True])

print("Sort the data frame first by ‘name’ in descending order, then by ‘score’ in ascending order:")
import pandas as pd

df1 = pd.DataFrame({'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],

'city': ['California', 'Los Angeles', 'California', 'California', 'California', 'Los Angeles', 'Los Angeles', 'Georgia', 'Georgia', 'Los Angeles']})

print("Original Dataframe \n")

print(df1)

g1 = df1.groupby(["city"]).size().reset_index(name='Number of people')

print("\nNew Dataframe \n")

print(g1)
import pandas as pd

import numpy as np

exam_data = {'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],

        'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],

        'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],

        'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']}

df = pd.DataFrame(exam_data)

print("Original DataFrame")

print(df)

print("\nNumber of NaN values in one or more columns:")

print(df.isnull().values.sum())
import pandas as pd

import numpy as np

exam_data = {'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],

        'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],

        'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],

        'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']}

df = pd.DataFrame(exam_data)

print("Original DataFrame")

print(df)

print("\nAfter removing first and second rows")

df = df.drop([0, 1])

print(df)

print("\nReset the Index:")

df = df.reset_index()

print(df)
import pandas as pd

d = {'col1': [1, 2, 3], 'col2': [4, 5, 6], 'col3': [7, 8, 9]}

df = pd.DataFrame(data=d)

print("Original DataFrame")

print(df)

df=df.rename(columns = {'col2':'Column2'})

print("New DataFrame after renaming second column:")

print(df)
import pandas as pd

d = {'col1': [1, 2, 3, 4, 7, 11], 'col2': [4, 5, 6, 9, 5, 0], 'col3': [7, 5, 8, 12, 1,11]}

df = pd.DataFrame(data=d)

print("Original DataFrame")

print(df)

print("\ntopmost n records within each group of a DataFrame:")

df1 = df.nlargest(3, 'col1')

print(df1)

df2 = df.nlargest(3, 'col2')

print(df2)

df3 = df.nlargest(3, 'col3')

print(df3)
import pandas as pd

d = {'col1': [1, 2, 3, 4, 7], 'col2': [4, 5, 6, 9, 5], 'col3': [7, 8, 12, 1, 11]}

df = pd.DataFrame(data=d)

print("Original DataFrame")

print(df)

print("\nAll columns except 'col3':")

df = df.loc[:, df.columns != 'col3']

print(df)
import numpy as np

from numpy import linalg as LA

a = np.array([[1, 0], [1, 2]])

print("Original 2-d array")

print(a)

print("Determinant of the said 2-D array:")

print(np.linalg.det(a))
import pandas as pd



ipl_data = {'Team': ['Riders', 'Riders', 'Devils', 'Devils', 'Kings',

   'kings', 'Kings', 'Kings', 'Riders', 'Royals', 'Royals', 'Riders'],

   'Rank': [1, 2, 2, 3, 3,4 ,1 ,1,2 , 4,1,2],

   'Year': [2014,2015,2014,2015,2014,2015,2016,2017,2016,2014,2015,2017],

   'Points':[876,789,863,673,741,812,756,788,694,701,804,690]}

df = pd.DataFrame(ipl_data)



print(df)
import pandas as pd



ipl_data = {'Team': ['Riders', 'Riders', 'Devils', 'Devils', 'Kings',

   'kings', 'Kings', 'Kings', 'Riders', 'Royals', 'Royals', 'Riders'],

   'Rank': [1, 2, 2, 3, 3,4 ,1 ,1,2 , 4,1,2],

   'Year': [2014,2015,2014,2015,2014,2015,2016,2017,2016,2014,2015,2017],

   'Points':[876,789,863,673,741,812,756,788,694,701,804,690]}

df = pd.DataFrame(ipl_data)



print(df.groupby('Team'))
## View Groups



df.groupby('Team').groups
df.groupby(['Team','Year']).groups
grouped = df.groupby('Year')



for name,group in grouped:

   print(name)

   print(group)
grouped = df.groupby('Year')

grouped.get_group(2014)
grouped = df.groupby('Year')

grouped['Points'].agg(np.mean)
grouped = df.groupby('Team')

grouped['Points'].agg([np.sum, np.mean, np.std])
grouped = df.groupby('Team')

score = lambda x: (x - x.mean()) / x.std()*10

grouped.transform(score)
df.groupby('Team').filter(lambda x: len(x) >= 3)