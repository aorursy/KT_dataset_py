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
import pandas as pd 

import numpy as np

import matplotlib.pyplot as plt

np.random.seed(0)

# array of normally distributed random numbers

values = np.random.randn(100)

# generate a pandas series

AP = pd.Series(values) 

 # hist computes distribution

AP.plot(kind='hist', title='Normally distributed random values')

plt.show()
#Let's check some of the data's statistics (mean, standard deviation, etc.)

AP.describe()
df = pd.DataFrame({'A': [1, 2, 1, 4, 3, 5, 2, 3, 4, 1],

                   'B': [12, 14, 11, 16, 18, 18, 22, 13, 21, 17],

                   'C': ['a', 'a', 'b', 'a', 'b', 'c', 'b', 'a', 'b', 'a']})
df
df.describe()
#Note that since C is not a numerical column, it is excluded from the output.

df['C'].describe()
#Appending a new row to DataFrame

import pandas as pd

df = pd.DataFrame(columns = ['A', 'B', 'C'])

df
#Appending a row by a single column value:

df.loc[0, 'A'] = 1
df
#Appending a row, given list of values:

df.loc[1] = [2, 3, 4]

df
#Appending a row given a dictionary:

df.loc[2] = {'A': 3, 'C': 9, 'B': 9}

df
#first input in .loc[] is the index. If you use an existing index, you will overwrite the values in that row

df.loc[1] = [5, 6, 7]
df
df.loc[0, 'B'] = 8

df
#Append a DataFrame to another DataFrame
