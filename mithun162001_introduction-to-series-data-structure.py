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
data=pd.Series([0.25,0.5,0.75,1.0])

print(data)
print(data.values)

data.index
print(data[1])

print(data[0:3])
data1= np.array([0.25,0.50,0.75,1.0])

print(data1)

data1[0]
data=pd.Series([0.25,0.50,0.75,1.0], index= ['ab','b','c','d'])  #manually add a index value to the data variable

print(data)
data2=pd.Series([0.25,0.5,0.75,1.0], index=[2, 0, 'a', 5])  #index values are not mandatory to be in chronological order, index values can be randomlu defined in Series object

print(data2)
population_dict={ 'California': 383325,

                  'Texas': 176423,

                  'New York': 182531,

                  'Florida': 98765}

print(population_dict)

print(population_dict.keys())

print(population_dict.values())
population_series= pd.Series(population_dict) #Here instead of manually giving values, we passed a dictionary to the series object

print(population_series)          # Keys of the dictionary becomes the index values of Series object and the values becomes the values of the Series object wiht the right index values.
print(population_series['California'])

print(population_series['California': 'New York'])
# pd.Series(data, index) Syntax to be followed
data4= pd.Series(5, index=[1,2,3])  #Data can be scalar, which is repeated to fill the specified index

print(data4)