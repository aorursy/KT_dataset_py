# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
print(os.listdir('../input'))

# Any results you write to the current directory are saved as output.
#We create Pandas dataframe.
data = pd.read_csv('../input/world-happiness/2015.csv')
#default value is 5. We will see first 7 dataset.
data.head(7)
#Default 5. We will see last 5 data in dataset.
data.tail()
#We will see the shape of data. First attribute is row of data and second attribute is columns(features) of data.
data.shape
#Give information about our data
data.info()
#value_count(): Frequency counts
print(data['Freedom'].value_counts())
#Describe the value of count, mean, std, min, 25%, 50%, 75%, max.

#count = number of entries.
#mean = average of entries.
#std = standart deviation
#min = minimum entry
#25% = first quantile
#50% = median or second quantile
#75% = third quantile
#max = maximum entry

data.describe()#ignore null entries
data.boxplot(column ='Economy (GDP per Capita)', by ='Family')
plt.show()
#Firstly, we create new data using our dataset.
data_new = data.head()
data_new
#Let's melt
#id_vars = what we do not wish to melt
#value_vars = what we want to melt
melted = pd.melt(frame = data_new,id_vars = 'Country', value_vars =  ['Happiness Rank', 'Happiness Score' ])
melted
#Index is name
#We want to make that columns are variable.
#Finally values in columns are value.
melted.pivot(index = 'Country', columns = 'variable', values = 'value')
### Firstly lets create 2 dataframe.
data1 = data.head()
data2 = data.tail()
conc_data_row = pd.concat([data1, data2], axis = 0, ignore_index = True)#axis =0 is add dataframe in row.
conc_data_row
#For horizontal 
data1 = data['Country'].head()
data2 = data['Happiness Rank'].head()
conc_data_col = pd.concat([data1, data2], axis = 1)#axis = 1 is add dataframe in column
conc_data_col
data.dtypes
#Lets convert object(str) to categorical and int to float
data['Country'] = data['Country'].astype('category')
data.dtypes