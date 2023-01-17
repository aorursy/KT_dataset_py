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
avocado = pd.read_csv('../input/avocados/Avocado.csv')

avocado
# display the first 5 rows

avocado.head()
# display the last 3 rows

avocado.tail(3)
# prints information about the rows from the starting column (start = 0) to the end (stop=18249) consecutively.

avocado.index
# prints out a list of column labels in the data frame

avocado.columns
# prints out the shape of the data frame (rows x columns)

avocado.shape
# For each year, display the total sum of small_bags. Store the new data frame into a variable.

avo_data = avocado.groupby(['year']).Small_Bags.sum()

avocado.groupby(['year']).Small_Bags.sum()
# Figure out the mean number of small bags sold per day for each year

avocado.groupby(['year']).Small_Bags.mean()
# Figure out the median number of small bags sold for each year

avocado.groupby(['year']).Small_Bags.median()
# Figure out the standard deviation of small bags sold for each year

avocado.groupby(['year']).Small_Bags.std()

# .size() gives you the number of elements in the frame

# In this case, it will display the number of dates listed per year

avocado.groupby(['year']).Small_Bags.size()

# value_counts() will will display the number of counts of quantity of small_bags sold for each year

avocado.groupby(['year']).Small_Bags.value_counts()

# Let's try grouping by years and display the total numbers of small bags and large bags 

new_avo= avocado.groupby('year')['Small_Bags', 'Large_Bags'].sum()

new_avo
# .loc[] method with a input of a specific index that we are looking for will return associated information about that index

new_avo.loc[2015] 
# .iloc[row_index, column_index] method will return information when row_index and column index are specified.

new_avo.iloc[1,1]
# The sort_values will take in a column labeland order the data frame accordingly\

# The second input, ascending =True is optional. You can also make the input = False in order to reverse the order.

avo_sorted_small = avocado.sort_values('Small_Bags', ascending = True)

avo_sorted_small
# .sort_index() method will sort the data frame according to the indices

avo_sorted_small.sort_index()
# .max() method will give you the highest value for all the columns available

avo_sorted_small.max()



# Similarly, .min() will give you the lowest value for all the columns

# avo_sorted_small.min()
# Group the number of small bags of avocadoes sold per year.

# Make a bar graph for it.

avocado.groupby('year')['Small_Bags'].sum().plot.bar()