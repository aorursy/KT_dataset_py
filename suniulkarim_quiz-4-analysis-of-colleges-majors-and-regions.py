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
# The index for salaries based on major is set to the 'Undergraduate Major'

sbm = pd.read_csv('/kaggle/input/college-salaries/degrees-that-pay-back.csv').set_index('Undergraduate Major')

# The index for salaries based on college type and region is set to the 'School Name'

sbc = pd.read_csv('/kaggle/input/college-salaries/salaries-by-college-type.csv').set_index('School Name')

sbr = pd.read_csv('/kaggle/input/college-salaries/salaries-by-region.csv').set_index('School Name')



#set_index() uses an existing column in the DataFrame and sets it as the DataFrame index
dataFrameList = [sbm,sbc,sbr] # dataFrameList stores our three DataFrames

for df in dataFrameList: # For each DataFrame in our list

    for col in df.columns: # For each column in that list

        i = 0

        while i < df.shape[0]: # i corresponds to the row

            if isinstance(df[col].iloc[i],str) and df[col].iloc[i][0] == "$": # Check to see that the value we are altering is a string and a dollar amounr

                df[col].iloc[i] = df[col].iloc[i].replace(',','') # Replace the comma with an empty string

                df[col].iloc[i] = df[col].iloc[i].replace('$','') # Replace the dollar sign with an empty string

                df[col].iloc[i] = float(df[col].iloc[i]) # Cast the conversion-friendly value into a float

            i = i + 1

# Note: do not feel uneasy if the above code is difficult for you to understand. 

# Methods will begin to make more sense as we go through our tutorial, the code was simply necessary at this point for us to proceed properly.
sbm.head()

# head() returns the first n rows of the DataFrame

# If no integer is provided, head() will default to returning the first 5 rows
sbm.tail(10)

# tail() returns the last n rows of the DataFrame, defaults to 5 as head() did
sbc.head()
sbc.tail()
sbc.index

# The index data attribute is an immutable ndarray, that implements an ordered and sliceable array - it stores the axis labels for any panda objects
sbc.columns

# The columns data attribute contains labels for the DataFrames' columns
sbc.shape[0]

# the shape data attribute contains the DataFrame's dimensions in the format shape[rows,columns]

# shape[0] allows us to access the number of rows
sbc.shape[1]

# shape[1] allows us to access the number of columns
sbc.head()
sbc.loc['Harvey Mudd College']

# .loc[] accesses a group of rows and columns by utilizing the label of a specific index
sbc.loc[['Cooper Union', 'Harvey Mudd College','Amherst College','Auburn University']]

# Note: Make sure to remember the double brackets when locating more than one item! Remember, you are locating a list of colleges.
sbc.iloc[100:-100]

# iloc[] allows for integer-based indexing

# Can be used in the form iloc[x:y] in order to find a subset of data containing the entire row

# Can also be used in the form iloc[x,y] in order to access specific rows and columns, and the two implementations 

# can be combined for even more specificity.
sbm[sbm['Mid-Career Median Salary'] > 100000]

# We do this by designating a condition, which goes inside the outer pair of brackets

# This will return only those rows that satisfy the condition

# NOTE: To combine filtering conditions in Pandas, use bitwise operators ('&' and '|') not pure Python ones ('and' and 'or')
sbc[sbc['Mid-Career Median Salary'] > 100000]
round(sbm['Mid-Career Median Salary'].mean(),2)

# First, we select only the subset of data we need, by designating the 'Mid-Career Median Salary' column, and then summarize utilizing mean()

# Round the value to two decimal places, for readability and since we are comparing dollar values
round(sbm['Mid-Career Median Salary'].median(),2)

# Same process, as above, but we instead summarize by utilizing the median (50th percentile)

# Calculating the median of a median salary may seem redundant, but the median we are provided is the median for the particular degree,

# while the new median we are calculating is the median of all of the degrees' mid-career median salary.
round(sbc['Mid-Career Median Salary'].mean(),2)
round(sbc['Mid-Career Median Salary'].median(),2)
round(sbm['Mid-Career Median Salary'].std(),2)

# Utilizing the std() function allows us to obtain the standard deviation

# The standard deviation allows us to see the spread of values from the average, which can be used to gauge variability depending on our choices
round(sbc['Mid-Career Median Salary'].std(),2)
sbm = sbm.sort_values('Mid-Career Median Salary',ascending=False) #ascending is a boolean, 

                                                #when false DataFrame will be sorted in descending order

# sort_values allows us to sort our DataFrame by axis items

sbm.head()
sbc = sbc.sort_values('School Name')

sbc.head()
sbm['Starting Median Salary'].max()

#First, we will select the 'Starting Median Salary' column, and then use .max() in order to find its maximum value
sbm['Starting Median Salary'].idxmax()

# Similar set up to the previous code cell

# .idxmax() returns the row label of the maximum value, rather than the maximum value itself
sbr.head()
sbr.groupby('Region')['Starting Median Salary'].mean()



# We facilitate split-apply-combine functions through .groupby(), which groups a DataFrame by a Series of columns

# We group our colleges together by their region, then we apply the .mean() method to the 'Starting Median Salary'

# and finally, that data is returned to us as a new dataset



# Although we could find the mean first and then select the 'Starting Median Salary' column, this is less efficient

# Because it means we will have to take the mean of every column and then select from that data, 

# rather than only compute for the specific data we are interested in
sbr.groupby('Region')['Starting Median Salary'].size()

# Split by the Region

# Apply the .size() function to the 'Starting Median Salary' column

# Combine the data into a new dataset
sbr['Region'].value_counts()

# We select the 'Region' column, and then apply .value_counts(), which is a function that returns a Series containing counts of unique values

# Notice that .groupby() above returned a Series with the same name as the column, while

# .value_counts() on its own returned a Series named after the Region, .value_counts() is also in descending order of values,

# while the .groupby() implementation is ascending in alphabetical order
sbc.groupby('School Type')['Starting Median Salary'].mean().plot.bar()

# To begin, we split the schools up into their respective types

# Then, we apply the mean() to the 'Starting Median Salary' column, which gives us a new dataset

# Then, we plot() the dataset, and use a .bar, which present us with a bar plot to easily visualize the data
sbc['School Type'].value_counts().plot.pie()

# We first create a new dataset, which utilizes .valuecounts() in order to obtain a Series containing the counts of unique elements

# Then we use .plot(), specifying pie, in order to create a pie chart
sbc[sbc['School Type'] == 'Ivy League'].sort_values('Starting Median Salary')['Starting Median Salary'].plot.barh()

# First, we use conditional selection to only select Ivy League Schools

# Then, we want to sort our values by 'Starting Median Salary', and then select only that column for our dataset as well using ['Starting Median Salary']

# We then simply proceed as before, and utilize plot(), this time specifying barh to indicate we want a 