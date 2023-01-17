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
# Read the data into a dataframe.

NY = pd.read_csv("/kaggle/input/nyc-jobs.csv")
# Describe the data

NY.describe()
# Check the top 5 rows

NY.head()
# Check the last 10 rows

NY.tail()
# Check for index

NY.index
# Display the columns

NY.columns
# Check data types

NY.dtypes
# Covert Salary Range From data type to integer

NY= NY.astype({'Salary Range From':'int32'})
# Check

print(NY["Salary Range From"].head())
# Covert Salary Range To data type to integer

NY= NY.astype({'Salary Range To':'int32'})
# Check

print(NY["Salary Range To"].head())
# Grouping

NY.groupby('Agency')['Salary Range From','Salary Range To'].mean()
# Grouping with multiple stats

NY.groupby('Agency').agg(

    {

         'Salary Range From':"mean",

         'Salary Range To':"mean",    

         '# Of Positions': "count"  

    }

)
NY_groupby = NY.groupby('Agency').agg(

    {

         'Salary Range From':"mean",

         'Salary Range To':"mean",    

         '# Of Positions': "count"  

    }

)

NY_Sort = NY_groupby.sort_values(['# Of Positions'],ascending=False)

print(NY_Sort)
NY_Sort2 = NY_groupby.sort_values(['Salary Range To'],ascending=False)

print(NY_Sort2)
# Groupby multiple columns

NY.groupby(['Agency','Job Category']).agg(

    {

         'Salary Range From':"mean",

         'Salary Range To':"mean",    

         '# Of Positions': "count"  

    }

)
NY_multiple_groupby = NY.groupby(['Agency','Job Category']).agg(

    {

         'Salary Range From':"mean",

         'Salary Range To':"mean",    

         '# Of Positions': "count"  

    }

)

NY_Sort3 = NY_multiple_groupby.sort_values(['# Of Positions'],ascending=False)

print(NY_Sort3)
NY_groupby_2 = NY.groupby('Business Title').agg(

    {

         'Salary Range From':"mean",

         'Salary Range To':"mean",    

         '# Of Positions': "count"  

    }

)

NY_Sort4 = NY_groupby_2.sort_values(['Salary Range To'],ascending=False)

print(NY_Sort4)