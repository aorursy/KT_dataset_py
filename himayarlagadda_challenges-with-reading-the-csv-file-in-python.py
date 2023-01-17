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
"""



### Challenges with reading the csv file:



1. Read the data except the first few rows in the file.

2. Read files present in multiple directories.

3. If the dataset is bigger in size, we can read few rows only.

4. If the dataset has large number of columns, read only specific columns



"""
"""

1. Read the data except the first few rows in the file

"""



#import pandas library to read, manipulate and clean the data in python

import pandas as pd



# big_mart_sales_top_row_error.csv consists of comments in the first few rows

# While reading file, pass the parameter skiprows = n (number of rows to skip)



data_without_comments = pd.read_csv('../input/samplebigmartdata/big_mart_sales_top_row_error.csv', skiprows = 5 )

print(data_without_comments)

"""

2. Reading files from multiple directories

"""



# Use the glob library to list the files in a dircetory



import glob



#list all the files in the folder



for directory in glob.glob('../input/salesdata/multi-directory/*'):

    print(directory)                       
#iterate through multi-directory folder to see the list of csv files inside each sub-folder



for directory in glob.glob('../input/salesdata/multi-directory/*'):

    for files in glob.glob(directory + '/*'):

        print(files)
#concatenate the files



dataframe_list = []



#iterate through each folder



for directory in glob.glob('../input/salesdata/multi-directory/*'):

    

    #iterate through each file

    

    for file in glob.glob(directory + '/*'):

        dataframe_list.append(pd.read_csv(file))

        

# concatenate the dataframes



final_data = pd.concat(dataframe_list)

print(final_data)





"""

3. Read first 'N' rows when the data is large

"""



#specify the number of rows to read



read_sample_from_data = pd.read_csv('../input/salesdata/multi-directory/1985/1985.csv', nrows=100)



print(read_sample_from_data)

"""

4. Read specific columns



If the dataset has large number of columns, it is impossible to go through all of them at the same time, so we can read only specific columns at a time



"""



read_specific_columns = pd.read_csv('../input/salesdata/multi-directory/1985/1985.csv', usecols=['Item_Identifier', 'Item_Type', 'Item_MRP'])



print(read_specific_columns)