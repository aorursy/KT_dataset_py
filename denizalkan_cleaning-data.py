import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))


data = pd.read_csv('../input/StudentsPerformance.csv')
data.head() #Top 5 Rows
data.tail() #Last 5 Rows
data.columns # Data Columns
data.shape # Rows and Column Count
data.info() # Column and İnfo about Columns

print(data['gender'].value_counts(dropna = False))
# Female : 518
# Male : 482 
# Name : gender, dtype : int64
# Dropna = False -> Show Nan value

print(data['lunch'].value_counts())
# Standart : 645, Free : 355 , int64

data.describe()
data.boxplot(column='math score',by = 'gender')

# Filtering the data
data[np.logical_and(data['gender']=='male', data['math score'] > 85)] # 66 Rows
data[np.logical_and(data['gender']=='female', data['math score'] > 85)]  # 37 Rows

data.corr()

data_top5 = data.head()
data_top5
melted_data=pd.melt(frame=data_top5,id_vars='gender',value_vars=['math score','reading score','writing score'])
melted_data
# Return 15 rows

#Pivoting the Data = Reverse of Melting

# melted_data.pivot(index = 'gender', columns = 'variable',values='value')
# Error : Index contains duplicate entries, cannot reshape. 
# So, if you want to Pivoting the data,  İndex must be uniq
 
header = data.head(3)
footer = data.tail(3)
conc_data = pd.concat([header,footer], axis = 0, ignore_index=False)
# ignore_index = True Return   ->  1-2-3-4-5-6 
# igoner_index = False Return  ->   0-1-2-797-798-799  
conc_data


# DATA TYPE
data.dtypes
# Change the Data type of Column
data['math score'] = data['math score'].astype('float')
data.dtypes # Data Types of Columns
# Change the data types
data['math score'] = data['math score'].astype('int64')
data.dtypes # Data Types of Columns

data.info()

# Assert

assert 1==1 # return nothing when true
# assert 1==2 #return error

data['gender'].notnull().all()         # Return True
assert data['gender'].notnull().all()  # Return Nothing