# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np
cust = pd.read_csv('../input/Customers.csv')
# to display the top five row 

cust.head()

#cust[:5]
cust.index

# It shows the informations about how the index are arranged 
# To get the names of all columns in the DataFrame

cust.columns
# To get the names of all columns in the DataFrame in the form of list

cust.columns.tolist()
# To know the datatype of all columns in the DataFrame

cust.dtypes
cust.ndim

# Dimention of DataFrame
# To get the file level information about the dataset

cust.info()
# To know the sattistical summary of dataset

cust.describe()

# It gives the information about total number of observation, mean, max, median, standard deviation, percentiles

# Note : it gives the result of only numerical variable columns
# It gives the information about number of columns of which datatype

cust.get_dtype_counts()
# To show the number of observations are presents in particular Attributes/variables/columns

# Indirectly we can observe the missing values 

cust.count()
# To get the number of observation and number of columns

cust.shape
# To display the number of rows only

cust.shape[0]
# To show the number of columns/ no. of attributes

cust.shape[1]
# firstly find out the number of rows mussing in 'Customer_value'

cust['Customer_Value'].isnull().sum()

# a= cust['Customer_Value'].isnull().sum()
# find the total length of observation

cust.shape[0]

# b= cust.shape[0]
#divide the missing length of 'Customer_Value' by 'Total length of observation'

cust['Customer_Value'].isnull().sum()/cust.shape[0]

# a/b
# to display in terms of percentage multiply by 100

cust['Customer_Value'].isnull().sum()/cust.shape[0]*100
round(((cust['Customer_Value'].isnull().sum()/cust.shape[0])*100),3)
# To fing the how many duplicate observation in the dataframe

cust.Customer_ID.duplicated().sum()
# create a subset of duplicate value

duplicate= cust[cust.Customer_ID.duplicated()]

duplicate.head()
# to get the number of rows and columns of duplicate subset

duplicate.shape
# create a subset of unique values value

unique_values= cust[cust.Customer_ID.duplicated()==False]

unique_values.head()
# to get the number of rows and columns of unique subset

unique_values.shape
cust_10000= cust[cust['Customer_Value']> 10000]

cust_10000.shape[0]
# Reset the index values and drops the original index number

cust_10000= cust[cust['Customer_Value']> 10000].reset_index(drop= True)

cust_10000.head()
from numpy import where as IF

#customer_value = CV

CV= cust.Customer_Value

cust['Customer_value_segment']= IF(CV > 25000, 'HIgh Srgment', IF((CV > 10000) & (CV < 25000),

                                                                  'Medium Segment','Low Segment'))

cust.head()
cust = cust.assign(average_revenue_per_trip= cust.Customer_Value/cust.Buy_Times,

                  Balance_Points=cust.Points_Earned - cust.Points_Redeemed )

cust.head()
import datetime as dt

today = pd.to_datetime(dt.datetime.now().date())

day_difference = today - pd.to_datetime(cust.Recent_Date,format= '%Y%m%d')

day_difference
# Select only 'last_region', 'last_state', 'last_city' and 'Customer_Value' columns

demography= cust[['Last_region', 'Last_state', 'Last_city', 'Customer_Value']]

demography.head()
demography.groupby(['Last_region'])['Customer_Value'].sum()/demography.Customer_Value.sum()
demography.groupby(['Last_region', 'Last_state'])['Customer_Value'].sum()/demography.Customer_Value.sum()
demography.groupby(['Last_region', 'Last_state', 'Last_city'])['Customer_Value'].sum()/demography.Customer_Value.sum()
cust.groupby(['Last_state', 'Last_city'])['Customer_ID'].count()
cust.groupby(['Last_state', 'Last_city'])['Buy_Times'].count()
cust.groupby(['Last_state', 'Last_city'])['average_revenue_per_trip'].mean()