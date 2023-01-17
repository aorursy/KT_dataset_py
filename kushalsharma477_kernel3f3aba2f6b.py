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
import pandas as pd #import pandas for getting and processing data from csv file

sales = pd.read_csv("../input/Sales_data_1.csv") #get data from csv

sales.head() #for checking top rows of dataset and check dataset is imported properly or not
sales.info() #for checking datatype of the columns
sales.shape #for total no of rows and columns 
sales.isnull().sum() #check null values
sales[sales.isnull().any(1)] #when row have all null column
sales.dropna(inplace=True) #delete permanently
sales[sales.isnull().any(1)] #check null rows are delete permanently or not

#or

sales.isnull().sum()
sales.duplicated() #cehcked duplicated rows
sales[sales.duplicated()]
sales.loc[sales['Order ID'] == 'Order ID'] #where not usable string available
sales.loc[sales['Order ID'] == 'Order ID'].index #index of that rows
sales.drop(sales.loc[sales['Order ID'] == 'Order ID'].index,inplace=True) #drop that rows
sales.shape #deleted
sales[sales.duplicated()==True] #check duplicate
sales.drop_duplicates(inplace=True) #deleted
sales[sales['Order ID'].duplicated()] #check if order id is duplicate into orderid column
sales['Order ID'].unique().size #check unique rows count
sales[sales.duplicated(['Order ID'],keep=False)] #check all duplicate values together
sales.isnull().sum() #check again null values
sales.info() #check datatype of columns
sales['Order ID'] = sales['Order ID'].astype('int64')

sales['Quantity Ordered'] = sales['Quantity Ordered'].astype('int32')

sales['Price Each'] = sales['Price Each'].astype('float64')

sales['Order Date'] = pd.to_datetime(sales['Order Date'])
sales.info()
sales['Revenue'] = sales['Quantity Ordered'] * sales['Price Each'] #created revenue
sales['Day'] = sales['Order Date'].dt.day #created day column
city_split = sales['Purchase Address'].apply(lambda x: x.split(',')[1]) #apply function for every value

city_split #find cities from address
sales['City_Name'] = city_split
sales.head()
sales['Purchase Address'].apply(lambda x: x.split(',')[2])  #purchase address
sales['State_Name'] = sales['Purchase Address'].apply(lambda x: x.split(',')[2].split(' ')[1]) 
#now we can create common column for address for the uniquness then we can delete city and state

total_data = sales['City_Name']+' '+ sales['State_Name']

total_data

sales['City_State'] = sales['City_Name']+' '+ sales['State_Name'] 
sales.drop(columns = 'City_Name')

sales.drop(columns = 'State_Name') #drop non usable column
#data is cleaned now we can get this dataset into new csv for creating graph plot

sales.to_csv("sales_new.csv",index=False) 
sales_new =pd.read_csv("sales_new.csv")
sales_new.head()