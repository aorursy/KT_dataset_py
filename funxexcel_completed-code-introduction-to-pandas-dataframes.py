#Import Package

import pandas as pd
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#Import Dataset

customers = pd.read_csv('/kaggle/input/retail-data-customer-summary-learn-pandas-basics/Retail_Data_Customers_Summary.csv')
#the Dataframe name "customers" returns first 5 and last 5 rows

customers
#If you use head(), it returns the top 5 rows

customers.head()
#To find the top 5 customers, we need to sort by tran_amount_2015 in descending order

customers.sort_values(by = ['tran_amount_2015'], ascending = False).head()
#To find the top 5 customers, we need to sort by tran_amount_2014 in descending order

#Then look at the tran_amount_2015 colums to identify those who did not buy

customers.sort_values(by = ['tran_amount_2014'], ascending = False).head()
#We will store in a new dataframe name

customers_2014 = customers.sort_values(by = ['tran_amount_2014'], ascending = False).head()
#Then we will select the relevent columns to answer the question

customers_2014[['customer_id','tran_amount_2014','tran_amount_2015']]
#Then we ### Select Multiple Columnwill select the relevent columns to answer the question

customers_2014[['customer_id','tran_amount_2014','tran_amount_2015']].sort_values(by = 'tran_amount_2015', na_position = 'first')
#Returns as error as this id is not present in any column name

customers['CS4074']
#so we use .loc to slice/select data by rows

#Still gives an error, as it is unable to find this id in a row (but which row?)

customers.loc['CS4074']
#The default index is 0,1,2,3,4,5 ...

customers.head()
#We can select one row by mentioning the index number

customers.loc[0]
#We need to set index before we could easily select rows (just like setting a primary key in SQL)

customers.set_index('customer_id') 
#When we see above the index is set. However, when we check its reset to default

customers.head()
#If you want to make permanent change to the dataframe, we use the inplace argument

customers.set_index('customer_id', inplace = True) 
#Now the change is permanent

customers.head()
#Now if look for details of a customer, we will be able to find it

customers.loc['CS4074']
#To get details of 5 customers, we use a list 

customers.loc[['CS4074', 'CS5057', 'CS2945', 'CS4798', 'CS4424']]
#We can extend .loc to select both rows and columns (begin with list customers from rows then select columns)

customers.loc[['CS4074', 'CS5057', 'CS2945', 'CS4798', 'CS4424'],['tran_amount_2015','transactions_2015']]
customers.reset_index(inplace = True)
#We can no longer select customers using (.loc())

customers.loc[['CS4074', 'CS5057', 'CS2945', 'CS4798', 'CS4424'],['tran_amount_2015','transactions_2015']]
customers.head()