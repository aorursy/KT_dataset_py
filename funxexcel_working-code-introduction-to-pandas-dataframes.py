#Import Packages

#Import Dataset (/kaggle/input/retail-data-customer-summary-learn-pandas-basics/Retail_Data_Customers_Summary.csv)

#the Dataframe name "customers" returns first 5 and last 5 rows

#If you use head(), it returns the top 5 rows

#To find the top 5 customers, we need to sort by tran_amount_2015 in descending order

#To find the top 5 customers, we need to sort by tran_amount_2014 in descending order

#Then look at the tran_amount_2015 colums to identify those who did not buy

#We will store in a new dataframe name

#Then we will select the relevent columns to answer the question

#Then we ### Select Multiple Columnwill select the relevent columns to answer the question

#Returns as error as this id is not present in any column name

#so we use .loc to slice/select data by rows

#Still gives an error, as it is unable to find this id in a row (but which row?)

#The default index is 0,1,2,3,4,5 ...

#We can select one row by mentioning the index number

#We need to set index before we could easily select rows (just like setting a primary key in SQL)

#When we see above the index is set. However, when we check its reset to default

#If you want to make permanent change to the dataframe, we use the inplace argument

 
#Now the change is permanent

#Now if look for details of a customer, we will be able to find it

#To get details of 5 customers, we use a list 

#We can extend .loc to select both rows and columns (begin with list customers from rows then select columns)

#We can no longer select customers using (.loc())
