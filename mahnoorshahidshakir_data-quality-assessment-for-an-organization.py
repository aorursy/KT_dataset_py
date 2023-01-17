# importing pandas library for i/o and dataframes 

import pandas as pd



# loading dataset and extracting sheets'

dataset = pd.ExcelFile('../input/raw-data-provided-by-organization/Raw_Data_provided_by_Organisation.xlsx')



# parsing sheets

Transactions = dataset.parse('Transactions', header=0)

NewCustomerList = dataset.parse('NewCustomerList')

CustomerDemographic = dataset.parse('CustomerDemographic')

CustomerAddress = dataset.parse('CustomerAddress')
# display data inside sheet

print(Transactions.head())
# Display columns of dataset Transactions

print(Transactions.info())
# checking the shape of your data

print(Transactions.shape)
# looking for the null values

total_null_values = Transactions.isnull().sum()



# calculating total values

total_values = Transactions.count().sort_values(ascending=True) 



# calculating the percentage of null values

null_values_percentage = total_null_values/total_values *100



# converting to dataframe of missing values

missing_values = pd.concat({'Total Values' : total_values, 'Null_values': total_null_values, 'Percentage of Missing Values': null_values_percentage}, axis=1)



# display missing values

print(missing_values)
# checking a single product id and its details

bool_series = Transactions['product_id'] == 0



product_id_0 = Transactions[bool_series]



#view the product details

print(product_id_0[['brand', 'product_line','product_class']])
# looking for duplicated values

duplicated_values = Transactions.duplicated()



# number of duplicated values in dataset

print("The number of duplicated records in Transactions dataset is {}".format(duplicated_values.sum()))
# display data of sheet NewCustomerList

print(NewCustomerList.head())
# display data of sheet Customer Demographic

print(CustomerDemographic.head())
# display data of sheet Customer Address

print(CustomerAddress.head())
# Display columns of dataset NewCustomerList

print(NewCustomerList.info())
# checking the shape of your data

print(NewCustomerList.shape)
# Display columns of dataset CustomerDemographic

print(CustomerDemographic.info())
# Display columns of dataset CustomerAddress

print(CustomerAddress.info())
# checking the shape of your data

print(CustomerDemographic.shape)
# checking the shape of your data

print(CustomerAddress.shape)
# looking for the null values

total_null_values = NewCustomerList.isnull().sum()



# calculating total values

total_values = NewCustomerList.count().sort_values(ascending=True) 



# calculating the percentage of null values

null_values_percentage = total_null_values/total_values *100



# converting to dataframe of missing values

missing_values_NewCustomerList = pd.concat({'Total Values' : total_values, 'Null_values': total_null_values, 'Percentage of Missing Values': null_values_percentage}, axis=1)



# display missing values

print(missing_values_NewCustomerList)
# looking for the null values

total_null_values = CustomerDemographic.isnull().sum()



# calculating total values

total_values = CustomerDemographic.count().sort_values(ascending=True) 



# calculating the percentage of null values

null_values_percentage = total_null_values/total_values *100



# converting to dataframe of missing values

missing_values_CustomerDemographic = pd.concat({'Total Values' : total_values, 'Null_values': total_null_values, 'Percentage of Missing Values': null_values_percentage}, axis=1)



# display missing values

print(missing_values_CustomerDemographic)
CustomerDemographic['DOB']
# looking for duplicated values

duplicated_values = NewCustomerList.duplicated()



# number of duplicated values in dataset

print("The number of duplicated records in NewCustomerList dataset is {}".format(duplicated_values.sum()))
# looking for duplicated values

duplicated_values = CustomerDemographic.duplicated()



# number of duplicated values in dataset

print("The number of duplicated records in CustomerDemographic dataset is {}".format(duplicated_values.sum()))
# looking for duplicated values

duplicated_values = CustomerAddress.duplicated()



# number of duplicated values in dataset

print("The number of duplicated records in CustomerAddress dataset is {}".format(duplicated_values.sum()))