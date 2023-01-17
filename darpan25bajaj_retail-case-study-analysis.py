import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import os
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
customer = pd.read_csv("/kaggle/input/Customer.csv")

prod_info = pd.read_csv("/kaggle/input/prod_cat_info.csv")

transaction = pd.read_csv("/kaggle/input/Transactions.csv")
customer.shape
prod_info.shape
transaction.shape
customer.head(2)
prod_info.head(2)
# renaming "prod_sub_cat_code" column in 'prod_info' table to make it similar to 'transaction' table

# to merge the both the tables easily

prod_info.rename(columns={"prod_sub_cat_code":"prod_subcat_code"},inplace=True)
transaction.head()
# merge transaction and prod_info table and create a new table "prod_concat"

prod_concat = pd.merge(left=transaction, right=prod_info,on=["prod_cat_code","prod_subcat_code"],how="left")
prod_concat
prod_concat.isnull().sum()
customer.head()
#merge "prod_concat" and "customer" table and create the final table "customer_final"

customer_final = pd.merge(left=prod_concat, right=customer,right_on="customer_Id", left_on="cust_id", how="left")
customer_final.head()
customer_final.shape
transaction.shape
print('''Rows of both the 'customer_final' and 'transaction' table are same. That means all the transactions done at the 

         Retail Store are present in the final table ''')
customer_final.dtypes
customer_final.isnull().sum()
# converting "DOB" and "tran_date" from object dtype to dates

customer_final["DOB"] = pd.to_datetime(customer_final["DOB"], format="%d-%m-%Y")
customer_final['DOB'].head(10)
customer_final["tran_date"] = pd.to_datetime(customer_final["tran_date"])
customer_final["tran_date"].head(10)
customer_final.duplicated().sum()
# dropping duplicate rows

customer_final.drop_duplicates(inplace=True)
customer_final.duplicated().sum()
#column names of "customer_final" dataframe

customer_final.columns
# data types of all columns of "customer_final" dataframe

customer_final.dtypes
# top 10 observations

customer_final.head(10)
#bottom 10 observations

customer_final.tail(10)
customer_final.describe()
customer_final.loc[:,customer_final.dtypes=="object"].describe()
conti_customer = customer_final.loc[:,['prod_subcat_code','prod_cat_code', 'Qty', 'Rate', 'Tax', 'total_amt']]
conti_customer.columns
for var in conti_customer.columns:

    conti_customer[var].plot(kind='hist')

    plt.title(var)

    plt.show()
category_customer = customer_final.loc[:,customer_final.dtypes=='object']
category_customer.head()
plt.figure(figsize=(8,8))

sns.countplot(category_customer['Gender'])

plt.show()
plt.figure(figsize=(8,8))

sns.countplot(category_customer['Store_type'])

plt.xlabel('Store Type')

plt.show()
plt.figure(figsize=(8,8))

sns.countplot(category_customer['prod_cat'])

plt.xlabel('Product Category')

plt.show()
plt.figure(figsize=(8,8))

category_customer.groupby('prod_subcat')['prod_subcat'].count().plot(kind='barh')

plt.xlabel('Count')

plt.ylabel('Product Subcategory')

plt.show()
customer_final.sort_values(by="tran_date")
min_date = customer_final["tran_date"].min()
max_date = customer_final["tran_date"].max()
print("Time period of the available transaction data is from "+ pd.Timestamp.strftime(min_date,format="%d-%m-%Y") + " to " + pd.Timestamp.strftime(max_date,format="%d-%m-%Y"))
customer_final.head()
#count of transaction_ids where total_amt was negative

negative_transaction = customer_final.loc[customer_final["total_amt"] < 0,"transaction_id"].count()
print("Count of transactions where the total amount of transaction was negative is",negative_transaction)
#groupby the data set on the basis of "Gender" and "prod_cat"

product_gender = customer_final.groupby(["Gender","prod_cat"])[["Qty"]].sum().reset_index()
product_gender
#converting to pivot table for better view

product_gender.pivot(index="Gender",columns="prod_cat",values="Qty")
customer_final.head(2)
customer_group = customer_final.groupby('city_code')['customer_Id'].count().sort_values(ascending =False)
customer_group
plt.figure(figsize=(8,5))

customer_group.plot(kind="bar")

plt.xlabel("City Code")

plt.ylabel("No. of customers")

plt.yticks(np.arange(0, 3500, step=500))

plt.show()
percentage = round((customer_group[4.0] / customer_group.sum()) * 100,2)
percentage
print("City code 4.0 has the maximum customers and the percentage of customers from that city is ",percentage)
customer_final.head(2)
customer_final.groupby("Store_type")["Qty","Rate"].sum().sort_values(by="Qty",ascending=False)
print('e-Shop store sell the maximum products by value and by quantity')
store_group = round(customer_final.pivot_table(index = "prod_cat",columns="Store_type", values="total_amt", aggfunc='sum'),2)
store_group
store_group.loc[["Clothing","Electronics"],"Flagship store"]
# if we have to find total amount of both 'Clothing' and 'Electronics' from ' Flagship Store'

store_group.loc[["Clothing","Electronics"],"Flagship store"].sum()
gender_group = round(customer_final.pivot_table(index = "prod_cat",columns="Gender", values="total_amt", aggfunc='sum'),2)
gender_group
male_earning = gender_group.loc["Electronics","M"]
print("The total amount earned from Male customers under the Electronics category is",male_earning)
#creating a new dataframe that does not contain transactions with negative values

pos_trans = customer_final.loc[customer_final["total_amt"]>0,:]
pos_trans
# creating a dataframe that contains unique transactions 

unique_trans = pos_trans.groupby(['customer_Id','prod_cat','prod_subcat'])['transaction_id'].count().reset_index()
unique_trans
# now finding the customers which have unique transactions greater than 10

unique_trans_count = unique_trans.groupby('customer_Id')['transaction_id'].count().reset_index()
unique_trans_count.head()
unique_trans_count[unique_trans_count['transaction_id'] > 10]
print('There are no unique transactions greater than 10')
now = pd.Timestamp('now')

customer_final['DOB'] = pd.to_datetime(customer_final['DOB'], format='%m%d%y')    # 1

customer_final['DOB'] = customer_final['DOB'].where(customer_final['DOB'] < now, customer_final['DOB'] -  np.timedelta64(100, 'Y'))   # 2

customer_final['AGE'] = (now - customer_final['DOB']).astype('<m8[Y]')
customer_final.head()
customer_final['Age_cat'] = pd.cut(customer_final['AGE'],bins=[24,35,46,57],labels=['25-35','36-46','47-57'],include_lowest=True)
customer_final.head()
# grouping the dataframe 'customer_final' on the basis of 'Age_cat' and 'prod_cat'

customer_25_35 = customer_final.groupby(['Age_cat','prod_cat'])['total_amt'].sum()
customer_25_35
customer_25_35.loc['25-35',['Books','Electronics']]
print("Total amount spent on 'Electronics' and 'Books' product categories is", 

      customer_25_35.loc['25-35',['Books','Electronics']].sum().round(2))
customer_final.head()
# filtering out data that belongs to the 'age_cat' = 25-35

customer_total_amount_25_35 = customer_final[customer_final['Age_cat']=='25-35']
customer_total_amount_25_35.head()
# getting all the data with transaction date between 1st Jan 2014 to 1st Mar 2014?

total_amount = customer_total_amount_25_35[(customer_total_amount_25_35['tran_date'] >='2014-01-01') & (customer_total_amount_25_35['tran_date'] <='2014-03-01')]
total_amount
print('The total amount spent by customers aged 25-35 between 1st Jan 2014 to 1st Mar 2014 is',

      total_amount['total_amt'].sum())