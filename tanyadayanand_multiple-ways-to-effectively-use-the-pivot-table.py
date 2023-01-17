# Importing necessary libraries



import pandas as pd

import numpy as np
# Importing and Loading the data into data frame



market_df = pd.read_csv("../input/market.csv")

customer_df = pd.read_csv("../input/customer.csv")

product_df = pd.read_csv("../input/product.csv")

shipping_df = pd.read_csv("../input/shipping.csv")

orders_df = pd.read_csv("../input/order.csv")



# Merging the dataframes to create a master_df

df_1 = pd.merge(market_df, customer_df, how='inner', on='Cust_id')

df_2 = pd.merge(df_1, product_df, how='inner', on='Prod_id')

df_3 = pd.merge(df_2, shipping_df, how='inner', on='Ship_id')

master_df = pd.merge(df_3, orders_df, how='inner', on='Ord_id')

master_df.head()
#Identifying Missing Values in Column

master_df.isnull().sum()
#a single index

#Using pandas.DataFrame.pivot_table

master_df.pivot_table(index = 'Customer_Segment')

#Same as above - results in same output

#Using pandas.pivot_table

pd.pivot_table(master_df, index = 'Customer_Segment')
#multiple indexes

master_df.pivot_table(index =['Customer_Segment','Product_Category'])
#Single value

master_df.pivot_table(values = 'Sales', index = 'Customer_Segment')
#multiple value

master_df.pivot_table(values = ['Order_Quantity','Sales'], index = 'Customer_Segment')
#Single aggrigate function(mean) and single value

master_df.pivot_table(values = 'Sales', index = 'Customer_Segment', aggfunc = 'mean')
#Single aggrigate function(sum) and single value

master_df.pivot_table(values = 'Order_Quantity', index = 'Region', aggfunc = 'sum')
#Sum aggregate function is applied to both the values

master_df.pivot_table(values = ['Order_Quantity','Sales'], index = 'Product_Category', aggfunc='sum')
#multiple Aggregating Function applied to single column

master_df.pivot_table(values = 'Sales', index = 'Product_Category', aggfunc=['sum', 'count'])
#Sum and Mean aggregate function is applied to both the values

master_df.pivot_table(values = ['Order_Quantity','Sales'], index = 'Product_Category', aggfunc=[np.sum, np.mean])
#different aggregate applied to different values

master_df.pivot_table(index = 'Product_Category', aggfunc = {'Order_Quantity':sum, 'Sales':'mean'})
#Single column

#Grouping by both rows and column

master_df.pivot_table(values = 'Profit', 

                      index = 'Product_Category', 

                      columns = 'Customer_Segment', 

                      aggfunc = 'sum')
#multiple columns

master_df.pivot_table(values = 'Profit', 

                      index = 'Customer_Segment', 

                      columns = ['Product_Category','Ship_Mode'], 

                      aggfunc = 'count')
#Margin

master_df.pivot_table(values = 'Profit', 

 index = 'Product_Category', 

 columns = 'Customer_Segment', 

 aggfunc = 'sum', margins=True)
#margins_name

master_df.pivot_table(values = 'Profit', 

                      index = 'Product_Category', 

                      columns = 'Customer_Segment', 

                      aggfunc = 'sum', 

                      margins=True,

                      margins_name ='TOTAL')
#Displaying NaN values in the table

#These can be imputed using fill_value

master_df.pivot_table(values = 'Product_Base_Margin', 

                      index = 'Customer_Name', 

                     columns = 'Customer_Segment', 

                      aggfunc = 'mean')
#imputing with mean using fill_value

master_df.pivot_table(values = 'Product_Base_Margin', 

                      index = 'Customer_Name', 

                     columns = 'Customer_Segment', 

                      aggfunc = 'mean', fill_value=np.mean(master_df['Product_Base_Margin']))