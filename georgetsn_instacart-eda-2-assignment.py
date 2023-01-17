import pandas as pd               # for data manipulation
import matplotlib.pyplot as plt   # for plotting 
import seaborn as sns             # an extension of matplotlib for statistical graphics
import numpy as np
## step 0 - import products.csv from directory '../input/products.csv'
products = pd.read_csv('../input/products.csv')
## step 1 - .groupby( ) all available products (from products data frame) by their "aisle_id", then select to find the size of each group
aisle_top = products.groupby('aisle_id')[['product_id']].count()
### step 2 - Rename the column of aisle_top as: 'total_products'
aisle_top.columns=['total_products']
# Before you move on to step 3, have a look at your produced results so far.
# Check the results below
aisle_top.head()
## step 3 - Sort the values of total_products so to get the aisles with most products first.
aisle_top_sort = aisle_top.sort_values(by='total_products', ascending=False)

## step 4 - Select the first 10 rows of the data frame. Remember that index in Python starts from 0
aisle_top_sort = aisle_top_sort.iloc[0:10]
### Before you move on to the final step, how can you ensure that the aisle_top has only 10 aisles?
print('The rows and columns:',aisle_top_sort.shape)

#Reviewing with .shape shows us 10rows and 1column, but this doesn't ensure we have 10 distinct aisles
#Coding mistakes from before might have led us into duplicate indexes therefore duplicate aisles
#We use .value_counts() on [aisle_top_sort.index] and then .sum() those results
#Now we can be sure not only we have 10 aisles (10rows in DataFrame) but each aisle is distinct

a = aisle_top_sort.index.value_counts()
print('Distinct aisles:',a.sum())
# Have a look at the produced data frame before you plot it (visualize it).
# Are your results fine?
aisle_top_sort.head()
## step 5 - Visualize the results. Place index on x-axis
plt.figure(figsize=(17,10))
sns.barplot(aisle_top_sort.index, aisle_top_sort.total_products, order=aisle_top_sort.index)
plt.title('Amount of Products per Aisle (Top10 Aisles)',fontsize=20)
plt.xlabel('Aisle Number ID (Top10)', size=15)
plt.ylabel('Volume of Products', size=15)
# Modify the limits
plt.ylim(830,1270)
plt.show()
## step 0 - Import the order_products__prior.csv from directory '../input/order_products__prior.csv'
order_products_prior = pd.read_csv('../input/order_products__prior.csv')
## step 1 - Filter order_products_prior and keep only these products with more than 30 purchases
avg_pos = order_products_prior.groupby('product_id').filter(lambda x: x.product_id.count()>30)
## step 2 -  .groupby( ) products and for add_to_cart_order column aggregate the values with the mean function.
avg_pos = avg_pos.groupby('product_id')[['add_to_cart_order']].mean()
avg_pos.head()
### step 3 - Rename column of avg_pos as: 'mean_add_to_cart_order'
avg_pos.columns=['mean_add_to_cart_order']
avg_pos.head()
## step 4 -  Use the proper method to sort the products by their mean_add_to_cart_order. Sort them in ascending order
avg_pos_asc = avg_pos.sort_values(by='mean_add_to_cart_order', ascending=True)
## step 5 - And now use again the same method to sort the products in descending order (store the results in a new DataFrame)
avg_pos_des = avg_pos.sort_values(by='mean_add_to_cart_order', ascending=False)
## step 6 - Store the product_id of the product with the highest mean_add_to_cart_order
id_low = avg_pos_des.index[0]
## step 7 -  Import products.csv and find the name of the product with the highest mean_add_to_cart_order
products = pd.read_csv('../input/products.csv')
products[products.product_id== id_low ]
### step 8 - Create a sns.barplot for the 10 products with the lowest mean_add_to_cart_order


#We already have the means in ascending order in [avg_pos_asc] and easily with an iloc we get the lowest 10 (the first 10 in [avg_pos_asc])
#In order to show in our barplot the actual names of the products (not just the IDs) and make it more understandable we merge with [products]
#The result [merged] is a DataFrame with the means as well as all the information for the 10 products with the lowest means
#We intentionally set to x-axes the [means] and to y-axes the [names] in order to make our barplot readable and presentable


low10 = avg_pos_asc.iloc[0:10].reset_index()
merged = pd.merge(low10,products)

plt.figure(figsize=(20,12))
graph=sns.barplot(merged.mean_add_to_cart_order, merged.product_name)
graph.set_xticks(np.arange(0, 2.5, step=0.2))
plt.xlim(0,2.4)
plt.title('10 products with the Lowest Mean Add to Cart',fontsize=20)
plt.xlabel('Mean Add to Cart', size=15)
plt.ylabel('Product Names',size=15)

plt.show()
