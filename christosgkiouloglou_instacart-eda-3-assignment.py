#load packages
import pandas as pd               # for data manipulation
import matplotlib.pyplot as plt   # for plotting 
import seaborn as sns             # an extension of matplotlib for statistical graphics

#load data
orders = pd.read_csv('../input/orders.csv' )
products = pd.read_csv('../input/products.csv')
order_products_prior = pd.read_csv('../input/order_products__prior.csv')
reorder = order_products_prior.groupby('product_id').filter(lambda x: x.shape[0] >40)
reorder = reorder.groupby('product_id')[['reordered']].mean()
reorder.columns = ['reorder_ratio']
reorprob_results=reorder
x=products['product_name']
x.index += 1 
reorprob_results=pd.concat([reorprob_results,x],axis=1)
reorprob_results = reorprob_results.sort_values(by='reorder_ratio', ascending=False)
top_10=reorprob_results.head(10)
top_10=reorprob_results.head(10)
plt.figure(figsize=(15,8))
sns.barplot(top_10.product_name, top_10.reorder_ratio )
plt.xticks(size=12, rotation=90)
plt.show()
prd=pd.merge(orders,order_products_prior)
order_size=pd.DataFrame(prd.groupby(['user_id','order_number'])['product_id'].count())
order_size.columns=['size']
results = pd.DataFrame(order_size.groupby(['user_id'])['size'].mean())
results.columns=['order_size_avg']
results['order_size_smallest']= pd.DataFrame(order_size.groupby(['user_id'])['size'].min())
results['order_size_biggest']= pd.DataFrame(order_size.groupby(['user_id'])['size'].max())
results.head()
top10_order_size_avg=results.sort_values(by='order_size_avg',ascending=False)
top10_order_size_avg.head()
plt.hist(results.order_size_avg,bins=100)
plt.show()
