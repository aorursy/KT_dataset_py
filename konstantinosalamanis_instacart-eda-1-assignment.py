import pandas as pd               # for data manipulation
import matplotlib.pyplot as plt   # for plotting 
import seaborn as sns             # an extension of matplotlib for statistical graphics
products = pd.read_csv('../input/products.csv')
products.head()
products.shape
products.info()
orders = pd.read_csv('../input/orders.csv' )
orders.head(10)
orders.days_since_prior_order.value_counts()
plt.figure(figsize=(15,5))

sns.countplot(x="days_since_prior_order", data=orders, color='grey')

plt.ylabel('Total Orders')
plt.xlabel('Days since prior order')
plt.title('Days passed since previous order')

plt.show()
order_volume = orders.user_id.value_counts()
order_volume.tail()
plt.figure(figsize=(15,5))
graph = sns.countplot(x=order_volume, data=orders, color='green')
plt.ylabel('Number of orders', fontsize=10)
plt.xlabel('User ID', fontsize=10)
plt.title('Number of costumers per volume of orders')
plt.show()

plt.figure(figsize=(15,5))
graph = sns.countplot(x=order_volume, data=orders)
graph.set( xticks=[0, 100], xticklabels=['4 orders', '100 orders' ] )
plt.show()