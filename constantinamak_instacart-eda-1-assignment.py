import pandas as pd               # for data manipulation
import matplotlib.pyplot as plt   # for plotting 
import seaborn as sns             # an extension of matplotlib for statistical graphics
products = pd.read_csv('../input/products.csv')
products.head()
products.shape
products.info()
orders = pd.read_csv('../input/orders.csv' )
orders.head()
orders.days_since_prior_order.value_counts()
plt.figure(figsize=(15,5))

sns.countplot(x='days_since_prior_order', data=orders, color='red')

plt.ylabel('Total Orders')
plt.xlabel('Days since prior order')
plt.title('Days passed since previous order')

plt.show()
order_volume = orders.user_id.value_counts()
order_volume.tail()
plt.figure(figsize=(15,5))
graph = sns.countplot(order_volume)
plt.show()
plt.figure(figsize=(15,5))
graph = sns.countplot(order_volume)
graph.set( xticks=list(range(0,101,4)), xticklabels=list( range(0,101,4) ) )
plt.show()