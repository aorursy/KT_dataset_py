import pandas as pd               # for data manipulation
import matplotlib.pyplot as plt   # for plotting 
import seaborn as sns             # an extension of matplotlib for statistical graphics
products = pd.________('../input/products.csv')
products.____()
products._____
products.____()
orders = __.______('../input/orders.csv' )
orders.____()
orders.days_since_prior_order.___________()
plt.figure(figsize=(15,5))

sns.countplot(x="___________", data=_______, color='red')

plt.ylabel('Total Orders')
plt.xlabel('Days since prior order')
plt.title('Days passed since previous order')

plt.show()
order_volume = _____._____.value_counts()
order_volume.tail()
plt.figure(figsize=(15,5))
graph = sns.countplot(_______)
plt.show()
plt.figure(figsize=(15,5))
graph = sns.countplot(_____)
graph.set( xticks=[0, 96], xticklabels=[__, __] )
plt.show()