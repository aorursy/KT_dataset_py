import pandas as pd               # for data manipulation
import matplotlib.pyplot as plt   # for plotting 
import seaborn as sns             # an extension of matplotlib for statistical graphics
products = pd.read_csv('../input/products.csv')
products.head(1)
products.shape
products.info()
orders = pd.read_csv('../input/orders.csv' )
orders.head(100)
orders.days_since_prior_order.count()
plt.figure(figsize=(15,5))

sns.countplot(x="days_since_prior_order", data=orders, color='red')

plt.ylabel('Total Orders')
plt.xlabel('Days since prior order')
plt.title('Days passed since previous order')

plt.show()