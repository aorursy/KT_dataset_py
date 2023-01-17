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

sns.countplot(x="days_since_prior_order", data=orders, color='red')

plt.ylabel('Total Orders')
plt.xlabel('Days since prior order')
plt.title('Days passed since previous order')

plt.show()
#By using the value_counts() method on [orders.user_id] we get how many times each [user_id] appears, therefore how many orders each user_id (each customer) made
#Storing the result as is to [order_volume] generates a pandas.Series with the index being the [user_id] and the values of the Series how many orders each [user_id] made
#The problem here is that tha values corresponding to the amount of orders each customer made, take the label [user_id] while the actual [user_id]s are the indexes
#We can better see that by converting the Series to Dataframe using the [to_frame] method and inspecting with [head()]
#Nevertheless the code works as is since the values in the Series are accurate but we need to understand this now in order to present our results correctly


order_volume = orders.user_id.value_counts()
order_volume.tail()

df = order_volume.to_frame()
df.head()

plt.figure(figsize=(15,5))
graph = sns.countplot(x=order_volume)

plt.show()
#In the previous figure we can see that the axes are labeled 'user_id' and 'count' because of the label [user_id] being on the wrong values as explained before
#This is not representative of what we are trying to show and is misguiding to the reader as the axes represent entirely different things
#The x-axes represents the amount of orders made with bars above the x-axes values.
#Each bar corresponds to the values on the y-axes showing how many customers made that amount of orders.
#Therefore we label the graph accordingly, so for example we can see from the 1st bar that almost [24000 customers(y-axes)] made [4 orders(x-axes)] in total.

plt.figure(figsize=(15,5))
graph = sns.countplot(x=order_volume)
graph.set( xticks=[0, 96], xticklabels=[4, 100] )
graph.set_title('Distribution of Customers per volume of Orders')
graph.set_xlabel('Amount of Orders')
graph.set_ylabel('Amount of Customers')
plt.show()
