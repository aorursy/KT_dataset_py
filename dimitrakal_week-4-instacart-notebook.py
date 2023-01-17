import pandas as pd               # for data manipulation
import matplotlib.pyplot as plt   # for plotting 
import seaborn as sns             # an extension of matplotlib for statistical graphics
orders = pd.read_csv('../input/orders.csv' )
orders.shape
orders.info()
#the argument in .head() represents how many first rows we want to get.
orders.head(12)
#1. Import departments.csv from directory: ../input/departments.csv'
departments = pd.read_csv('../input/departments.csv')
departments.head(10)
departments.shape
departments.info()
orders.days_since_prior_order.max()
orders.days_since_prior_order.mean()
orders.days_since_prior_order.median()
# alternative syntax: orders.days_since_prior_order.plot(kind='box')
orders.boxplot('days_since_prior_order')
orders.head()
order_hours = orders.order_hour_of_day.value_counts()
order_hours
#alternative syntax : order_hours.plot(kind='bar')
order_hours.plot.bar()
#Remember that the alias that we have defined for seaborn is the sns.
sns.countplot(x="order_hour_of_day", data=orders)
# Step one - define the dimensions of the plot (15 for x axis, 5 for y axis)
plt.figure(figsize=(15,5))

# Step two - define the plot that we want to produce with seaborn
# Here we also define the color of the bar chart as 'red'
sns.countplot(x="order_hour_of_day", data=orders, color='red')

# Step three - we define the name of the axes and we add a title to our plot
# fontsize indicates the size of the titles
plt.ylabel('Total Orders', fontsize=10)
plt.xlabel('Hour of day', fontsize=10)
plt.title("Frequency of order by hour of day", fontsize=15)

# Step four - we produce our plot
plt.show()
sns.countplot(x='order_dow' , data=orders )
plt.figure(figsize=(10,10))
sns.countplot(x="order_dow", data=orders, color='blue')
plt.ylabel('Total orders', fontsize=10)
plt.xlabel('Day of the week', fontsize=10)
plt.title("Frequensy of day of the week", fontsize=15)
plt.show()
orders_first = orders[orders['order_number']==1]
orders_first.head()
orders_second = orders[orders['order_number']==2]
orders_second.head()
#create a subplot which contains two plots; one down the other
fig, axes = plt.subplots(nrows=2, ncols=1,figsize=(15,8))

#assign each plot to the appropiate axes
sns.countplot(ax=axes[0], x='order_dow', data=orders_first, color='red')
sns.countplot(ax=axes[1], x='order_dow', data=orders_second, color='red')

# produce the final plot
plt.show()
orders.head(15)
order_count = orders.order_number.value_counts()
order_count
# Set size 15x5 and bar color red
plt.figure(figsize=(15,5))
sns.countplot(x='order_number', data=orders, color='red')
plt.ylabel('Total Customers', fontsize=10)
plt.xlabel('Total Orders', fontsize=10)
plt.show()
#import the required function
import matplotlib.ticker as ticker

plt.figure(figsize=(15,5))

#assign plot in a variable
ax = sns.countplot(x='order_number', data=orders, color='red')
plt.ylabel('Total Customers', fontsize=10)
plt.xlabel('Total Orders', fontsize=10)

#select the step for xticks (2)
ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())

plt.show()