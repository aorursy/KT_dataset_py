import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import re
import seaborn as sns
color = sns.color_palette()

# Limit floats output to 3 decimal points
pd.set_option('display.float_format', lambda x: '%.3f' % x)

plt.style.use('fivethirtyeight')
%matplotlib inline 

#Supress unnecessary warnings for readability and cleaner presentation
import warnings
warnings.filterwarnings('ignore') 

# Increase default figure and font sizes for easier viewing.
plt.rcParams['figure.figsize'] = (8, 6)
plt.rcParams['font.size'] = 14

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
from subprocess import check_output
print(check_output(["ls", "../input/instacart-market-basket-analysis/"]).decode("utf8"))
order_products_train_df = pd.read_csv("../input/instacart-market-basket-analysis/order_products__train.csv")
order_products_prior_df = pd.read_csv("../input/instacart-market-basket-analysis/order_products__prior.csv")
orders_df = pd.read_csv("../input/instacart-market-basket-analysis/orders.csv")
products_df = pd.read_csv("../input/instacart-market-basket-analysis/products.csv")
aisles_df = pd.read_csv("../input/instacart-market-basket-analysis/aisles.csv")
departments_df = pd.read_csv("../input/instacart-market-basket-analysis/departments.csv")
orders_df.head()
# Check for
orders_df.isnull().sum()
# Set NaN to zeros

orders_df = orders_df.reset_index()
orders_df.isnull().sum()
orders_df.head()
products_df.head()
departments_df.head()
aisles_df.head()
order_products_prior_df.head()
order_products_train_df.head()
print(aisles_df.shape, products_df.shape, departments_df.shape, order_products_prior_df.shape,
      order_products_train_df.shape, orders_df.shape)
orders_df.columns
combine_dataset = orders_df.groupby('eval_set')['order_id'].aggregate({'Total_orders': 'count'}).reset_index()

combine_dataset
combine_dataset  = combine_dataset.groupby(['eval_set']).sum()['Total_orders'].sort_values(ascending=False)

sns.set_style('whitegrid')
f, ax = plt.subplots(figsize=(10,10))
sns.barplot(combine_dataset.index, combine_dataset.values, palette="RdBu")
plt.ylabel('Number of Orders', fontsize=14)
plt.title('Types of Datasets', fontsize=16)
plt.show()
# Approach with inner joins on 'products_df' & 'departments_df'
# Aisle by 'department_id' and 'aisle_id'

product_combine = products_df.reset_index().set_index('department_id').join(departments_df, how="inner")
product_combine = product_combine.reset_index().set_index('aisle_id').join(aisles_df, how="inner")

product_combine.head()
product_combine.head()
"""
product_combine = product_combine.reset_index().set_index('product_id')
product_combine.sort_index(axis=0, ascending= True, kind= 'quicksort', inplace= True)
"""
order_products_train_df.head()
orders_df.columns.values
dayofweek = orders_df.groupby('order_id')['order_dow'].aggregate("sum").reset_index()

dayofweek = dayofweek.order_dow.value_counts()
sns.set_style('darkgrid')
f, ax = plt.subplots(figsize=(10, 10))
sns.barplot(dayofweek.index, dayofweek.values, palette="RdBu")
plt.ylabel('Number of Orders', fontsize=13)
plt.xlabel('Days of Order in a Week', fontsize=13)
plt.title('Number of Orders from Each Day of the Week', fontsize = 16)
plt.show()
orders_df.columns.values
orders_df.head()
timeofday = orders_df.groupby('order_id')['order_hour_of_day'].aggregate("sum").reset_index()

timeofday = timeofday.order_hour_of_day.value_counts()

sns.set_style('darkgrid')
f, ax = plt.subplots(figsize=(15,10))
sns.barplot(timeofday.index, timeofday.values, palette="Blues_d")
plt.ylabel('Number of Orders', fontsize=14)
plt.xlabel('Time of Day of Orders', fontsize=14)
plt.show()
# Selecting a small sample size for kernel density axes 
smallset = orders_df[0:100000]

# Use KDE plot to depict the probability densities at different values in continuous variable.
day_vs_hours = sns.jointplot(x="order_hour_of_day", y="order_dow", data=smallset, kind="kde", color="dodgerblue")
day_vs_hours.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+")
day_vs_hours.ax_joint.collections[0].set_alpha(0)
day_vs_hours.set_axis_labels("Hour of Day (24 hour format)", "Day of the week")
day_vs_sincepriororder = sns.jointplot(smallset.days_since_prior_order, smallset.order_dow, data=smallset, kind="kde", color="dodgerblue")
day_vs_sincepriororder.set_axis_labels("Days Since Last Order", "Day of Week")
orders_df.columns
prior_order_dist.head()
# Generating a dataframe with one column 'days_since_prior_order'

prior_order_dist = orders_df[['days_since_prior_order']]

sns.set_style('darkgrid')
f, ax = plt.subplots(figsize=(15,10))
sns.barplot(prior_order_dist.index, palette="Reds_d")
plt.ylabel('Number of Orders', fontsize=14)
plt.xlabel('Days Since Last Order', fontsize=14)
plt.show()
# Merging product_id, aisle_id, department_id from products_df, aisles_df, departments_df into order_products_prior_df

# This will allow me to pull out and aggregate column values to generate product distribution by department.


order_products_prior_df = pd.merge(order_products_prior_df, products_df, on='product_id', how='left')
order_products_prior_df = pd.merge(order_products_prior_df, aisles_df, on='aisle_id', how='left')
order_products_prior_df = pd.merge(order_products_prior_df, departments_df, on='department_id', how='left')
order_products_prior_df.head()
plt.figure(figsize=(10,10))
temp_series = order_products_prior_df['department'].value_counts()
labels = (np.array(temp_series.index))
sizes = (np.array((temp_series / temp_series.sum())*100))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=200)
plt.title("Departments Distribution", fontsize=15)
plt.show()
order_products_prior_df.head()
order_products_prior_df.columns
freq_rereorder = order_products_prior_df.groupby('reordered')['product_id'].aggregate({'Total_Products': 'count'}).reset_index()

freq_rereorder['Ratios'] = freq_rereorder["Total_Products"].apply(lambda x: x / freq_rereorder['Total_Products'].sum())

freq_rereorder
freq_rereorder  = freq_rereorder.groupby(['reordered']).sum()['Total_Products'].sort_values(ascending=False)

sns.set_style('whitegrid')
f, ax = plt.subplots(figsize=(5, 8))
sns.barplot(freq_rereorder.index, freq_rereorder.values, palette='muted')
plt.ylabel('Number of Products', fontsize=13)
plt.xlabel('Reorder Frequency', fontsize=13)
plt.ticklabel_format(style='plain', axis='y')

plt.show()
order_products_train_df.head(5)
order_products_prior_df.head(5)
# Combine files together via concatenation in dataframe 'order_products_all'
# Double check new sum.

order_products_all = pd.concat([order_products_train_df, order_products_prior_df], axis = 0)

print("order_products_all size is : ", order_products_all.shape)
order_products_all.columns
# Aggregate columns product_id, Reorder_Sum, and Reorder_Total:
mostreordered = order_products_all.groupby('product_id')['reordered'].aggregate({'Reorder_Sum': sum,'Reorder_Total': 'count'}).reset_index()

# Add column for probability for reorder for each product_id:
mostreordered['Probability_of_Reorder'] = mostreordered['Reorder_Sum']/mostreordered['Reorder_Total']

mostreordered
# Add product names associated with their ID's:
mostreordered = pd.merge(mostreordered,products_df[['product_id','product_name']])

# Sort from highest probability:
mostreordered = mostreordered.sort_values(['Probability_of_Reorder'], ascending=False)

mostreordered
order_products_all.columns
order_products_prior_df.columns