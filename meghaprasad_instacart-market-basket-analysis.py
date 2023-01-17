import pandas as pd
import numpy as np
import seaborn as sns
import sklearn as sk
import sklearn.tree as tree
from IPython.display import Image  
#import pydotplus
import matplotlib.pyplot as plt
import squarify 
%matplotlib inline
pd.set_option('display.width', 1000)
# limited dataset to ensure kaggle is happy
orders = pd.read_csv('../input/orders.csv', nrows= 2000000)
products = pd.read_csv('../input/products.csv', nrows= 2000000)
departments = pd.read_csv('../input/departments.csv')
aisles = pd.read_csv('../input/aisles.csv')
order_products_prior = pd.read_csv('../input/order_products__prior.csv', nrows= 2000000)
order_products_train = pd.read_csv('../input/order_products__train.csv', nrows= 2000000)
orders.head(2)
products.head(2)
departments.head(2)
aisles.head(2)
order_products_prior.head(2)
order_products_train.head(2)
df_train = order_products_train.copy().merge(orders, left_on='order_id', right_on='order_id')
df_train.head(2)
df_prior = order_products_prior.copy().merge(orders, left_on='order_id', right_on='order_id')
df_prior.head(2)
orders.eval_set.unique()
order_set = orders.copy()
order_set = orders[orders.eval_set != 'test']
order_set.eval_set.unique()
order_set =orders.drop(['eval_set'], axis =1)
order_set.head(2)
len(order_set)
order_products = pd.concat([order_products_prior, order_products_train])
order_products.head()
len(order_products)
df = orders.merge(order_products, on='order_id')
Top_products = pd.DataFrame({'Size': df.groupby('product_id').size()}).sort_values('Size', ascending=False)\
.reset_index()[:2000]
Top_products = Top_products.merge(products, on='product_id')
df = df.loc[df['product_id'].isin(Top_products.product_id)]
product_orders_by_hour = pd.DataFrame({'Count': df.groupby(['product_id', 'order_hour_of_day']).size()})\
.reset_index()
product_orders_by_hour.head(24)
product_orders_by_hour['pct'] = product_orders_by_hour.groupby('product_id')['Count'].apply(lambda x: x/x.sum()*100)
product_orders_by_hour.head(24)
order_count = product_orders_by_hour.groupby('order_hour_of_day')['Count'].sum()
order_count= order_count.reset_index()
sns.factorplot(x ='order_hour_of_day',y ='Count',  data = order_count, kind ='bar', aspect = 3)
def MeanHour(x):
    return sum(x['order_hour_of_day'] * x['Count'])/sum(x['Count'])
MeanHour = pd.DataFrame({'MeanHour': product_orders_by_hour.groupby('product_id').apply(MeanHour)}).reset_index()
MeanHour.head(3)
Morning = MeanHour.sort_values('MeanHour')[:25]
Morning = Morning.merge(products, on='product_id')
Morning.head()
Late = MeanHour.sort_values('MeanHour', ascending=False)[:25]
Late = Late.merge(products, on='product_id')
Late.head()
# Create MorningPct table to get count of product_id with MeanHour
MorningPct = product_orders_by_hour.merge(Morning, on='product_id')
MorningPct=MorningPct.sort_values(['MeanHour', 'order_hour_of_day'])
# Create larePct table to get count of product_id with MeanHour
LatePct = product_orders_by_hour.merge(Late, on='product_id')
LatePct =LatePct.sort_values(['MeanHour', 'order_hour_of_day'], ascending=False)
Morning_ProductName = list(MorningPct['product_name'].unique())
Morning_ProductName = '\n'.join(Morning_ProductName)
Late_ProductName = list(LatePct['product_name'].unique())
Late_ProductName = '\n'.join(Late_ProductName)
fig, ax = plt.subplots(figsize=(12, 8))

MorningPct.groupby('product_id').plot(x='order_hour_of_day', y='pct', ax=ax,legend=False,alpha =0.2,aa=True, color='green',
                                       linewidth=1.0,)

LatePct.groupby('product_id').plot(x='order_hour_of_day', y='pct', ax=ax, legend= False,alpha=0.2, aa=True,color='red',
                                   linewidth=1.0,)
plt.margins(x=0.5, y=0.05)

label_font_size = 13
plt.xlabel('Hour of Day Ordered', fontsize= label_font_size)
plt.ylabel('Percent of Orders by Product', fontsize=label_font_size)


tick_font_size = 10
ax.tick_params(labelsize=tick_font_size)
plt.xticks(range(0, 25, 2))
plt.yticks(range(0, 16, 5))
plt.xlim([-2, 28])

text_font_size = 9
ax.text(0.01, 1.0, Morning_ProductName,verticalalignment='top', horizontalalignment='left',transform=ax.transAxes,
        color='green', fontsize=text_font_size)
ax.text(0.99, 1.0, Late_ProductName,verticalalignment='top', horizontalalignment='right',
        transform=ax.transAxes,
        color='red', fontsize=text_font_size);
order_size_reorder = df_prior.groupby(['order_id']).agg({'add_to_cart_order':'max','reordered':'sum'})\
                    .rename(columns={'reordered':'Count of Reordered Items', 'add_to_cart_order': 'Order Size'})
order_size_reorder.head(2)
avg_order_size = order_size_reorder.groupby('Order Size').agg({'Count of Reordered Items': 'mean'}).reset_index()
sns.factorplot(x='Order Size', y='Count of Reordered Items', data=avg_order_size, aspect=4, kind='point')
order_size_reorder.corr()
order_size_by_order_number = order_size_reorder.copy()
order_size_by_order_number
order_size_by_order_number = order_size_by_order_number.merge(orders, left_on='order_id', right_on='order_id')
order_size_by_order_number.head()
order_size_by_order_number = order_size_by_order_number.groupby('order_number').agg({'Order Size': 'mean'}) \
                            .reset_index()
order_size_by_order_number['order_number_bin'] = pd.cut(order_size_by_order_number['order_number'],\
                            bins=[0,10,20,30,40,50,60,70,80,90,100])
sns.factorplot(x='order_number_bin', y='Order Size', data=order_size_by_order_number, aspect=4)
first60_orders = df_prior.copy().loc[df_prior.order_number <= 60]
first60_orders = first60_orders.merge(products, left_on='product_id', right_on='product_id')
first60_orders = first60_orders.merge(departments, left_on='department_id', right_on='department_id')
first60_orders = first60_orders.merge(aisles, left_on='aisle_id', right_on='aisle_id')
first60_orders
first60_orders.drop(['eval_set', 'order_dow', 'order_hour_of_day','department_id', 'aisle_id'], axis=1, inplace=True)
order_number_by_aisle = first60_orders.groupby(['order_number', 'aisle']) \
        .agg({'order_number':'count', 'days_since_prior_order':'mean','reordered':'mean', \
              'add_to_cart_order':'mean','product_name':'first', 'department':'first'}) \
        .rename(columns={'order_number':'count', 'days_since_prior_order':'avg_days_prior', \
                         'add_to_cart_order':'avg_cart_position'}) \
        .sort_values(by=['order_number','count'], ascending=False).reset_index()
order_number_by_aisle.sort_values(by='count', ascending=False)
order_size_change = order_number_by_aisle.groupby('aisle')\
                    .agg({'count': lambda x: x.max()-x.min(), 'department': 'first'}) \
                    .rename(columns={'count':'spread'})
order_size_change
order_size_change.nlargest(3, 'spread')
order_size_change.nsmallest(3, 'spread')
top3_aisle_order_number = order_number_by_aisle.groupby('order_number').head(3)
top3_aisle_order_number
sns.factorplot(x='order_number', y='count', hue='aisle', data=top3_aisle_order_number, kind='bar', aspect=3,\
               legend_out=False)
order_number_by_aisle.department.unique()
order_number_by_aisle.head()
dept_price_est = {
    'produce': 2.00, 
    'dairy eggs': 3.00, 
    'beverages': 2.00, 
    'snacks': 3.00, 
    'bakery': 3.00,
    'babies': 5.00, 
    'deli': 3.00,
    'frozen': 5.00, 
    'dry goods pasta':3.00, 
    'bakery': 3.00,
    'meat seafood': 5.00,
    'canned goods': 2.00,
    'pantry': 3.00,
    'breakfast': 3.00,
    'missing': 2.00,
    'international': 5.00,
    'household': 7.00, 
    'pets': 10.00, 
    'other': 3.00, 
    'personal care': 4.00, 
    'alcohol': 2.00, 
    'bulk': 2.00
}
orders_by_dept = first60_orders.groupby(['user_id','order_number', 'department']) \
        .agg({'product_name':'count'}) \
        .rename(columns={'product_name':'item_count'}).reset_index()
orders_by_dept.sort_values(by='user_id')
orders_by_dept['amount'] = orders_by_dept['department'].apply(lambda x: dept_price_est[x])
orders_by_dept['amount'] = orders_by_dept['amount']*orders_by_dept['item_count']
orders_by_dept_data = orders_by_dept.groupby(['order_number', 'department']).agg({'item_count':'mean', \
                                                                                  'amount':'mean'}).reset_index()
orders_by_dept_data
user_orders = orders_by_dept.groupby(['user_id', 'order_number']).agg({'item_count':'sum', 'amount':'sum'})\
                .rename(columns={'amount': 'item_total'}).reset_index()
user_orders
user_orders['delivery_fee'] = user_orders['order_number'].apply(lambda x: 5.99 if x > 1 else 0)
user_orders['service_fee'] = user_orders['item_total'].apply(lambda x: x*0.1 if x >= 12 else 0)
user_orders['order_total'] = user_orders['item_total'] + user_orders['delivery_fee'] + user_orders['service_fee']
user_orders['instacart_margin'] = user_orders['item_total'] * 0.05
user_orders['instacart_earnings'] = user_orders['instacart_margin'] + user_orders['service_fee']\
                                    + (user_orders['service_fee'] * 0.05)
user_orders
avg_total_by_order_number = user_orders.groupby('order_number') \
                            .agg({'item_total': 'mean', 'order_total': 'mean', 'instacart_earnings':'mean'}) \
                            .reset_index()
avg_total_by_order_number
sns.set(font_scale=2)
fig, ax = plt.subplots(figsize=(30,10))
plt.plot(avg_total_by_order_number.order_number, avg_total_by_order_number.order_total, color='red')
plt.plot(avg_total_by_order_number.order_number, avg_total_by_order_number.item_total, color='orange')
plt.title("Order Total (USD) by Order Number", loc='center', fontsize=14, fontweight=0, color='black')
plt.xlabel("Order Number")
plt.ylabel("Amount (USD)")
plt.legend()
plt.axvline(31,linestyle='--')
sns.set(font_scale=1)
sns.factorplot(x='order_number', y='instacart_earnings', data=avg_total_by_order_number, aspect=4, color='green')
plt.title("Order Total (USD) by Order Number", loc='center', fontsize=14, fontweight=0, color='black')
plt.xlabel("Order Number")
plt.ylabel("Amount (USD)")
plt.axvline(31,linestyle='--')
order_number_by_aisle['avg_days_prior'].fillna(0)
avg_days_prior_by_order = order_number_by_aisle.groupby('order_number').agg({'avg_days_prior':'mean'})
avg_days_prior_by_order['days_cumsum'] = avg_days_prior_by_order['avg_days_prior'].cumsum()
avg_days_prior_by_order
sns.factorplot(x='order_number', y='days_cumsum', data=avg_days_prior_by_order.reset_index(), aspect=4)
plt.title("Day Intervals Between Orders", loc='center', fontsize=14, fontweight=0, color='black')
plt.xlabel("Order Number")
plt.ylabel("No. of Days")
product_reorder = df_train.groupby(['product_id','reordered']).agg({'add_to_cart_order': 'mean'})
import math
product_reorder['add_to_cart_order'] = product_reorder['add_to_cart_order'].apply(lambda x: math.ceil(x))
product_reorder.reset_index(inplace = True)
X = product_reorder.drop(['product_id', 'reordered'],axis=1)
Y = product_reorder.reordered
dt = tree.DecisionTreeClassifier(max_depth=2)
dt.fit(X,Y)
dt_feature_names = list(X.columns)
dt_target_names = [str(s) for s in Y.unique()]
tree.export_graphviz(dt, out_file='add-to-cart-order-tree.png', 
    feature_names=dt_feature_names, class_names=dt_target_names,
    filled=True)  
#graph = pydotplus.graph_from_dot_file('add-to-cart-order-tree.png')
#Image(graph.create_png())
reordered_count = product_reorder.groupby(['add_to_cart_order','reordered']) \
                    .agg({'reordered':'count'}) \
                    .rename(columns={'reordered':'reorder_count'})
reordered_count['add_to_cart_order_total'] = reordered_count.groupby(level=0)['reorder_count'].transform('sum')
reordered_count['reorder_prob'] = reordered_count['reorder_count'] / reordered_count['add_to_cart_order_total']
reordered_count.head(2)
reordered_count.reset_index(inplace=True)
sns.factorplot(y='reorder_count', x='add_to_cart_order', hue='reordered', \
               data=reordered_count.loc[reordered_count.add_to_cart_order < 35] \
               ,aspect=3, legend_out=False)
reordered_count['add_to_cart_order_bin'] = pd.cut(reordered_count['add_to_cart_order'],\
                            bins=[0,8.5,9.5,10.5, 100])
sns.factorplot(y='reorder_prob', x='add_to_cart_order_bin', hue='reordered', data=reordered_count, kind='bar',aspect=3)
top9_cart_position_products = df_train.loc[df_train.add_to_cart_order <= 9.5]
top9_cart_position_products = top9_cart_position_products.merge(products, left_on='product_id', right_on='product_id') 
top9_cart_position_products = top9_cart_position_products.merge(departments, left_on='department_id', right_on='department_id') 
top9_cart_position_products.drop(['aisle_id', 'department_id', 'product_id'], axis=1, inplace=True)
top9_cart_position_products.sort_values(by='add_to_cart_order').head(5)
top9_cart_position_products.groupby(['department', 'product_name']) \
                                .agg({'department': 'count'}).rename(columns={'department':'product_count'})
top9_cart_position_dept = top9_cart_position_products.groupby(['department']) \
                                .agg({'department': 'count'}).rename(columns={'department':'product_count'})
top9_cart_position_dept.reset_index(inplace=True)
top9_cart_position_dept.head(2)
fig = plt.gcf()
ax = fig.add_subplot()
fig.set_size_inches(10, 8)
colors = ["#e0301e", "#602320","#a32020","#eb8c00","#dc6900","#4d7358","#e39e54", "#e8d174", "#326ada","#433e90","#a19c9c"]
#colors = ['#dc6900', "#ffeead", "#ff6f69", "#ffcc5c", "#602320", "#e0301e", "#c1242f","#65c25e", "#6d7371", "#b9bab6"]
squarify.plot(sizes=top9_cart_position_dept.product_count, label=top9_cart_position_dept.department, alpha=.6, color=colors)
plt.title("Products Added to Cart Prior 9.5th Position by Departments",fontsize=18,fontweight="bold")
plt.axis('off')
plt.show()
top9_cart_position_products.groupby(['department', 'product_name']) \
    .agg({'department': 'count'}).rename(columns={'department':'product_count'}) \
    .reset_index().sort_values('product_count', ascending=False).head(5)
product_df=products.copy().merge(departments, on ='department_id',how ='left').merge(aisles, on ='aisle_id',how ='left')
product_df.head()
PO =df_train.merge(product_df , on ='product_id', how ='left')
Top_Aisle = PO.copy()
sns.distplot(Top_Aisle['days_since_prior_order'].fillna(0).astype(int));
user_days_since= Top_Aisle.groupby(['user_id', 'order_id']).apply(lambda x: x.iloc[0]['days_since_prior_order']).rename('days_since').reset_index()
user_days_since = user_days_since.groupby('user_id').apply(lambda x: x['days_since'].mean()).rename('mean_time').reset_index()
user_days_since.head()
weekly_users = user_days_since[user_days_since['mean_time'] < 8]
monthly_users = user_days_since[user_days_since['mean_time'] > 21]
monthly_data = Top_Aisle.merge(monthly_users, on='user_id',how='inner')
weekly_data = Top_Aisle.merge(weekly_users, on='user_id',how='inner')
WeeklyDF=weekly_data.groupby('department').size().rename('counts').reset_index().sort_values('counts', ascending=False)
WeeklyDF.head()
MonthlyDF=monthly_data.groupby('department').size().rename('counts').reset_index().sort_values('counts', ascending=False)
MonthlyDF.head()
sns.factorplot(x ='department',y ='counts',  data = MonthlyDF, kind ='bar', aspect = 5)
sns.factorplot(x ='department',y ='counts',  data = WeeklyDF, kind ='bar', aspect = 5)
reorder_by_dept = PO.copy()
reorder_by_dept.head()
reorder_by_dept = reorder_by_dept.groupby('department_id')\
                    .agg({'department':'first', 'days_since_prior_order':'mean', 'reordered':'mean'})
reorder_by_dept.head()
reorder_by_dept['days_since_prior_order'] = reorder_by_dept['days_since_prior_order'].apply(lambda x: math.ceil(x))
sns.factorplot(x='department', y='days_since_prior_order', data=reorder_by_dept, kind='bar', aspect=4)