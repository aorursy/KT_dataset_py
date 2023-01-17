import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from IPython.display import display

%matplotlib inline

plt.style.use('ggplot')
orders = pd.read_csv('../input/orders.csv', index_col='order_id', dtype={'order_id':'int32', 

                                                               'user_id':'int32',

                                                               'eval_set':'category', 

                                                               'order_dow':'category', 

                                                               'order_hour_of_day':'int8', #could also be category

                                                               'days_since_prior_order':'float16'})

products = pd.read_csv('../input/products.csv', index_col='product_id', dtype={'product_id':'int32', 

                                                                   'product_name':'object', 

                                                                   'aisle_id':'int16', 

                                                                   'department_id':'int16'})

order_products_train = pd.read_csv('../input/order_products__train.csv', dtype={'order_id':'int32',

                                                                                     'product_id':'int32',

                                                                                     'add_to_cart_order':'int8',

                                                                                     'reordered':'uint8'})

order_products_prior = pd.read_csv('../input/order_products__prior.csv', dtype={'order_id':'int32',

                                                                                     'product_id':'int32',

                                                                                     'add_to_cart_order':'int8',

                                                                                     'reordered':'uint8'})

aisles = pd.read_csv('../input/aisles.csv', index_col='aisle_id', 

                     dtype={'aisle_id':'int16', 'aisle':'category'})

departments = pd.read_csv('../input/departments.csv', index_col='department_id', 

                          dtype={'department_id':'int8', 'department':'category'})
plt.figure(figsize=(12, 8))

plt.title('Order Count by Hour of Day (24hr)', fontweight='bold')

plt.ylabel('number of orders')

plt.xlabel('hour of day')

plt.hist(orders.order_hour_of_day, bins=np.arange(25), width=0.9, facecolor='green', alpha=0.6)

plt.grid(axis='x')

plt.show()
plt.clf()

days=['sun', 'mon', 'tue', 'wed', 'thu', 'fri', 'sat']

plt.figure(figsize=(8,5))

plt.hist(orders.order_dow.astype('int8'), bins=np.arange(8), width=0.9, facecolor='green', alpha=0.6)

plt.xticks(np.arange(7), ('sun', 'mon', 'tue', 'wed', 'thu', 'fri', 'sat'))

plt.title('Order Count by day of week', fontweight='bold')

plt.xlabel('day of week')

plt.ylabel('number of orders')

plt.show()

plt.figure(figsize=(12, 8))

plt.title('Order Count by Hour of Day (24hr)', fontweight='bold')

plt.ylabel('frequency')

plt.xlabel('days since prior order')

plt.hist(orders.days_since_prior_order.dropna(), bins=np.arange(32), width=0.9, facecolor='green', alpha=0.6)

plt.grid(axis='x')

plt.xticks(np.arange(31))

plt.show()
unique, counts = np.unique(orders.user_id.value_counts().values, return_counts=True)

plt.clf()

ind = np.arange(100)

fig=plt.figure(figsize=(15,8))

fig.add_axes()



ax1 = fig.add_subplot(111)

ax1.bar(unique, counts, color='g', alpha=0.8)

ax1.set_title('number of prior orders', fontweight='bold')

ax1.set_xlabel('count of user_id', fontsize=15)

ax1.set_ylabel('frequency', fontsize=15)



plt.show()
y_train = order_products_train.groupby('order_id').add_to_cart_order.max().value_counts().sort_index().values

x_train = order_products_train.groupby('order_id').add_to_cart_order.max().value_counts().sort_index().index.values

y_prior = order_products_prior.groupby('order_id').add_to_cart_order.max().value_counts().sort_index().values

x_prior = order_products_prior.groupby('order_id').add_to_cart_order.max().value_counts().sort_index().index.values
plt.clf()

import matplotlib.patches as mpatches



fig, (ax1, ax2, ax3) = plt.subplots(3,1, figsize=(15,15))



ax1.bar(x_train, y_train, color='b', alpha=0.8)

ax1.set_title('number of items per order', fontweight='bold')

ax1.set_ylabel('frequency (log)', fontsize=15)



ax1.set_xlim(0,130)

ax1.set_yscale('log')

ax1.set_xticks(np.arange(0,130, 5))

ax1.bar(x_prior, y_prior, color='r', alpha=0.4)

red_patch = mpatches.Patch(color='red', label='Prior Data')

purple_patch = mpatches.Patch(color='purple', label='Train Data')

ax1.legend(handles=[red_patch, purple_patch], fontsize='xx-large')



ax2.bar(x_prior, y_prior, color='r', alpha=0.8)

#ax2.set_title('prior data')

ax2.set_xlim(0,60)

ax2.set_xticks(np.arange(0,60, 5))

ax2.text(30,100000,

         'x max is 127, but x-scale limited to 60 \nin order to compare prior/train more easily.',

         bbox=dict(facecolor='white', alpha=0.6), fontsize='x-large')



ax3.bar(x_train, y_train, color='purple', alpha=0.8)

#ax3.set_title('train data')

ax3.set_xlim(0, 60)

ax3.set_xticks(np.arange(0,60,5))

ax3.set_xlabel('item count', fontsize=15)







plt.show()
top_n=10

print ('Top {0} Bestselling Items'.format(top_n))

temp = pd.merge(left=order_products_train.product_id.value_counts().nlargest(top_n).to_frame('count'), right=products[['product_name']], left_index=True, right_index=True)



display(temp)

temp.plot(x='product_name', y='count', kind='bar', figsize=(12,8), width=.9, fontsize='large', alpha=0.8)
temp = order_products_train['reordered'].value_counts().to_frame('count')

temp['proportion (%)'] = (temp['count']/temp['count'].sum()*100).round(2)

display(temp)



plt.clf()

ax = temp.plot(y='count', kind='bar', figsize=(2,4), legend=False, width=.9, alpha=0.8)

ax.set_xticklabels(['reordered', 'new'])

plt.show()



print ('{0}% of the ordered items are reordered'.format(temp.iloc[0,1]))
temp = pd.merge(left=order_products_train.groupby('product_id').reordered.mean().to_frame('proportion_reordered'), 

                right=products, left_index=True, right_index=True)

temp = pd.merge(left=temp,

                right=order_products_train.product_id.value_counts().to_frame('count'),  

                left_index=True, right_index=True)

temp = pd.merge(left=temp,

                right=pd.DataFrame(order_products_train.groupby('product_id').reordered.sum().to_frame(), dtype='int64'),  

                left_index=True, right_index=True)

temp.index.rename('product_id', inplace=True)





display(temp[temp['count']>35].nlargest(10, 'proportion_reordered').sort_values(['proportion_reordered'], ascending=False))



plt.clf()

ax = temp[temp['count']>35].nlargest(10, 'proportion_reordered').plot(x='product_name', 

                                                                      y='proportion_reordered', 

                                                                      kind='bar', 

                                                                      legend=False,

                                                                      figsize=(12,5),

                                                                      width=.9, rot=45, alpha=0.8)

ax.set_ylim(.85, .95)

plt.show()
depts = pd.DataFrame({'department_sum':temp.groupby('department_id')['count'].sum(),

              'department_reordered':temp.groupby('department_id')['reordered'].sum()})



depts['proportion_reordered'] = depts['department_reordered']/depts['department_sum']



depts = pd.merge(left=depts,

               right = departments,

               left_index=True,

               right_index=True)



display(depts.sort_values('proportion_reordered', ascending=False))



plt.clf()

ax = depts.sort_values('proportion_reordered', ascending=False).plot(x='department', 

                                                                     y='proportion_reordered', 

                                                                     kind='bar', 

                                                                     color='purple', alpha=0.8, legend=False,

                                                                     figsize=(12,5), width=.9, rot=45)

ax.set_ylim(0,1)

plt.show()
ais = pd.DataFrame({'aisle_sum':temp.groupby('aisle_id')['count'].sum(),

              'aisle_reordered':temp.groupby('aisle_id')['reordered'].sum()})



ais['proportion_reordered'] = ais['aisle_reordered']/ais['aisle_sum']



ais = pd.merge(left=ais,

               right = aisles,

               left_index=True,

               right_index=True)



display(ais.nlargest(10, 'proportion_reordered').sort_values('proportion_reordered', ascending=False))



plt.clf()

ax = ais.nlargest(10, 'proportion_reordered').plot(x='aisle', y='proportion_reordered', 

                                                   kind='bar', color='purple', alpha=0.8, 

                                                   legend=False,figsize=(12,5), width=.9, rot=75)

ax.set_ylim(.65, .8)

plt.show()
temp = pd.DataFrame({'num_ordered':order_products_train.groupby(['product_id']).order_id.count(),

              'first':order_products_train[(order_products_train.add_to_cart_order==1)].groupby('product_id').order_id.count()})

temp['pct_first'] = temp['first']/temp['num_ordered']

temp = pd.merge(left=temp,

               right=products,

               left_index=True,

               right_index=True)

temp.dropna(inplace=True)

temp['first'] = temp['first'].astype(int)

display(temp[temp['num_ordered']>35].nlargest(10, 'pct_first').sort_values('pct_first', ascending=False))



temp = temp[['first', 'num_ordered', 'department_id']].groupby('department_id').sum()

temp['pct_first'] = temp['first']/temp['num_ordered']

pd.merge(left=temp, right=departments, left_index=True, right_index=True).sort_values('pct_first', ascending=False)
plt.clf()

ax = pd.merge(left=order_products_train,

        right=orders, 

        left_on='order_id',

        right_index=True).groupby('days_since_prior_order').reordered.mean().plot.area(figsize=(12,7), alpha=.6)

ax.set_ylim(.4, .85)

plt.show()
temp = pd.merge(left=order_products_train.groupby('product_id').reordered.mean().to_frame('reorder_rate'),

        right=order_products_train.product_id.value_counts().where(lambda x : x>1).to_frame('n').dropna(),

        left_index=True,

        right_index=True)



display(temp.head())





plt.clf()

ax = temp.plot.scatter(y='reorder_rate', x='n', figsize=(15,7), alpha=0.2)

sns.regplot(x="n", y="reorder_rate", data=temp, scatter=False, logx=True, color='purple', ci=None)

ax.set_xlim(0, 2000)

ax.set_title('reorder rate and how often a product has been ordered')

plt.show()
products['organic'] = np.where(products['product_name'].str.contains('rganic'), 1, 0)
temp = pd.merge(left=order_products_train,

               right=products[['organic']],

               left_on='product_id',

               right_index=True)



temp = pd.merge(left=temp.groupby('organic').count()['order_id'].to_frame('count'),

         right=temp[['reordered', 'organic']].groupby('organic').sum().astype(int),

         left_index=True,

         right_index=True)



temp['percent_of_sales'] = temp['count']/temp['count'].sum()

temp['reorder_rate'] = temp['reordered']/temp['count']



display(temp)



temp = temp[['percent_of_sales', 'reorder_rate']].transpose()

temp.columns = ['non_organic', 'organic']

temp.plot.bar(alpha=0.8, rot=0, figsize=(10,8))
def get_dept_counts(department):

    # add names, product counts of departments/aisles to order_products

    temp = pd.merge(left=order_products_train.product_id.value_counts().to_frame('count'), 

             right=pd.merge(left=pd.merge(left=products, right=departments, left_on='department_id', right_index=True),

                          right=aisles, left_on='aisle_id', right_index=True), left_index=True, right_index=True)

    # add count of reordered

    temp = pd.merge(left=temp,

                right=pd.DataFrame(order_products_train.groupby('product_id').reordered.sum().to_frame(), dtype='int64'),  

                left_index=True, right_index=True)

    # subset temp for 'department' and set index to aisle name

    temp = pd.merge(left=temp[temp.department==department].drop(['department_id', 'organic'], axis=1).groupby('aisle_id').sum(),

             right=aisles, left_index=True, right_index=True).set_index('aisle')

    return temp



def show_dept(department):

    display(get_dept_counts(department))

    ax = get_dept_counts(department).plot.bar(width=0.75, alpha=0.8, rot=75)

    ax.set_title(department)

    plt.show()
show_dept('produce')
def stacked(department):

    foo = get_dept_counts(department)

    foo['reorder_rate'] = foo['reordered']/foo['count']

    foo['not_reorder_rate'] = 1-foo['reorder_rate']

    ax = foo[['reorder_rate', 'not_reorder_rate']].plot.bar(stacked=True, alpha=0.7, color=['green', 'gray'])

    ax.set_yticks(np.arange(0,1.1, 0.1))

    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    plt.show()
stacked('produce')
temp = pd.merge(left=order_products_prior[['order_id', 'product_id', 'reordered']],

        right=orders[['user_id', 'order_number']][orders.order_number > 1],

        left_on='order_id', right_index=True)

temp = pd.merge(left=orders[orders.order_number > 1].user_id.value_counts().to_frame('count'),

                right=temp.groupby('user_id').reordered.mean().to_frame('reorder_rate'),

                left_index=True, right_index=True)

temp.index.names = ['user_id']

print('temp is a DataFrame with user_id as index, number of orders per user, and reorder_rate for user')
temp[temp.reorder_rate==1].sort_values('count', ascending=False).nlargest(10, 'count')
def get_user(user_id):

    return pd.merge(left= pd.merge(left=orders[(orders.user_id==user_id) & (orders.eval_set=='prior')],

                   right=order_products_prior,

                   left_index=True,

                   right_on='order_id'),

                   right= products[['product_name']],

                   left_on='product_id',

                   right_index=True).sort_values(['order_number', 'add_to_cart_order']).drop(['user_id', 'product_id'], axis=1).reset_index(drop=True)
get_user(99753).head()
for i in temp[temp.reorder_rate==1].sort_values('count', ascending=False).nlargest(10, 'count').index.values:

    print(get_user(i).product_name.unique())