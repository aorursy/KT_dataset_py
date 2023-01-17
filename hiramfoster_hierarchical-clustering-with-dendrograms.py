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

                                                               'order_dow':'int8', 

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
display(products.head())
temp = pd.merge(left=products,

         right=order_products_train.product_id.value_counts().to_frame('count'), 

         left_index=True, right_index=True)



temp = pd.merge(left=temp, 

                    right=pd.DataFrame(order_products_train.groupby('product_id').reordered.sum().to_frame(), dtype='int64'),  

                    left_index=True, right_index=True)



temp['reorder_rate'] = temp['reordered']/temp['count']



temp = pd.merge(left=temp, 

                right=order_products_train.groupby('product_id').add_to_cart_order.mean().to_frame('add_to_cart_mean'),

                left_index=True, right_index=True)



temp = pd.merge(left=temp, 

                right=pd.merge(left=order_products_train, 

                               right=orders[['order_dow', 'order_hour_of_day', 'days_since_prior_order']], 

                               left_on='order_id', right_index=True).groupby('product_id').order_dow.mean().to_frame(),

                left_index=True, right_index=True)



temp = pd.merge(left=temp, 

                right=pd.merge(left=order_products_train, 

                               right=orders[['order_dow', 'order_hour_of_day', 'days_since_prior_order']], 

                               left_on='order_id', right_index=True).groupby('product_id').order_hour_of_day.mean().to_frame(),

                left_index=True, right_index=True)



temp = pd.merge(left=temp, 

                right=pd.merge(left=order_products_train, 

                               right=orders[['order_dow', 'order_hour_of_day', 'days_since_prior_order']], 

                               left_on='order_id', right_index=True).groupby('product_id').days_since_prior_order.mean().to_frame(),

                left_index=True, right_index=True)

display(temp.head())

temp.shape
temp = temp[temp['count'] > 10]

temp.shape
temp.drop(['product_name', 'department_id', 'aisle_id', 'reordered'], axis=1, inplace=True)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

temp_scaled = scaler.fit_transform(temp)

print('done.')
def fancy_dendrogram(*args, **kwargs):

    max_d = kwargs.pop('max_d', None)

    if max_d and 'color_threshold' not in kwargs:

        kwargs['color_threshold'] = max_d

    annotate_above = kwargs.pop('annotate_above', 0)

    plt.figure(figsize=(15,10))

    ddata = dendrogram(*args, **kwargs)



    if not kwargs.get('no_plot', False):

        plt.title('Hierarchical Clustering Dendrogram (truncated)')

        plt.xlabel('sample index or (cluster size)')

        plt.ylabel('distance')

        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):

            x = 0.5 * sum(i[1:3])

            y = d[1]

            if y > annotate_above:

                plt.plot(x, y, 'o', c=c)

                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),

                             textcoords='offset points',

                             va='top', ha='center')

        if max_d:

            plt.axhline(y=max_d, c='k')

    return ddata
from scipy.cluster.hierarchy import dendrogram, ward



linked_array = ward(temp_scaled)



fancy_dendrogram(

    linked_array,

    truncate_mode='lastp',

    p=30,

    leaf_rotation=90.,

    leaf_font_size=12.,

    show_contracted=True,

    annotate_above=10,

    max_d=80

)



plt.show()
print('distances for the last 5 merges:\n{}'.format(linked_array[-5:,2]))
from scipy.cluster.hierarchy import fcluster

max_d = 80

clusters = fcluster(linked_array, max_d, criterion='distance')

    

labels, counts = np.unique(clusters, return_counts=True)



temp['clusters'] = clusters



print('reorder rates for each cluster\n')

for i in range(1,len(np.unique(clusters))+1):

    print('\nlabel: {}'.format(i))

    print('n: {}'.format(counts[i-1]))

    print('rr: {}'.format(round(temp[temp['clusters'] == i].reorder_rate.mean()*100, 2))) 
label = 3

print('reorder rate for cluster {0}: {1}'.format(label, round(temp[temp['clusters'] == label].reorder_rate.mean()*100,2)))

pd.merge(right=temp[temp['clusters'] == label], left=products, left_index=True, right_index=True).head()
temp.drop('clusters', axis=1, inplace=True)



from sklearn.preprocessing import RobustScaler

robust_scaler = RobustScaler()

temp_robust = robust_scaler.fit_transform(temp)
linked_array = ward(temp_robust)



fancy_dendrogram(

    linked_array,

    truncate_mode='lastp',

    p=30,

    leaf_rotation=90.,

    leaf_font_size=12.,

    show_contracted=True,

    annotate_above=10,

    max_d=300

)



plt.show()
from scipy.cluster.hierarchy import fcluster

max_d = 300

clusters = fcluster(linked_array, max_d, criterion='distance')



labels, counts = np.unique(clusters, return_counts=True)



temp['clusters'] = clusters



print('reorder rates for each cluster\n')

for i in range(1,len(np.unique(clusters))+1):

    print('\nlabel: {}'.format(i))

    print('n: {}'.format(counts[i-1]))

    print('rr: {}'.format(round(temp[temp['clusters'] == i].reorder_rate.mean()*100, 2))) 
temp_scaled = scaler.fit_transform(temp[['count', 'reorder_rate']])

print('done.')
linked_array = ward(temp_scaled)



fancy_dendrogram(

    linked_array,

    truncate_mode='lastp',

    p=30,

    leaf_rotation=90.,

    leaf_font_size=12.,

    show_contracted=True,

    annotate_above=10,

    max_d=50

)



plt.show()
max_d = 50

clusters = fcluster(linked_array, max_d, criterion='distance')



labels, counts = np.unique(clusters, return_counts=True)



temp['clusters'] = clusters



print('reorder rates for each cluster\n')

for i in range(1,len(np.unique(clusters))+1):

    print('\nlabel: {}'.format(i))

    print('n: {}'.format(counts[i-1]))

    print('rr: {}'.format(round(temp[temp['clusters'] == i].reorder_rate.mean()*100, 2))) 