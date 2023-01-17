from sklearn.decomposition import NMF
import pandas as pd

import numpy as np



file_path = '../input/'



load_data_dtype = {'order_id': np.uint32,

                   'user_id': np.uint32,

                   'eval_set': 'category',

                   'order_number': np.uint8,

                   'order_dow': np.uint8,

                   'order_hour_of_day': np.uint8,

                   # pandas 'gotcha'; leave as float:

                   'days_since_prior_order': np.float16,

                   'product_id': np.uint16,

                   'add_to_cart_order': np.uint8,

                   'reordered': np.bool

                   }
df_aisles = pd.read_csv(file_path + 'aisles.csv')

df_departments = pd.read_csv(file_path + 'departments.csv')

df_products = pd.read_csv(file_path + 'products.csv')



# Specify dtype to reduce memory utilization

df_order_products_prior = pd.read_csv(file_path + 'order_products__prior.csv',

                                      dtype=load_data_dtype

                                      )

df_order_products_train = pd.read_csv(file_path + 'order_products__train.csv',

                                      dtype=load_data_dtype

                                      )

df_orders = pd.read_csv(file_path + 'orders.csv',

                        dtype=load_data_dtype

                        )



# df_prior = full products from all prior orders 

df_prior = pd.merge(df_orders[df_orders['eval_set'] == 'prior'],

              df_order_products_prior,

              on='order_id'

              )



# Useful DataFrame for aisle and department feature construction

df_ad = pd.merge(df_prior, df_products, how='left',

                 on='product_id').drop('product_name', axis=1)
from sklearn.model_selection import train_test_split



# Names of dataset partitions

dsets = ['train',

         'test',

         'kaggle']



users = dict.fromkeys(dsets)



# Use sklearn utility to partition project users into train and test user lists.

users['train'], users['test'] = train_test_split(list(df_orders[df_orders.eval_set == 'train']['user_id']),

                                                 test_size=0.2,

                                           random_state=20190502)



# Kaggle submissions test set

users['kaggle'] = list(df_orders[df_orders.eval_set == 'test']['user_id'])#.to_list()
# Split DataFrames we will use in feature construction into dicts of DataFrames

prior = dict.fromkeys(dsets)

orders = dict.fromkeys(dsets)

ad = dict.fromkeys(dsets)



for ds in dsets:

    prior[ds] = df_prior[df_prior['user_id'].isin(users[ds])]

    orders[ds] = df_orders[df_orders['user_id'].isin(users[ds]) & (df_orders.eval_set == 'prior')]

    ad[ds] = df_ad[df_ad['user_id'].isin(users[ds])]
# Create MultiIndex of all (nonempty) (user, product) pairs

# for pandas 0.24:

# up_index[ds], _ = pd.MultiIndex.from_frame(prior[ds][['user_id', 'product_id']]).sortlevel()

# for pandas 0.23.4:



up_index = dict.fromkeys(dsets)



for ds in dsets:

    up_index[ds], _ = pd.MultiIndex.from_tuples(list(prior[ds][['user_id', 'product_id']].values),

                                                names=prior[ds][['user_id', 'product_id']].columns).sortlevel()

    up_index[ds] = up_index[ds].drop_duplicates()
UP_count = dict.fromkeys(dsets)

for ds in dsets:

    UP_count[ds] = (prior[ds]

                    .groupby(['user_id', 'product_id'])['order_id']

                    .count()

                    .rename('UP_count'))
UP_count_sparse, rows, columns = UP_count['test'].to_sparse().to_coo(row_levels=['user_id'],

                                             column_levels=['product_id'],

                                             sort_labels=True)
# from sklearn import metrics



# def get_score(model, data, scorer=metrics.explained_variance_score):

#     """ Estimate performance of the model on the data """

#     prediction = model.inverse_transform(model.transform(data))

#     return scorer(data, prediction)
factor = NMF(n_components=10)
W = factor.fit_transform(UP_count_sparse)

H = factor.components_
H.shape
factor.reconstruction_err_
import datetime
print(datetime.datetime.now())

Ks = list(range(10, 120, 10))

errors_test = []

for K in Ks:

    factor = NMF(n_components=K).fit(UP_count_sparse)

    errors_test.append(factor.reconstruction_err_)

    print(datetime.datetime.now())

print(errors_test)
import matplotlib.pyplot as plt



plt.plot(Ks, errors_test)

plt.show()
%who
del (ad, 

     df_ad,

     df_aisles,

     df_departments,

     df_order_products_prior,

     df_order_products_train,

     df_orders,

     df_prior,

     df_products    

    )
product_similarity_ = dict.fromkeys(dsets)

for ds in dsets:

    product_similarity[ds] = (prior[ds][['user_id', 'product_id']]

                              .drop_duplicates()

                              .sort_values(by=['user_id', 'product_id']))
product_similarity['train'].info()
from itertools import combinations
def col_comb(gp, r):

    return pd.DataFrame(list(combinations(gp.values, r)), 

                            columns=['row', 'col'])



product_user = (product_similarity['test']

                .groupby('user_id')

                .product_id

                .apply(col_comb, 2)

                .reset_index(level=1, drop=True)

                .reset_index()

                .groupby(['row','col'])

                .count())
UP_bar.head(100).groupby('user_id')['product_id'].apply(lambda prod : list(combinations(prod.values,2)))
help(combinations)