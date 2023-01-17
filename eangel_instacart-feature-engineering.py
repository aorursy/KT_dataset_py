!ls ../input
import pandas as pd

import numpy as np

pd.options.display.latex.repr=True



file_path = '../input/'



load_data_dtype = {

    'order_id': np.uint32,

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

                                      dtype=load_data_dtype)

df_order_products_train = pd.read_csv(file_path + 'order_products__train.csv',

                                      dtype=load_data_dtype)

df_orders = pd.read_csv(file_path + 'orders.csv', dtype=load_data_dtype)



# df_prior = full products from all prior orders

df_prior = pd.merge(df_orders[df_orders['eval_set'] == 'prior'],

                    df_order_products_prior,

                    on='order_id')



# # Useful DataFrame for aisle and department feature construction

# df_ad = pd.merge(df_prior, df_products, how='left',

#                  on='product_id').drop('product_name', axis=1)
from sklearn.model_selection import train_test_split



# Names of dataset partitions

dsets = ['train', 'test', 'kaggle']



users = dict.fromkeys(dsets)



# Use sklearn utility to partition project users into train and test user lists.

users['train'], users['test'] = train_test_split(list(

    df_orders[df_orders.eval_set == 'train']['user_id']),

                                                 test_size=0.2,

                                                 random_state=20190502)



# Kaggle submissions test set

users['kaggle'] = list(

    df_orders[df_orders.eval_set == 'test']['user_id'])  #.to_list()
# Split DataFrames we will use in feature construction into dicts of DataFrames

prior = dict.fromkeys(dsets)

orders = dict.fromkeys(dsets)

orders_full = dict.fromkeys(dsets)



# ad = dict.fromkeys(dsets)



for ds in dsets:

    prior[ds] = df_prior[df_prior['user_id'].isin(users[ds])]

    orders[ds] = df_orders[df_orders['user_id'].isin(users[ds])

                           & (df_orders.eval_set == 'prior')]

    orders_full[ds] = df_orders[df_orders['user_id'].isin(users[ds])]

#     ad[ds] = df_ad[df_ad['user_id'].isin(users[ds])]
pd.__version__
# Create Index of all users

# for pandas 0.24:

# u_index[ds], _ = pd.MultiIndex.from_frame(orders[ds]['user_id']).sortlevel()

# for pandas 0.23.4:



u_index = dict.fromkeys(dsets)



for ds in dsets:

    u_index[ds], _ = pd.Index(list(orders[ds]['user_id'].values),

                              name='user_id').sortlevel()

    u_index[ds] = u_index[ds].drop_duplicates()
# Create MultiIndex of all (nonempty) (user, product) pairs

# and empty DataFrame with that MultiIndex for joins with

# features with user index or product index

# for pandas 0.24:

# up_index[ds], _ = pd.MultiIndex.from_frame(prior[ds][['user_id', 'product_id']]).sortlevel()

# for pandas 0.23.4:



up_index = dict.fromkeys(dsets)

up_empty_df = dict.fromkeys(dsets)



for ds in dsets:

    up_index[ds], _ = pd.MultiIndex.from_tuples(

        list(prior[ds][['user_id', 'product_id']].values),

        names=prior[ds][['user_id', 'product_id']].columns).sortlevel()

    up_index[ds] = up_index[ds].drop_duplicates()

    up_empty_df[ds] = pd.DataFrame(index=up_index[ds])
# The ultimate orders

ultimate = dict.fromkeys(dsets)



ultimate['train'] = df_orders[(df_orders['eval_set'] == 'train')

                              & df_orders['user_id'].isin(users['train'])]

# 'eval_set' == 'train' is correct here since that is *Kaggle's* train:

ultimate['test'] = df_orders[(df_orders['eval_set'] == 'train')

                             & df_orders['user_id'].isin(users['test'])]

ultimate['kaggle'] = df_orders[(df_orders['eval_set'] == 'test')

                               & df_orders['user_id'].isin(users['kaggle'])]
# Build y['train'] and y['test']

# df_present = ultimate train and test orders

df_y = pd.merge(df_orders[df_orders['eval_set'] == 'train'],

                df_order_products_train,

                on='order_id')
y = dict.fromkeys(dsets)



y['train'] = (

    pd.DataFrame(

        [[True]],

        index=pd.MultiIndex.from_tuples(

            list(

                # (user, product) pairs of purchases in 'train' df -> list

                df_y[df_y['user_id'].isin(users['train'])]

                [['user_id', 'product_id']].values)))

    # Fill unpurchased items in overall up_index as False

    .reindex(up_index['train']).fillna(False))



y['test'] = (

    pd.DataFrame(

        [[True]],

        index=pd.MultiIndex.from_tuples(

            list(

                # (user, product) pairs of purchases in 'test' df -> list

                df_y[df_y['user_id'].isin(users['test'])]

                [['user_id', 'product_id']].values)))

    # Fill unpurchased items in overall up_index as False

    .reindex(up_index['test']).fillna(False))



y['kaggle'] = pd.DataFrame(data=['foo'])
pd.set_option('io.hdf.default_format', 'table')
store = pd.HDFStore('io.h5')
for dset, dframe in y.items():

    store['/y/' + str(dset)] = dframe
store.close()
store.is_open
# Cleanup y

del df_y, y
# dimensions

users_num = df_orders['user_id'].max()

products_num = df_products['product_id'].max()
# Make a dict to collect groups of features (e.g. profiles, clusterings, etc)

groups_dict = {}
from astropy.stats import circmean, circvar



def angle_transform(series, period):

    return series.multiply(2 * np.pi / period).sub(np.pi).astype('float16')
from collections import defaultdict



# dictionary to store given user features

u_given_dict = defaultdict(dict)
# Compute each feature separately for 'train', 'test,', and 'kaggle' in dsets

for ds in dsets:



    # ultimate order_dow

    u_given_dict['U_ultimate_order_dow'][ds] = angle_transform(

        ultimate[ds].set_index('user_id').order_dow, 7)



    # ultimate order_hour_of_day

    u_given_dict['U_ultimate_order_hour_of_day'][ds] = angle_transform(

        ultimate[ds].set_index('user_id').order_hour_of_day, 24)



    # ultimate days_since_prior_order

    u_given_dict['U_ultimate_days_since_prior_order'][ds] = (

        ultimate[ds].set_index('user_id').days_since_prior_order)
# Rename feature columns/pandas Series object by u_given_dict key name pointing to it.



for ds in dsets:

    for k, v in u_given_dict.items():

        v[ds].rename(k, inplace=True)
# Combine given user features; store as key 'U_given'



groups_dict['U_given'] = {

    ds: pd.concat([u_given_dict[k][ds] for k in u_given_dict.keys()], axis=1)

    for ds in dsets

}
# dictionary to store user features

u_dict = defaultdict(dict)



for ds in dsets:



    # number of orders a given user has placed

    u_dict['U_orders_num'][ds] = (

        prior[ds]

        .groupby(by='user_id')['order_number']

        .max().apply(pd.to_numeric,

                     downcast='unsigned'))



    # number of total items a given user has purchased

    u_dict['U_items_total'][ds] = (

        prior[ds].groupby('user_id')['product_id'].count().apply(

            pd.to_numeric, downcast='unsigned'))



    # mean basket size for a given user

    u_dict['U_order_size_mean'][ds] = (u_dict['U_items_total'][ds].div(

        u_dict['U_orders_num'][ds]).astype('float16'))



    # std basket size for a given user

    u_dict['U_order_size_std'][ds] = (prior[ds].groupby([

        'user_id', 'order_number'

    ]).add_to_cart_order.max().groupby('user_id').std().astype('float16'))



    # number of unique products a given user has purchased

    u_dict['U_unique_products'][ds] = (

        prior[ds].groupby('user_id')['product_id'].nunique().apply(

            pd.to_numeric, downcast='unsigned'))



    # number of total items a given user has purchased which are reorders

    u_dict['U_reordered_num'][ds] = (

        prior[ds].groupby('user_id')['reordered'].sum().apply(

            pd.to_numeric, downcast='unsigned'))



    # mean reorders per basket

    u_dict['U_reorder_size_mean'][ds] = (u_dict['U_reordered_num'][ds].div(

        u_dict['U_orders_num'][ds]).astype('float16'))



    # std reorders per basket

    u_dict['U_reorder_size_std'][ds] = (prior[ds].groupby([

        'user_id', 'order_number'

    ]).reordered.sum().groupby('user_id').std().astype('float16'))



    # proportion of items a given user has purchased which are reorders

    u_dict['U_reordered_ratio'][ds] = (u_dict['U_reordered_num'][ds].div(

        u_dict['U_items_total'][ds]).astype('float16'))



    # mean order_dow

    u_dict['U_order_dow_mean'][ds] = pd.concat(

        [

            orders[ds]['user_id'],

            angle_transform(

                # load-bearing .rename(). Fix.

                orders[ds]['order_dow'].rename('U_order_dow_mean'),

                7)

        ],

        axis=1).groupby('user_id').aggregate(circmean).astype(

            'float16').U_order_dow_mean



    # var order_dow

    u_dict['U_order_dow_var'][ds] = pd.concat(

        [

            orders[ds]['user_id'],

            angle_transform(

                # load-bearing .rename(). Fix.

                orders[ds]['order_dow'].rename('U_order_dow_var'),

                7)

        ],

        axis=1).groupby('user_id').aggregate(circvar).astype(

            'float16').U_order_dow_var



    # ultimate score for order_dow using circstd = sqrt(-2ln(circvar))

    u_dict['U_order_dow_score'][ds] = (

        u_given_dict['U_ultimate_order_dow'][ds]

        .sub(u_dict['U_order_dow_mean'][ds])

        .div(u_dict['U_order_dow_var'][ds]

             .apply(lambda x: np.sqrt(-2 * np.log(x))))

        .fillna(0)

        .clip(-20, 20)

        .astype('float16'))



    # mean order_hour_of_day

    u_dict['U_order_hour_of_day_mean'][ds] = (

        pd.concat(

            [orders[ds]['user_id'],

                angle_transform(

                    orders[ds]['order_hour_of_day']

                    # load-bearing .rename(). Fix.

                    .rename('U_order_hour_of_day_mean'),

                    24)

            ],

            axis=1)

        .groupby('user_id')

        .aggregate(circmean)

        .astype('float16')

        .U_order_hour_of_day_mean)



    # var order_hour_of_day

    u_dict['U_order_hour_of_day_var'][ds] = (

        pd.concat(

            [

                orders[ds]['user_id'],

                angle_transform(

                    orders[ds]['order_hour_of_day']

                    # load-bearing .rename(). Fix.

                    .rename('U_order_hour_of_day_var'),

                    24)

            ],

            axis=1)

        .groupby('user_id')

        .aggregate(circvar)

        .astype('float16')

        .U_order_hour_of_day_var)



    # ultimate score for order_hour_of_day using circstd = sqrt(-2ln(circvar))

    u_dict['U_order_hour_of_day_score'][ds] = (

        u_given_dict['U_ultimate_order_hour_of_day'][ds]

        .sub(u_dict['U_order_hour_of_day_mean'][ds])

        .div(u_dict['U_order_hour_of_day_var'][ds]

             .apply(lambda x: np.sqrt(-2 * np.log(x))))

        .fillna(0)

        .clip(-20, 20)

        .astype('float16')

    )



    # mean days since prior order (mean user order time interval)

    u_dict['U_days_since_prior_order_mean'][ds] = (

        orders_full[ds]

        .groupby('user_id')

        .days_since_prior_order

        .mean()

        .astype('float16')

    )



    # std days since prior order (std user order time interval)

    u_dict['U_days_since_prior_order_std'][ds] = (

        orders_full[ds]

        .groupby('user_id')

        .days_since_prior_order

        .std()

        .astype('float16')

    )
# Rename feature columns/pandas Series object by u_dict key name pointing to it.



for ds in dsets:

    for k, v in u_dict.items():

        v[ds].rename(k, inplace=True)
# Combine user features; store as key 'U'



groups_dict['U'] = {ds : pd.concat([u_dict[k][ds] for k in u_dict.keys()], axis=1) for ds in dsets}
# dictionary to store product features

p_dict = defaultdict(dict)



for ds in dsets:



    # number of total purchases

    p_dict['P_orders_num'][ds] = (

        prior[ds]

        .groupby('product_id')['order_id']

        .count()

        .apply(pd.to_numeric, downcast='unsigned'))



    # number of purchasers

    p_dict['P_unique_users'][ds] = (

        prior[ds]

        .groupby('product_id')['user_id']

        .nunique()

        .apply(pd.to_numeric, downcast='unsigned'))



    # reorder ratio

    p_dict['P_reorder_ratio'][ds] = (

        prior[ds]

        .groupby(['product_id'])['reordered']

        .mean()

        .astype('float16'))



    # mean order_hour_of_day

    p_dict['P_order_hour_of_day_mean'][ds] = angle_transform(

        prior[ds]

        .set_index('product_id')

        .order_hour_of_day,

        24).groupby('product_id').aggregate(circmean)



    # var order_hour_of_day

    p_dict['P_order_hour_of_day_var'][ds] = angle_transform(

        prior[ds]

        .set_index('product_id')

        .order_hour_of_day,

        24).groupby('product_id').aggregate(circvar)



    # mean order_dow

    p_dict['P_order_dow_mean'][ds] = angle_transform(

        prior[ds]

        .set_index('product_id')

        .order_hour_of_day,

        7).groupby('product_id').aggregate(circmean)



    # var order_dow

    p_dict['P_order_dow_var'][ds] = angle_transform(

        prior[ds]

        .set_index('product_id')

        .order_hour_of_day,

        7).groupby('product_id').aggregate(circvar)
# Rename feature columns/pandas Series objects by p_dict key name pointing to it.



for ds in dsets:

    for k, v in p_dict.items():

        v[ds].rename(k, inplace=True)
# Combine product features; store as key 'P'



groups_dict['P'] = {

    ds: pd.concat([p_dict[k][ds] for k in p_dict.keys()], axis=1)

    for ds in dsets

}
# dictionary to store user-product features

up_dict = defaultdict(dict)



for ds in dsets:



    # number of times particular user has ordered particular product

    up_dict['UP_orders_num'][ds] = (

        prior[ds]

        .groupby(['user_id', 'product_id'])['order_id']

        .count()

        .apply(pd.to_numeric, downcast='unsigned'))



    # number of orders since previous purchase of product by user

    # fill_value = infty?

    up_dict['UP_orders_since_previous'][ds] = (

        prior[ds].groupby(['user_id'])['order_number']

        .max()

        - prior[ds]

        .groupby(['user_id', 'product_id'])['order_number']

        .max()

        .apply(pd.to_numeric, downcast='unsigned'))



    # days since user last ordered product

    # groups of days_since_prior_order by user_id

    days_gpby_user = (

        orders_full[ds]

        .groupby('user_id')

        .days_since_prior_order

    )



    # given 'order_number' is UP_orders_since_previous

    # sum last orders_ago+1 days_since_prior_order

    def days_ago(row):

        orders_ago = int(row['order_number'])

        user = row['user_id']

        return (days_gpby_user

                .get_group(user)

                .iloc[-(orders_ago + 1):]

                .sum())



    # apply days_ago to UP_orders_since_previous

    up_dict['UP_days_since_prior_order'][ds] = (pd.Series(

        data=up_dict['UP_orders_since_previous'][ds]

        .reset_index()

        .apply(days_ago, axis=1)

        .values,

        index=up_dict['UP_orders_since_previous'][ds].index)

    .astype('uint16'))



    # clean-up

    del days_gpby_user



    # normalize above by user's days_since_prior_order

    # maybe use t-score instead?

    up_dict['UP_days_since_prior_order_score'][ds] = (

        up_dict['UP_days_since_prior_order'][ds]

        .sub(up_empty_df[ds].join(

            u_dict['U_days_since_prior_order_mean'][ds]).iloc[:, 0])

        .div(up_empty_df[ds].join(

            u_dict['U_days_since_prior_order_std'][ds]).iloc[:, 0])

        .fillna(0).clip(-20, 20).astype('float16'))
for ds in dsets:



    # reordered as `bool`

    up_dict['UP_reordered'][ds] = (

        prior[ds]

        .groupby(['user_id', 'product_id'])['reordered']

        .any())



    # fraction of baskets in which a given product appears for a given user,

    # count of orders in which product appears divided by total orders

    up_dict['UP_order_ratio'][ds] = (

        prior[ds].groupby(['user_id', 'product_id'])['order_number']

        .count()

        .div(prior[ds].groupby(['user_id'])['order_number']

             .max())

        .astype('float16')

    )



    # products in user's penultimate (previous) order as `bool`

    # (`train` and `test` sets contain ultimate order)



    up_dict['UP_penultimate'][ds] = (

        prior[ds].groupby(['user_id', 'product_id'])

        .order_number

        .max() 

        == prior[ds].groupby(['user_id'])

        .order_number

        .max()

        .reindex(up_index[ds], level=0)

    )



    # products in user's antepenultimate order as `bool`

    # index = UP pair (not distinct) with data = order_number

    past_orders = (

        prior[ds][['user_id', 'order_number', 'product_id']]

        .set_index(['user_id', 'product_id'])

    )

    

    # all UP pairs with max order_number - 1

    max_order_number_sub1 = (

        prior[ds].groupby(['user_id'])

        .order_number

        .max()

        .sub(1)

        .reindex(up_index[ds], level=0)

        .to_frame()

    )

    

    # intersection

    up_dict['UP_antepenultimate'][ds] = (

        pd.merge(

            past_orders,

            max_order_number_sub1,

            on=['user_id', 'product_id', 'order_number'])

        .reindex(up_index[ds], fill_value=False)

        .astype('bool')

        .iloc[:, 0]

    )

    

    # cleanup

    del past_orders, max_order_number_sub1



    # ultimate score for order_dow using circstd = sqrt(-2ln(circvar))

    # using (U_ultimate - P_order_dow_mean) / P_order_dow_std

    # broadcast to up_index

    # intuitively, how 'far' is a user's ultimate order dow from the mean dow product is ordered

    up_dict['UP_order_dow_score'][ds] = (

        pd.DataFrame(

            data=(up_empty_df[ds]

                      .join(u_given_dict['U_ultimate_order_dow'][ds])

                      .iloc[:, 0]

                  .sub(up_empty_df[ds]

                       .join(p_dict['P_order_dow_mean'][ds])

                       .iloc[:, 0])

                  .div(up_empty_df[ds]

                       .join(p_dict['P_order_dow_var'][ds]

                             .apply(lambda x: 

                                    np.sqrt(-2 * np.log(x))))

                       .iloc[:, 0])

                  ),

            index=up_index[ds])    

        .fillna(0)

        .clip(-20, 20)

        .astype('float16')

        .iloc[:, 0]

    )

        

    # ultimate score for order_hour_of_day using circstd = sqrt(-2ln(circvar))

    # using (U_ultimate - P_order_hour_of_day_mean) / P_order_hour_of_day_std

    # broadcast to up_index

    # intuitively, how 'far' is a user's ultimate order hour_of_day from the mean hour_of_day product is ordered

    # ndarray instead of pandas; couldn't resolve an arithmetic issue

    up_dict['UP_order_hour_of_day_score'][ds] = (

        pd.DataFrame(

            data=(up_empty_df[ds]

                      .join(u_given_dict['U_ultimate_order_hour_of_day'][ds])

                      .iloc[:, 0]

                  .sub(up_empty_df[ds]

                       .join(p_dict['P_order_hour_of_day_mean'][ds])

                       .iloc[:, 0])

                  .div(up_empty_df[ds]

                       .join(p_dict['P_order_hour_of_day_var'][ds]

                             .apply(lambda x: 

                                    np.sqrt(-2 * np.log(x))))

                       .iloc[:, 0])

                  ),

            index=up_index[ds])    

        .fillna(0)

        .clip(-20, 20)

        .astype('float16')

        .iloc[:, 0]

    )
# Rename feature columns/pandas Series objects by up_dict key name pointing to it.



for ds in dsets:

    for k, v in up_dict.items():

        v[ds].rename(k, inplace=True)
# Combine user-product features; store as key 'UP'



groups_dict['UP'] = {

    ds: pd.concat([up_dict[k][ds] for k in up_dict.keys()], axis=1)

    for ds in dsets

}
# scipy sparse matrix of number of times particular user has ordered particular product

UP_count_matrix = dict.fromkeys(dsets)



for ds in dsets:

    UP_count_matrix[ds], _, _ = (groups_dict['UP'][ds]['UP_orders_num'].apply(

        pd.to_numeric, downcast='unsigned').to_sparse().to_coo())
from sklearn.decomposition import LatentDirichletAllocation



LDA_features = dict.fromkeys(dsets)



for ds in dsets:

    lda = LatentDirichletAllocation(n_components=10,

                                    max_iter=10,

                                    learning_decay=0.85,

                                    n_jobs=1,

                                    learning_method='online')



    LDA_features[ds] = lda.fit_transform(UP_count_matrix[ds])
groups_dict['LDA'] = {

    ds: pd.DataFrame(data=LDA_features[ds],

                     index=u_index[ds],

                     columns=[

                         'LDA_' + str(k + 1)

                         for k in range(LDA_features[ds].shape[1])

                     ]).astype('float16')

    for ds in dsets

}
# # dictionary to store aisle features

# a_dict = defaultdict(dict)



# for ds in dsets:



#     # mean order_hour_of_day

#     a_dict['A_order_hour_of_day_mean'][ds] = angle_transform(ad[ds].set_index('aisle_id')

#                                                 .order_hour_of_day,

#                                                 24

#                                                 ).groupby('aisle_id').aggregate(circmean)



#     # std order_hour_of_day

#     a_dict['A_order_hour_of_day_var'][ds] = angle_transform(ad[ds].set_index('aisle_id')

#                                                .order_hour_of_day,

#                                                24

#                                                ).groupby('aisle_id').aggregate(circvar)



#     # mean order_dow

#     a_dict['A_order_dow_mean'][ds] = angle_transform(ad[ds].set_index('aisle_id')

#                                         .order_dow,

#                                         7

#                                         ).groupby('aisle_id').aggregate(circmean)



#     # var order_dow

#     a_dict['A_order_dow_var'][ds] = angle_transform(ad[ds].set_index('aisle_id')

#                                        .order_dow,

#                                        7

#                                        ).groupby('aisle_id').aggregate(circvar)



#     # reorder ratio

#     a_dict['A_reorder_ratio'][ds] = (ad[ds].groupby(['aisle_id'])['reordered']

#                        .mean()

#                        .astype('float16')

#                        )
# # Rename feature columns/pandas Series objects by a_dict key name pointing to it.



# for ds in dsets:

#     for k, v in a_dict.items():

#         v[ds].rename(k, inplace=True)
# # Combine aisle features into a_features

# # Reindex to products index for join with up_index



# #a_features = {ds : pd.DataFrame(index=groups_dict['P'][ds].index).join(

# #    pd.concat([a_dict[k][ds] for k in a_dict.keys()], axis=1)) for ds in dsets}



# # a_features = {ds : pd.concat([a_dict[k][ds] for k in a_dict.keys()], axis=1) for ds in dsets}



# groups_dict['A'] = {ds : 

#     # "dict" from product_id -> aisle_id (index=product_id, col=aisle_id)

#                     df_ad[['aisle_id', 'product_id']]

#                     .drop_duplicates()

#                     .set_index('product_id')

#                     .sort_index()

#     # join with aisle features with aisle_id as column

#                     .join(

#                         pd.concat([feature[ds] for feature in a_dict.values()], axis=1),

#         on='aisle_id')

#     .drop('aisle_id', axis=1)

#                     for ds in dsets}



# for ds in dsets:

#     groups_dict['A'][ds].index.rename('product_id', inplace=True)
# # dictionary to store department features

# d_dict = defaultdict(dict)



# for ds in dsets:

    

#     # mean order_hour_of_day

#     d_dict['D_order_hour_of_day_mean'][ds] = angle_transform(ad[ds].set_index('department_id')

#                                                 .order_hour_of_day,

#                                                 24

#                                                 ).groupby('department_id').aggregate(circmean)



#     # std order_hour_of_day

#     d_dict['D_order_hour_of_day_var'][ds] = angle_transform(ad[ds].set_index('department_id')

#                                                .order_hour_of_day,

#                                                24

#                                                ).groupby('department_id').aggregate(circvar)



#     # mean order_dow

#     d_dict['D_order_dow_mean'][ds] = angle_transform(ad[ds].set_index('department_id')

#                                         .order_dow,

#                                         7

#                                         ).groupby('department_id').aggregate(circmean)



#     # var order_dow

#     d_dict['D_order_dow_var'][ds] = angle_transform(ad[ds].set_index('department_id')

#                                        .order_dow,

#                                        7

#                                        ).groupby('department_id').aggregate(circvar)



#     # reorder ratio

#     d_dict['D_reorder_ratio'][ds] = (ad[ds].groupby(['department_id'])['reordered']

#                        .mean()

#                        .astype('float16')

#                        )
# # Rename feature columns/pandas Series objects by d_dict key name pointing to it.



# for ds in dsets:

#     for k, v in d_dict.items():

#         v[ds].rename(k, inplace=True)
# # Combine department features into a_features

# # Reindex to products index for join with up_index



# #a_features = {ds : pd.DataFrame(index=groups_dict['P'][ds].index).join(

# #    pd.concat([d_dict[k][ds] for k in d_dict.keys()], axis=1)) for ds in dsets}



# # a_features = {ds : pd.concat([d_dict[k][ds] for k in d_dict.keys()], axis=1) for ds in dsets}



# groups_dict['D'] = {ds : 

#     # "dict" from product_id -> department_id (index=product_id, col=department_id)

#                     df_ad[['department_id', 'product_id']]

#                     .drop_duplicates()

#                     .set_index('product_id')

#                     .sort_index()

#     # join with department features with department_id as column

#                     .join(

#                         pd.concat([feature[ds] for feature in d_dict.values()], axis=1),

#         on='department_id')

#     .drop('department_id', axis=1)

#                     for ds in dsets}



# for ds in dsets:

#     groups_dict['D'][ds].index.rename('product_id', inplace=True)
# Cleanup intermediate dicts

del (

    u_given_dict,

    u_dict,

    p_dict,

    up_dict,

    #     a_dict,

    #     d_dict

)



# Cleanup dataframes

del (  #df_ad,

    df_aisles, df_departments, df_order_products_prior,

    df_order_products_train, df_orders, df_prior, df_products)
%who
# Concatenate list of elements of groups_dict for each dset

X = {

    ds: pd.concat([

        pd.DataFrame(index=up_index[ds]).join(group[ds])

        for group in groups_dict.values()

    ],

                  axis=1)

    for ds in dsets

}
# Nulls make sklearn unhappy

[X[ds].isnull().any().any() for ds in dsets]
X['train'].info()
cols = [

    'U_orders_num', 'U_items_total', 'U_unique_products', 'U_reordered_num',

    'P_orders_num', 'P_unique_users', 'UP_orders_num',

    'UP_orders_since_previous'

]



for ds in dsets:

    X[ds][cols] = X[ds][cols].apply(pd.to_numeric,

                                    errors='coerce',

                                    downcast='unsigned')



# for ds in dsets:

#     X[ds]['UP_order_dow_score'] = np.nan_to_num(X[ds]['UP_order_dow_score'])

#     X[ds]['UP_order_hour_of_day_score'] = np.nan_to_num(X[ds]['UP_order_hour_of_day_score'])
X['train'].info()
store.open()
store.is_open
for dset, dframe in X.items():

    store['/X/' + str(dset)] = dframe
store.keys()
store.close()