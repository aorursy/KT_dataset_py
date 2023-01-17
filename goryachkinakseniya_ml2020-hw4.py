import logging
import os
import numpy as np
import time
import pandas as pd
from typing import Optional, Dict, Tuple
from implicit.als import AlternatingLeastSquares
from scipy import sparse
from datetime import timedelta
from sklearn.preprocessing import LabelEncoder
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from os.path import join as pjoin
from datetime import datetime

MAILING_DATETIME = datetime(2020, 4, 4)
RANDOM_STATE = 1
N_PURCHASES_ROWS = None
N_ALS_ITERATIONS = 15
SECONDS_IN_DAY = 60 * 60 * 24
def drop_column_multi_index_inplace(df):
    df.columns = ['_'.join(t) for t in df.columns]

def make_sum_csr(df,index_col,value_col,col_to_sum):
    print(df[col_to_sum].values.shape)
    print(df[index_col].values.shape)
    print(df[value_col].values.shape)
    coo = sparse.coo_matrix((df[col_to_sum].values,(df[index_col].values,df[value_col].values)))
    csr = coo.tocsr(copy=False)
    return csr
    
    
def make_count_csr(df,index_col,value_col):
    col_to_sum_name = '__col_to_sum__'
    df['__col_to_sum__'] = 1
    csr = make_sum_csr(df,index_col=index_col,value_col=value_col,col_to_sum=col_to_sum_name)
    df.drop(columns=col_to_sum_name, inplace=True)
    return csr


def make_latent_feature(df,index_col,value_col,n_factors,n_iterations,sum_col=None):
    if sum_col is None:
        csr = make_count_csr(df, index_col=index_col, value_col=value_col)
    else:
        csr = make_sum_csr(df,index_col=index_col,value_col=value_col,col_to_sum=sum_col)

    model = AlternatingLeastSquares(
        factors=n_factors,
        dtype=np.float32,
        iterations=n_iterations,
        regularization=0.1,
        use_gpu=False,  # True if n_factors >= 32 else False,

    )
    np.random.seed(RANDOM_STATE)
    model.fit(csr.T)

    return model.user_factors

INTERVALS = ['year', 'month', 'day', 'dayofweek', 'dayofyear', 'hour']

def make_client_features(clients):
    print('Preparing features')
    min_datetime = clients['first_issue_date'].min()
    days_from_min_to_issue = (
            (clients['first_issue_date'] - min_datetime)
            .dt.total_seconds() /
            SECONDS_IN_DAY
    ).values
    days_from_min_to_redeem = (
            (clients['first_redeem_date'] - min_datetime)
            .dt.total_seconds() /
            SECONDS_IN_DAY
    ).values

    age = clients['age'].values
    age[age < 0] = -2
    age[age > 100] = -3
    
    print('Combining features')
    gender = clients['gender'].values
    features = pd.DataFrame({
        'client_id': clients['client_id'].values,
        'gender_M': (gender == 'M').astype(int),
        'gender_F': (gender == 'F').astype(int),
        'gender_U': (gender == 'U').astype(int),
        'age': age,
        'days_from_min_to_issue': days_from_min_to_issue,
        'days_from_min_to_redeem': days_from_min_to_redeem,
        'issue_redeem_delay': days_from_min_to_redeem - days_from_min_to_issue})
    features = features.fillna(-1)
    print(f'Client features are created. Shape = {features.shape}')
    return features
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

N_FACTORS = {
    'product_id': 32,
    'level_1': 2,
    'level_2': 3,
    'level_3': 4,
    'level_4': 5,
    'segment_id': 4,
    'brand_id': 10,
    'vendor_id': 10,
}

N_ITERATIONS = N_ALS_ITERATIONS


def make_product_features(products,purchases):
    
    print('Creating purchases-products matrix')
    purchases_products = pd.merge(purchases,products,on='product_id')
    
    print('Purchases-products matrix is ready')

    del purchases
    del products

    print('Creating latent features')
    latent_features = make_latent_features(purchases_products)

    print('Creating usual features')
    usual_features = make_usual_features(purchases_products)

    print('Combining features')
    features = pd.merge(latent_features,usual_features,on='client_id')

    print(f'Product features are created. Shape = {features.shape}')
    return features


def make_usual_features(purchases_products):
    pp_gb = purchases_products.groupby('client_id')
    usual_features = pp_gb.agg(
        {
            'netto': ['median', 'max', 'sum'],
            'is_own_trademark': ['sum', 'mean'],
            'is_alcohol': ['sum', 'mean'],
            'level_1': ['nunique'],
            'level_2': ['nunique'],
            'level_3': ['nunique'],
            'level_4': ['nunique'],
            'segment_id': ['nunique'],
            'brand_id': ['nunique'],
            'vendor_id': ['nunique']})
    drop_column_multi_index_inplace(usual_features)
    usual_features.reset_index(inplace=True)

    return usual_features


def make_latent_features(purchases_products):
    latent_feature_matrices = []
    latent_feature_names = []
    for col, n_factors in N_FACTORS.items():
        print(f'Creating latent features for {col}')

        counts_subject_by_client = (
            purchases_products
            .groupby('client_id')[col]
            .transform('count')
        )
        share_col = f'{col}_share'
        purchases_products[share_col] = 1 / counts_subject_by_client

        latent_feature_matrices.append(
            make_latent_feature(
                purchases_products,
                index_col='client_id',
                value_col=col,
                n_factors=n_factors,
                n_iterations=N_ITERATIONS,
                sum_col=share_col
            )
        )

        purchases_products.drop(columns=share_col, inplace=True)

        latent_feature_names.extend(
            [f'{col}_f{i+1}' for i in range(n_factors)]
        )

        print(f'Features for {col} were created')

    # Add features that show how much client likes product in category
    print(f'Creating latent features for product in 4th category')
    col = 'product_id'
    counts_products_by_client_and_category = (
        purchases_products
            .groupby(['client_id', 'level_4'])[col]
            .transform('count')
    )
    share_col = f'{col}_share_by_client_and_cat'
    purchases_products[share_col] = 1 / counts_products_by_client_and_category

    n_factors = N_FACTORS[col]
    latent_feature_matrices.append(
        make_latent_feature(
            purchases_products,
            index_col='client_id',
            value_col=col,
            n_factors=n_factors,
            n_iterations=N_ITERATIONS,
            sum_col=share_col
        )
    )

    purchases_products.drop(columns=share_col, inplace=True)

    latent_feature_names.extend(
        [f'product_by_cat_f{i+1}' for i in range(n_factors)]
    )

    print(f'Features for {col} were created')

    latent_features = pd.DataFrame(
        np.hstack(latent_feature_matrices),
        columns=latent_feature_names
    )
    latent_features.insert(0, 'client_id', np.arange(latent_features.shape[0]))

    return latent_features
ORDER_COLUMNS = [
    'transaction_id',
    'datetime',
    'regular_points_received',
    'express_points_received',
    'regular_points_spent',
    'express_points_spent',
    'purchase_sum',
    'store_id'
]

FLOAT32_MAX = np.finfo(np.float32).max
POINT_TYPES = ('regular', 'express')
POINT_EVENT_TYPES = ('spent', 'received')
WEEK_DAYS = [
    'Monday',
    'Tuesday',
    'Wednesday',
    'Thursday',
    'Friday',
    'Saturday',
    'Sunday'
]
TIME_LABELS = ['Night', 'Morning', 'Afternoon', 'Evening']


def make_purchase_features_for_last_days(purchases,n_days):
    print(f'Creating purchase features for last {n_days} days...')
    cutoff = MAILING_DATETIME - timedelta(days=n_days)
    purchases_last = purchases[purchases['datetime'] >= cutoff]
    purchase_last_features = make_purchase_features(purchases_last)
    print(f'Purchase features for last {n_days} days are created')
    return purchase_last_features


def make_purchase_features(purchases):
    print('Creating purchase features...')
    n_clients = purchases['client_id'].nunique()

    print('Creating really purchase features...')
    purchase_features = make_really_purchase_features(purchases)
    print('Really purchase features are created')

    print('Creating small product features...')
    product_features = make_small_product_features(purchases)
    print('Small product features are created')

    print('Preparing orders table...')

    orders = purchases.reindex(columns=['client_id'] + ORDER_COLUMNS)
    del purchases
    orders.drop_duplicates(inplace=True)
    print(f'Orders table is ready. Orders: {len(orders)}')

    print('Creating order features...')
    order_features = make_order_features(orders)
    print('Order features are created')

    print('Creating store features...')
    store_features = make_store_features(orders)
    print('Store features are created')

    print('Creating order interval features...')
    order_interval_features = make_order_interval_features(orders)
    print('Order interval features are created')

    print('Creating features for orders with express points spent ...')
    orders_with_express_points_spent_features = \
        make_features_for_orders_with_express_points_spent(orders)
    print('Features for orders with express points spent are created')


    features = (
        purchase_features
        .merge(order_features, on='client_id')
        .merge(product_features, on='client_id')
        .merge(store_features, on='client_id')
        .merge(order_interval_features, on='client_id')
        .merge(orders_with_express_points_spent_features, on='client_id')
    )

    print('Creating ratio time features...')
    ratio_time_features = make_ratio_time_features(features)
    print('Ratio time features are created')

    features = features.merge(ratio_time_features, on='client_id')

    assert len(features) == n_clients, \
        f'n_clients = {n_clients} but len(features) = {len(features)}'

    features['days_from_last_order_share'] = \
        features['days_from_last_order'] / features['orders_interval_median']

    features['most_popular_store_share'] = (
        features['store_transaction_id_count_max'] /
        features['transaction_id_count']
    )

    features['ratio_days_from_last_order_eps_to_median_interval_eps'] = (
        features['days_from_last_express_points_spent'] /
        features['orders_interval_median_eps']
    )

    features['ratio_mean_purchase_sum_eps_to_mean_purchase_sum'] = (
        features['median_purchase_sum_eps'] /
        features['purchase_sum_median']
    )

    print(f'Purchase features are created. Shape = {features.shape}')
    return features


def make_really_purchase_features(purchases):
    simple_purchases = purchases.reindex(
        columns=['client_id', 'product_id', 'trn_sum_from_iss']
    )
    prices_bounds = [0, 98, 195, 490, 950, 1900, 4400, FLOAT32_MAX]
    agg_dict = {}
    for i, lower_bound in enumerate(prices_bounds[:-1]):
        upper_bound = prices_bounds[i + 1]
        name = f'price_from_{lower_bound}'
        simple_purchases[name] = (
            (simple_purchases['trn_sum_from_iss'] >= lower_bound) &
            (simple_purchases['trn_sum_from_iss'] < upper_bound)
        ).astype(int)
        agg_dict[name] = ['sum', 'mean']

    agg_dict.update(
        {
            'trn_sum_from_iss': ['median'],  # median product price
            'product_id': ['count', 'nunique']
        }
    )
    simple_features = simple_purchases.groupby('client_id').agg(agg_dict)
    drop_column_multi_index_inplace(simple_features)
    simple_features.reset_index(inplace=True)

    p_gb = purchases.groupby(['client_id', 'transaction_id'])
    purchase_agg = p_gb.agg(
        {
            'product_id': ['count'],
            'product_quantity': ['max']
        }
    )
    drop_column_multi_index_inplace(purchase_agg)
    purchase_agg.reset_index(inplace=True)
    o_gb = purchase_agg.groupby('client_id')
    complex_features = o_gb.agg(
        {
            # mean products in order
            'product_id_count': ['mean', 'median'],
            # mean max number of one product
            'product_quantity_max': ['mean', 'median']
        }
    )
    drop_column_multi_index_inplace(complex_features)
    complex_features.reset_index(inplace=True)
    features = pd.merge(
        simple_features,
        complex_features,
        on='client_id'
    )


    return features


def make_order_features(orders):
    orders = orders.copy()

    o_gb = orders.groupby('client_id')

    agg_dict = {
            'transaction_id': ['count'],  # number of orders
            'regular_points_received': ['sum', 'max', 'median'],
            'express_points_received': ['sum', 'max', 'median'],
            'regular_points_spent': ['sum', 'min', 'median'],
            'express_points_spent': ['sum', 'min', 'median'],
            'purchase_sum': ['sum', 'max', 'median'],
            'store_id': ['nunique'],  # number of unique stores
            'datetime': ['max']  # datetime of last order
        }

    # is regular/express points spent/received
    for points_type in POINT_TYPES:
        for event_type in POINT_EVENT_TYPES:
            col_name = f'{points_type}_points_{event_type}'
            new_col_name = f'is_{points_type}_points_{event_type}'
            orders[new_col_name] = (orders[col_name] != 0).astype(int)
            agg_dict[new_col_name] = ['sum']

    features = o_gb.agg(agg_dict)
    drop_column_multi_index_inplace(features)
    features.reset_index(inplace=True)

    features['days_from_last_order'] = (
        MAILING_DATETIME - features['datetime_max']
    ).dt.total_seconds() // SECONDS_IN_DAY
    features.drop(columns=['datetime_max'], inplace=True)

    # proportion of regular/express points spent to all transactions
    for points_type in POINT_TYPES:
        for event_type in POINT_EVENT_TYPES:
            col_name = f'is_{points_type}_points_{event_type}_sum'
            new_col_name = f'proportion_count_{points_type}_points_{event_type}'
            features[new_col_name] = (
                    features[col_name] / features['transaction_id_count']
            )

    express_col = f'is_express_points_spent_sum'
    regular_col = f'is_regular_points_spent_sum'
    new_col_name = f'ratio_count_express_to_regular_points_spent'
    features[new_col_name] = (
            features[express_col] / features[regular_col]
    ).replace(np.inf, FLOAT32_MAX)

    for points_type in POINT_TYPES:
        spent_col = f'is_{points_type}_points_spent_sum'
        received_col = f'is_{points_type}_points_received_sum'
        new_col_name = f'ratio_count_{points_type}_points_spent_to_received'
        features[new_col_name] = (
                features[spent_col] / features[received_col]
        ).replace(np.inf, 1000)


    for points_type in POINT_TYPES:
        spent_col = f'{points_type}_points_spent_sum'
        orders_sum_col = f'purchase_sum_sum'
        new_col_name = f'ratio_sum_{points_type}_points_spent_to_purchases_sum'
        features[new_col_name] = features[spent_col] / features[orders_sum_col]

    new_col_name = f'ratio_sum_express_points_spent_to_sum_regular_points_spent'
    regular_col = f'regular_points_spent_sum'
    express_col = f'express_points_spent_sum'
    features[new_col_name] = features[express_col] / features[regular_col]

    return features


def make_features_for_orders_with_express_points_spent(orders):

    orders_with_eps = orders.loc[orders['express_points_spent'] != 0]

    o_gb = orders_with_eps.groupby(['client_id'])
    features = o_gb.agg(
        {
            'purchase_sum': ['median'],
            'datetime': ['max']
        }
    )
    drop_column_multi_index_inplace(features)
    features.reset_index(inplace=True)
    features['days_from_last_express_points_spent'] = (
            MAILING_DATETIME - features['datetime_max']
    ).dt.days
    features.drop(columns=['datetime_max'], inplace=True)
    features.rename(
        columns={
            'purchase_sum_median': 'median_purchase_sum_eps'
        },
        inplace=True)

    order_int_features = make_order_interval_features(orders_with_eps)
    renamings = {
        col: f'{col}_eps'
        for col in order_int_features
        if col != 'client_id'
    }
    order_int_features.rename(columns=renamings, inplace=True)

    features = pd.merge(
        features,
        order_int_features,
        on='client_id')

    features = features.merge(
        pd.Series(orders['client_id'].unique(), name='client_id'),
        how='right')

    return features


def make_time_features(orders):
    # np.unique returns sorted array
    client_ids = np.unique(orders['client_id'].values)

    orders['weekday'] = np.array(WEEK_DAYS)[
        orders['datetime'].dt.dayofweek.values
    ]

    time_bins = [-1, 6, 11, 18, 24]

    orders['part_of_day'] = pd.cut(
        orders['datetime'].dt.hour,
        bins=time_bins,
        labels=TIME_LABELS
    ).astype(str)

    time_part_encoder = LabelEncoder()
    orders['part_of_day'] = time_part_encoder.fit_transform(orders['part_of_day'])

    time_part_columns_name = time_part_encoder.inverse_transform(
        np.arange(len(time_part_encoder.classes_))
    )

    time_part_cols = make_count_csr(orders,index_col='client_id',value_col='part_of_day')[client_ids, :]  # drop empty rows

    time_part_cols = pd.DataFrame(
        time_part_cols.toarray(),
        columns=time_part_columns_name)
    time_part_cols['client_id'] = client_ids

    weekday_encoder = LabelEncoder()
    orders['weekday'] = weekday_encoder.fit_transform(orders['weekday'])

    weekday_column_names = weekday_encoder.inverse_transform(
        np.arange(len(weekday_encoder.classes_))
    )
    weekday_cols = make_count_csr(
        orders,
        index_col='client_id',
        value_col='weekday')[client_ids, :]  # drop empty rows
    weekday_cols = pd.DataFrame(
        weekday_cols.toarray(),
        columns=weekday_column_names)
    weekday_cols['client_id'] = client_ids

    time_part_features = pd.merge(
        left=time_part_cols,
        right=weekday_cols,
        on='client_id')
    time_part_features.columns = [
        f'{col}_orders_count' if col != 'client_id' else col
        for col in time_part_features.columns
    ]

    return time_part_features


def make_small_product_features(purchases):
    cl_pr_gb = purchases.groupby(['client_id', 'product_id'])
    product_agg = cl_pr_gb.agg({
        'product_quantity': ['sum']})

    drop_column_multi_index_inplace(product_agg)
    product_agg.reset_index(inplace=True)

    cl_gb = product_agg.groupby(['client_id'])
    features = cl_gb.agg({'product_quantity_sum': ['max']})

    drop_column_multi_index_inplace(features)
    features.reset_index(inplace=True)

    return features


def make_store_features(orders):
    cl_st_gb = orders.groupby(['client_id', 'store_id'])
    store_agg = cl_st_gb.agg({
        'transaction_id': ['count']})

    drop_column_multi_index_inplace(store_agg)
    store_agg.reset_index(inplace=True)

    cl_gb = store_agg.groupby(['client_id'])
    simple_features = cl_gb.agg(
        {
            'transaction_id_count': ['max', 'mean', 'median']
        }
    )

    drop_column_multi_index_inplace(simple_features)
    simple_features.reset_index(inplace=True)
    simple_features.columns = (
        ['client_id'] +
        [
            f'store_{col}'
            for col in simple_features.columns[1:]
        ]
    )

    latent_features = make_latent_store_features(orders)

    features = pd.merge(
        simple_features,
        latent_features,
        on='client_id'
    )

    return features




def make_latent_store_features(orders):
    n_factors = 8
    latent_feature_names = [f'store_id_f{i + 1}' for i in range(n_factors)]

    latent_feature_matrix = make_latent_feature(
        orders,
        index_col='client_id',
        value_col='store_id',
        n_factors=n_factors,
        n_iterations=N_ALS_ITERATIONS)

    latent_features = pd.DataFrame(
        latent_feature_matrix,
        columns=latent_feature_names
    )
    latent_features.insert(0, 'client_id', np.arange(latent_features.shape[0]))

    return latent_features


def make_order_interval_features(orders):
    orders = orders.sort_values(['client_id', 'datetime'])

    last_order_client = orders['client_id'].shift(1)
    is_same_client = last_order_client == orders['client_id']
    orders['last_order_datetime'] = orders['datetime'].shift(1)

    orders['orders_interval'] = np.nan
    orders.loc[is_same_client, 'orders_interval'] = (
        orders.loc[is_same_client, 'datetime'] -
        orders.loc[is_same_client, 'last_order_datetime']
    ).dt.total_seconds() / SECONDS_IN_DAY

    cl_gb = orders.groupby('client_id', sort=False)
    features = cl_gb.agg(
        {
            'orders_interval': [
                'mean',  # mean interval between orders
                'median',
                'std',  # constancy of orders
                'min',
                'max',
                'last',  # interval between last 2 orders
            ]
        }
    )
    drop_column_multi_index_inplace(features)
    features.reset_index(inplace=True)
    features.fillna(-3, inplace=True)

    return features


def make_ratio_time_features(features):
    time_labels = TIME_LABELS + WEEK_DAYS
    columns = [f'{col}_orders_count' for col in time_labels]
    share_columns = [f'{col}_share' for col in columns]

    time_features = features.reindex(columns=columns).values
    orders_count = features['transaction_id_count'].values

    share_time_features = time_features / orders_count.reshape(-1, 1)

    share_time_features = pd.DataFrame(
        share_time_features,
        columns=share_columns
    )
    share_time_features['client_id'] = features['client_id']

    return share_time_features
def make_z(treatment, target):
    y = target
    w = treatment
    z = y * w + (1 - y) * (1 - w)
    return z


def calc_uplift(prediction):
    uplift = 2 * prediction - 1
    return uplift


def get_feature_importances(est, columns):
    return pd.DataFrame({
        'column': columns,
        'importance': est.feature_importances_
    }).sort_values('importance', ascending=False)
def uplift_fit(model, X_train, treatment_train, target_train):
    z = make_z(treatment_train, target_train)
    model = clone(model)
    model.fit(X_train, z)
    return model

def uplift_predict(model, X_test, z=True):
    predict_z = model.predict_proba(X_test)[:, 1]
    uplift = calc_uplift(predict_z)
    if z: return predict_z
    else: return uplift
def score_uplift(prediction,treatment,target,rate = 0.3):
    """
    Подсчет Uplift Score
    """
    order = np.argsort(-prediction)
    treatment_n = int((treatment == 1).sum() * rate)
    treatment_p = target[order][treatment[order] == 1][:treatment_n].mean()
    control_n = int((treatment == 0).sum() * rate)
    control_p = target[order][treatment[order] == 0][:control_n].mean()
    score = treatment_p - control_p
    return score


def score_roc_auc(prediction,treatment,target):
    y_true = make_z(treatment, target)
    score = roc_auc_score(y_true, prediction)
    return score


def uplift_metrics(prediction,treatment,target,rate_for_uplift = 0.3):
    scores = {
        'roc_auc': score_roc_auc(prediction, treatment, target),
        'uplift': score_uplift(prediction, treatment, target, rate_for_uplift)
    }
    return scores
def load_clients():
    return pd.read_csv('/kaggle/input/x5-uplift-valid/data/clients2.csv',
                       parse_dates=['first_issue_date', 'first_redeem_date'])

def prepare_clients():
    print('Preparing clients...')
    clients = load_clients()
    client_encoder = LabelEncoder()
    clients['client_id'] = client_encoder.fit_transform(clients['client_id'])
    print('Clients are ready')
    return clients, client_encoder


def load_products():
    return pd.read_csv('/kaggle/input/x5-uplift-valid/data/products.csv')


def prepare_products():
    print('Preparing products...')
    products = load_products()
    product_encoder = LabelEncoder()
    products['product_id'] = product_encoder. \
        fit_transform(products['product_id'])

    products.fillna(-1, inplace=True)

    for col in [
        'level_1', 'level_2', 'level_3', 'level_4',
        'segment_id', 'brand_id', 'vendor_id'
    ]:
        products[col] = LabelEncoder().fit_transform(products[col].astype(str))
    print('Products are ready')
    return products, product_encoder


def load_purchases():
    print('Loading purchases...')
    purchases_train = pd.read_csv('/kaggle/input/x5-uplift-valid/train_purch/train_purch.csv',
        nrows=N_PURCHASES_ROWS)
    purchases_test = pd.read_csv('/kaggle/input/x5-uplift-valid/test_purch/test_purch.csv',
        nrows=N_PURCHASES_ROWS)
    purchases = pd.concat([purchases_train, purchases_test])
    print('Purchases are loaded')
    return purchases


def prepare_purchases(client_encoder,product_encoder):
    print('Preparing purchases...')
    purchases = load_purchases()

    print('Handling n/a values...')
    purchases.dropna(
        subset=['client_id', 'product_id'],
        how='any',
        inplace=True
    )
    purchases.fillna(-1, inplace=True)

    print('Label encoding...')
    purchases['client_id'] = client_encoder.transform(purchases['client_id'])
    purchases['product_id'] = product_encoder.transform(purchases['product_id'])
    for col in ['transaction_id', 'store_id']:
        purchases[col] = LabelEncoder(). \
            fit_transform(purchases[col].astype(str))

    print('Date and time conversion...')
    purchases['datetime'] = pd.to_datetime(
        purchases['transaction_datetime'],
        format='%Y-%m-%d %H:%M:%S'
    )
    purchases.drop(columns=['transaction_datetime'], inplace=True)

    print('Purchases are ready')
    return purchases


def load_train():
    return pd.read_csv('/kaggle/input/x5-uplift-valid/data/train.csv',
        index_col='client_id')


def load_test():
    return pd.read_csv('/kaggle/input/x5-uplift-valid/data/test.csv',
        index_col='client_id')
import logging
import pickle
from datetime import timedelta
from os.path import join as pjoin

import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier


log_format = '[%(asctime)s] %(name)-25s %(levelname)-8s %(message)s'
logging.basicConfig(
    format=log_format,
    level=logging.INFO
)
def prepare_features():
    print('Loading data...')
    clients, client_encoder = prepare_clients()
    products, product_encoder = prepare_products()
    purchases = prepare_purchases(client_encoder, product_encoder)
    del product_encoder
    print('Data is loaded')

    print('Preparing features...')
    purchase_features = make_purchase_features(purchases)

    purchases_ids = purchases.reindex(columns=['client_id', 'product_id'])
    del purchases
    product_features = make_product_features(products, purchases_ids)
    del purchases_ids

    client_features = make_client_features(clients)

    print('Combining features...')
    features = (
        client_features
            .merge(purchase_features, on='client_id', how='left')
           
            .merge(product_features, on='client_id', how='left')
    )
    del client_features
    del purchase_features
    del product_features

    features.fillna(-2, inplace=True)

    features['client_id'] = client_encoder.inverse_transform(features['client_id'])
    del client_encoder

    print('Features are ready')

    return features


def save_submission(indices_test, test_pred, filename):
    df_submission = pd.DataFrame({'pred': test_pred}, index=indices_test)
    df_submission.to_csv(filename)
features = prepare_features()
print('Saving features...')
with open('features.pkl', 'wb') as f:
    pickle.dump(features, f, protocol=pickle.HIGHEST_PROTOCOL)
print('Features are saved')
print('Loading features...')
with open('features.pkl', 'rb') as f:
    features: pd.DataFrame = pickle.load(f)
print('Features are loaded')

print(f'Features shape: {features.shape}')

print('Preparing data sets...')
features.set_index('client_id', inplace=True)
train = load_train()
test = load_test()
indices_train = train.index
indices_test = test.index

X_train = features.loc[indices_train]
treatment_train = train.loc[indices_train, 'treatment_flg'].values
target_train = train.loc[indices_train, 'target'].values

X_test = features.loc[indices_test]

indices_learn, indices_valid = train_test_split(train.index,test_size=0.3,random_state=RANDOM_STATE + 1)

X_learn = features.loc[indices_learn]
treatment_learn = train.loc[indices_learn, 'treatment_flg'].values
target_learn = train.loc[indices_learn, 'target'].values

X_valid = features.loc[indices_valid]
treatment_valid = train.loc[indices_valid, 'treatment_flg'].values
target_valid = train.loc[indices_valid, 'target'].values
print('Data sets prepared')
X_train
clf_ = LGBMClassifier(
        boosting_type='rf',
        n_estimators=15000,
        num_leaves=40,
        max_depth=3,
        max_bin=110,
        # reg_lambda=1,
        learning_rate=0.001,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        bagging_freq=1,
        bagging_fraction=0.5,
        importance_type='split',
        is_unbalance=True,
        min_child_samples=20,
        min_child_weight=0.001,
        min_split_gain=0.0,
        objective='binary',
        reg_alpha=0.0,
        reg_lambda=0.0,
        silent=True,
        subsample=1.0,
        subsample_for_bin=200000,
        subsample_freq=0
    )
print('Build model for learn data set...')
clf = uplift_fit(clf_, X_learn, treatment_learn, target_learn)
print('Model is ready')
learn_pred = uplift_predict(clf, X_learn)
learn_scores = uplift_metrics(learn_pred, treatment_learn, target_learn)
print(f'Learn scores: {learn_scores}')
valid_pred = uplift_predict(clf, X_valid)
valid_scores = uplift_metrics(valid_pred, treatment_valid, target_valid)
print(f'Valid scores: {valid_scores}')
test_pred = uplift_predict(clf, X_test, z = True)
print('Saving submission...')
save_submission(indices_test,test_pred,'submission_without_my.csv')
print('Submission is ready')
test_pred
df_train = pd.read_csv('/kaggle/input/x5-uplift-valid/data/train.csv', index_col='client_id')
df_clients = pd.read_csv('/kaggle/input/x5-uplift-valid/data/clients2.csv', index_col='client_id')
df_test = pd.read_csv('/kaggle/input/x5-uplift-valid/data/test.csv', index_col='client_id')
df_products = pd.read_csv('/kaggle/input/x5-uplift-valid/data/products.csv')
df_pursh = pd.read_csv('/kaggle/input/x5-uplift-valid/train_purch/train_purch.csv')
df_pursh_test = pd.read_csv('/kaggle/input/x5-uplift-valid/test_purch/test_purch.csv')
indices_train = df_train.index
df_clients_train = df_clients.loc[indices_train]
df_train_all = df_clients_train.merge(df_train, right_index=True, left_index=True)
df_train_all.shape
indices_test = df_test.index
df_clients_test = df_clients.loc[indices_test]
df_test_all = df_clients_test.merge(df_test, right_index=True, left_index=True)
df_test_all.shape
df_test_all = df_test_all.drop(['client_id.1'], axis = 1)
df_train_all = df_train_all.drop(['client_id.1'], axis = 1)
#Метод для вычисления аплифта
def uplift_score(data):
    return data[data.treatment_flg == 1].target.mean() - data[data.treatment_flg == 0].target.mean()
max_uplf = 0
#найдем возраст, до которого и после аплифт наиболее отличается: 
for i in range(18,60, 1):
    if max_uplf < (uplift_score(df_train_all[df_train_all.age>i]) - uplift_score(df_train_all[df_train_all.age<=i])):
        max_uplf = (uplift_score(df_train_all[df_train_all.age>i]) - uplift_score(df_train_all[df_train_all.age<=i]))
        print(i)
last_cols = ['regular_points_received', 'purchase_sum']
last_train_month = df_pursh[df_pursh['transaction_datetime'] > '2019-02-18'].groupby(['client_id','transaction_id'])[last_cols].last()
last_test_month = df_pursh_test[df_pursh_test['transaction_datetime'] > '2019-02-18'].groupby(['client_id','transaction_id'])[last_cols].last()
features =  pd.concat([last_train_month.groupby('client_id')['purchase_sum'].count(),
                       last_train_month.groupby('client_id').sum()],axis = 1)

features_test =  pd.concat([last_test_month.groupby('client_id')['purchase_sum'].count(),
                       last_test_month.groupby('client_id').sum()],axis = 1)

features.columns = ['last_month_trans_count', 'regular_points_received_sum_last_month', 'purchase_sum_last_month']
features_test.columns = ['last_month_trans_count', 'regular_points_received_sum_last_month', 'purchase_sum_last_month']
df_test['target'] = 1
merged_test = pd.concat([df_test,df_clients,features_test],axis = 1,sort = True)
merged_test = merged_test[~merged_test['target'].isnull()].copy()
merged_test['first_issue_date'] = merged_test['first_issue_date'].apply(pd.to_datetime).astype(int)/10**9
merged_test['first_redeem_date'] = merged_test['first_redeem_date'].apply(pd.to_datetime).astype(int)/10**9


merged_train = pd.concat([df_train,df_clients,features],axis = 1,sort = True)
merged_train = merged_train[~merged_train['target'].isnull()].copy()
merged_train['first_issue_date'] = merged_train['first_issue_date'].apply(pd.to_datetime).astype(int)/10**9
merged_train['first_redeem_date'] = merged_train['first_redeem_date'].apply(pd.to_datetime).astype(int)/10**9
merged_train.pop('client_id.1')
merged_train.pop('age')
merged_train.pop('gender')
merged_test.pop('client_id.1')
merged_test.pop('age')
merged_test.pop('gender')
df_train_all = df_train_all.merge(merged_train, right_index=True, left_index=True)
df_test_all = df_test_all.merge(merged_test, right_index=True, left_index=True)
df_new_features = df_train_all[['first_issue_date_y', 'first_redeem_date_y', 'last_month_trans_count', 'regular_points_received_sum_last_month', 'purchase_sum_last_month']]
df_new_features_test = df_test_all[['first_issue_date_y', 'first_redeem_date_y', 'last_month_trans_count', 'regular_points_received_sum_last_month', 'purchase_sum_last_month']]
X_learn_with_my = X_learn.merge(df_new_features.loc[indices_learn], right_index=True, left_index=True)
X_valid_with_my = X_valid.merge(df_new_features.loc[indices_valid], right_index=True, left_index=True)
X_test_with_my = X_test.merge(df_new_features_test, right_index=True, left_index=True)
clf_ = LGBMClassifier(
        boosting_type='rf',
        n_estimators=15000,
        num_leaves=40,
        max_depth=3,
        max_bin=110,
        # reg_lambda=1,
        learning_rate=0.001,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        bagging_freq=1,
        bagging_fraction=0.5,
        importance_type='split',
        is_unbalance=True,
        min_child_samples=20,
        min_child_weight=0.001,
        min_split_gain=0.0,
        objective='binary',
        reg_alpha=0.0,
        reg_lambda=0.0,
        silent=True,
        subsample=1.0,
        subsample_for_bin=200000,
        subsample_freq=0
    )
print('Build model for learn data set...')
clf = uplift_fit(clf_, X_learn_with_my, treatment_learn, target_learn)
print('Model is ready')
learn_pred = uplift_predict(clf, X_learn_with_my)
learn_scores = uplift_metrics(learn_pred, treatment_learn, target_learn)
print(f'Learn scores: {learn_scores}')
valid_pred = uplift_predict(clf, X_valid_with_my)
valid_scores = uplift_metrics(valid_pred, treatment_valid, target_valid)
print(f'Valid scores: {valid_scores}')
test_pred_with_my = uplift_predict(clf, X_test_with_my, z = True)
save_submission(indices_test,test_pred_with_my,'submission_with_my.csv')