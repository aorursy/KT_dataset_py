#installing sklearn extra to run KMedoids

!pip install https://github.com/scikit-learn-contrib/scikit-learn-extra/archive/master.zip



# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from scipy.stats import zscore



#data visualisation

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

from plotly.subplots import make_subplots

import plotly.graph_objects as go



#clustering 

from sklearn.cluster import KMeans, AgglomerativeClustering

from sklearn_extra.cluster import KMedoids



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os, gc, sys

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2    

    for col in df.columns:

        col_type = df[col].dtypes

        if col_type in numerics:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)    

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df





# function to read the data and merge it (ignoring some columns, this is a very fst model)

def read_data():

    print('Reading files...')

    INPUT_DIR = "/kaggle/input/m5-forecasting-accuracy"

    calendar = pd.read_csv(os.path.join(INPUT_DIR, 'calendar.csv'))

    calendar = reduce_mem_usage(calendar)

    print('Calendar has {} rows and {} columns'.format(calendar.shape[0], calendar.shape[1]))

#     sell_prices = pd.read_csv(os.path.join(INPUT_DIR, 'sell_prices.csv'))

#     sell_prices = reduce_mem_usage(sell_prices)

#     print('Sell prices has {} rows and {} columns'.format(sell_prices.shape[0], sell_prices.shape[1]))

    sales_train_validation = pd.read_csv(os.path.join(INPUT_DIR, 'sales_train_validation.csv'))

    print('Sales train validation has {} rows and {} columns'.format(sales_train_validation.shape[0], sales_train_validation.shape[1]))

    submission = pd.read_csv(os.path.join(INPUT_DIR, 'sample_submission.csv'))

    return calendar, sales_train_validation, submission







def melt_and_merge(calendar, sales_train_validation, submission, nrows = 55000000, merge = False):

    

    # melt sales data, get it ready for training

    data = pd.melt(sales_train_validation, id_vars = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], var_name = 'day', value_name = 'demand')

    print('Melted sales train validation has {} rows and {} columns'.format(sales_train_validation.shape[0], sales_train_validation.shape[1]))

    

    # get only a sample for fst training

    data = data.loc[nrows:]

    

    # drop some calendar features

    calendar.drop(['weekday', 'wday', 'month', 'year'], inplace = True, axis = 1)

    

    

    if merge:

        # notebook crash with the entire dataset (maybee use tensorflow, dask, pyspark xD)

        data = pd.merge(data, calendar, how = 'left', left_on = ['day'], right_on = ['d'])

        data.drop(['d', 'day'], inplace = True, axis = 1)

        print('Our final dataset to train has {} rows and {} columns'.format(data.shape[0], data.shape[1]))

    else: 

        pass

    

    gc.collect()

    

    #convert date to pandas datetime

    data['date'] = pd.to_datetime(data['date'])



    

    return data

        

calendar, sales_train_validation, submission = read_data()

data = melt_and_merge(calendar, sales_train_validation, submission, nrows = 25000000, merge = True)
a = data.groupby([pd.Grouper(key='date', freq='M'),'state_id'])['demand'].sum().reset_index(name='sum_of_sales')

px.line(a, x='date', y='sum_of_sales',  color='state_id', title='Sum of Sales Monthly per State')
a = data.groupby([pd.Grouper(key='date', freq='W'),'state_id'])['demand'].sum().reset_index(name='sum_of_sales')

px.line(a, x='date', y='sum_of_sales',  color='state_id', title='Sum of Sales Weekly per State')
a = data.groupby([pd.Grouper(key='date', freq='M'),'state_id'])['demand'].sum().reset_index(name='sum_of_sales')

px.box(a, x='state_id', y='sum_of_sales', title=' Sales vs. State name ', color='state_id')
a = data.groupby([pd.Grouper(key='date', freq='M'),'cat_id'])['demand'].sum().reset_index(name='sum_of_sales')

px.line(a, x='date', y='sum_of_sales',  color='cat_id', title='Sum of Sales monthly based on category')
a = data.groupby([pd.Grouper(key='date', freq='W'),'cat_id'])['demand'].sum().reset_index(name='sum_of_sales')

px.line(a, x='date', y='sum_of_sales',  color='cat_id', title='Sum of Sales Weekly based on category')
a = data.groupby([pd.Grouper(key='date', freq='M'),'cat_id'])['demand'].sum().reset_index(name='sum_of_sales')

px.box(a, x='cat_id', y='sum_of_sales', title=' Sales vs. Category name ', color='cat_id')
a = data.groupby([pd.Grouper(key='date', freq='W'),'store_id'])['demand'].sum().reset_index(name='sum_of_sales')

px.line(a, x='date', y='sum_of_sales',  color='store_id', title='Sum of Sales Weekly based on Store')
a = data.groupby(['date', 'store_id'])['demand'].sum().reset_index(name='sum_of_sales')

px.line(a, x='date', y='sum_of_sales',  color='store_id', title='Sum of Sales per Store ')
a = data.groupby([pd.Grouper(key='date', freq='M'),'store_id'])['demand'].sum().reset_index(name='sum_of_sales')

px.box(a, x='store_id', y='sum_of_sales', title=' Sales vs. Store name in each state', color='store_id')
a = data.groupby([pd.Grouper(key='date', freq='M'), 'state_id', 'dept_id'])['demand'].sum().reset_index(name='sum_of_sales')

px.line(a, x='date', y='sum_of_sales',  facet_col='state_id', color='dept_id', title='Sum of Sales per state/dept id')
df = data.groupby([pd.Grouper(key='date', freq='M'),'id'])['demand'].sum().reset_index(name='sales')

df.head()
pivot_df = df.pivot(index='id', columns='date', values='sales')

sales_matrix = pivot_df.values

#changing values of 0 to 0.0001 to deal with 0 divition

max_values = np.max(sales_matrix, axis=1)[:,None]

max_values = np.array(max_values, dtype='float64')

max_values[np.argmin(max_values)][0] = 0.0001 
#normalize with the highest number of sales per product to transform into relative

normalized_sales_matrix = np.divide(sales_matrix, max_values)

pivot_df[pivot_df.columns] = normalized_sales_matrix

pivot_df.head()
di = dict()

for column in pivot_df.columns:

    di[column] = pivot_df[pivot_df[column] == 0].shape[0]

cols = sorted(di, key=di.get, reverse=True)
nclusters = 10

def cluster(matrix, n_clusters=8):

    kmeans = KMeans(n_clusters)

    kmedoids = KMedoids(n_clusters)

    labels_kmeans = kmeans.fit_predict(matrix)

    print(labels_kmeans)

    labels_kmedoids = kmedoids.fit_predict(matrix)

    print(labels_kmedoids)

    cluster_df = pd.DataFrame()

    cluster_df['id'] = pivot_df.index

    cluster_df['cluster_kmeans'] = labels_kmeans

    cluster_df['cluster_kmedoids'] = labels_kmedoids

    return cluster_df



cluster_df = cluster(pivot_df[cols].values, nclusters)

pivot_df['cluster_means'] = cluster_df['cluster_kmeans'].values

pivot_df['cluster_kmedoids'] = cluster_df['cluster_kmedoids'].values

pivot_df.head()
fig, ax = plt.subplots(nrows=int(nclusters/2), ncols=2, figsize=(15,18))



x, y = 0, 0

for j in range(0, nclusters):

    if j == 0:

        x = 0

        y = 0

    elif j % 2 == 0:

        x += 1 

        y = 0

    else:

        y += 1

    a = pivot_df[pivot_df['cluster_means'] == j]

    print(a.shape)

    for i in range(0, 50):

        

        ax[x, y].plot(df['date'].unique(), a.iloc[i, :-2].values, color='lightgrey');

    ax[x, y].plot(df['date'].unique(), a.iloc[:, :-2].values.mean(axis=0), color='red')

    ax[x, y].set_title('cluster:' + str(j))



plt.ylim(0, 5000)

plt.show()
fig, ax = plt.subplots(nrows=int(nclusters/2), ncols=2, figsize=(15,18))



x, y = 0, 0

for j in range(0, nclusters):

    if j == 0:

        x = 0

        y = 0

    elif j % 2 == 0:

        x += 1 

        y = 0

    else:

        y += 1

    a = pivot_df[pivot_df['cluster_kmedoids'] == j]

    print(a.shape)



    for i in range(0, 50):

        

        ax[x, y].plot(df['date'].unique(), a.iloc[i, :-2].values, color='lightgrey');

    ax[x, y].plot(df['date'].unique(), a.iloc[:, :-2].values.mean(axis=0), color='red')

    ax[x, y].set_title('cluster:' + str(j))



plt.ylim(0, 5000)

plt.show()