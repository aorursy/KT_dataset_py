# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
#########################################
############## models ###################
from catboost import *
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import SGDRegressor
#########################################

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import time
import joblib 
import gc
from tqdm import tqdm_notebook

########################################
############ Text libraries ############
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import NMF
from sklearn.mixture import GaussianMixture
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
#If you don't have pymystem you can install it with !pip install pymystem3==0.2.0
#!pip install pymystem3==0.2.0
from pymystem3 import Mystem
from string import punctuation
########################################

import matplotlib.pyplot as plt
%matplotlib inline

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
DATA_FOLDER = '/kaggle/input/competitive-data-science-predict-future-sales'
transactions    = pd.read_csv(os.path.join(DATA_FOLDER, 'sales_train.csv'))
items           = pd.read_csv(os.path.join(DATA_FOLDER, 'items.csv'))
item_categories = pd.read_csv(os.path.join(DATA_FOLDER, 'item_categories.csv'))
shops           = pd.read_csv(os.path.join(DATA_FOLDER, 'shops.csv'))
test            = pd.read_csv(os.path.join(DATA_FOLDER, 'test.csv'))
transactions.describe()
#train = transactions.drop('item_name', axis=1)
for df in [transactions, items, item_categories, shops]:
    print(df.shape, df.head().T)
t = time.time()
transactions['date'] = pd.to_datetime(transactions['date'])
transactions['weekday'] = transactions['date'].dt.day_name()
transactions['year'] = transactions['date'].dt.year
transactions['month'] = transactions['date'].dt.month
transactions['day'] = transactions['date'].dt.day
time.time() - t
shops
# go in a loop, where first value is a shop id with typos and second - corresponding good shop id
# shop id occurs only in train and test dataframes
for shop_with_typo, original_shop_id in [(0, 57), (1, 58), (10, 11), (40, 39)]:
    transactions.loc[transactions.shop_id == shop_with_typo, 'shop_id'] = original_shop_id
    test.loc[test.shop_id == shop_with_typo, 'shop_id'] = original_shop_id

# drop bad ids, they are unnececary now
shops.drop([0, 1, 10], inplace=True)
shops.loc[shops.shop_name == 'Сергиев Посад ТЦ "7Я"', 'shop_name' ] = 'СергиевПосад ТЦ "7Я"'
shops.loc[shops.shop_name == '!Якутск Орджоникидзе, 56 фран', 'shop_name'] = 'Якутск Орджоникидзе, 56 фран'
shops.loc[shops.shop_name == '!Якутск ТЦ "Центральный" фран', 'shop_name'] = 'Якутск ТЦ "Центральный" фран'
shops.shop_name.apply(lambda x: x.split()[0]).head(10)
shops['city_id'] = shops.shop_name.apply(lambda x: x.split()[0]).factorize(sort=True)[0]
print('Number of unique cities is:', shops.city_id.nunique())
shops.head()
# you can use regex if your are more comfortable with it. for me this way is faster
# here we use strip to throw away spaces
def simple_splitter(category):
    if '-' in category:
        return category.split('-')[0].strip()
    elif '(' in category:
        return category.split('(')[0].strip()
    else: 
        return category

item_categories['meta_category_id'] = \
    item_categories.item_category_name.apply(simple_splitter).factorize(sort=True)[0]
print('Number of meta categories:', item_categories.meta_category_id.nunique())
item_categories.head()
vectorizer = CountVectorizer().fit(items.item_name.tolist())
print('Unique words in vectorizer:', len(vectorizer.get_feature_names()))
print('Some examples of vectorized words:', vectorizer.get_feature_names()[::200])
# create lemmatizer and stopwords list
stemmer = Mystem() 
ru_stopwords = stopwords.words('russian')

# preprocess function from 
# https://www.kaggle.com/alxmamaev/how-to-easy-preprocess-russian-text
# it simply lemmatize and stems sentences (using apriory grammar knowledge)
# and then filter out most frequent uninformative words
def preprocess_text(text):
    tokens = stemmer.lemmatize(text.lower())
    tokens = [token for token in tokens if token not in ru_stopwords\
              and token != " " \
              and token.strip() not in punctuation]
    
    text = " ".join(tokens)
    
    return text

t = time.time()
vectorizer = CountVectorizer().fit(items.item_name.apply(preprocess_text).tolist())
print('Time spent:', time.time() - t)
print('Unique words in vectorizer on stemmed words:', len(vectorizer.get_feature_names()))
print('Some examples of vectorized on stemmed words:', vectorizer.get_feature_names()[::100])
plt.figure(figsize=(16, 4))
counts = vectorizer.transform(items.item_name.apply(preprocess_text).tolist())
transformer = TfidfTransformer(smooth_idf=False)
tfidf = transformer.fit_transform(counts)
# we calculate mean among all vector dimensions and plot them
# we can expect that some peaks are very informative words
plt.plot(tfidf.toarray().mean(axis=0))
# here we use list comprehension to get all useful features by name, we also set threshold value at 0.005
informative_words = [vectorizer.get_feature_names()[ind] for ind in
                         np.where(tfidf.toarray().mean(axis=0) > 0.010)[0]]
print('Number of informative words:', len(informative_words))
print('Examples of informative words:', informative_words)
# note that normally clustering algorithms are used on original data,
# that is unfeasible in this case, so we use already dim. reduced vectors
# note that we train on all items, so our similar items_name will be encoded in training set 
# hyperparameters optimization, grid search on number of clusters
gmm_data = NMF(50, random_state=69).fit_transform(tfidf)
# gmms = []
# for i in tqdm_notebook(range(8)):
#     gmms.append(GaussianMixture(n_components=2**i, random_state=69).fit(gmm_data))
# plt.figure(figsize=(16, 8))
# plt.plot([gmm.aic(gmm_data) for gmm in gmms], label="AIC")
# plt.plot([gmm.bic(gmm_data) for gmm in gmms], label="BIC")
# plt.legend()
# del gmms
# now lets fit our best model
t = time.time()
gmm = GaussianMixture(n_components=2**5, random_state=69, verbose=2).fit(gmm_data)
tfidf_probs = gmm.predict_proba(gmm_data)
# i prefer using gaussian mixture model for clustering, because i can get
# probability distribution over all clusters
# here we create 3 new columns: first, second and third nearest clusters
items['tf-idf_cluster1'] = tfidf_probs.argsort(axis=1)[:, -1]
items['tf-idf_cluster2'] = tfidf_probs.argsort(axis=1)[:, -2]
items['tf-idf_cluster3'] = tfidf_probs.argsort(axis=1)[:, -3]
print('Processed in:', time.time() - t)
t = time.time()
# now let's add smaller dimensionallity reduction of tfidf in items dataframe
# dimensionality reduction with matrix factorization
nmf = NMF(5, random_state=69).fit_transform(tfidf)

# vectorizer = CountVectorizer().fit(informative_words)
# tfidf = TfidfTransformer(smooth_idf=True).fit_transform(
#     vectorizer.transform(items.item_name.apply(preprocess_text).tolist()))
# extend columns in items dataframe
# for i in range(len(vectorizer.get_feature_names())):
#     items["tfidf_"+str(i+1)] = tfidf[:, i].toarray()
for i in range(nmf.shape[1]):
    items['nmf_'+str(i+1)] = nmf[:, i]
print('Proccessed in: ', time.time() - t)
items.head().T
transactions
transactions = pd.merge(transactions, shops, on='shop_id', how='left')
items = pd.merge(items, item_categories, on='item_category_id', how='left')
transactions = pd.merge(transactions, items, on='item_id', how='left')
transactions = transactions.drop(['item_name', 'shop_name', 'item_category_name'], axis=1)
transactions.head()
transactions.date_block_num.value_counts().plot(kind='bar')
transactions.item_id.value_counts().iloc[:20].plot(kind='bar')
transactions['is_pocket'] = 0
transactions.loc[transactions.item_id==20949, 'is_pocket'] = 1
# printing prices statistics
# there are negative values, lets filter them
transactions.item_price.describe()
transactions = transactions[transactions.item_price > 0.0].copy()
transactions.item_price.describe()
# apply log-trick to get better vizualisation on item price
transactions.item_price.transform(lambda x: np.log(1+x)).hist(bins=22)
transactions = transactions[np.log(transactions.item_price+1) < 10.3].copy()
transactions.item_price.describe()
plt.boxplot(transactions.item_cnt_day.values)
transactions = transactions[transactions.item_cnt_day <= 20].copy()
plt.figure(figsize=(16, 8))
# using revenue for visualization
transactions['revenue'] = transactions['item_price']*transactions['item_cnt_day']
transactions.groupby('date_block_num').revenue.agg('sum').plot()
plt.figure(figsize=(16, 8))
# using revenue for visualization
transactions.groupby(transactions.date.dt.dayofweek).revenue.agg('sum').plot()
# how many unique items are present in training set?
print(transactions.item_id.nunique())
# how many unique shops are present in training set?
print(transactions.shop_id.nunique())
# how many rows in training set?
print(transactions.item_id.shape[0])

# how many unique items are present in test set?
print(test.item_id.nunique())
# how many unique shops are present in test set?
print(test.shop_id.nunique())
# how many rows in test set?
print(test.item_id.shape[0])
# let's see how frequent are items in our training set
# here we get group by on each item in date block and then count how much time this item is present
itemfreqbydateblock = transactions.groupby(['date_block_num', 'item_id'], as_index=False
                   ).date_block_num.apply(np.size).reset_index()
# then we calculate mean over date block num for all items
itemfreqmean = itemfreqbydateblock.groupby('item_id', as_index=False)[0].apply(np.mean)
# here is some statistics about item frequencies in our training set
itemfreqmean.describe()
# now let's repeat our steps but for the test part
test['date_block_num'] = 34
itemfreqbydateblock_test = test.groupby(['date_block_num', 'item_id'], as_index=False
                   ).date_block_num.apply(np.size).reset_index()
# then we calculate mean over date block num for all items
itemfreqmean_test = itemfreqbydateblock_test.groupby('item_id', as_index=False)[0].apply(np.mean)
# here is some statistics about item frequencies in our test set
itemfreqmean_test.describe()
# check for new items to predict
len(list(set(test.item_id) - set(test.item_id).intersection(set(transactions.item_id))))
# what are these 366 new items
t = time.time()
new_item_ids = np.array(list(set(test.item_id) - set(test.item_id).intersection(set(transactions.item_id))))
for i_id in new_item_ids[:5]:
    # find similar
    pass
# print 10 most frequent categories
print(item_categories.iloc[
    items.iloc[new_item_ids].item_category_id.value_counts().head(10).index
        ].item_category_name)

similar_items = []
for i_id in new_item_ids:
    items.iloc[i_id]
    condition = np.ones_like(items.item_id)
    for i, cluster in enumerate(items.loc[i_id, ['tf-idf_cluster'+str(i+1) for i in range(3)]].values):
        condition *= items['tf-idf_cluster'+str(i+1)] == cluster
    # now we get all similar items by cluster
    # next step is to calculate l2 distance to our new item
    # (NMF uses L2 distance as objective function, so we also should use it)
    tfidf = items[condition.astype(bool)].loc[:, ['nmf_'+str(i+1) for i in range(5)]]
    # we take first 5 similar items
    # note that we don't take the first item, because it obviously is i_id
    similar_items.append(((tfidf - tfidf.loc[i_id])**2).sum(axis=1).sort_values().index.values[1:6] )

print('Processed in:', time.time() - t)
similar_ids = pd.Series(np.concatenate(similar_items))
print('Found {} similar items'.format(similar_ids.unique().size))
transactions = transactions[transactions.item_id.isin(pd.concat([test.item_id, similar_ids]).unique())]
transactions = transactions[transactions.shop_id.isin(test.shop_id.unique())]
transactions.shape
from itertools import product
import time
t = time.time()
grid = []
cols  = ['date_block_num', 'shop_id', 'item_id']
# For every month we create a grid from all shops/items combinations from that month
for i in transactions.date_block_num.unique():
    sales = transactions[transactions.date_block_num == i]
    grid.append( np.array(list( product( [i],
                                          sales.shop_id.unique(),
                                          sales.item_id.unique()) ), dtype = np.int16) )

grid = pd.DataFrame( np.vstack(grid), columns = cols )
grid.sort_values( cols, inplace = True )
print('Processed in:', time.time() - t)
t = time.time()
#get aggregated values for (shop_id, item_id, month)
gb = transactions.groupby(cols).agg({'item_cnt_day': ['sum'],
                                     'item_price':np.mean}).reset_index()
gb.columns = [*list(gb.columns)[:-2], ('target', ''), ('price_avg', '')]
#fix column names
gb.columns = [col[0] if col[-1]=='' else col[-1] for col in gb.columns.values]
#join aggregated data to the grid
all_data = pd.merge(grid, gb, how='left',on=cols).fillna(0)
#sort the data
all_data.sort_values(cols, inplace=True)
print('Processed in:', time.time() - t)
all_data = pd.merge(all_data, items, on='item_id',how='left')
all_data = pd.merge(all_data, shops, on='shop_id',how='left')
all_data.drop(['item_name', 'shop_name', 'item_category_name'], axis=1, inplace=True)
# add some other forgotted features
all_data['is_pocket'] = 0
all_data.loc[all_data.item_id==20949, 'is_pocket'] = 1
# following the suggestion on coursera
# all_data["target"] = all_data["target"].fillna(0.3343).clip(0, 20)
all_data['month'] = all_data.date_block_num % 12
all_data['days'] = all_data['month'].apply(lambda x: [31,28,31,30,31,30,31,31,30,31,30,31][x]).astype(np.int8)
#all_data["item_shop_first_sale"] = all_data["date_block_num"] - all_data.groupby(["item_id","shop_id"])["date_block_num"].transform('min')
#all_data["item_first_sale"] = all_data["date_block_num"] - all_data.groupby(["item_id"])["date_block_num"].transform('min')
all_data.head()
# change dtypes so the database is less heavy
uints = ['item_category_id', 'date_block_num', 'meta_category_id', 'city_id', 'is_pocket', 'month']
for col in all_data.columns:
    if all_data[col].dtype == np.float64:
        all_data[col] = all_data[col].astype(np.float32)
    elif all_data[col].dtype == np.int64:
        all_data[col] = all_data[col].astype(np.int32)
    elif col in uints:
        all_data[col] = all_data[col].astype(np.uint8)
    
all_data.dtypes
# non regularized mean encoding
summ = all_data.groupby(['item_id', 'date_block_num'])['target'].transform('mean')
np.corrcoef(all_data['target'].values, summ.values)[0][1]
# expanding mean encoding
cumsum = all_data.groupby(['item_id', 'date_block_num']).target.cumsum() - all_data.target
cumcnt = all_data.groupby(['item_id', 'date_block_num']).cumcount()

np.corrcoef(all_data['target'].values, (cumsum/cumcnt).fillna(0.3343).values)[0][1]
for type_id in tqdm_notebook(['item_id', 'shop_id', 'item_category_id']):
    for column_id, aggregator, aggtype in [('item_price',np.mean,'avg'),('item_cnt_day',np.sum,'sum'),('item_cnt_day',np.mean,'avg')]:      
#         mean_df = transactions.groupby([type_id,'date_block_num']).aggregate(aggregator).reset_index()[[column_id,type_id,'date_block_num']]
#         mean_df.columns = [type_id+'_'+aggtype+'_'+column_id, type_id, 'date_block_num']
#         all_data = pd.merge(all_data, mean_df, on=['date_block_num',type_id], how='left')
        if column_id == 'item_price': 
            col = 'price_avg'
            fill_value = all_data[col].median() if aggtype == 'avg' else 0.0
        elif column_id == 'item_cnt_day': 
            col = 'target'
            fill_value = all_data[col].mean() if aggtype == 'avg' else 0.0
        cumsum = all_data.groupby([type_id, 'date_block_num'])[col].cumsum() - all_data[col]
        if aggtype == 'avg':
            cumcnt = all_data.groupby([type_id, 'date_block_num']).cumcount()
            all_data[type_id+'_'+aggtype+'_'+column_id] = cumsum/cumcnt
        else:
            all_data[type_id+'_'+aggtype+'_'+column_id] = cumsum
        all_data[type_id+'_'+aggtype+'_'+column_id].fillna(fill_value, inplace=True)
# Here we print columns indexes for convinience
[(i, c) for i, c in enumerate(all_data.columns.tolist())]
lag_variables  = list(all_data.columns[19:])+['target']
lags = [1, 2, 3, 6]
for lag in tqdm_notebook(lags):
    sales_new_df = all_data.copy()
    sales_new_df.date_block_num += lag
    sales_new_df = sales_new_df[['date_block_num','shop_id','item_id']+lag_variables]
    for col in sales_new_df.columns:
        sales_new_df[col] = sales_new_df[col].astype(np.float32)
    sales_new_df.columns = ['date_block_num','shop_id','item_id']+ [lag_feat+'_lag_'+str(lag) for lag_feat in lag_variables]
    all_data = pd.merge(all_data, sales_new_df,on=['date_block_num','shop_id','item_id'] ,how='left')
all_data.head()
# also add subtractions of lagged values
all_data['delta_target_lag_1'] = all_data['target_lag_1'] - all_data['target_lag_2']
all_data['delta_target_lag_2'] = all_data['target_lag_1'] - all_data['target_lag_3']
all_data['delta_target_lag_5'] = all_data['target_lag_1'] - all_data['target_lag_6']
all_data = all_data[all_data['date_block_num']>12]
for feat in all_data.columns:
    if 'item_cnt' in feat:
        all_data[feat].fillna(0.0, inplace=True)
    if 'target' in feat:
        all_data[feat].fillna(0.0, inplace=True)
    elif 'item_price' in feat:
        all_data[feat].fillna(all_data[feat].median(), inplace=True)
all_data
all_data.columns
cols_to_drop = \
['price_avg',
 'item_id_avg_item_price',
 'item_id_sum_item_cnt_day',
 'item_id_avg_item_cnt_day',
 'shop_id_avg_item_price',
 'shop_id_sum_item_cnt_day',
 'shop_id_avg_item_cnt_day',
 'item_category_id_avg_item_price',
 'item_category_id_sum_item_cnt_day',
 'item_category_id_avg_item_cnt_day'
]
all_data.to_pickle('all_data.gz')
#all_data = pd.read_pickle('all_data.gz')
training = all_data[all_data['date_block_num'] < 33].drop(cols_to_drop,axis=1)
validation = all_data[all_data['date_block_num'] == 33].drop(cols_to_drop,axis=1)
#del all_data
gc.collect()
train = Pool(training.iloc[:, training.columns != 'target'], training['target'].values,
            [0, 1, 2, 3, 4, 5, 6, 12, 13, 14, 15])
val = Pool(validation.iloc[:, validation.columns != 'target'], validation['target'].values,
          [0, 1, 2, 3, 4, 5, 6, 12, 13, 14, 15])
# alldata = Pool(all_data.drop(cols_to_drop,axis=1).iloc[:, training.columns != 'target'],
#               all_data.drop(cols_to_drop,axis=1)['target'].values,
#               [0, 1, 2, 3, 4, 5, 6, 12, 13, 14, 15])
# print statistics about columns
for col in training.columns:
    print(col, training[col].isna().value_counts())
    print(training[col].min(), training[col].max())
    print(training[col].dtype)
[(i, c) for i, c in enumerate(training.iloc[:, training.columns != 'target'].columns.tolist())]
# manually hyperparameters tuning
# depth=10, lr=0.02, subsample=1.0, rsm=0.1, border_count=64, l2leafreg=None
# depth=15, lr=0.01, subsample=1.0, rsm=0.1, border_count=64, l2leafreg=None
# depth=16, lr=0.013, subsample=1.0, rsm=0.1, border_count=64, l2leafreg=None
tunned_model = CatBoostRegressor(
    random_seed=63,
    iterations=250,
    learning_rate=0.013,
    #l2_leaf_reg=1,
    #bagging_temperature=1,
    random_strength=1,
    depth=16,
    eval_metric = 'RMSE',
    boosting_type='Plain',
    bootstrap_type='Bernoulli',
    subsample=1.0,
    one_hot_max_size=20,
    rsm=0.1,
    leaf_estimation_iterations=5,
    max_ctr_complexity=2,
    border_count=64,
    thread_count=-1,)

tunned_model.fit(
    train,
    logging_level='Silent',
    eval_set=val,
    use_best_model=True,
    plot=True
)
# serialize catboost model
# i tried with their method (save_model), but it has a bug and 
# don't save feature_importances_
joblib.dump(tunned_model, 'cboost.cbm')
cb = joblib.load('cboost.cbm')
cb.feature_importances_
# let's plot feature importances
plt.figure(figsize=(16, 8))
plt.xticks(rotation=90)
indexes = cb.feature_importances_.argsort()[::-1]
plt.bar(training.iloc[:, training.columns != 'target'].columns[indexes],
        cb.feature_importances_[indexes])
# we will use this features for KNN
training.iloc[:, ~training.columns.isin(training.iloc[:, 
                    training.columns != 'target'].columns[indexes].tolist()[-44:]+['target'])]
# for knn we will use last 12 months
knn_data = training[training.date_block_num > 21].iloc[:, ~training.columns.isin(training.iloc[:, 
                    training.columns != 'target'].columns[indexes].tolist()[-44:]+['target'])]
#
sc = StandardScaler().fit(knn_data)
knn_data = sc.transform(knn_data)
# notice that we normalize target value, so we predict in the range [0, 1]
knn_target = training[training.date_block_num > 21].target.values
#knn = KNeighborsRegressor(n_neighbors=10, weights = "distance", metric="braycurtis", n_jobs=-1).fit(knn_data,
#                                     training.target/20)
# perform grid search of number of neighbours on subset of training
# for better results we repeat process 5 times and average score
# this may take a while
# best value is 64, just trust me, it will take tooo long to run
rmse_score = 1.3
for n in tqdm_notebook([3, 8, 16, 32, 64, 128]):
    rmses = []
    for i in range(5):
        indices = np.random.choice(np.arange(knn_data.shape[0]), 20000)
        knn_tmp = knn_data[indices].copy()
        
        # bray curtis is a good metric as much as cosine distance in this problem
        # weights are chosen to distance because they provide better results
        knn = KNeighborsRegressor(n_neighbors=n, weights = 'distance', metric='braycurtis', n_jobs=-1).fit(
                                         knn_tmp,
                                         knn_target[indices])
        rmse = np.sqrt(mean_squared_error(validation.target, knn.predict(knn_val_data)*20))
        rmses.append(rmse)
    # here we find best rmse_score on average and corresponding to it number of neighbours
    if rmse_score > np.mean(rmses):
        rmse_score = np.mean(rmses)
        best_n = n
        print(rmse_score, best_n)
# run our model
knn_val_data = sc.transform(validation.iloc[:, ~validation.columns.isin(validation.iloc[:, 
                    validation.columns != 'target'].columns[indexes].tolist()[-44:]+['target'])])
t = time.time()
indices = np.random.choice(np.arange(knn_data.shape[0]), 25000)
knn_tmp = knn_data[indices].copy()
knn = KNeighborsRegressor(n_neighbors=128, weights = 'distance', metric='braycurtis', n_jobs=-1).fit(
                                         knn_tmp,
                                         knn_target[indices])
print('RMSE score on validation:', np.sqrt(mean_squared_error(validation.target, knn.predict(knn_val_data))))
print('Processed in:', time.time() - t)
# Serializing knn and standard scaler
#joblib.dump([knn, sc], 'knn_model')
knn, sc = joblib.load('knn_model')
knn_preds = knn.predict(knn_val_data)
cb_preds = cb.predict(validation.iloc[:, validation.columns != 'target'])
# scatterplot to see how much predictions are correlated
plt.figure(figsize=(16, 4))
plt.scatter(knn_preds, cb_preds)
# score on catboost model
print('RMSE score on validation:', np.sqrt(mean_squared_error(validation.target.values, 
                                                              cb_preds)))
# learn a second level model 
t = time.time()
meta_data = np.hstack([knn_preds.reshape(-1, 1), cb_preds.reshape(-1, 1)])
meta_learner = SGDRegressor(random_state=69, 
                            early_stopping=False,
                            max_iter=1000,
                            learning_rate='adaptive',
                            eta0 = 0.001,
                            tol=1e-3).fit(meta_data, validation.target.values)
print('RMSE score on validation:', np.sqrt(mean_squared_error(validation.target.values, 
                                                              meta_learner.predict(meta_data))))
print('Processed in:', time.time() - t)
# serializing our model
#joblib.dump(meta_learner, 'meta_learner')
meta_learner = joblib.load('meta_learner')
test = pd.merge(test, items, on='item_id', how='left')
test= pd.merge(test, shops, on='shop_id',how='left')
test.drop(['item_name', 'shop_name', 'item_category_name'], axis=1, inplace=True)
# add some other forgotted features
test['date_block_num'] = 34
# actually this feature is bad on test data (not in real-life scenario)
test['is_pocket'] = 0
test.loc[test.item_id==20949, 'is_pocket'] = 1 
# following the suggestion on coursera, clipping values
#test["target"] = all_data["target"].fillna(0.3343).clip(0, 20)
test['month'] = test.date_block_num % 12
test['days'] = test['month'].apply(lambda x: [31,28,31,30,31,30,31,31,30,31,30,31][x]).astype(np.int8)
for lag in tqdm_notebook(lags):
    sales_new_df = all_data.copy()
    sales_new_df.date_block_num += lag
    sales_new_df = sales_new_df[['date_block_num','shop_id','item_id']+lag_variables]
    sales_new_df.columns = ['date_block_num','shop_id','item_id']+ [lag_feat+'_lag_'+str(lag) for lag_feat in lag_variables]
    test = pd.merge(test, sales_new_df,on=['date_block_num','shop_id','item_id'] ,how='left')
# also add subtractions of lagged values
test['delta_target_lag_1'] = test['target_lag_1'] - test['target_lag_2']
test['delta_target_lag_2'] = test['target_lag_1'] - test['target_lag_3']
test['delta_target_lag_5'] = test['target_lag_1'] - test['target_lag_6']
_test = set(test.drop(['ID'], axis=1).columns)
_training = set(training.drop('target',axis=1).columns)
for i in _test:
    assert i in _training, i
for i in _training:
    assert i in _test, i
assert _training == _test
test = test.drop(['ID'], axis=1)
for feat in test.columns:
    if 'target' in feat:
        test[feat]=test[feat].fillna(0)
    if 'item_cnt' in feat:
        test[feat]=test[feat].fillna(0)
    elif 'item_price' in feat:
        test[feat]=test[feat].fillna(test[feat].median())
test.describe()
# This may take a while, KNN predictions are so long, even with n_jobs=-1,  ~3 minutes on kaggle server
# Catboost model is enough to pass final project, though
pred = cb.predict(test[training.iloc[:, training.columns != 'target'].columns])
knn_pred = knn.predict(sc.transform(test.iloc[:, ~test.columns.isin(test.iloc[:, 
                    test.columns != 'target'].columns[indexes].tolist()[-44:]+['target'])]))
pred = meta_learner.predict(np.hstack([knn_pred.reshape(-1, 1), pred.reshape(-1, 1)])).clip(0.0, 20.0)
# Alternatively, use
pred = cb.predict(test[training.iloc[:, training.columns != 'target'].columns]).clip(0.0, 20.0)
pd.Series(pred).describe()
sub_df = pd.DataFrame({'ID':test.index,'item_cnt_month': pred})
sub_df.head(10)
sub_df.to_csv('submission.csv',index=False)