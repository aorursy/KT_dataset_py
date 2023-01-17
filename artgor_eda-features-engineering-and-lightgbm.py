import os
import pandas_profiling as pp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
plt.style.use('ggplot')

import datetime
import pandas_profiling as pp
from scipy import stats
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud
from collections import Counter
from nltk.corpus import stopwords
from nltk.util import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer
stop = set(stopwords.words('russian'))
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import KFold
kf = KFold(n_splits=5)
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
periods_train = pd.read_csv('../input/periods_train.csv')
sub = pd.read_csv('../input/sample_submission.csv')
train.head()
train.info()
train.describe(include='all')
periods_train.head()
pp.ProfileReport(train)
train['activation_date'] = pd.to_datetime(train['activation_date'])
train['date'] = train['activation_date'].dt.date
train['weekday'] = train['activation_date'].dt.weekday
train['day'] = train['activation_date'].dt.day
count_by_date_train = train.groupby('date')['deal_probability'].count()
mean_by_date_train = train.groupby('date')['deal_probability'].mean()

test['activation_date'] = pd.to_datetime(test['activation_date'])
test['date'] = test['activation_date'].dt.date
test['weekday'] = test['activation_date'].dt.weekday
test['day'] = test['activation_date'].dt.day
count_by_date_test = test.groupby('date')['item_id'].count()
fig, (ax1, ax3) = plt.subplots(figsize=(26, 8), ncols=2, sharey=True)
count_by_date_train.plot(ax=ax1, legend=False, label='Ads count')
ax1.set_ylabel('Ads count', color='b')
ax2 = ax1.twinx()
mean_by_date_train.plot(ax=ax2, color='g', legend=False, label='Mean deal_probability')
ax2.set_ylabel('Mean deal_probability', color='g')
count_by_date_test.plot(ax=ax3, color='r', legend=False, label='Ads count test')
plt.grid(False)

ax1.title.set_text('Trends of deal_probability and number of ads')
ax3.title.set_text('Trends of number of ads for test data')
ax1.legend(loc=(0.8, 0.35))
ax2.legend(loc=(0.8, 0.2))
ax3.legend(loc="upper right")
fig, ax1 = plt.subplots(figsize=(16, 8))
plt.title("Ads count and deal_probability by day of week.")
sns.countplot(x='weekday', data=train, ax=ax1)
ax1.set_ylabel('Ads count', color='b')
plt.legend(['Projects count'])
ax2 = ax1.twinx()
sns.pointplot(x="weekday", y="deal_probability", data=train, ci=99, ax=ax2, color='black')
ax2.set_ylabel('deal_probability', color='g')
plt.legend(['deal_probability'], loc=(0.875, 0.9))
plt.grid(False)
a = train.groupby(['parent_category_name', 'category_name']).agg({'deal_probability': ['mean', 'count']}).reset_index().sort_values([('deal_probability', 'mean')], ascending=False).reset_index(drop=True)
a
city_ads = train.groupby('city').agg({'deal_probability': ['mean', 'count']}).reset_index().sort_values([('deal_probability', 'mean')], ascending=False).reset_index(drop=True)
print('There are {0} cities in total.'.format(len(train.city.unique())))
print('There are {1} cities with more that {0} ads.'.format(100, city_ads[city_ads['deal_probability']['count'] > 100].shape[0]))
print('There are {1} cities with more that {0} ads.'.format(1000, city_ads[city_ads['deal_probability']['count'] > 1000].shape[0]))
print('There are {1} cities with more that {0} ads.'.format(10000, city_ads[city_ads['deal_probability']['count'] > 10000].shape[0]))
city_ads[city_ads['deal_probability']['count'] > 1000].head()
city_ads[city_ads['deal_probability']['count'] > 1000].tail()
print('Лабинск')
train.loc[train.city == 'Лабинск'].groupby('category_name').agg({'deal_probability': ['mean', 'count']}).reset_index().sort_values([('deal_probability', 'count')], ascending=False).reset_index(drop=True).head(5)
print('Миллерово')
train.loc[train.city == 'Миллерово'].groupby('category_name').agg({'deal_probability': ['mean', 'count']}).reset_index().sort_values([('deal_probability', 'count')], ascending=False).reset_index(drop=True).head()
plt.hist(train['deal_probability']);
plt.title('deal_probability');
text = ' '.join(train['title'].values)
wordcloud = WordCloud(max_font_size=None, stopwords=stop, background_color='white',
                      width=1200, height=1000).generate(text)
plt.figure(figsize=(12, 8))
plt.imshow(wordcloud)
plt.title('Top words for title')
plt.axis("off")
plt.show()
train['description'] = train['description'].apply(lambda x: str(x).replace('/\n', ' ').replace('\xa0', ' '))
text = ' '.join(train['description'].values)
text = [i for i in ngrams(text.lower().split(), 3)]
print('Common trigrams.')
Counter(text).most_common(40)
train[train.description.str.contains('↓')]['description'].head(10).values
train['has_image'] = 1
train.loc[train['image'].isnull(),'has_image'] = 0
print('There are {} ads with images. Mean deal_probability is {:.3}.'.format(len(train.loc[train['has_image'] == 1]), train.loc[train['has_image'] == 1, 'deal_probability'].mean()))
print('There are {} ads without images. Mean deal_probability is {:.3}.'.format(len(train.loc[train['has_image'] == 0]), train.loc[train['has_image'] == 0, 'deal_probability'].mean()))
plt.scatter(train.item_seq_number, train.deal_probability, label='item_seq_number vs deal_probability');
plt.xlabel('item_seq_number');
plt.ylabel('deal_probability');
train['params'] = train['param_1'].fillna('') + ' ' + train['param_2'].fillna('') + ' ' + train['param_3'].fillna('')
train['params'] = train['params'].str.strip()
text = ' '.join(train['params'].values)
text = [i for i in ngrams(text.lower().split(), 3)]
print('Common trigrams.')
Counter(text).most_common(40)
sns.set(rc={'figure.figsize':(15, 8)})
train_ = train[train.price.isnull() == False]
train_ = train.loc[train.price < 100000.0]
sns.boxplot(x="parent_category_name", y="price", hue="user_type",  data=train_)
plt.title("Price by parent category and user type")
plt.xticks(rotation='vertical')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()
train['price'] = train.groupby(['city', 'category_name'])['price'].apply(lambda x: x.fillna(x.median()))
train['price'] = train.groupby(['region', 'category_name'])['price'].apply(lambda x: x.fillna(x.median()))
train['price'] = train.groupby(['category_name'])['price'].apply(lambda x: x.fillna(x.median()))
plt.hist(train['price']);
plt.hist(stats.boxcox(train['price'] + 1)[0]);
#Let's transform test in the same way as train.
test['params'] = test['param_1'].fillna('') + ' ' + test['param_2'].fillna('') + ' ' + test['param_3'].fillna('')
test['params'] = test['params'].str.strip()

test['description'] = test['description'].apply(lambda x: str(x).replace('/\n', ' ').replace('\xa0', ' '))
test['has_image'] = 1
test.loc[test['image'].isnull(),'has_image'] = 0

test['price'] = test.groupby(['city', 'category_name'])['price'].apply(lambda x: x.fillna(x.median()))
test['price'] = test.groupby(['region', 'category_name'])['price'].apply(lambda x: x.fillna(x.median()))
test['price'] = test.groupby(['category_name'])['price'].apply(lambda x: x.fillna(x.median()))
train['price'] = stats.boxcox(train.price + 1)[0]
test['price'] = stats.boxcox(test.price + 1)[0]
train['user_price_mean'] = train.groupby('user_id')['price'].transform('mean')
train['user_ad_count'] = train.groupby('user_id')['price'].transform('sum')

train['region_price_mean'] = train.groupby('region')['price'].transform('mean')
train['region_price_median'] = train.groupby('region')['price'].transform('median')
train['region_price_max'] = train.groupby('region')['price'].transform('max')

train['region_price_mean'] = train.groupby('region')['price'].transform('mean')
train['region_price_median'] = train.groupby('region')['price'].transform('median')
train['region_price_max'] = train.groupby('region')['price'].transform('max')

train['city_price_mean'] = train.groupby('city')['price'].transform('mean')
train['city_price_median'] = train.groupby('city')['price'].transform('median')
train['city_price_max'] = train.groupby('city')['price'].transform('max')

train['parent_category_name_price_mean'] = train.groupby('parent_category_name')['price'].transform('mean')
train['parent_category_name_price_median'] = train.groupby('parent_category_name')['price'].transform('median')
train['parent_category_name_price_max'] = train.groupby('parent_category_name')['price'].transform('max')

train['category_name_price_mean'] = train.groupby('category_name')['price'].transform('mean')
train['category_name_price_median'] = train.groupby('category_name')['price'].transform('median')
train['category_name_price_max'] = train.groupby('category_name')['price'].transform('max')

train['user_type_category_price_mean'] = train.groupby(['user_type', 'parent_category_name'])['price'].transform('mean')
train['user_type_category_price_median'] = train.groupby(['user_type', 'parent_category_name'])['price'].transform('median')
train['user_type_category_price_max'] = train.groupby(['user_type', 'parent_category_name'])['price'].transform('max')
test['user_price_mean'] = test.groupby('user_id')['price'].transform('mean')
test['user_ad_count'] = test.groupby('user_id')['price'].transform('sum')

test['region_price_mean'] = test.groupby('region')['price'].transform('mean')
test['region_price_median'] = test.groupby('region')['price'].transform('median')
test['region_price_max'] = test.groupby('region')['price'].transform('max')

test['region_price_mean'] = test.groupby('region')['price'].transform('mean')
test['region_price_median'] = test.groupby('region')['price'].transform('median')
test['region_price_max'] = test.groupby('region')['price'].transform('max')

test['city_price_mean'] = test.groupby('city')['price'].transform('mean')
test['city_price_median'] = test.groupby('city')['price'].transform('median')
test['city_price_max'] = test.groupby('city')['price'].transform('max')

test['parent_category_name_price_mean'] = test.groupby('parent_category_name')['price'].transform('mean')
test['parent_category_name_price_median'] = test.groupby('parent_category_name')['price'].transform('median')
test['parent_category_name_price_max'] = test.groupby('parent_category_name')['price'].transform('max')

test['category_name_price_mean'] = test.groupby('category_name')['price'].transform('mean')
test['category_name_price_median'] = test.groupby('category_name')['price'].transform('median')
test['category_name_price_max'] = test.groupby('category_name')['price'].transform('max')

test['user_type_category_price_mean'] = test.groupby(['user_type', 'parent_category_name'])['price'].transform('mean')
test['user_type_category_price_median'] = test.groupby(['user_type', 'parent_category_name'])['price'].transform('median')
test['user_type_category_price_max'] = test.groupby(['user_type', 'parent_category_name'])['price'].transform('max')
def target_encode(trn_series=None, 
                  tst_series=None, 
                  target=None, 
                  min_samples_leaf=1, 
                  smoothing=1,
                  noise_level=0):
    """
    
    https://www.kaggle.com/ogrellier/python-target-encoding-for-categorical-features
    Smoothing is computed like in the following paper by Daniele Micci-Barreca
    https://kaggle2.blob.core.windows.net/forum-message-attachments/225952/7441/high%20cardinality%20categoricals.pdf
    trn_series : training categorical feature as a pd.Series
    tst_series : test categorical feature as a pd.Series
    target : target data as a pd.Series
    min_samples_leaf (int) : minimum samples to take category average into account
    smoothing (int) : smoothing effect to balance categorical average vs prior  
    """ 
    assert len(trn_series) == len(target)
    assert trn_series.name == tst_series.name
    temp = pd.concat([trn_series, target], axis=1)
    # Compute target mean 
    averages = temp.groupby(by=trn_series.name)[target.name].agg(["mean", "count"])
    # Compute smoothing
    smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))
    # Apply average function to all target data
    prior = target.mean()
    # The bigger the count the less full_avg is taken into account
    averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing
    averages.drop(["mean", "count"], axis=1, inplace=True)
    # Apply averages to trn and tst series
    ft_trn_series = pd.merge(
        trn_series.to_frame(trn_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=trn_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_trn_series.index = trn_series.index 
    ft_tst_series = pd.merge(
        tst_series.to_frame(tst_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=tst_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_tst_series.index = tst_series.index
    return ft_trn_series, ft_tst_series
train['parent_category_name'], test['parent_category_name'] = target_encode(train['parent_category_name'], test['parent_category_name'], train['deal_probability'])
train['category_name'], test['category_name'] = target_encode(train['category_name'], test['category_name'], train['deal_probability'])
train['region'], test['region'] = target_encode(train['region'], test['region'], train['deal_probability'])
train['image_top_1'], test['image_top_1'] = target_encode(train['image_top_1'], test['image_top_1'], train['deal_probability'])
train['city'], test['city'] = target_encode(train['city'], test['city'], train['deal_probability'])
train['param_1'], test['param_1'] = target_encode(train['param_1'], test['param_1'], train['deal_probability'])
train['param_2'], test['param_2'] = target_encode(train['param_2'], test['param_2'], train['deal_probability'])
train['param_3'], test['param_3'] = target_encode(train['param_3'], test['param_3'], train['deal_probability'])
train.drop(['date', 'day', 'user_id'], axis=1, inplace=True)
test.drop(['date', 'day', 'user_id'], axis=1, inplace=True)
train['len_title'] = train['title'].apply(lambda x: len(x))
train['words_title'] = train['title'].apply(lambda x: len(x.split()))
train['len_description'] = train['description'].apply(lambda x: len(x))
train['words_description'] = train['description'].apply(lambda x: len(x.split()))
train['len_params'] = train['params'].apply(lambda x: len(x))
train['words_params'] = train['params'].apply(lambda x: len(x.split()))

train['symbol1_count'] = train['description'].str.count('↓')
train['symbol2_count'] = train['description'].str.count('\*')
train['symbol3_count'] = train['description'].str.count('✔')
train['symbol4_count'] = train['description'].str.count('❀')
train['symbol5_count'] = train['description'].str.count('➚')
train['symbol6_count'] = train['description'].str.count('ஜ')
train['symbol7_count'] = train['description'].str.count('.')
train['symbol8_count'] = train['description'].str.count('!')
train['symbol9_count'] = train['description'].str.count('\?')
train['symbol10_count'] = train['description'].str.count('  ')
train['symbol11_count'] = train['description'].str.count('-')
train['symbol12_count'] = train['description'].str.count(',')

test['len_title'] = test['title'].apply(lambda x: len(x))
test['words_title'] = test['title'].apply(lambda x: len(x.split()))
test['len_description'] = test['description'].apply(lambda x: len(x))
test['words_description'] = test['description'].apply(lambda x: len(x.split()))
test['len_params'] = test['params'].apply(lambda x: len(x))
test['words_params'] = test['params'].apply(lambda x: len(x.split()))

test['symbol1_count'] = test['description'].str.count('↓')
test['symbol2_count'] = test['description'].str.count('\*')
test['symbol3_count'] = test['description'].str.count('✔')
test['symbol4_count'] = test['description'].str.count('❀')
test['symbol5_count'] = test['description'].str.count('➚')
test['symbol6_count'] = test['description'].str.count('ஜ')
test['symbol7_count'] = test['description'].str.count('.')
test['symbol8_count'] = test['description'].str.count('!')
test['symbol9_count'] = test['description'].str.count('\?')
test['symbol10_count'] = test['description'].str.count('  ')
test['symbol11_count'] = test['description'].str.count('-')
test['symbol12_count'] = test['description'].str.count(',')
vectorizer=TfidfVectorizer(stop_words=stop, max_features=2000)
vectorizer.fit(train['title'])
train_title = vectorizer.transform(train['title'])
test_title = vectorizer.transform(test['title'])
train.drop(['title', 'params', 'description', 'user_type', 'activation_date'], axis=1, inplace=True)
test.drop(['title', 'params', 'description', 'user_type', 'activation_date'], axis=1, inplace=True)
pd.set_option('max_columns', 60)
train.head()
%%time
X_meta = np.zeros((train_title.shape[0], 1))
X_test_meta = []
for fold_i, (train_i, test_i) in enumerate(kf.split(train_title)):
    print(fold_i)
    model = Ridge()
    model.fit(train_title.tocsr()[train_i], train['deal_probability'][train_i])
    X_meta[test_i, :] = model.predict(train_title.tocsr()[test_i]).reshape(-1, 1)
    X_test_meta.append(model.predict(test_title))
    
X_test_meta = np.stack(X_test_meta)
X_test_meta_mean = np.mean(X_test_meta, axis = 0)
X_full = csr_matrix(hstack([train.drop(['item_id', 'deal_probability', 'image'], axis=1), X_meta]))
X_test_full = csr_matrix(hstack([test.drop(['item_id', 'image'], axis=1), X_test_meta_mean.reshape(-1, 1)]))

X_train, X_valid, y_train, y_valid = train_test_split(X_full, train['deal_probability'], test_size=0.20, random_state=42)
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())
#took parameters from this kernel: https://www.kaggle.com/the1owl/beep-beep
params = {'learning_rate': 0.05, 'max_depth': 6, 'boosting': 'gbdt', 'objective': 'regression', 'metric': ['auc','rmse'], 'is_training_metric': True, 'seed': 19, 'num_leaves': 63, 'feature_fraction': 0.9, 'bagging_fraction': 0.8, 'bagging_freq': 5}
model = lgb.train(params, lgb.Dataset(X_train, label=y_train), 2000, lgb.Dataset(X_valid, label=y_valid), verbose_eval=50, early_stopping_rounds=20)
pred = model.predict(X_test_full)
#clipping is necessary.
sub['deal_probability'] = np.clip(pred, 0, 1)
sub.to_csv('sub.csv', index=False)