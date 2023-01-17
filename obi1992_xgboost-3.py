import numpy as np
import pandas as pd
from subprocess import check_output
from tqdm import tqdm
import time
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
import xgboost as xgb
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
train_df = pd.read_csv('../input/train.tsv', sep='\t')
test_df = pd.read_csv('../input/test.tsv', sep='\t')
feature_names = []
# check percentage of missing values
train_df.isna().sum() / train_df.shape[0]
### exploratory analysis ###
# Lets check how the distribution of test and vaidation set looks like ...
start = time.time()
fig, ax = plt.subplots(nrows=5, sharex=True, sharey=True)
sns.distplot(train_df.loc[train_df['item_condition_id']==3, 'price'].tolist(), ax=ax[0], color='blue', label='Validation')
sns.distplot(train_df.loc[train_df['item_condition_id']==1, 'price'].tolist(), ax=ax[1], color='green', label='Test')
sns.distplot(train_df.loc[train_df['item_condition_id']==2, 'price'].tolist(), ax=ax[2], color='green', label='Test')
sns.distplot(train_df.loc[train_df['item_condition_id']==4, 'price'].tolist(), ax=ax[3], color='green', label='Test')
sns.distplot(train_df.loc[train_df['item_condition_id']==5, 'price'].tolist(), ax=ax[4], color='green', label='Test')
ax[0].legend(loc=0)
ax[1].legend(loc=0)
plt.show()
end = time.time()
print("Time taken by above cell is {}.".format((end-start)))
# temp codes
# sns.distplot(train_df['price'].tolist(), color='blue', label='Validation')
# plt.show()

# temp_train_price = np.log(train_df['price']+1)
# sns.distplot(temp_train_price, color='blue', label='Validation')
# plt.show()
##### create dummy variables denoting if a record has certain category label ###
# output: added columns in dataframe, derived from count matrix reduced by svm
# replace all nan with ''
print('number of missing in category_name:',train_df['category_name'].isnull().sum())
train_df['category_name'].fillna('', inplace=True)
test_df['category_name'].fillna('', inplace=True)
print('number of missing in category_name after imputation:',train_df['category_name'].isnull().sum())
start = time.time()
def tokenize_category_name(text):
    return text.lower().strip().split('/')

count_vec = CountVectorizer(
    tokenizer=tokenize_category_name, ngram_range=(1,1), stop_words=None)
full_count = count_vec.fit_transform(
    train_df['category_name'].values.tolist() + 
    test_df['category_name'].values.tolist())
train_count = count_vec.transform(train_df['category_name'].values.tolist())
test_count = count_vec.transform(test_df['category_name'].values.tolist())

n_comp = 20
svd_obj = TruncatedSVD(n_components=n_comp, algorithm='arpack')
svd_obj.fit(full_count.asfptype()) # change int to float, as svd does not accept int type
train_svd = pd.DataFrame(svd_obj.transform(train_count.asfptype()))
test_svd = pd.DataFrame(svd_obj.transform(test_count.asfptype()))
train_svd.columns = ['svd_cat_'+str(i) for i in range(n_comp)]
test_svd.columns = ['svd_cat_'+str(i) for i in range(n_comp)]
feature_names += ['svd_cat_'+str(i) for i in range(n_comp)]

train_df = pd.concat([train_df, train_svd], axis=1)
test_df = pd.concat([test_df, test_svd], axis=1)

print("time taken {}".format((time.time() - start)/60))
train_df.head(3)
### if brand name is missing ###
# output: add dummy series in dataframe
train_df['brand_miss'] = train_df['brand_name'].isnull().astype(float)
test_df['brand_miss'] = test_df['brand_name'].isnull().astype(float)
print('number of records in train set with missing brand name: {}'.format(train_df['brand_miss'].sum()))
print('number of records in test set with missing brand name: {}'.format(test_df['brand_miss'].sum()))
feature_names += ['brand_miss']
### create dummy variables denoting brands ###
# output: added columns in dataframe, derived from count matrix reduced by svm
# replace all nan with ''
print('number of missing in category_name:',train_df['brand_name'].isnull().sum())
train_df['brand_name'].fillna('', inplace=True)
test_df['brand_name'].fillna('', inplace=True)
print('number of missing in category_name after imputation:',train_df['brand_name'].isnull().sum())
start = time.time()
def tokenize_brand_name(text):
    return [text.lower().strip(' ')]

count_vec = CountVectorizer(tokenizer=tokenize_brand_name, ngram_range=(1,1), stop_words=None)
# count_vec.get_feature_names() # can use this to check what tokens are counted
full_count = count_vec.fit_transform(
    train_df['brand_name'].values.tolist() + 
    test_df['brand_name'].values.tolist())
train_count = count_vec.transform(train_df['brand_name'].values.tolist())
test_count = count_vec.transform(test_df['brand_name'].values.tolist())

n_comp = 10
svd_obj = TruncatedSVD(n_components=n_comp, algorithm='arpack')
svd_obj.fit(full_count.asfptype()) # change int to float, as svd does not accept int type
train_svd = pd.DataFrame(svd_obj.transform(train_count.asfptype()))
test_svd = pd.DataFrame(svd_obj.transform(test_count.asfptype()))
train_svd.columns = ['svd_brand_'+str(i) for i in range(n_comp)]
test_svd.columns = ['svd_brand_'+str(i) for i in range(n_comp)]
feature_names += ['svd_brand_'+str(i) for i in range(n_comp)]

train_df = pd.concat([train_df, train_svd], axis=1)
test_df = pd.concat([test_df, test_svd], axis=1)

print("time taken {}".format((time.time() - start) / 60))

### add dummies variables for item_condition_id ###
train_cond_id_dummies = pd.get_dummies(
    train_df['item_condition_id'], dummy_na=False, drop_first=True)
test_cond_id_dummies = pd.get_dummies(
    test_df['item_condition_id'], dummy_na=False, drop_first=True)
train_cond_id_dummies.columns = ['cond_id_'+str(i) for i in range(train_cond_id_dummies.shape[1])]
test_cond_id_dummies.columns = ['cond_id_'+str(i) for i in range(test_cond_id_dummies.shape[1])]
feature_names += train_cond_id_dummies.columns.tolist()
train_df = pd.concat([train_df, train_cond_id_dummies], axis=1)
test_df = pd.concat([test_df, test_cond_id_dummies], axis=1)
### if the item_description is not available ###
train_df['itemDest_miss'] = (train_df['item_description'] == 'No description yet').astype(float)
test_df['itemDest_miss'] = (test_df['item_description'] == 'No description yet').astype(float)
feature_names += ['itemDest_miss']
print('number of records with no item description: {}'.format(train_df['itemDest_miss'].sum()))
### add dummies variables for tfidf of item description, reduced by svd ###
# replace all nan with ''
train_df['item_description'].fillna('', inplace=True)
test_df['item_description'].fillna('', inplace=True)
start = time.time()

tfidf_vec = TfidfVectorizer(ngram_range=(1,2), stop_words=None)
# count_vec.get_feature_names() # can use this to check what tokens are counted
full_tfidf = tfidf_vec.fit_transform(train_df['item_description'].values.tolist() + 
                                     test_df['item_description'].values.tolist())
train_tfidf = tfidf_vec.transform(train_df['item_description'].values.tolist())
test_tfidf = tfidf_vec.transform(test_df['item_description'].values.tolist())

n_comp = 20
svd_obj = TruncatedSVD(n_components=n_comp, algorithm='arpack')
svd_obj.fit(full_tfidf.asfptype()) # change int to float, as svd does not accept int type
train_svd = pd.DataFrame(svd_obj.transform(train_tfidf.asfptype()))
test_svd = pd.DataFrame(svd_obj.transform(test_tfidf.asfptype()))
train_svd.columns = ['svd_itemDes_tfidf_'+str(i) for i in range(n_comp)]
test_svd.columns = ['svd_itemDes_tfidf_'+str(i) for i in range(n_comp)]
feature_names += ['svd_itemDes_tfidf_'+str(i) for i in range(n_comp)]

train_df = pd.concat([train_df, train_svd], axis=1)
test_df = pd.concat([test_df, test_svd], axis=1)

print("time taken {}".format((time.time() - start)/60))
### add dummies variables for tfidf of item name (name), reduced by svd ###
# replace all nan with ''
print('number of missing in name:',train_df['name'].isnull().sum())
train_df['name'].fillna('', inplace=True)
test_df['name'].fillna('', inplace=True)
print('number of missing in name after imputation:',train_df['name'].isnull().sum())
start = time.time()

tfidf_vec = TfidfVectorizer(ngram_range=(1,2), stop_words=None)
# count_vec.get_feature_names() # can use this to check what tokens are counted
full_tfidf = tfidf_vec.fit_transform(train_df['name'].values.tolist()+test_df['name'].values.tolist())
train_tfidf = tfidf_vec.transform(train_df['name'].values.tolist())
test_tfidf = tfidf_vec.transform(test_df['name'].values.tolist())

n_comp = 20
svd_obj = TruncatedSVD(n_components=n_comp, algorithm='arpack')
svd_obj.fit(full_tfidf.asfptype()) # change int to float, as svd does not accept int type
train_svd = pd.DataFrame(svd_obj.transform(train_tfidf.asfptype()))
test_svd = pd.DataFrame(svd_obj.transform(test_tfidf.asfptype()))
train_svd.columns = ['svd_name_tfidf_'+str(i) for i in range(n_comp)]
test_svd.columns = ['svd_name_tfidf_'+str(i) for i in range(n_comp)]
feature_names += ['svd_name_tfidf_'+str(i) for i in range(n_comp)]

train_df = pd.concat([train_df, train_svd], axis=1)
test_df = pd.concat([test_df, test_svd], axis=1)

print("time taken {}".format((time.time() - start)/60))
feature_names += ['shipping']
feature_names.__len__()
feature_names
# prepare train and validation
train = train_df.copy()
test = test_df.copy()
#y = np.log(train['price'].values + 1)
y = train['price'].values

Xtr, Xv, ytr, yv = train_test_split(train[feature_names].values, y, test_size=0.2, random_state=1987)
dtrain = xgb.DMatrix(Xtr, label=ytr, feature_names=feature_names)
dvalid = xgb.DMatrix(Xv, label=yv, feature_names=feature_names)
dtest = xgb.DMatrix(test[feature_names].values, feature_names=feature_names)
watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
Xtr.shape
start = time.time()
xgb_par = {'min_child_weight': 20, 'eta': 0.05, 'colsample_bytree': 0.5, 'max_depth': 13,
           'subsample': 0.9, 'lambda': 2.0, 'nthread': -1, 'booster' : 'gbtree', 'silent': 1,
           'eval_metric': 'rmse', 'objective': 'reg:linear'}

model_3 = xgb.train(xgb_par, dtrain, 80, watchlist, early_stopping_rounds=20, 
                    maximize=False, verbose_eval=20)
print('Modeling RMSLE %.5f' % model_3.best_score)
print("Time taken in training is {}.".format((time.time() - start)/60))
# Plot the important variables ##
from matplotlib import pyplot as plt
fig, ax = plt.subplots(figsize=(12,6))
xgb.plot_importance(model_3, max_num_features=20, height=0.8, ax=ax)
plt.title("Feature importance - top 20")
plt.show()
start = time.time()
yvalid = model_3.predict(dvalid)
ytest = model_3.predict(dtest)
end = time.time()
print("Time taken in prediction is {}.".format(end - start))
# Lets check how the distribution of test and vaidation set looks like ...
start = time.time()
fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True)
sns.distplot(yvalid, ax=ax[0], color='blue', label='Validation')
sns.distplot(ytest, ax=ax[1], color='green', label='Test')
ax[0].legend(loc=0)
ax[1].legend(loc=0)
plt.show()
end = time.time()
print("Time taken by above cell is {}.".format((end-start)))
start = time.time()
if test.shape[0] == ytest.shape[0]:
    print('Test shape OK.')
#test['price'] = np.exp(ytest) - 1
test['price'] = ytest
#test[['test_id', 'price']].to_csv('xgb_4.csv', index=False)
test[['test_id', 'price']].to_csv('xgb_4_2.csv', index=False)
end = time.time()
print("Time taken in training is {}.".format(end - start))
ytest
