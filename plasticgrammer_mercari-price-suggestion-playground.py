import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import os

sns.set_style('darkgrid')
sns.set_palette('bone')
pd.options.display.float_format = '{:,.2f}'.format

print(os.listdir("../input"))
train = pd.read_table('../input/train.tsv')
test = pd.read_table('../input/test_stg2.tsv')

print(train.shape, test.shape)
train.info()
train.head()
train['train_id'] = train['train_id'].astype(str)
test['test_id'] = test['test_id'].astype(str)
print(train['price'].describe())
plt.subplot(1, 2, 1)
train['price'].plot.hist(bins=50, edgecolor='white', range=[0,200], figsize=(15,5))
plt.xlabel('price')
plt.ylabel('frequency')
plt.title('Price Distribution')

plt.subplot(1, 2, 2)
np.log1p(train['price']).plot.hist(bins=50, edgecolor='white')
plt.xlabel('log(price+1)')
plt.ylabel('frequency')
plt.title('Log(Price) Distribution')
plt.show()
# correcting skew
train['price'] = np.log1p(train['price'])
train['name'].to_frame().head(10)
print('unique name count:', train['name'].nunique(), 'of', len(train))
print('unique brand count:', train['brand_name'].nunique())
df = train['brand_name'].value_counts().to_frame('brand_count')
df.head(10)
train['category_name'].head(10).to_frame()
catsCount = train['category_name'].apply(lambda x: len(str(x).split('/')))
maxCount = max(catsCount)
print('max categories:', maxCount)
print(train['category_name'][catsCount == maxCount].unique())

del catsCount
# separate category
def separate_category_feat(df):
    df['general_cat'], df['subcat_1'], df['subcat_2'] = \
        zip(*df['category_name'].apply(lambda x: str(x).split('/',3) if x == x else ('None', 'None', 'None')))
subset = train['category_name'].to_frame().copy()
separate_category_feat(subset)
    
subset[['category_name','general_cat','subcat_1','subcat_2']].head(10)
for c in ['general_cat','subcat_1','subcat_2']:
    print(f'unique {c} count:', subset[c].nunique())
    
subset['general_cat'].value_counts().plot.bar()

del subset
train['item_description'].to_frame().head(10)
# remove missing values in item description (in train)
train = train[pd.notnull(train['train_id']) & pd.notnull(train['item_description'])]
all_df = pd.concat([train, test], sort=False)

del train,test
# fillna
all_df['brand_name'].fillna('None', inplace=True)
all_df['item_description'].fillna('None', inplace=True)
separate_category_feat(all_df)
all_df['nm_word_len'] = all_df['name'].map(lambda x: len(x.split()))
all_df['desc_word_len'] = all_df['item_description'].map(lambda x: len(x.split()))
all_df['nm_len'] = all_df['name'].map(lambda x: len(x))
all_df['desc_len'] = all_df['item_description'].map(lambda x: len(x))
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(stop_words='english', 
                             min_df=100,
                             max_features=50000,
                             #tokenizer=tokenize,
                             ngram_range=(1, 2))
all_desc = all_df['item_description'].values
vz = vectorizer.fit_transform(list(all_desc))
tfidf = pd.DataFrame(index=vectorizer.get_feature_names(), data=vectorizer.idf_, columns=['tfidf'])
tfidf.sort_values(by=['tfidf'], ascending=True).head(10)
del all_desc
gc.collect()
from sklearn.cluster import MiniBatchKMeans

num_clusters = 30 # need to be selected wisely

kmeans_model = MiniBatchKMeans(n_clusters=num_clusters,
                               init='k-means++',
                               n_init=1,
                               init_size=1000, batch_size=1000, verbose=0, max_iter=1000)
all_df['kmeans_cluster30'] = kmeans_model.fit_predict(vz)

sorted_centroids = kmeans_model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(num_clusters):
    print("Cluster %d:" % i)
    aux = ''
    for j in sorted_centroids[i, :10]:
        aux += terms[j] + ' | '
    print(aux)
    print()
del kmeans_model, vz, vectorizer
gc.collect()
all_df.drop(['item_description'], axis=1, inplace=True)
all_df.info()
all_df.drop(['name'], axis=1, inplace=True)
all_df.drop(['brand_name'], axis=1, inplace=True)
all_df.drop(['category_name'], axis=1, inplace=True)
import sys
print(pd.DataFrame([[val for val in dir()], [sys.getsizeof(eval(val)) for val in dir()]],
                   index=['name','size']).T.sort_values('size', ascending=False).reset_index(drop=True)[:10])
cols = [c for c in all_df.columns if c not in ['train_id','test_id','price']]
for i, t in all_df.loc[:, cols].dtypes.iteritems():
    if t == object:
        #all_df = pd.concat([all_df, pd.get_dummies(all_df[i].astype(str), prefix=i)], axis=1)
        #all_df.drop(i, axis=1, inplace=True)
        all_df[i] = pd.factorize(all_df[i])[0]
train = all_df[all_df['price'].notnull()].drop(['test_id'], axis=1)
test = all_df[all_df['price'].isnull()].drop(['train_id','price'], axis=1)

X_train = train.drop(['price','train_id'], axis=1)
Y_train = train['price']
X_test  = test.drop(['test_id'], axis=1)
train_id  = train['train_id']
test_id  = test['test_id']

print(X_train.shape, X_test.shape)
del train, test, all_df
gc.collect()
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
params={'learning_rate': 0.2,
        'objective':'regression',
        'metric':'rmse',
        'num_leaves': 31,
        'verbose': 1,
        'random_state':42,
        'bagging_fraction': 0.7,
        'feature_fraction': 0.7
       }

folds = GroupKFold(n_splits=5)

oof_preds = np.zeros(X_train.shape[0])
sub_preds = np.zeros(X_test.shape[0])
for fold_, (trn_, val_) in enumerate(folds.split(X_train, Y_train, groups=train_id)):
    trn_x, trn_y = X_train.iloc[trn_], Y_train.iloc[trn_]
    val_x, val_y = X_train.iloc[val_], Y_train.iloc[val_]
    
    reg = lgb.LGBMRegressor(**params, n_estimators=3000)
    reg.fit(trn_x, trn_y, eval_set=[(val_x, val_y)], early_stopping_rounds=50, verbose=500)
    
    oof_preds[val_] = reg.predict(val_x, num_iteration=reg.best_iteration_)
    sub_preds += reg.predict(X_test, num_iteration=reg.best_iteration_) / folds.n_splits

pred = sub_preds
submission = pd.DataFrame({
    "test_id": test_id,
    "price": np.expm1(pred),
})
submission.to_csv("./submission.csv", index=False)
submission.head(10)
