import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.preprocessing import OneHotEncoder

from collections import Counter

from sklearn.decomposition import TruncatedSVD

from sklearn.model_selection import train_test_split

import re

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import cross_validate

from sklearn.metrics import mean_squared_log_error

from sklearn.metrics import make_scorer

from sklearn.model_selection import KFold





%matplotlib inline
train = pd.read_csv('../input/train.tsv',delimiter='\t',index_col=0)
train.head()
train = train.drop('item_description',axis=1)

train['item_condition_id'] = train['item_condition_id'].astype(np.int8)

train['shipping'] = train['shipping'].astype(np.int8)

train.info()
len(train)
len(train.category_name.unique())
len(train.brand_name.unique())
plt.hist(train.price,bins=50,range=(0,2000),color='b',alpha=0.6)

plt.xlabel('Price [$]')

plt.ylabel('Number')

#plt.yscale('log')

plt.show()
plt.hist(np.log10(train[train['price']>0].price),bins=25,range=(0.5,3.1),color='b',alpha=0.6)

plt.xlabel(r'log_{10}(Price) [$]')

plt.ylabel('Number')

#plt.yscale('log')

plt.show()
plt.hist(train[train['shipping']==0].price,bins=100,range=(0,100),

         normed=True,label='Shipping=0',color='b',alpha=0.6)

plt.hist(train[train['shipping']==1].price,bins=100,range=(0,100),

         normed=True,label='Shipping=1',color='r',alpha=0.6)



plt.xlabel('Price [$]')

plt.ylabel('Fraction')

#plt.yscale('log')

plt.legend(loc='upper right')

plt.show()
print(train['item_condition_id'].value_counts())

train.groupby('item_condition_id')['price'].describe()
train['category_name'] = train['category_name'].fillna('')



counter = Counter()

cats = [str(x).split('/') for x in train.category_name]

for c in cats:

    counter.update(c)



print('Number of categories: {}'.format(len(counter)))

print(counter.most_common())
train['brand_name'] = train['brand_name'].fillna('')



counter = Counter()

cats = [str(x).split('/') for x in train.brand_name]

for c in cats:

    counter.update(c)



print('Number of brands: {}'.format(len(counter)))

print(counter.most_common())
train.groupby('brand_name').price.mean().sort_values()
train.groupby('category_name').price.mean().sort_values()
def rmsle(ytrue,y):

    return np.sqrt(mean_squared_log_error(ytrue,y))
kf = KFold(n_splits=5,random_state=123)

i = 0

def get_val(series,x):

    try:

        return series[x]

    except:

        pass

    return series['']

    

for train_idx,val_idx in kf.split(train):

    print('Fold {}'.format(i))

    i+=1

    cols = [1,2,3,4,5]

    X_train = train.iloc[train_idx,cols]

    X_val = train.iloc[val_idx,cols]



    brand_price = X_train.groupby('brand_name').price.mean()

    cat_price = X_train.groupby('category_name').price.mean()

    

    X_train['brand_price'] = [brand_price[x] for x in X_train.brand_name ]

    X_train['category_price'] = [cat_price[x] for x in X_train.category_name ]



    X_val['brand_price'] = [get_val(brand_price,x) for x in X_val.brand_name ]

    X_val['category_price'] = [get_val(cat_price,x) for x in X_val.category_name ]

    

    print('RMSLE, train, brand price: {:0.4}'

          .format(rmsle(X_train.price,X_train.brand_price)))

    print('RMSLE, train, category price: {:0.4}'

          .format(rmsle(X_train.price,X_train.category_price)))

    print('RMSLE, test, brand price: {:0.4}'

          .format(rmsle(X_val.price,X_val.brand_price)))

    print('RMSLE, test, category price: {:0.4}'

          .format(rmsle(X_val.price,X_val.category_price)))
kf = KFold(n_splits=5,random_state=123)

i = 0

def get_val(series,x):

    try:

        return series[x]

    except:

        pass

    return series['Cat: Brand:']



train['CatBrand'] = 'Cat:'+train.category_name+' Brand:'+train.brand_name



for train_idx,val_idx in kf.split(train):

    print('Fold {}'.format(i))

    i+=1

    cols = [1,2,3,4,5]

    X_train = train.iloc[train_idx,:]

    X_val = train.iloc[val_idx,:]



    cb_price = X_train.groupby('CatBrand').price.mean()

    

    cb_price_train = np.array([cb_price[x] for x in X_train.CatBrand ])

    cb_price_val = np.array([get_val(cb_price,x) for x in X_val.CatBrand])



    X_val['cb_price'] = [get_val(cb_price,x) for x in X_val.CatBrand ]

    

    print('RMSLE, train, cat/brand price: {:0.4}'

          .format(rmsle(X_train.price,cb_price_train)))

    print('RMSLE, test, cat/brand price: {:0.4}'

          .format(rmsle(X_val.price,cb_price_val)))

from sklearn.model_selection import KFold



kf = KFold(n_splits=5,random_state=123)

i = 0

def get_val(series,x,default):

    try:

        return series[x]

    except:

        pass

    return default



train['All4'] = ['Cat:'+c+' Brand:'+b+ \

                ' condition:'+str(i) + ' shipping:'+str(s)

                for c,b,i,s in zip(train.category_name,train.brand_name,

                                   train.item_condition_id,train.shipping)]



for train_idx,val_idx in kf.split(train):

    print('Fold {}'.format(i))

    i+=1

    cols = [1,2,3,4,5]

    X_train = train.iloc[train_idx,:]

    X_val = train.iloc[val_idx,:]



    cb_price = X_train.groupby('All4').price.mean()



    mean = train.price.mean()

    

    cb_price_train = np.array([cb_price[x] for x in X_train.All4 ])

    cb_price_val = np.array([get_val(cb_price,x,mean) for x in X_val.All4])

    

    print('RMSLE, train, all feature price: {:0.4}'

          .format(rmsle(X_train.price,cb_price_train)))

    print('RMSLE, test, all feature price: {:0.4}'

          .format(rmsle(X_val.price,cb_price_val)))

train = pd.read_csv('../input/train.tsv',delimiter='\t',index_col=0)

train = train.drop('item_description',axis=1)

cvec = CountVectorizer(min_df=25,stop_words='english')



X_tr,X_te = train_test_split(train,test_size=0.3,random_state=234)



names_tr = X_tr.name

names_tr = [n.lower() for n in names_tr]

names_tr = [ re.sub(r'[^A-Za-z]',' ',n) for n in names_tr]



names_te = X_te.name

names_te = [n.lower() for n in names_te]

names_te = [ re.sub(r'[^A-Za-z]',' ',n) for n in names_te]

cvec.fit(names_tr)



X_tr_names = cvec.transform(names_tr)

X_te_names = cvec.transform(names_te)

    

print(len(cvec.vocabulary_))

svd = TruncatedSVD(n_components=50,n_iter=10)

svd.fit(X_tr_names)

X_tr_svd = svd.transform(X_tr_names)

X_te_svd = svd.transform(X_te_names)
y_tr = X_tr['price']

X_tr = X_tr.loc[:,['item_condition_id','category_name','brand_name','shipping','price']]

y_te = X_te['price']

X_te = X_te.loc[:,['item_condition_id','category_name','brand_name','shipping','price']]

cat_counts = X_tr.groupby('category_name')['price'].count()

to_keep = []

for i in range(len(cat_counts)):

    if (cat_counts.iloc[i]>10):

        to_keep.append(cat_counts.index.values[i])

def filter_vals(x,alist):

    if x in alist:

        return x

    return ''

X_tr.loc[:,'category_name'] = [filter_vals(x,to_keep) for x in X_tr['category_name']]  

X_te.loc[:,'category_name'] = [filter_vals(x,to_keep) for x in X_te['category_name']]    
brand_counts = X_tr.groupby('brand_name')['price'].count()

to_keep = []

for i in range(len(brand_counts)):

    if (brand_counts.iloc[i]>10):

        to_keep.append(brand_counts.index.values[i])



X_tr.loc[:,'brand_name'] = [filter_vals(x,to_keep) for x in X_tr['brand_name']]

X_te.loc[:,'brand_name'] = [filter_vals(x,to_keep) for x in X_te['brand_name']]  
brands_sorted = X_tr.groupby('brand_name')['price'].mean().sort_values()

cat_sorted = X_tr.groupby('category_name')['price'].mean().sort_values()



brand_dict = {}

cat_dict = {}

for i in range(len(brands_sorted)):

    brand_dict[brands_sorted.index.values[i]] = i

    

for i in range(len(cat_sorted)):

    cat_dict[cat_sorted.index.values[i]] = i

    

X_tr['brand'] = X_tr['brand_name'].map(brand_dict)

X_tr['category'] = X_tr['category_name'].map(cat_dict)

X_te['brand'] = X_te['brand_name'].map(brand_dict)

X_te['category'] = X_te['category_name'].map(cat_dict)
X_tr.head()
X_tr = X_tr.loc[:,['item_condition_id','shipping','brand','category']]

X_te = X_te.loc[:,['item_condition_id','shipping','brand','category']]

X_tr_fin = np.concatenate((X_tr,X_tr_svd),axis=1)

X_te_fin = np.concatenate((X_te,X_te_svd),axis=1)
from sklearn.ensemble import RandomForestRegressor



rfc = RandomForestRegressor(n_estimators=50,min_samples_leaf=10,max_depth=10)

n_reduced = int(2./3*len(X_tr_fin))

X_tr_fin = X_tr_fin[:n_reduced,:]

y_tr = y_tr.iloc[:n_reduced]

rfc.fit(X_tr_fin,y_tr)

def rmsle(ytrue,y):

    return np.sqrt(mean_squared_log_error(ytrue,y))



y_tr_pred = rfc.predict(X_tr_fin)

score_tr = rmsle(y_tr,y_tr_pred)



y_te_pred = rfc.predict(X_te_fin)

score_te = rmsle(y_te,y_te_pred)

print(score_tr)

print(score_te)
rfc.feature_importances_