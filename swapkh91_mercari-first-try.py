# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.feature_extraction.text import TfidfVectorizer

import string

import matplotlib.pyplot as plt

import xgboost as xgb

import time

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.linear_model import Ridge, LogisticRegression



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

# print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('../input/train.tsv', sep='\t')

train_df.head(4)
test_df = pd.read_csv('../input/test.tsv', sep='\t')

test_df.head(4)
# train_df = train_df.set_index("train_id")

# test_df = test_df.set_index("test_id")
all_data_na = (train_df.isnull().sum() / len(train_df)) * 100

all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]

missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})

missing_data
def if_brand(row):

    if row == row:

        return 1

    else:

        return 0

    

train_df['if_brand'] = train_df.brand_name.apply(lambda row : if_brand(row))

test_df['if_brand'] = test_df.brand_name.apply(lambda row : if_brand(row))

train_df.head()
keys = train_df.brand_name.dropna().unique()

values = list(range(len(keys)))

brand_dict = dict(zip(keys, values))



def brand_label(row):

    try:

        return brand_dict[row]

    except:

        return np.nan



train_df['brand_label'] = train_df.brand_name.apply(lambda row: brand_label(row))

test_df['brand_label'] = test_df.brand_name.apply(lambda row: brand_label(row))

train_df.head()
def if_description(row):

    if row == 'No description yet':

        a = 0

    else:

        a = 1

    return a



train_df['is_description'] = train_df.item_description.apply(lambda row : if_description(row))

test_df['is_description'] = test_df.item_description.apply(lambda row : if_description(row))

train_df.head()
def if_catname(row):

    if row == row:

        return 1

    else:

        return 0

    

train_df['if_cat'] = train_df.category_name.apply(lambda row : if_catname(row))

test_df['if_cat'] = test_df.category_name.apply(lambda row : if_catname(row))

train_df.head()
def cat_split(row):

    try:

        text = row

        txt1, txt2, txt3 = text.split('/')

        return txt1, txt2, txt3

    except:

        return np.nan, np.nan, np.nan





train_df["cat_1"], train_df["cat_2"], train_df["cat_3"] = zip(*train_df.category_name.apply(lambda val: cat_split(val)))

test_df["cat_1"], test_df["cat_2"], test_df["cat_3"] = zip(*test_df.category_name.apply(lambda val: cat_split(val)))

train_df.head()
keys = train_df.cat_1.unique().tolist() + test_df.cat_1.unique().tolist()

keys = list(set(keys))

values = list(range(len(keys)))

cat1_dict = dict(zip(keys, values))



keys2 = train_df.cat_2.unique().tolist() + test_df.cat_2.unique().tolist()

keys2 = list(set(keys2))

values2 = list(range(len(keys2)))

cat2_dict = dict(zip(keys2, values2))



keys3 = train_df.cat_3.unique().tolist() + test_df.cat_3.unique().tolist()

keys3 = list(set(keys3))

values3 = list(range(len(keys3)))

cat3_dict = dict(zip(keys3, values3))
def cat_lab(row,cat1_dict = cat1_dict, cat2_dict = cat2_dict, cat3_dict = cat3_dict):

    txt1 = row['cat_1']

    txt2 = row['cat_2']

    txt3 = row['cat_3']

    return cat1_dict[txt1], cat2_dict[txt2], cat3_dict[txt3]



train_df["cat_1_label"], train_df["cat_2_label"], train_df["cat_3_label"] = zip(*train_df.apply(lambda val: cat_lab(val), axis =1))

test_df["cat_1_label"], test_df["cat_2_label"], test_df["cat_3_label"] = zip(*test_df.apply(lambda val: cat_lab(val), axis =1))

train_df.head(4)
def compute_tfidf(description):

    description = str(description)

    description.translate(string.punctuation)



    tfidf_sum=0

    words_count=0

    for w in description.lower().split():

        words_count += 1

        if w in tfidf_dict:

            tfidf_sum += tfidf_dict[w]

    

    if words_count > 0:

        return tfidf_sum/words_count

    else:

        return 0



tfidf = TfidfVectorizer(

    min_df=10, max_features=180000, strip_accents='unicode', lowercase =True,

    analyzer='word', token_pattern=r'\w+', ngram_range=(1, 3), use_idf=True, 

    smooth_idf=True, sublinear_tf=True, stop_words='english')
tfidf.fit_transform(train_df['item_description'].apply(str))

tfidf_dict = dict(zip(tfidf.get_feature_names(), tfidf.idf_))

train_df['tfidf'] = train_df['item_description'].apply(compute_tfidf)
tfidf.fit_transform(test_df['item_description'].apply(str))

tfidf_dict = dict(zip(tfidf.get_feature_names(), tfidf.idf_))

test_df['tfidf'] = test_df['item_description'].apply(compute_tfidf)
print(train_df.isnull().sum())

train_df.fillna(0, inplace=True)

test_df.fillna(0, inplace=True)

print(train_df.isnull().sum())
# plt.figure(figsize=(20, 15))

# plt.scatter(train_df['tfidf'], train_df['price'])

# plt.title('Train price X item_description TF-IDF', fontsize=15)

# plt.xlabel('Price', fontsize=15)

# plt.ylabel('TF-IDF', fontsize=15)

# plt.xticks(fontsize=15)

# plt.yticks(fontsize=15)

# plt.legend(fontsize=15)

# plt.show()
train = train_df.copy()

test = test_df.copy()



do_not_use_for_training = ['cat_1','test_id','cat_2','cat_3','train_id','name', 'category_name', 'brand_name', 'price', 'item_description']

feature_names = [f for f in train.columns if f not in do_not_use_for_training]
y = np.log(train['price'].values + 1)
# Xtr, Xv, ytr, yv = train_test_split(train[feature_names].values, y, test_size=0.2, random_state=1987)

# dtrain = xgb.DMatrix(Xtr, label=ytr)

# dvalid = xgb.DMatrix(Xv, label=yv)

# dtest = xgb.DMatrix(test[feature_names].values)

# watchlist = [(dtrain, 'train'), (dvalid, 'valid')]



# xgb_par = {'min_child_weight': 20, 'eta': 0.05, 'colsample_bytree': 0.5, 'max_depth': 15,

#             'subsample': 0.9, 'lambda': 2.0, 'nthread': -1, 'booster' : 'gbtree', 'silent': 1,

#             'eval_metric': 'rmse', 'objective': 'reg:linear'}



# model_1 = xgb.train(xgb_par, dtrain, 80, watchlist, early_stopping_rounds=20, maximize=False, verbose_eval=20)

# print('Modeling RMSLE %.5f' % model_1.best_score)
model = Ridge(solver="saga", fit_intercept=True, random_state=205)
x_train = train[feature_names].values
model.fit(x_train, y)
pred = model.predict(X=test[feature_names].values)
test["price"] = np.expm1(pred)
test[["test_id", "price"]].to_csv("submission_Ridge.csv", index = False)
# x_train,y_train = train.drop(['price'],axis =1),train.price
# m = RandomForestRegressor(n_jobs=-1,min_samples_leaf=3,n_estimators=200)
# m.fit(x_train, y_train)

# m.score(x_train,y_train)
# preds = m.predict(test[feature_names].values)

# preds = pd.Series(np.exp(preds))
# submit = pd.concat([test.test_id,preds],axis=1)
# submit.columns = ['test_id','price']

# submit.to_csv('submission_mercari_forest.csv', index=False)
# yvalid = model_1.predict(dvalid)

# ytest = model_1.predict(dtest)
# test['price'] = np.exp(ytest) - 1

# test[['test_id', 'price']].to_csv('submission_mercari_2.csv', index=False)