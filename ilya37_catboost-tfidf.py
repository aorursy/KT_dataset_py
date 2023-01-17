import numpy as np

import pandas as pd

from subprocess import check_output

from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer

from catboost import Pool, CatBoostRegressor

from sklearn import preprocessing

import gc

from scipy.sparse import hstack, csr_matrix
#train = pd.read_csv('../input/mercari-price-suggestion-challenge/train.tsv', nrows=10, sep='\t')

#test = pd.read_csv('../input/mercari-price-suggestion-challenge/test_stg2.tsv',nrows=10,  sep='\t')

train = pd.read_csv('../input/mercari-price-suggestion-challenge/train.tsv', sep='\t')

test = pd.read_csv('../input/mercari-price-suggestion-challenge/test_stg2.tsv',  sep='\t')



print("finish loading")
def split_cat(text):

    try: return text.split("/")

    except: return ("No Label", "No Label", "No Label")



train['general_cat'], train['subcat_1'], train['subcat_2'] = zip(*train['category_name'].apply(lambda x: split_cat(x)))

test['general_cat'], test['subcat_1'], test['subcat_2'] = zip(*test['category_name'].apply(lambda x: split_cat(x)))



train = train.drop("category_name", axis=1)

test = test.drop("category_name", axis=1)



train['brand_name'] = train['brand_name'].fillna('nobrand')

test['brand_name'] = test['brand_name'].fillna('nobrand')



train['brand_name'] = train['brand_name'].fillna('nobrand')

test['brand_name'] = test['brand_name'].fillna('nobrand')



train['name'] = train['name'].fillna('name')

test['name'] = test['name'].fillna('name')



train['item_description'] = train['item_description'].fillna('descr')

test['item_description'] = test['item_description'].fillna('descr')
corpus_name = np.hstack([train.name,test.name ])

vectorizer_name = CountVectorizer(stop_words='english')

vectorizer_name.fit(corpus_name)



train_name = vectorizer_name.transform(train.name.values)

test_name = vectorizer_name.transform(test.name.values)



del vectorizer_name

gc.collect()



transformer = TfidfTransformer()

tr_name = transformer.fit_transform(train_name)

te_name = transformer.fit_transform(test_name)



del train_name, test_name

gc.collect()
corpus_desc = np.hstack([train.item_description, test.item_description])

vectorizer_desc = CountVectorizer(stop_words='english')

vectorizer_desc.fit(corpus_desc)



train_desc = vectorizer_desc.transform(train.item_description.values)

test_desc = vectorizer_desc.transform(test.item_description.values)



del vectorizer_desc

gc.collect()



tr_desc = transformer.fit_transform(train_desc)

te_desc = transformer.fit_transform(test_desc)



del train_desc, test_desc

gc.collect()
train['cor_name_Tfidf'] = np.mean(tr_name,1)

train['cor_desc_Tfidf'] = np.mean(tr_desc,1)



test['cor_name_Tfidf'] = np.mean(te_name,1)

test['cor_desc_Tfidf'] = np.mean(te_desc,1)



del tr_name, tr_desc, te_name, te_desc

gc.collect()
shape = train.shape[0]



names = pd.concat([train.name, test.name])

names = names.astype('category').cat.codes

train['names_cat'] = names[:shape]

test['names_cat'] = names[shape:]



descs = pd.concat([train.item_description, test.item_description])

descs = descs.astype('category').cat.codes

train['descs_cat'] = descs[:shape]

test['descs_cat'] = descs[shape:]
del shape, names, descs

gc.collect()
x_train = train.drop(['name','item_description','train_id','price'],1)

x_test = test.drop(['name','item_description','test_id',],1)

y_train = np.abs(train.price.values)



x_train = x_train.fillna(0)

x_test = x_test.fillna(0)
train_data = Pool(x_train, y_train, cat_features=[0,1,3,4,5,8,9])

test_data = Pool(x_test, cat_features=[0,1,3,4,5,8,9])

idx = test.test_id.values

del x_train, x_test, y_train, train, test

gc.collect()
params = {'depth': 11, 'iterations': 500, 'l2_leaf_reg': 9, 

        'learning_rate': 0.98, 'random_seed': 1111,

        'loss_function': 'MAE'}



#params = {'depth': 2, 'iterations': 2, 'l2_leaf_reg': 9, 

#        'learning_rate': 0.98, 'random_seed': 1111,

#        'loss_function': 'MAE'}



model = CatBoostRegressor(**params)

model.fit(train_data)
print('prediction')

y_pred = model.predict(test_data)



sub = pd.DataFrame()

sub['test_id'] = idx

sub['price'] = np.abs(y_pred)

sub.to_csv('submission.csv', index=False, float_format='%.3f')

print('finish prediction')
