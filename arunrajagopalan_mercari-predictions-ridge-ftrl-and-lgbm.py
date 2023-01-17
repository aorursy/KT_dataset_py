import pandas as pd
import numpy as np
import scipy

from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
#from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer

import gc
data = {
    'train': pd.read_csv("../input/train.tsv", sep='\t'),
    'test': pd.read_csv("../input/test.tsv", sep='\t'),
}
data['train'].info()
data['test'].info()
from sklearn.model_selection import train_test_split

y_train = np.log1p(data['train']['price'])
X_train, X_valid, y_train, y_valid = train_test_split(data['train'], y_train, test_size=0.2, random_state=42)
n_train = X_train.shape[0]
n_valid = X_valid.shape[0]
n_test = data['test'].shape[0]

full_data = pd.concat([X_train, X_valid, data['test']], axis=0)

del data['train']
#del data['test']
gc.collect()
def split_cat(text):
    try: 
        return text.split("/")
    except: 
        return ("Unknown", "Unknown", "Unknown")
full_data['general_cat'], full_data['subcat_1'], full_data['subcat_2'] = zip(*full_data['category_name'].apply(lambda x: split_cat(x)))
full_data["brand_name"] = full_data["brand_name"].fillna("unknown")
full_data["item_description"] = full_data["item_description"].fillna("No description yet")
print("There are %d General categories." % full_data['general_cat'].nunique())
print("There are %d cat1 categories." % full_data['subcat_1'].nunique())
print("There are %d cat2 categories." % full_data['subcat_2'].nunique())
full_data['general_cat'] = full_data['general_cat'].astype('category')
full_data['subcat_1'] = full_data['subcat_1'].astype('category')
full_data['subcat_2'] = full_data['subcat_2'].astype('category')
full_data['brand_name'] = full_data['brand_name'].astype('category')
full_data['item_condition_id'] = full_data['item_condition_id'].astype('category')
from nltk.corpus import stopwords
import re

# Define helpers for text normalization
stopwords = {x: 1 for x in stopwords.words('english')}
non_alphanums = re.compile(u'[^A-Za-z0-9]+')

def norm_text(text):
    return u" ".join(
        [x for x in [y for y in non_alphanums.sub(' ', text).lower().strip().split(" ")] if len(x) > 1 and x not in stopwords])
import wordbatch
from wordbatch.models import FTRL
from wordbatch.extractors import WordBag

wb = wordbatch.WordBatch(norm_text, extractor=(WordBag, {"hash_ngrams": 2, "hash_ngrams_weights": [1.5, 1.0],
                                                                  "hash_size": 2 ** 29, "norm": None, "tf": 'binary',
                                                                  "idf": None,
                                                                  }), procs=8)
wb.dictionary_freeze= True
X_name = wb.fit_transform(full_data['name'])
del(wb)
X_name = X_name[:, np.array(np.clip(X_name.getnnz(axis=0) - 1, 0, 1), dtype=bool)]

wb = wordbatch.WordBatch(norm_text, extractor=(WordBag, {"hash_ngrams": 2, "hash_ngrams_weights": [1.0, 0.5],
                                                                  "hash_size": 2 ** 29, "norm": "l2", "tf": 'binary',
                                                                  "idf": None,
                                                                  }), procs=8)
wb.dictionary_freeze= True
X_description = wb.fit_transform(full_data['item_description'])
del(wb)
X_description = X_description[:, np.array(np.clip(X_name.getnnz(axis=0) - 1, 0, 1), dtype=bool)]
X_description.shape
from scipy.sparse import csr_matrix, hstack

#NAME_MIN_DF = 10
#MAX_FEATURES_ITEM_DESCRIPTION = 3

#cv = CountVectorizer(min_df=NAME_MIN_DF)
#X_name = cv.fit_transform(full_data['name'])
  
#tv = TfidfVectorizer(max_features=MAX_FEATURES_ITEM_DESCRIPTION,
                     #ngram_range=(1, 3),
                     #stop_words='english')
#X_description = tv.fit_transform(full_data['item_description'])

lb = LabelBinarizer(sparse_output=True)
X_brand = lb.fit_transform(full_data['brand_name'])
X_cat = lb.fit_transform(full_data['general_cat'])
X_subcat1 = lb.fit_transform(full_data['subcat_1'])
X_subcat2 = lb.fit_transform(full_data['subcat_2'])

X_dummies = csr_matrix(pd.get_dummies(full_data[['item_condition_id', 'shipping']],
                                      sparse=True).values)

print(print(X_dummies.shape, X_description.shape, X_brand.shape, X_cat.shape,
            X_subcat1.shape, X_subcat2.shape,X_name.shape))

X = hstack((X_dummies, X_name, X_brand, X_cat, X_subcat1, X_subcat2, X_description)).tocsr()
X_train = X[:n_train]
y_train = y_train.reshape(-1, 1)

X_valid = X[n_train:n_train+n_valid]
y_valid = y_valid.reshape(-1, 1)

X_test = X[n_train+n_valid:]
print(X.shape, X_train.shape, X_valid.shape, X_test.shape)
ridge_model = Ridge(solver='auto', fit_intercept=True, alpha=0.4,
    max_iter=200, normalize=False, tol=0.01, random_state = 42)
ridge_model.fit(X_train, y_train)
'''def rmsle(real, predicted):
    sum=0.0
    for x in range(len(predicted)):
        if predicted[x]<0 or real[x]<0: #check for negative values
            continue
        p = np.log(predicted[x]+1)
        r = np.log(real[x]+1)
        sum = sum + (p - r)**2
    return (sum/len(predicted))**0.5'''
def rmsle(Y, Y_pred):
    assert Y.shape == Y_pred.shape
    return np.sqrt(np.mean(np.square(Y_pred - Y )))
from sklearn.metrics import mean_absolute_error

y_valid_pred = ridge_model.predict(X_valid)
#y_valid_pred = y_valid_pred.reshape(-1, 1)
#y_valid = y_valid.reshape(-1, 1)
print("RMSL error on valid set:", rmsle(y_valid, y_valid_pred))
print("MAE on valid set:", mean_absolute_error(y_valid, y_valid_pred))
ridge_preds = ridge_model.predict(X_test)
ftrl_model = FTRL(alpha=0.01, beta=0.1, L1=0.00001, L2=1.0, D=X_train.shape[1], 
             iters=60, inv_link="identity", threads=1)

ftrl_model.fit(X_train, y_train.reshape(-1))
y_valid_pred = ftrl_model.predict(X_valid)
y_valid_pred = y_valid_pred.reshape(-1, 1)
#y_valid = y_valid.reshape(-1, 1)
print("RMSL error on valid set:", rmsle(y_valid, y_valid_pred))
print("MAE on valid set:", mean_absolute_error(y_valid, y_valid_pred))
ftrl_preds = ftrl_model.predict(X_test)
from wordbatch.models import FM_FTRL

fm_ftrl_model = FM_FTRL(alpha=0.01, beta=0.1, L1=0.00001, L2=0.1, D=X_train.shape[1], alpha_fm=0.01, 
        L2_fm=0.0, init_fm=0.01, D_fm=200, e_noise=0.0001, iters=18, inv_link="identity", threads=4)

fm_ftrl_model.fit(X_train, y_train.reshape(-1))
y_valid_pred = fm_ftrl_model.predict(X_valid)
y_valid_pred = y_valid_pred.reshape(-1, 1)
#y_valid = y_valid.reshape(-1, 1)
print("RMSL error on valid set:", rmsle(y_valid, y_valid_pred))
print("MAE on valid set:", mean_absolute_error(y_valid, y_valid_pred))
ft_ftrl_preds = fm_ftrl_model.predict(X_test)
import lightgbm as lgb

lgb_label = y_train.ravel()
lgb_y_valid = y_valid.ravel()
lgb_train = lgb.Dataset(X_train, label=lgb_label)
lgb_eval = lgb.Dataset(X_valid, lgb_y_valid, reference=lgb_train)

params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': {'l2', 'rmse'},
    'learning_rate': 0.6,
    'feature_fraction': 0.65,
    'bagging_fraction': 0.6,
    'bagging_freq': 5,
    'verbose': 1,
    'data_random_seed': 42,
    'max_depth': 3,
    'nthread': 4,
    'min_data_in_leaf': 100,
    'max_bin': 31
}

print('Start training...')
# train
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=5500,
                valid_sets=lgb_eval,
                early_stopping_rounds=1000)
from sklearn.metrics import mean_squared_error

'''print('Start predicting...')
# predict
y_pred = gbm.predict(X_valid, num_iteration=gbm.best_iteration)
# eval
#print('The rmse of prediction is:', mean_absolute_error(y_valid, y_pred) ** 0.5)
y_pred = y_pred.reshape(-1, 1)
print("RMSL error on valid set:", rmsle(y_valid, y_pred))
print("MAE on valid set:", mean_absolute_error(y_valid, y_pred))'''
'''from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dropout, Dense, concatenate, GRU, Embedding, Flatten, Activation
# from keras.layers import Bidirectional
from keras.optimizers import Adam
from keras.models import Model
from keras import backend as K
from nltk.corpus import stopwords

print("Transforming text data to sequences...")
raw_text = np.hstack([full_data.item_description.str.lower(), full_data.name.str.lower()])

print(" Fitting tokenizer...")
tok_raw = Tokenizer()
tok_raw.fit_on_texts(raw_text)

print(" Transforming text to sequences...")
full_data['seq_item_description'] = tok_raw.texts_to_sequences(full_data.item_description.str.lower())
full_data['seq_name'] = tok_raw.texts_to_sequences(full_data.name.str.lower())

del tok_raw'''
'''MAX_NAME_SEQ = 10 
MAX_ITEM_DESC_SEQ = 75 

def get_rnn_data(dataset):
    X = {
        'name': pad_sequences(dataset.seq_name, maxlen=MAX_NAME_SEQ),
        'item_desc': pad_sequences(dataset.seq_item_description, maxlen=MAX_ITEM_DESC_SEQ),
        'brand_name': np.array(dataset.brand_name),
        #'category': np.array(dataset.category),
        'item_condition': np.array(dataset.item_condition_id),
        'num_vars': np.array(dataset[["shipping"]]),
        'general_cat': np.array(dataset.general_cat),
        'subcat_1': np.array(dataset.subcat_1),
        'subcat_2': np.array(dataset.subcat_2),
    }
    return X'''
'''train = full_data[:n_train]
valid = full_data[n_train:n_train+n_valid]
test = full_data[n_train+n_valid:]

X_train = get_rnn_data(train)
Y_train = y_train.reshape(-1, 1)

X_valid = get_rnn_data(valid)
Y_valid = y_valid.reshape(-1, 1)

X_test = get_rnn_data(test)'''
#np.random.seed(42)
#ridge_preds = ridge_model.predict(X_test)
lgbm_preds = gbm.predict(X_test, num_iteration=gbm.best_iteration)
lgbm_preds = lgbm_preds.reshape(-1, 1)
ftrl_preds = ftrl_preds.reshape(-1, 1)
ft_ftrl_preds = ft_ftrl_preds.reshape(-1, 1)

preds = ridge_preds*0.05 + lgbm_preds*0.06 + ftrl_preds*0.25 + ft_ftrl_preds*0.64
#preds = ridge_preds*0.1 + ftrl_preds*0.2 + ft_ftrl_preds*0.7

data['test']["price"] = np.expm1(preds)
data['test'][["test_id", "price"]].to_csv("submission_ridge_FT_FTRL_LGBM.csv", index = False)
y_valid.shape
