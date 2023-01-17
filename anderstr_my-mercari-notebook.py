import numpy as np 
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn import metrics
from sklearn import preprocessing
import sklearn_pandas
import random
#from nltk.corpus import stopwords
import wordbatch
from wordbatch.extractors import WordBag
from wordbatch.models import FTRL, FM_FTRL
import re
import gc
import time
develop = False
develop_size = 0

NUM_BRANDS = 4500
NUM_CATEGORIES = 1200
MIN_PRICE = 3
MAX_PRICE = 2000
PREDICT_BATCH_SIZE = 350000
start_time = time.time()
def printtime(text):
    print("[{:.3f}] {}".format(time.time()-start_time, text))
train = pd.read_csv('../input/train.tsv', sep='\t')
if develop_size > 0:
    train = train.sample(develop_size, random_state=42)
printtime("train loaded")
def ngrams(text, ngram_range):
    words = text.split()
    for wordcount in range(ngram_range[0], ngram_range[1]+1):
        for i in range(0, len(words)-wordcount+1):
            yield " ".join(words[i:i+wordcount])
            
def match_brand_name(row, top_brands):
    if row['brand_name'] != "unknown":
        return row['brand_name']
    if row['name'] in top_brands:
        return row['name']
    # Substrings of wordcount 1, 2, 3
    for subname in reversed(list(ngrams(row['name'], (1, 3)))):
        if subname in top_brands:
            return subname
    return row['brand_name']

def fill_brand(data, top_brands):
    data['brand_name'] = data[['brand_name', 'name']].apply(lambda x : match_brand_name(x, top_brands), axis=1)
    
def split_cat(text):
    try:
        return text.split("/")
    except:
        return ("unknown", "unknown", "unknown")
    
#stop = set(stopwords.words('english'))
non_alphanums = re.compile(u'[^A-Za-z0-9]+')
def normalize_text(text):
    return u" ".join(
        [x for x in [y for y in non_alphanums.sub(' ', text).lower().strip().split(" ")] \
         #if len(x) > 1 and x not in stop])
         if len(x) > 1])
def calc_stats(data):
    def topn_values(col, n):
        return set(col.value_counts().loc[lambda x: x.index != 'unknown'].index[:n].values)
    return {
        'top_brands': topn_values(data['brand_name'], NUM_BRANDS),
        'top_cat1': topn_values(data['cat1'], NUM_CATEGORIES),
        'top_cat2': topn_values(data['cat2'], NUM_CATEGORIES),
        'top_cat3': topn_values(data['cat3'], NUM_CATEGORIES),
    }
stats = None
def preprocess(data, train=False):
    global stats
    
    # Split category
    data['cat1'], data['cat2'], data['cat3'] = zip(*data['category_name'].apply(lambda x: split_cat(x)))
    
    # Fill missing values
    data['name'].fillna("unknown", inplace=True)
    data['item_description'] = data['item_description']\
        .fillna("unknown")\
        .apply(lambda x: x.replace('[rm]', ''))\
        .apply(lambda x: x.replace('No description yet', 'unknown'))
    data['has_brand'] = data['brand_name'].apply(lambda x: 'no_brand' if pd.isnull(x) else 'has_brand')
    data['brand_name'].fillna("unknown", inplace=True)
    data['item_condition_id'] = data['item_condition_id'].astype(str).fillna("unknown")
    data['shipping'] = data['item_condition_id'].astype(str).fillna("unknown")
    
    # Category condition combinations
    data['cat1_cond'] = data['cat1'] + "_" + data['item_condition_id']
    data['cat2_cond'] = data['cat2'] + "_" + data['item_condition_id']
    data['cat3_cond'] = data['cat3'] + "_" + data['item_condition_id']
    
    # 2-letter units and karats
    unit_regex = r'(\d+)[\s-]([a-z]{2})(\s)'
    unit_repl = r'\1\2\3'
    karats_regex = r'(\d)([\s-]?)(karat|karats|carat|carats|kt)([^\w])'
    karats_repl = r'\1k\4'
    data['name'] = data['name'].str.replace(unit_regex, unit_repl)
    data['item_description'] = data['item_description'].str.replace(unit_regex, unit_repl)
    data['name'] = data['name'].str.replace(karats_regex, karats_repl)
    data['item_description'] = data['item_description'].str.replace(karats_regex, karats_repl)
    
    # Calc top value stats in training phase
    if train:
        stats = calc_stats(data)
    
    # Detect brands
    fill_brand(data, stats['top_brands'])
    
    # Cut
    data.loc[~data['brand_name'].isin(stats['top_brands']), 'brand_name'] = 'unknown'
    data.loc[~data['cat1'].isin(stats['top_cat1']), 'cat1'] = 'unknown'
    data.loc[~data['cat2'].isin(stats['top_cat2']), 'cat2'] = 'unknown'
    data.loc[~data['cat3'].isin(stats['top_cat3']), 'cat3'] = 'unknown'
    
    # Text lengths
    data['description_length'] = preprocessing.scale(data['item_description'].apply(lambda x: len(x.split())).astype(float))
    data['name_length'] = preprocessing.scale(data['name'].apply(lambda x: len(x.split())).astype(float))
    
    # Categorical values
    data['cat1_cat'] = data['cat1'].astype('category')
    data['cat2_cat'] = data['cat2'].astype('category')
    data['cat3_cat'] = data['cat3'].astype('category')
    data['item_condition_id_cat'] = data['item_condition_id'].astype('category')
train.drop(train[train['price'] < MIN_PRICE].index, inplace=True)
printtime("dropped erroneous prices < 3")

X_train = train
y_train = np.log1p(train['price'])
X_test = []
y_test = []
if develop:
    X_train, X_test, y_train, y_test = train_test_split(train, y_train, random_state=42, test_size=0.1)
    X_train.is_copy = False
    X_test.is_copy = False
%%time
preprocess(X_train, train=True)
printtime("preprocess training data done")
cat1_cv = CountVectorizer(token_pattern='.+')
cat2_cv = CountVectorizer(token_pattern='.+')
cat3_cv = CountVectorizer(token_pattern='.+')
brand_cv = CountVectorizer(token_pattern='.+')
condition_cv = CountVectorizer(token_pattern='.+')
shipping_cv = CountVectorizer(token_pattern='.+')
has_brand_cv = CountVectorizer(token_pattern='.+')
cat1_cond_cv = CountVectorizer(token_pattern='.+')
cat2_cond_cv = CountVectorizer(token_pattern='.+')
cat3_cond_cv = CountVectorizer(token_pattern='.+')
mapper = sklearn_pandas.DataFrameMapper([
    ('name', TfidfVectorizer(ngram_range=(1, 2))),
    ('cat1', cat1_cv),
    ('cat2', cat2_cv),
    ('cat3', cat3_cv),
    ('cat1_cond', cat1_cond_cv),
    ('cat2_cond', cat2_cond_cv),
    ('cat3_cond', cat3_cond_cv),
    ('brand_name', brand_cv),
    ('item_description', TfidfVectorizer(ngram_range=(1, 2), max_features=100000)),
    ('item_condition_id', condition_cv),
    ('shipping', shipping_cv),
    ('description_length', None),
    ('name_length', None),
    ('has_brand', has_brand_cv)
], sparse=True)
model_ridge = Pipeline([
    ('map', mapper),
    ('reg', Ridge(alpha=3.0))
])
%%time
model_ridge.fit(X_train, y_train)
len(mapper.transformed_names_)
%%time
wb_name = wordbatch.WordBatch(normalize_text, extractor=(WordBag, {
    "hash_ngrams": 2, 
    "hash_ngrams_weights": [1.5, 1.0],
    "hash_size": 2 ** 29, 
    "norm": None, 
    "tf": 'binary', 
    "idf": None,
    }), procs=8)
wb_name.dictionary_freeze = True
wb_name.fit(X_train['name'])
%%time
wb_desc = wordbatch.WordBatch(normalize_text, extractor=(WordBag, {
    "hash_ngrams": 2, 
    "hash_ngrams_weights": [1.0, 1.0],
    "hash_size": 2 ** 28,
    "norm": "l2",
    "tf": 1.0,
    "idf": None
    }), procs=8)
wb_desc.dictionary_freeze = True
wb_desc.fit(X_train['item_description'])
mask_name = None
mask_desc = None
mask_total = None
def transform_fm(data, train=False):
    global mask_name
    global mask_desc
    global mask_total
    X_cat1 = cat1_cv.transform(data['cat1'])
    X_cat2 = cat2_cv.transform(data['cat2'])
    X_cat3 = cat3_cv.transform(data['cat3'])
    X_cat1_cond = cat1_cond_cv.transform(data['cat1_cond'])
    X_cat2_cond = cat2_cond_cv.transform(data['cat2_cond'])
    X_cat3_cond = cat3_cond_cv.transform(data['cat3_cond'])
    X_brand = brand_cv.transform(data['brand_name'])
    X_condition = shipping_cv.transform(data['shipping'])
    X_shipping = condition_cv.transform(data['item_condition_id'])
    X_name_length = csr_matrix(data['name_length']).transpose()
    X_desc_length = csr_matrix(data['description_length']).transpose()
    X_name = wb_name.transform(data['name'])
    print(X_name.shape)
    if train:
        mask_name = np.array(np.clip(X_name.getnnz(axis=0) - 1, 0, 1), dtype=bool)
    X_name = X_name[:, mask_name]
    print(X_name.shape)
    X_desc = wb_desc.transform(data['item_description'])
    print(X_desc.shape)
    if train:
        mask_desc = np.array(np.clip(X_desc.getnnz(axis=0) - 1, 0, 1), dtype=bool)
    X_desc = X_desc[:, mask_desc]
    print(X_desc.shape)
    X_has_brand = has_brand_cv.transform(data['has_brand'])
    X = hstack((X_name, X_desc, X_cat1, X_cat2, X_cat3, X_cat1_cond, X_cat2_cond, X_cat3_cond, X_brand, X_condition, X_shipping, X_name_length, X_desc_length, X_has_brand)).tocsr()
    print(X.shape)
    if train:
        mask_total = np.array(np.clip(X.getnnz(axis=0) - 1, 0, 1), dtype=bool)
    X = X[:, mask_total]
    print(X.shape)
    return X
%%time
X_train_fm = transform_fm(X_train, train=True)
%%time
model_fm = FM_FTRL(alpha=0.01, beta=0.01, L1=0.00001, L2=0.1, D=X_train_fm.shape[1], alpha_fm=0.01, L2_fm=0.0, init_fm=0.01,
                    D_fm=200, e_noise=0.0001, iters=15, inv_link="identity", threads=4)
model_fm.fit(X_train_fm, y_train)
def predict(X_test, X_test_fm):
    y_ridge = model_ridge.predict(X_test)
    y_fm = model_fm.predict(X_test_fm)
    
    w = 0.20
    y = w*y_ridge+(1-w)*y_fm
    y = np.clip(np.expm1(y), MIN_PRICE, MAX_PRICE)
    return y
%%time
if not develop:
    printtime("Processing test data")
    chunk = 1
    for X_test in pd.read_csv('../input/test.tsv', sep='\t', chunksize=PREDICT_BATCH_SIZE):
        preprocess(X_test)
        X_test_fm = transform_fm(X_test)
        y = predict(X_test, X_test_fm)

        results = pd.DataFrame({'test_id': X_test['test_id'], 'price': y}, columns=['test_id', 'price'])
        
        # write first chunk to new file, then append remaining chunks
        mode = 'w' if chunk == 1 else 'a'
        write_header = chunk == 1
        results.to_csv('submission_rige_ftrl.csv', index=False, header=write_header, mode=mode)
        
        printtime("Processed test data batch " + str(chunk))
        chunk += 1
        
else: # develop
    preprocess(X_test)
    printtime("preprocess test data done")

    X_test_fm = transform_fm(X_test)
    printtime("prepare FM data done")
    
    y = predict(X_test, X_test_fm)
    printtime("predictions done")
if develop:
    y_test_exp = np.expm1(y_test.values)
    #print("RMSLE Ridge   ", metrics.mean_squared_log_error(y_test_exp, np.expm1(y_ridge)) ** 0.5)
    #print("RMSLE FM      ", metrics.mean_squared_log_error(y_test_exp, np.expm1(y_fm)) ** 0.5)
    #print("RMSLE FTRL    ", metrics.mean_squared_log_error(y_test_exp, np.expm1(y_ftrl)) ** 0.5)
    #print("RMSLE", metrics.mean_squared_log_error(y_test_exp, np.expm1(y_ridge2)) ** 0.5)
    print("RMSLE Ensemble", metrics.mean_squared_log_error(y_test_exp, y) ** 0.5)
    #print("MAE", metrics.mean_absolute_error(y_test_exp, y))
    #mask50 = y_test_exp < 50
    #print("MAE<50", metrics.mean_absolute_error(y_test_exp[mask50], y[mask50]))
printtime("kernel done")