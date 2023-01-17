#Flag to set whether all training set should be used to predict prices for test set (True)
SUBMISSION = True
print("SUBMISSION: {}".format(SUBMISSION))
import os 
import multiprocessing as mp
from joblib import Parallel, delayed

os.environ['MKL_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['JOBLIB_START_METHOD'] = 'forkserver'
import gc
import re
import math
from time import time
from collections import Counter
import tensorflow as tf
import pandas as pd
import numpy as np
import csv
from fastcache import clru_cache as lru_cache
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
t_start = time()

def print_elapsed(text=''):
    took = (time() - t_start) / 60.0
    print('==== "%s" elapsed %.3f minutes' % (text, took))
#Competition metric
def rmsle(y, y0):
    assert len(y) == len(y0)
    return np.sqrt(np.mean(np.power(np.log1p(y) - np.log1p(y0), 2)))
print('Loading GloVE...')

GLOVE_PATH = '../input/glove-global-vectors-for-word-representation/glove.6B.50d.txt'
embeddings_df = pd.read_table(GLOVE_PATH, sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)

word_embeddings_matrix = embeddings_df.values.astype(np.float32)
print('GloVE Word embeddings shape:', word_embeddings_matrix.shape)
word_embedding_vocab = {t: i for (i, t) in enumerate(embeddings_df.index.tolist())}

del(embeddings_df)

print_elapsed()
print('Reading train data...')
df_train = pd.read_table('../input/mercari-price-suggestion-challenge/train.tsv', engine='c')
print('Train set size: {}'.format(len(df_train)))
print_elapsed()
print('Generating features with statistics for item description textual content')

acronyms_regex = re.compile('([A-Z\-0-9]{2,})')
hashtag_regex = re.compile(r'(#[a-z]{2,})')

#Extracts statistics for each description, words lengths, like percentage of upper-case words, hashtags, etc
def extract_counts(text):
    text_size_words_counts = len(text.split(' '))
    text_size_words_log_counts = math.log1p(text_size_words_counts)
    full_uppercase_perc = len(acronyms_regex.findall(text)) / float(text_size_words_counts)
    exclamation_log_count = math.log1p(text.count('!'))
    star_log_count = math.log1p(text.count('*'))
    percentage_log_count = math.log1p(text.count('%'))
    price_removed_marker_log_count = math.log1p(text.count('[rm]'))
    hashtag_log_count = math.log1p(len(hashtag_regex.findall(text)))    
    return [text_size_words_log_counts,
            full_uppercase_perc,
            exclamation_log_count,
            star_log_count,            
            percentage_log_count,
            price_removed_marker_log_count,
            hashtag_log_count]

item_descr_counts = np.vstack(df_train['item_description'].astype(str).apply(extract_counts).values)

item_descr_counts_scaler = StandardScaler(copy=True)
X_item_descr_counts = item_descr_counts_scaler.fit_transform(item_descr_counts)

del(item_descr_counts)

print_elapsed()
#Removing target attribute (price) from training set, to avoid data leak
price = df_train.pop('price')
print('Spliting train/validation set')
#Defining train / eval sets
valid_rate = 0.00001 if SUBMISSION else 0.1
valid_size = int(len(df_train)*valid_rate)

np.random.seed(100)
rows_idxs = np.arange(0,len(df_train))
np.random.shuffle(rows_idxs)

valid_idxs = rows_idxs[-valid_size:]

#Ignoring lower prices in the train set (minimum price is 3.0 on Mercari website)
train_zeroed_prices_idxs = price.iloc[np.in1d(price.index.values, valid_idxs, invert=True)][price < 3.0].index.values
train_idxs = price.iloc[np.in1d(price.index.values, np.hstack([valid_idxs, train_zeroed_prices_idxs]), 
                                invert=True)].index.values

#Validating the train / validation set split
assert len(df_train) == len(train_zeroed_prices_idxs) + len(train_idxs) + len(valid_idxs)
assert len(set(train_idxs).intersection(set(valid_idxs))) == 0

train_size = len(train_idxs)

del(rows_idxs)
print_elapsed()
print('Normalizing price')
price_log = np.log1p(price)

price_log_train = price_log.iloc[train_idxs].values
price_log_train_mean = price_log_train.mean()
price_log_train_std = price_log_train.std()
del(price_log_train)

y = (price_log - price_log_train_mean) / price_log_train_std
y = y.values.reshape(-1, 1)

print_elapsed()
print('Filling null values...')

df_train.name.fillna('unk_name', inplace=True)
df_train.category_name.fillna('unk_cat', inplace=True)
df_train.brand_name.fillna('unk_brand', inplace=True)
df_train.item_description.fillna('unk_descr', inplace=True)

print_elapsed()
print('Gessing null Brands from name and category...')

def concat_categories(x):
    return set(x.values)

#Getting categories for brands
brand_names_categories = dict(df_train[df_train['brand_name'] != 'unk_brand'][['brand_name','category_name']].astype('str').groupby('brand_name').agg(concat_categories).reset_index().values.tolist())

#Brands sorted by length (decreasinly), so that longer brand names have precedence in the null brand search
brands_sorted_by_size = list(sorted(filter(lambda y: len(y) >= 3, list(brand_names_categories.keys())), key = lambda x: -len(x)))

brand_name_null_count = len(df_train.loc[df_train['brand_name'] == 'unk_brand'])

#Try to guess the Brand based on Name and Category
def brandfinder(name, category):    
    for brand in brands_sorted_by_size:
        if brand in name and category in brand_names_categories[brand]:
            return brand
        
    return 'unk_brand'

train_names_unknown_brands = df_train[df_train['brand_name'] == 'unk_brand'][['name','category_name']].astype('str').values
train_estimated_brands = Parallel(n_jobs=4)(delayed(brandfinder)(name, category) for name, category in train_names_unknown_brands)
df_train.ix[df_train['brand_name'] == 'unk_brand', 'brand_name'] = train_estimated_brands

found = brand_name_null_count-len(df_train.loc[df_train['brand_name'] == 'unk_brand'])
print("Null brands found: %d from %d" % (found, brand_name_null_count))

print_elapsed()
print('Generating features from category statistics for price ...')

CAT_STATS_MIN_COUNT = 5
STD_SIGMAS = 2

df_train['price_log'] = price_log
cats_stats_df = df_train.iloc[train_idxs].groupby(['category_name', 'brand_name', 'shipping']).agg({'category_name': len,
                                                     'price_log': [np.median, np.mean, np.std]})
cats_stats_df.columns = ['price_log_median', 'price_log_mean', 'price_log_std','count']
#Removing categories without a minimum threshold of samples, to avoid price data leak 
cats_stats_df.drop(cats_stats_df[cats_stats_df['count'] < CAT_STATS_MIN_COUNT].index, inplace=True)
cats_stats_df['price_log_std'] = cats_stats_df['price_log_std'].fillna(0)
cats_stats_df['price_log_conf_variance'] = cats_stats_df['price_log_std'] / cats_stats_df['price_log_mean']
cats_stats_df['count_log'] = np.log1p(cats_stats_df['count'])
cats_stats_df['min_expected_log_price'] = (cats_stats_df['price_log_mean'] - cats_stats_df['price_log_std']*STD_SIGMAS).clip(lower=1.0)
cats_stats_df['max_expected_log_price'] = (cats_stats_df['price_log_mean'] + cats_stats_df['price_log_std']*STD_SIGMAS)
del(df_train['price_log'])

len(cats_stats_df)


def merge_with_cat_stats(df):
    return df.merge(cats_stats_df.reset_index(), how='left', 
            on=['category_name', 'brand_name', 'shipping'])[['price_log_median', 'price_log_mean', 'price_log_std', 
                                               'price_log_conf_variance', 'count_log', 'min_expected_log_price', 'max_expected_log_price']].fillna(0).values

train_cats_stats_features = merge_with_cat_stats(df_train)

cats_stats_features_scaler = StandardScaler(copy=True)
X_cats_stats_features_scaled = cats_stats_features_scaler.fit_transform(train_cats_stats_features)

del(train_cats_stats_features)

print_elapsed()
gc.collect()
#Joining the dense features
X_float_features = np.hstack([X_item_descr_counts, X_cats_stats_features_scaled])
#For Glove, spliting composite words separated by "-", because they are rare on Glove
regex_tokenizer = RegexpTokenizer(r'[a-z][\w&]*|[\d]+[\.]*[\w]*|[/!?*:%$"\'\-\+=\.,](?![/!?*:%$"\'-\+=\.,])')

def regex_tokenizer_nltk(text):
    return regex_tokenizer.tokenize(text.lower())
class Tokenizer:
    def __init__(self, min_df=10, limit_length_transform=None, tokenizer=str.split, vocabulary=None, 
                 unk_token_if_ootv=False, workers=1):
        self.min_df = min_df
        self.tokenizer = tokenizer
        self.limit_length_transform = limit_length_transform
        self.workers = workers
        self.vocab_idx = vocabulary
        self.unk_token_if_ootv = unk_token_if_ootv
        self.max_len = None  
        
    def tokenize(self, texts):
        #Multi-processing
        if self.workers>1:
            tokenized_texts = Parallel(n_jobs=self.workers)(delayed(self.tokenizer)(t) for t in texts)
        else:
            tokenized_texts = [self.tokenizer(t) for t in texts] 
        return tokenized_texts
                
    def fit(self, texts):
        doc_freq = Counter()

        max_len = 0

        if type(texts) is list:
            tokenized_texts = texts
        else: #str
                        
            tokenized_texts = self.tokenize(texts)
            
            for sentence in tokenized_texts:
                if self.vocab_idx == None:
                    doc_freq.update(set(sentence))
                max_len = max(max_len, len(sentence))
            
        self.max_len = max_len

        #If the vocabulary is not passed, build from text
        if self.vocab_idx == None:
            vocab = sorted([t for (t, c) in doc_freq.items() if c >= self.min_df])
            self.vocab_idx = {t: (i + 1) for (i, t) in enumerate(vocab)}     


    def fit_transform(self, texts):
        self.fit(texts)
        return self.transform(texts)


    def text_to_idx(self, tokenized):
        if self.unk_token_if_ootv:            
            return [self.vocab_idx[t] if t in self.vocab_idx else self.vocab_idx[UNK_TOKEN] for t in tokenized]
        else:
            return [self.vocab_idx[t] for t in tokenized if t in self.vocab_idx]
    
    def transform(self, texts):
        n = len(texts)
        max_length = self.limit_length_transform or self.max_len
        #Value 0 is reserved for the padding character (<PAD>)
        result = np.zeros(shape=(n, max_length), dtype=np.int32)
        
        if self.workers>1:
            tokenized_texts = Parallel(n_jobs=self.workers)(delayed(self.tokenizer)(t) for t in texts)
        else:
            tokenized_texts = [self.tokenizer(t) for t in texts]              
        
        for i, sentence in enumerate(tokenized_texts):
            text = self.text_to_idx(sentence[:max_length])
            result[i, :len(text)] = text

        return result
    
    def vocabulary_size(self):
        return len(self.vocab_idx) + 1
print('Generating cumulative sub-categories features...')

def paths(tokens):
    all_paths = ['/'.join(tokens[0:(i+1)]) for i in range(len(tokens))]
    return ' '.join(all_paths)

whitespace_regex = re.compile(r'\s+')
@lru_cache(1024)
def cat_process(cat):
    cat = cat.lower()
    cat = whitespace_regex.sub('', cat)
    split = cat.split('/')
    return paths(split)

df_train['category_name_cum'] = df_train.category_name.apply(cat_process)

cat_tok = Tokenizer(min_df=50)
cat_tok.fit(df_train.iloc[np.hstack([train_idxs, train_zeroed_prices_idxs])]['category_name_cum'])
X_cat = cat_tok.transform(df_train['category_name_cum'])
cat_voc_size = cat_tok.vocabulary_size()

print_elapsed()
print('Processing vocabulary for name and description....')

general_tokenizer = Tokenizer(min_df=30, tokenizer=regex_tokenizer_nltk, workers=4)

general_tokenizer.fit(df_train.iloc[np.hstack([train_idxs, train_zeroed_prices_idxs])]['name'] + " " \
                    + df_train.iloc[np.hstack([train_idxs, train_zeroed_prices_idxs])]['item_description'])
print("Text vocabulary size: {}".format(len(general_tokenizer.vocab_idx)))
print_elapsed()
print('Creating vocabulary and loading/generating word embeddings...')

words_vocab_with_embeddings = set(general_tokenizer.vocab_idx.keys()).intersection(set(word_embedding_vocab.keys()))
words_vocab_no_embeddings = set(general_tokenizer.vocab_idx.keys()) - words_vocab_with_embeddings
print('Found word embeddings for corpus: {} from {}'.format(len(words_vocab_with_embeddings), len(general_tokenizer.vocab_idx)))

UNK_TOKEN = '<UNK>'
PAD_TOKEN = '<PAD>'

#Adding words without embedding in the start of the vocabulary
words_vocab_general = {t: i for (i, t) in enumerate([PAD_TOKEN, UNK_TOKEN] + sorted(words_vocab_no_embeddings))}
words_without_embeddings_vocab_size = len(words_vocab_general)
words_with_embeddings_vocab_size = len(words_vocab_with_embeddings)

#Adding words with embedding in the end of the vocabulary
for (i, t) in enumerate(sorted(words_vocab_with_embeddings)):
    words_vocab_general[t] = i+ words_without_embeddings_vocab_size
    
#Creating inverted vocabulary index
custom_inv_vocab_words = dict([(idx,w) for w, idx in list(words_vocab_general.items())])
    
total_vocab_size = len(words_vocab_general)
print('Words without embedding: %d\tTotal vocabulary size: %d' % (words_without_embeddings_vocab_size, total_vocab_size))
    
embedding_size = word_embeddings_matrix.shape[1]  
print("Embedding size: %d" % (embedding_size))

np.random.seed(10)
max_abs_embedding_random_value = np.sqrt(2 / embedding_size)
glove_scaling_factor = word_embeddings_matrix.max() / max_abs_embedding_random_value

#For words available in this GloVE dataset, loading embeddings
words_with_embeddings_matrix = word_embeddings_matrix[[word_embedding_vocab[custom_inv_vocab_words[i]] 
                                                       for i in range(words_without_embeddings_vocab_size, total_vocab_size)]] \
                                / glove_scaling_factor

#For words NOT available in this GloVE dataset, generating random embeddings (according to GloVE distribution)
words_without_embeddings_matrix = np.random.normal(loc=words_with_embeddings_matrix.mean(), 
                                                   scale=words_with_embeddings_matrix.std(), 
                                                   size=(words_without_embeddings_vocab_size, embedding_size))

print("words_without_embeddings_matrix:", words_without_embeddings_matrix.shape) 
print("words_without_embeddings_matrix:", words_with_embeddings_matrix.shape) 

print_elapsed()
#Releasing original GloVE embeddings
del(word_embeddings_matrix)
#Maximum number of words of Name and Item_Description to be processed
NAME_MAX_LEN = 20
ITEM_DESCR_MAX_LEN = 70
print('Processing Title...')

name_tok = Tokenizer(min_df=0, limit_length_transform=NAME_MAX_LEN, tokenizer=regex_tokenizer_nltk, 
                     vocabulary=words_vocab_general, unk_token_if_ootv=True, workers=4)
name_tok.fit(df_train.iloc[np.hstack([train_idxs, train_zeroed_prices_idxs])]['name'])
X_name = name_tok.transform(df_train.name)
print(X_name.shape)

print_elapsed()
print('Processing Description...')

desc_tok = Tokenizer(min_df=0, limit_length_transform=ITEM_DESCR_MAX_LEN, tokenizer=regex_tokenizer_nltk, 
                     vocabulary=words_vocab_general, unk_token_if_ootv=True, workers=4)

desc_tok.fit(df_train.iloc[np.hstack([train_idxs, train_zeroed_prices_idxs])].item_description)
X_desc = desc_tok.transform(df_train.item_description)
print(X_desc.shape)

print_elapsed()
print('Processing Brands...')

df_train.brand_name = df_train.brand_name.str.lower()
df_train.brand_name = df_train.brand_name.str.replace(' ', '_')

brand_cnt = Counter(df_train.brand_name[df_train.brand_name != 'unk_brand'])
brands = sorted(b for (b, c) in brand_cnt.items() if c >= 20)
brands_idx = {b: (i + 1) for (i, b) in enumerate(brands)}

X_brand = df_train.brand_name.apply(lambda b: brands_idx.get(b, 0))
X_brand = X_brand.values.reshape(-1, 1) 
brand_voc_size = len(brands) + 1
print("Brands vocab. size: {}".format(brand_voc_size))

print_elapsed()
print('Processing Item condition and Shipping...')
X_item_cond = (df_train.item_condition_id - 1).astype('uint8').values.reshape(-1, 1)
X_shipping = df_train.shipping.astype('float32').values.reshape(-1, 1)
print_elapsed()
def prepare_batches(seq, step):
    n = len(seq)
    res = []
    for i in range(0, n, step):
        res.append(seq[i:i+step])
    return res
def train_model(session, epochs=4, batch_size=500, eval_each_epoch=True, dropout_input_words=0.0):
    print('\ntraining the model...')
    
    print_elapsed()
    
    training_size = train_idxs.shape[0]

    for i in range(int(np.ceil(epochs))):
        print("-----------------EPOCH: {}-------------------".format(i))
        t0 = time()
        np.random.seed(i)

        epoch_size = training_size
        #If the epoch is not int (eg. 2.5), reduces the last epoch size (number of steps)
        if i+1 - epochs > 0:
            epoch_size = int(training_size*(epochs%1))
            
        #Training dataset shuffling
        batches = prepare_batches(np.random.permutation(train_idxs)[:epoch_size], batch_size)
        
        for idx in batches:
            name_batch = X_name[idx]
            desc_batch = X_desc[idx]
            
            #If set, apply dropout to the input words (kind of data augmentation)
            if dropout_input_words > 0:            
                name_mask = (np.random.uniform(0,1, size=name_batch.shape) >= dropout_input_words).astype(np.int32)
                name_batch = name_batch * name_mask

                desc_mask = (np.random.uniform(0,1, size=desc_batch.shape) >= dropout_input_words).astype(np.int32)
                desc_batch = desc_batch * desc_mask
        
            feed_dict = {
                place_name: name_batch,
                place_desc: desc_batch,
                place_brand: X_brand[idx],
                place_cat: X_cat[idx],
                place_cond: X_item_cond[idx],
                place_ship: X_shipping[idx],
                place_float_stats: X_float_features[idx],
                place_y: y[idx],
                place_training: True
            }
            session.run(train_step, feed_dict=feed_dict)

        took = time() - t0
        print('epoch %d took %.3fs' % (i, took))
        
        if eval_each_epoch and i < epochs-1:
            train_set_evaluation(sess)
            print_elapsed()
            
            eval_set_evaluation(session)
            
        print_elapsed()
def eval_set_evaluation(session, save_results_to=None):
    print('\nEVAL SET Evaluation')

    y_pred_norm_eval = np.zeros(valid_size)

    EVAL_BATCH_SIZE = 5000

    batches = prepare_batches(valid_idxs, EVAL_BATCH_SIZE)

    for b, idx in enumerate(batches):
        feed_dict = {
            place_name: X_name[idx],
            place_desc: X_desc[idx],
            place_brand: X_brand[idx],
            place_cat: X_cat[idx],
            place_cond: X_item_cond[idx],
            place_ship: X_shipping[idx],
            place_float_stats: X_float_features[idx],
            place_training: False
        }
        batch_pred = session.run(out, feed_dict=feed_dict)
        start_idx = (b*EVAL_BATCH_SIZE)
        y_pred_norm_eval[start_idx:min(start_idx+EVAL_BATCH_SIZE, valid_size)] = batch_pred[:, 0]
        
    y_adjusted_pred_eval = np.expm1(y_pred_norm_eval * price_log_train_std + price_log_train_mean)

    print("Eval set RMSLE: {}".format(rmsle(price[valid_idxs].values, y_adjusted_pred_eval)))    

    if not SUBMISSION and save_results_to != None:
        print('Saving CNN eval results to "valid_cnn_predictions.csv"...')    
        df_out = pd.DataFrame()
        df_out['train_id'] = df_train.iloc[valid_idxs]['train_id']
        df_out['price'] = price[valid_idxs].values
        df_out['pred_price'] = y_adjusted_pred_eval
        df_out.to_csv(save_results_to, index=False)    

    return y_pred_norm_eval
def train_set_evaluation(session):
    print('\nTRAIN SET Evaluation')

    y_pred_norm = np.zeros(train_size)

    EVAL_BATCH_SIZE = 5000

    batches = prepare_batches(train_idxs, EVAL_BATCH_SIZE)

    for b, idx in enumerate(batches):
        feed_dict = {
            place_name: X_name[idx],
            place_desc: X_desc[idx],
            place_brand: X_brand[idx],
            place_cat: X_cat[idx],
            place_cond: X_item_cond[idx],
            place_ship: X_shipping[idx],
            place_float_stats: X_float_features[idx],
            place_training: False
        }
        batch_pred = session.run(out, feed_dict=feed_dict)
        start_idx = (b*EVAL_BATCH_SIZE)
        
        y_pred_norm[start_idx:min(start_idx+EVAL_BATCH_SIZE, train_size)] = batch_pred[:, 0]                

    y_pred_train = y_pred_norm * price_log_train_std + price_log_train_mean
    y_adjusted_pred_train = np.expm1(y_pred_train)

    print("Train set RMSLE: {}".format(rmsle(price[train_idxs].values, y_adjusted_pred_train)))
    
    return price[train_idxs].values, y_adjusted_pred_train
def save_test_set_predictions(preds, filename):
    df_out = pd.DataFrame()
    df_out['test_id'] = range(0,preds.shape[0])
    df_out['price'] = preds

    df_out.to_csv(filename, index=False)    
    
    print_elapsed('Predictions exported to {}'.format(filename))
def generate_test_submission(session):
    print('Reading and generating features for test data...')

    #TODO: Read data (read_table) in batches (ex: read smaller batches using chunksize) 
    #to better support larger testset in 2nd phase
    df_test = pd.read_csv('../input/mercari-price-suggestion-challenge/test_stg2.tsv', sep='\t')
    
    #Filling nulls
    df_test.name.fillna('unk_name', inplace=True)
    df_test.category_name.fillna('unk_cat', inplace=True)
    df_test.brand_name.fillna('unk_brand', inplace=True)
    df_test.item_description.fillna('unk_brand', inplace=True)
    
    #Guessing null Brands
    test_names_unknown_brands = df_test[df_test['brand_name'] == 'unk_brand'][['name','category_name']].astype('str').values
    test_estimated_brands = Parallel(n_jobs=7)(delayed(brandfinder)(name, category) for name, category in test_names_unknown_brands)
    df_test.ix[df_test['brand_name'] == 'unk_brand', 'brand_name'] = test_estimated_brands

    #Processing categories
    df_test.category_name = df_test.category_name.apply(cat_process)
    df_test.brand_name = df_test.brand_name.str.lower()
    df_test.brand_name = df_test.brand_name.str.replace(' ', '_')
    
    #Generating statistic features for Description 
    X_item_descr_counts_test = item_descr_counts_scaler.transform(np.vstack(df_test['item_description'].astype(str) \
                                                                            .apply(extract_counts).values))

    #Generating features of price statistics by categories
    X_cats_stats_features_scaled_test = cats_stats_features_scaler.transform(merge_with_cat_stats(df_test))
    
    #Joining dense features
    X_float_features_test = np.hstack([X_item_descr_counts_test, X_cats_stats_features_scaled_test])
    
    #Tokenizing category name
    X_cat_test = cat_tok.transform(df_test.category_name)
    X_name_test = name_tok.transform(df_test.name)

    #Tokenizing description
    X_desc_test = desc_tok.transform(df_test.item_description)

    #Tokenizing category name
    X_item_cond_test = (df_test.item_condition_id - 1).astype('uint8').values.reshape(-1, 1)
    X_shipping_test = df_test.shipping.astype('float32').values.reshape(-1, 1)

    #Processing brands
    X_brand_test = df_test.brand_name.apply(lambda b: brands_idx.get(b, 0))
    X_brand_test = X_brand_test.values.reshape(-1, 1)
    
    print_elapsed('Finish generating features for test set')
    
    
    print('Prediction for test set using the trained CNN model...')

    n_test = len(df_test)
    y_pred = np.zeros(n_test)

    test_idx = np.arange(n_test)
    batches = prepare_batches(test_idx, 5000)

    for idx in batches:
        feed_dict = {
            place_name: X_name_test[idx],
            place_desc: X_desc_test[idx],
            place_brand: X_brand_test[idx],
            place_cat: X_cat_test[idx],
            place_cond: X_item_cond_test[idx],
            place_ship: X_shipping_test[idx],            
            place_float_stats: X_float_features_test[idx],
            place_training: False
        }
        
        batch_pred = session.run(out, feed_dict=feed_dict)
        y_pred[idx] = batch_pred[:, 0]

    print_elapsed()
    
    return y_pred
gc.collect()
print('Defining the model...')

def conv1d(inputs, num_filters, filter_size, pool_size, is_training, reg=0.0, activation=None, padding='same'):

    out = tf.layers.conv1d(
        inputs=inputs, filters=num_filters, padding=padding,
        kernel_size=filter_size,
        strides=1,
        activation=activation,      
        kernel_initializer=tf.contrib.layers.xavier_initializer(),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(reg),
        )
    
    out = tf.layers.max_pooling1d(out, pool_size=pool_size, strides=1, padding='valid')
                                       
    return out

def dense(X, size, reg=0.0, activation=None):
    out = tf.layers.dense(X, units=size, activation=activation, 
                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                     kernel_regularizer=tf.contrib.layers.l2_regularizer(reg))
    return out

def embed(inputs, size, dim):
    std = np.sqrt(2 / dim)
    emb = tf.Variable(tf.random_uniform([size, dim], -std, std))
    lookup = tf.nn.embedding_lookup(emb, inputs)
    return lookup


dnn_settings = {
    #Training params
    'lr_initial': 0.002, 
    'lr_decay_rate': 0.94, 
    'lr_num_epochs_per_decay': 0.1,
    'batch_size': 256,
    'epochs': 3.0, #2.5, #3
    
    'dropout_input_words': 0.05, #Randomly set to zero (<PAD>) some words of the input text (data augmentation)
    'dropout_rate': 0.00,
    'l2_reg': 0.0, 
    'main_batch_norm_decay': 0.93,

     #Model params
    'cnn_filter_sizes': [3], 

    'name_seq_len': NAME_MAX_LEN,
    'name_num_filters': 128,
    'name_avg_embedding_num_words': 5,    

    'desc_seq_len': ITEM_DESCR_MAX_LEN,
    'desc_num_filters': 96, 
    'descr_avg_embedding_num_words': 20,

    'brand_embeddings_dim': 32, 

    'cat_embeddings_dim': 32, 
    'cat_seq_len': X_cat.shape[1],

    'word_embeddings_size': embedding_size,
    'word_embeddings_max_norm': 0.45,
}

print("SETTINGS:", dnn_settings)


print()


graph = tf.Graph()
graph.seed = 1

ALLOW_SOFT_PLACEMENT=True
LOG_DEVICE_PLACEMENT=False

with graph.as_default():
    
    #As GPUs were not available on Kaggle Kernelm using them only for local development 
    with tf.device('/cpu:0' if SUBMISSION else '/gpu:0'):    
    
        session_conf = tf.ConfigProto(
          allow_soft_placement=ALLOW_SOFT_PLACEMENT,
          log_device_placement=LOG_DEVICE_PLACEMENT,
          intra_op_parallelism_threads = 4,
          inter_op_parallelism_threads = 4)

        sess = tf.Session(config=session_conf)
        with sess.as_default():
            
            with tf.name_scope("embedding"):          
                
                #Loading GloVE word embeddings for available words 
                #Reference: https://ireneli.eu/2017/01/17/tensorflow-07-word-embeddings-2-loading-pre-trained-vectors/
                words_with_embedding_placeholder = tf.placeholder(tf.float32, [words_with_embeddings_vocab_size,
                                                                               dnn_settings['word_embeddings_size']])
                words_with_embedding_variable = tf.Variable(tf.constant(0.0, shape=[words_with_embeddings_vocab_size,
                                                                                    dnn_settings['word_embeddings_size']]),
                    #Best results were obtaining using Glove to initialize embeddings, 
                    #and letting them to be trained for the prediction task
                    trainable=True,                
                    name="words_with_embedding")
                words_with_embedding_init = words_with_embedding_variable.assign(words_with_embedding_placeholder)
                
                #Loading random word embeddings for OOTV words
                words_without_embedding_placeholder = tf.placeholder(tf.float32, [words_without_embeddings_vocab_size,
                                                                                  dnn_settings['word_embeddings_size']])
                words_without_embedding_variable = tf.Variable(tf.constant(0.0, shape=[words_without_embeddings_vocab_size,
                                                                                       dnn_settings['word_embeddings_size']]),
                    trainable=True,                
                    name="words_without_embedding")
                words_without_embedding_init = words_without_embedding_variable.assign(words_without_embedding_placeholder)
                
                #Creating a regularizer for embeddings values
                word_embedding_regularizer = tf.nn.l2_loss(words_without_embedding_variable)

                #Concatenating GloVE embeddings and random embeddings in a single variable
                words_embedding_variable = tf.concat([words_without_embedding_variable, words_with_embedding_variable], 
                                                    axis=0)

            
            #Model input features
            place_name = tf.placeholder(tf.int32, shape=(None, dnn_settings['name_seq_len']))
            place_desc = tf.placeholder(tf.int32, shape=(None, dnn_settings['desc_seq_len']))
            place_brand = tf.placeholder(tf.int32, shape=(None, 1))
            place_cat = tf.placeholder(tf.int32, shape=(None, dnn_settings['cat_seq_len']))
            place_ship = tf.placeholder(tf.float32, shape=(None, 1))
            place_cond = tf.placeholder(tf.uint8, shape=(None, 1))
            place_float_stats = tf.placeholder(dtype=tf.float32, shape=(None, X_float_features.shape[1]))
            
            #Output feature
            place_y = tf.placeholder(dtype=tf.float32, shape=(None, 1))
            
            #Flag to indicate whether the graph is being trained or used for inference
            place_training = tf.placeholder(tf.bool, shape=(), )
            
            #Creating embedding layer for categorical features Brands and Categories
            brand = embed(place_brand, brand_voc_size, dnn_settings['brand_embeddings_dim'])
            cat = embed(place_cat, cat_voc_size, dnn_settings['cat_embeddings_dim'])
                        
            #Looking up embeddings for each word            
            name = tf.nn.embedding_lookup(words_embedding_variable, place_name, max_norm=dnn_settings['word_embeddings_max_norm'])
            desc = tf.nn.embedding_lookup(words_embedding_variable, place_desc, max_norm=dnn_settings['word_embeddings_max_norm'])
            print("name.shape", name.shape)
            print("desc.shape", desc.shape)
            
            #Creating a special layer to average embeddings of the first words of the name and description
            #under the assumption that they are the most representative ones
            name_mean_embeddings = tf.reduce_mean(name[:,:dnn_settings['name_avg_embedding_num_words'],:], axis=1)
            print("name_mean_embeddings.shape", name_mean_embeddings.shape)
            desc_mean_embeddings = tf.reduce_mean(desc[:,:dnn_settings['descr_avg_embedding_num_words'],:], axis=1)
            print("desc_mean_embeddings.shape", desc_mean_embeddings.shape)
            
            
            conv_layers = []
            for filter_size in dnn_settings['cnn_filter_sizes']:
                with tf.name_scope("conv-maxpool-%s" % filter_size):
                    name_conv = conv1d(name, num_filters=dnn_settings['name_num_filters'], filter_size=filter_size, 
                                       pool_size=dnn_settings['name_seq_len'], 
                                       is_training=place_training, reg=dnn_settings['l2_reg'], activation=tf.nn.relu)                
                    name_conv = tf.contrib.layers.flatten(name_conv)
                    print(("conv-maxpool-%s NAME" % filter_size), name_conv.shape)
                    conv_layers.append(name_conv)

                    desc_conv = conv1d(desc, num_filters=dnn_settings['desc_num_filters'], filter_size=filter_size, 
                                       pool_size=dnn_settings['desc_seq_len'], 
                                       is_training=place_training, reg=dnn_settings['l2_reg'], activation=tf.nn.relu)
                    desc_conv = tf.contrib.layers.flatten(desc_conv)
                    print(("conv-maxpool-%s DESCRIPTION" % filter_size), desc_conv.shape)
                    conv_layers.append(desc_conv)

            brand = tf.contrib.layers.flatten(brand)
            print(brand.shape)

            #cat = tf.layers.average_pooling1d(cat, pool_size=dnn_settings['cat_seq_len'], strides=1, padding='valid')
            cat = tf.contrib.layers.flatten(cat)
            print(cat.shape)

            ship = place_ship
            print(ship.shape)

            cond = tf.one_hot(place_cond, 5)
            cond = tf.contrib.layers.flatten(cond)
            print(cond.shape)
            
            float_stats = place_float_stats
            float_stats = tf.contrib.layers.flatten(float_stats)
            print(float_stats.shape)
              
            #Joining all layers outputs for a sequence of Fully Connected layers
            out = tf.concat(conv_layers + [name_mean_embeddings, desc_mean_embeddings, brand, cat, ship, cond, float_stats], axis=1)
            print('concatenated dim:', out.shape)


            out = tf.contrib.layers.batch_norm(out, decay=dnn_settings['main_batch_norm_decay'], 
                                               center=True, scale=False, epsilon=0.001,           
                                               is_training=place_training)
            

            #out = tf.layers.dropout(out, rate=dropout_rate, training=place_training)
            out = dense(out, 256, reg=dnn_settings['l2_reg'], activation=tf.nn.relu)

            #out = tf.layers.dropout(out, rate=dropout_rate, training=place_training)
            out = dense(out, 128, reg=dnn_settings['l2_reg'], activation=tf.nn.relu)
            
            out = tf.contrib.layers.layer_norm(tf.concat([out, ship, cond, float_stats], axis=1))

            out = dense(out, 1, reg=dnn_settings['l2_reg'])

            reg_loss = tf.losses.get_regularization_loss() + dnn_settings['l2_reg']*word_embedding_regularizer
            
            #Computing the loss, with L2 regularization
            loss = tf.losses.mean_squared_error(place_y, out) + reg_loss
            rmse = tf.sqrt(loss)
            
            #Setting learning rate decay
            lr_decay_steps = int(train_size / dnn_settings['batch_size'] * dnn_settings['lr_num_epochs_per_decay'])
            
            global_step = tf.Variable(0, trainable=False)
            learning_rate_decay = tf.train.exponential_decay(dnn_settings['lr_initial'],
                                          global_step,
                                          lr_decay_steps,
                                          dnn_settings['lr_decay_rate'],
                                          staircase=True,
                                          name='exponential_decay_learning_rate')
            
            opt = tf.train.AdamOptimizer(learning_rate=learning_rate_decay,
                                         beta1=0.9,
                                         beta2=0.999,
                                         epsilon=1e-08
                                        )
            
            #Necessary to run update ops for batch_norm
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):                
                train_step = opt.minimize(loss=loss, global_step=global_step)

            print("Initializing variables")
            init = tf.global_variables_initializer()        
            sess.run(init)
            print_elapsed()
                  
            #Initializing word embeddings variable
            sess.run([words_without_embedding_init,
                      words_with_embedding_init], 
                     feed_dict={words_without_embedding_placeholder: words_without_embeddings_matrix,
                                words_with_embedding_placeholder: words_with_embeddings_matrix})

            #Training the model
            train_model(sess, epochs=dnn_settings['epochs'], batch_size=dnn_settings['batch_size'], 
                        dropout_input_words=dnn_settings['dropout_input_words'], 
                        eval_each_epoch=not SUBMISSION)
            print_elapsed()

            if not SUBMISSION:
                #Evaluating train set
                train_actual_prices_debug, train_pred_prices_debug = train_set_evaluation(sess)
                print_elapsed()

                #Evaluating validation set
                cnn_pred_eval = eval_set_evaluation(sess)
                print_elapsed()

            if SUBMISSION:
                #Generating output CSV file with the predictions for test set
                cnn_pred_test = generate_test_submission(sess)                
                cnn_pred_test_scaled = np.expm1(cnn_pred_test * price_log_train_std + price_log_train_mean)
                save_test_set_predictions(cnn_pred_test_scaled, 'submission_cnn.csv')
print_elapsed('Finished script')