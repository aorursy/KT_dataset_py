import time
start_time = time.time()

import gc
import numpy as np
import pandas as pd
from subprocess import check_output

develop = False
train_set_ratio = .99
split_seed = 123

#train_path = '../input/mercari-price-suggestion-challenge/train.tsv'
#test_path = '../input/mercari-price-suggestion-challenge/test.tsv'
train_path = '../input/train.tsv'
test_path = '../input/test.tsv'
#test_path = '../input/extend-test-set-for-testing/test_extend.tsv'

def rmsle(y, y_pred):
    assert y.shape == y_pred.shape
    return np.sqrt(np.square(np.log(y_pred + 1) - np.log(y + 1)).mean())
# # # # # # # # # # # # # # # # # # # # # # RNN # # # # # # # # # # # # # # # # # # # # # # # #
# import modules
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.cross_validation import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# load data
train_df = pd.read_csv(train_path, sep='\t')
test_df = pd.read_csv(test_path, sep='\t')
#test_df = pd.read_csv(test_path, delimiter=',')
print('size of train set:', train_df.shape)
print('size of test set:', test_df.shape)

# missing values imputation
def impute_missing(data_df):
    data_df.category_name.fillna(value="missing", inplace=True)
    data_df.brand_name.fillna(value="missing", inplace=True)
    data_df.item_description.fillna(value="missing", inplace=True)

impute_missing(train_df)
impute_missing(test_df)

# process categorical data
le = LabelEncoder()
le.fit(np.hstack([train_df.category_name, test_df.category_name]))
train_df.category_name = le.transform(train_df.category_name)
test_df.category_name = le.transform(test_df.category_name)

le.fit(np.hstack([train_df.brand_name, test_df.brand_name]))
train_df.brand_name = le.transform(train_df.brand_name)
test_df.brand_name = le.transform(test_df.brand_name)
del le

# process text variable
print("[{}] text to seq process...".format(time.time() - start_time))
raw_text = np.hstack([
    train_df.item_description.str.lower(),
    train_df.name.str.lower()
])
print("[{}] fitting tokenizer...".format(time.time() - start_time))
tok_raw = Tokenizer()
tok_raw.fit_on_texts(raw_text)

train_df["seq_item_description"] = tok_raw.texts_to_sequences(train_df.item_description.str.lower())
test_df["seq_item_description"] = tok_raw.texts_to_sequences(test_df.item_description.str.lower())
train_df["seq_name"] = tok_raw.texts_to_sequences(train_df.name.str.lower())
test_df["seq_name"] = tok_raw.texts_to_sequences(test_df.name.str.lower())
print("[{}] transforming text to seq...".format(time.time() - start_time))

# length of sequences
max_name_seq = np.max([np.max(train_df.seq_name.apply(lambda x: len(x))),
                       np.max(test_df.seq_name.apply(lambda x: len(x)))])
max_seq_item_description = np.max([np.max(train_df.seq_item_description.apply(lambda x: len(x))),
                                   np.max(test_df.seq_item_description.apply(lambda x: len(x)))])
print('[{}] finish calculating MAX_seq'.format(time.time() - start_time))

# EMBEDDINGS MAX VALUE
MAX_NAME_SEQ = 10
MAX_ITEM_DESC_SEQ = 75
#MAX_ITEM_DESC_SEQ = 100
MAX_TEXT = np.max([np.max(train_df.seq_name.max()),
                   np.max(test_df.seq_name.max()),
                   np.max(train_df.seq_item_description.max()),
                   np.max(test_df.seq_item_description.max())]) + 2
MAX_CATEGORY = np.max([train_df.category_name.max(), test_df.category_name.max()]) + 1
MAX_BRAND = np.max([train_df.brand_name.max(), test_df.brand_name.max()]) + 1
MAX_CONDITION = np.max([train_df.item_condition_id.max(), test_df.item_condition_id.max()]) + 1

# scale target variable
train_df["target"] = np.log(train_df.price + 1)
target_scaler = MinMaxScaler(feature_range=(-1, 1))
train_df["target"] = target_scaler.fit_transform(train_df.target.reshape(-1, 1))


# keras data definition
def get_keras_data(dataset):
    X = {
        'name': pad_sequences(dataset.seq_name, maxlen=MAX_NAME_SEQ)
        , 'item_desc': pad_sequences(dataset.seq_item_description, maxlen=MAX_ITEM_DESC_SEQ)
        , 'brand_name': np.array(dataset.brand_name)
        , 'category_name': np.array(dataset.category_name)
        , 'item_condition': np.array(dataset.item_condition_id)
        , 'num_vars': np.array(dataset[["shipping"]])
    }
    return X


# extract development test
if develop:
    dtrain, dvalid = train_test_split(train_df, train_size=train_set_ratio, random_state=split_seed)
    X_train = get_keras_data(dtrain)
    X_valid = get_keras_data(dvalid)
    X_test = get_keras_data(test_df)
else:
    dtrain = train_df
    #dvalid = train_df   # this might be the part that make it slow
    X_train = get_keras_data(dtrain)
    #X_valid = get_keras_data(dvalid)
    X_test = get_keras_data(test_df)
ids = test_df[["test_id"]]
del train_df, test_df; gc.collect()
print('[{}] finish forming keras data input for rnn model'.format(time.time() - start_time))


# keras model definition
from keras.layers import Input, Dropout, Dense, BatchNormalization, Activation, concatenate, GRU, Embedding, Flatten, \
    BatchNormalization
from keras.models import Model
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras import backend as K
from keras import regularizers

dr_r = .1

def get_callbacks(filepath, patience=2):
    es = EarlyStopping('val_loss', patience=patience, mode="min")
    msave = ModelCheckpoint(filepath, save_best_only=True)
    return [es, msave]


def rmsle_cust(y_true, y_pred):
    first_log = K.log(K.clip(y_pred, K.epsilon(), None) + 1.)
    second_log = K.log(K.clip(y_true, K.epsilon(), None) + 1.)
    return K.sqrt(K.mean(K.square(first_log - second_log), axis=-1))


def rmsle2(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_true - y_pred), axis=-1))


def get_model():
    # Inputs
    name = Input(shape=[X_train["name"].shape[1]], name="name")
    item_desc = Input(shape=[X_train["item_desc"].shape[1]], name="item_desc")
    brand_name = Input(shape=[1], name="brand_name")
    category_name = Input(shape=[1], name="category_name")
    item_condition = Input(shape=[1], name="item_condition")
    num_vars = Input(shape=[X_train["num_vars"].shape[1]], name="num_vars")

    # Embeddings layers
    emb_name = Embedding(MAX_TEXT, 50)(name)
    emb_item_desc = Embedding(MAX_TEXT, 50)(item_desc)
    emb_brand_name = Embedding(MAX_BRAND, 10)(brand_name)
    emb_category_name = Embedding(MAX_CATEGORY, 10)(category_name)
    emb_item_condition = Embedding(MAX_CONDITION, 5)(item_condition)

    # rnn layer
    rnn_layer1 = GRU(16)(emb_item_desc)
    rnn_layer2 = GRU(8)(emb_name)

    # main layer
    main_l = concatenate([
        Flatten()(emb_brand_name)
        , Flatten()(emb_category_name)
        , Flatten()(emb_item_condition)
        , rnn_layer1
        , rnn_layer2
        , num_vars
    ])
    main_l = Dropout(dr_r)(Dense(128, activation='relu')(main_l))

    # output
    output = Dense(1, activation="linear")(main_l)

    # model
    model = Model([name, item_desc, brand_name, category_name, item_condition, num_vars], output)
    model.compile(loss="mse", optimizer="adam", metrics=["mae", rmsle_cust])

    return model

model = get_model()
#model.summary()

# fitting the model
BATCH_SIZE = 2000
epochs = 2
steps = int(len(X_train['name']) / BATCH_SIZE) * epochs
lr_init, lr_fin = 0.009, 0.006
exp_decay = lambda init, fin, steps: (init / fin) ** (1/(steps - 1)) - 1
lr_decay = exp_decay(lr_init, lr_fin, steps)
log_subdir = '_'.join(['ep', str(epochs),
                       'bs', str(BATCH_SIZE),
                       'lrI', str(lr_init),
                       'lrF', str(lr_fin),
                       'dr', str(dr_r)
                       ])

model = get_model()
K.set_value(model.optimizer.lr, lr_init)
K.set_value(model.optimizer.decay, lr_decay)

if develop:
    model.fit(X_train, dtrain.target
              , epochs=epochs
              , batch_size=BATCH_SIZE
              , validation_data=(X_valid, dvalid.target)
              , verbose=1)
    print('[{}] finish fitting rnn model'.format(time.time() - start_time))
else:
    model.fit(X_train, dtrain.target
              , epochs=epochs
              , batch_size=BATCH_SIZE
              , validation_split=0
              , verbose=1)
    print('[{}] finish fitting rnn model'.format(time.time() - start_time))
    
# create predictions for validation set
if develop:
    val_preds_rnn = model.predict(X_valid)
    val_preds_rnn = target_scaler.inverse_transform(val_preds_rnn)
    val_preds_rnn = np.expm1(val_preds_rnn)
    y_true = np.array(dvalid.price.values).reshape(dvalid.shape[0], 1)  # price, not the target column
    v_rmsle = rmsle(y_true, val_preds_rnn)
    print('[{}] RMSLE of validation set with RNN model is: {}'.format(
        time.time() - start_time, v_rmsle))

# create predictions for test set
preds_rnn = model.predict(X_test, batch_size=BATCH_SIZE)
preds_rnn = target_scaler.inverse_transform(preds_rnn)
preds_rnn = np.expm1(preds_rnn)
print('[{}] finish prediction with rnn'.format(time.time() - start_time))

# clear
if develop:
    del dtrain, dvalid, X_train, X_valid, X_test; gc.collect()
else:
    del dtrain, X_train, X_test; gc.collect()
# # # # # # # # # # # # # # # # # WordBatch Linear Models # # # # # # # # # # # # # # # # # # # #
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import sys

#sys.path.insert(0, '../input/wordbatch/wordbatch/')
import wordbatch
from wordbatch.extractors import WordBag, WordHash
from wordbatch.models import FM_FTRL
from nltk.corpus import stopwords
import re

NUM_BRANDS = 4500
NUM_CATEGORIES = 1250

def rmsle(y, y0):
    assert len(y) == len(y0)
    return np.sqrt(np.mean(np.power(np.log1p(y) - np.log1p(y0), 2)))


def split_cat(text):
    try:
        return text.split("/")
    except:
        return ("No Label", "No Label", "No Label")


def handle_missing_inplace(dataset):
    dataset['general_cat'].fillna(value='missing', inplace=True)
    dataset['subcat_1'].fillna(value='missing', inplace=True)
    dataset['subcat_2'].fillna(value='missing', inplace=True)
    dataset['brand_name'].fillna(value='missing', inplace=True)
    dataset['item_description'].fillna(value='missing', inplace=True)


def cutting(dataset):
    pop_brand = dataset['brand_name'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_BRANDS]
    dataset.loc[~dataset['brand_name'].isin(pop_brand), 'brand_name'] = 'missing'
    pop_category1 = dataset['general_cat'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_CATEGORIES]
    pop_category2 = dataset['subcat_1'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_CATEGORIES]
    pop_category3 = dataset['subcat_2'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_CATEGORIES]
    dataset.loc[~dataset['general_cat'].isin(pop_category1), 'general_cat'] = 'missing'
    dataset.loc[~dataset['subcat_1'].isin(pop_category2), 'subcat_1'] = 'missing'
    dataset.loc[~dataset['subcat_2'].isin(pop_category3), 'subcat_2'] = 'missing'


def to_categorical(dataset):
    dataset['general_cat'] = dataset['general_cat'].astype('category')
    dataset['subcat_1'] = dataset['subcat_1'].astype('category')
    dataset['subcat_2'] = dataset['subcat_2'].astype('category')
    dataset['item_condition_id'] = dataset['item_condition_id'].astype('category')


# Define helpers for text normalization
stopwords = {x: 1 for x in stopwords.words('english')}
non_alphanums = re.compile(u'[^A-Za-z0-9]+')


def normalize_text(text):
    return u" ".join(
        [x for x in [y for y in non_alphanums.sub(' ', text).lower().strip().split(" ")] \
         if len(x) > 1 and x not in stopwords])


def construct_features(start_time):
    train = pd.read_table(train_path, engine='c')
    test = pd.read_table(test_path, engine='c')
    #test = pd.read_table(test_path, delimiter=',', engine='c')
    print('[{}] Finished to load data'.format(time.time() - start_time))
    print('Train shape: ', train.shape)
    print('Test shape: ', test.shape)
    nrow_test = train.shape[0]  # -dftt.shape[0]
    dftt = train[(train.price < 1.0)]
    train = train.drop(train[(train.price < 1.0)].index)
    del dftt['price']
    nrow_train = train.shape[0]
    # print(nrow_train, nrow_test)
    y = np.log1p(train["price"])
    merge: pd.DataFrame = pd.concat([train, dftt, test])
    #merge: pd.DataFrame = pd.concat([train, test])
    #submission: pd.DataFrame = test[['test_id']]

    del train
    del test
    gc.collect()

    merge['general_cat'], merge['subcat_1'], merge['subcat_2'] = \
        zip(*merge['category_name'].apply(lambda x: split_cat(x)))
    merge.drop('category_name', axis=1, inplace=True)
    print('[{}] Split categories completed.'.format(time.time() - start_time))

    handle_missing_inplace(merge)
    print('[{}] Handle missing completed.'.format(time.time() - start_time))

    cutting(merge)
    print('[{}] Cut completed.'.format(time.time() - start_time))

    to_categorical(merge)
    print('[{}] Convert categorical completed'.format(time.time() - start_time))

    wb = wordbatch.WordBatch(normalize_text, extractor=(WordBag, {"hash_ngrams": 2, "hash_ngrams_weights": [1.5, 1.0],
                                                                  "hash_size": 2 ** 21, "norm": None, "tf": 'binary',
                                                                  "idf": None,
                                                                  }), procs=8)
    wb.dictionary_freeze = True
    X_name = wb.fit_transform(merge['name'])
    del (wb)
    X_name = X_name[:, np.array(np.clip(X_name.getnnz(axis=0) - 3, 0, 1), dtype=bool)]
    print('[{}] Vectorize `name` completed.'.format(time.time() - start_time))

    wb = CountVectorizer()
    X_category1 = wb.fit_transform(merge['general_cat'])
    X_category2 = wb.fit_transform(merge['subcat_1'])
    X_category3 = wb.fit_transform(merge['subcat_2'])
    print('[{}] Count vectorize `categories` completed.'.format(time.time() - start_time))

    wb = wordbatch.WordBatch(normalize_text, extractor=(WordBag, {"hash_ngrams": 2, "hash_ngrams_weights": [1.0, 1.0],
                                                                  "hash_size": 2 ** 21, "norm": "l2", "tf": 1.0,
                                                                  "idf": None}), procs=8)
    wb.dictionary_freeze = True
    X_description = wb.fit_transform(merge['item_description'])
    del (wb)
    X_description = X_description[:, np.array(np.clip(X_description.getnnz(axis=0) - 3, 0, 1), dtype=bool)]
    print('[{}] Vectorize `item_description` completed.'.format(time.time() - start_time))

    lb = LabelBinarizer(sparse_output=True)
    X_brand = lb.fit_transform(merge['brand_name'])
    print('[{}] Label binarize `brand_name` completed.'.format(time.time() - start_time))

    X_dummies = csr_matrix(pd.get_dummies(merge[['item_condition_id', 'shipping']],
                                          sparse=True).values)
    print('[{}] Get dummies on `item_condition_id` and `shipping` completed.'.format(time.time() - start_time))
    print(X_dummies.shape, X_description.shape, X_brand.shape, X_category1.shape, X_category2.shape, X_category3.shape,
          X_name.shape)
    sparse_merge = hstack((X_dummies, X_description, X_brand, X_category1, X_category2, X_category3, X_name)).tocsr()

    print('[{}] Create sparse merge completed'.format(time.time() - start_time))
    return sparse_merge, y, nrow_train, nrow_test


def main(start_time):
    from time import gmtime, strftime
    print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))

    sparse_merge, y, nrow_train, nrow_test = construct_features(start_time)

    # Remove features with document frequency <=1
    print(sparse_merge.shape)
    mask = np.array(np.clip(sparse_merge.getnnz(axis=0) - 3, 0, 1), dtype=bool)
    sparse_merge = sparse_merge[:, mask]
    X = sparse_merge[:nrow_train]
    X_test = sparse_merge[nrow_test:]
    print('X size:', X.shape)
    print('X_test size:', X_test.shape)
    print(sparse_merge.shape)

    gc.collect()
    if develop:
        train_X, valid_X, train_y, valid_y = train_test_split(X, y, train_size=train_set_ratio, random_state=split_seed)
    else:
        train_X, train_y = X, y
    del X, y; gc.collect()

    # FM_FTRL
    model = FM_FTRL(alpha=0.01, beta=0.01, L1=0.00001, L2=0.1, 
                    D=sparse_merge.shape[1], alpha_fm=0.01, L2_fm=0.0,
                    init_fm=0.01, D_fm=200, e_noise=0.0001, iters=17, 
                    inv_link="identity", threads=4)
    model.fit(train_X, train_y)
    print('[{}] Train ridge v2 completed'.format(time.time() - start_time))
    if develop:
        val_predsFM = np.expm1(model.predict(X=valid_X))
        print("FM_FTRL dev RMSLE:", rmsle(np.expm1(valid_y), val_predsFM))
    predsFM = np.expm1(model.predict(X_test))
    print('[{}] Predict FM_FTRL completed'.format(time.time() - start_time))

    # delete
    del train_X, train_y, sparse_merge
    if develop:
        del valid_X, X, y
    gc.collect()

    if develop:
        results = val_predsFM, predsFM, valid_y
    else:
        results = predsFM

    return results

results_linear_models = main(start_time)
# combination
if develop:
    val_predsFM, predsFM, valid_y = results_linear_models
    # validation sizeof two models still not the same yet
    #val_preds_ensemble = val_preds_rnn.reshape(val_preds_rnn.shape[0],) * .8 + val_predsFM * .2
    #print("ensemble model validation RMSLE: ", rmsle(np.expm1(valid_y).as_matrix(), val_preds_ensemble))
    preds_ensemble = preds_rnn.reshape(preds_rnn.shape[0],) * .6 + predsFM * .4
else:
    predsFM = results_linear_models
    preds_ensemble = preds_rnn.reshape(preds_rnn.shape[0],) * .6 + predsFM * .4
# submit
submission = ids
submission["price"] = preds_ensemble
submission.to_csv("ensemble_rnn_wordbatch_ridge_submission_v10.csv", index=False)
print('[{}] finish everything'.format(time.time()-start_time))
