import pandas as pd

import numpy as np

import matplotlib

matplotlib.use('nbagg')

from collections import Counter

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer



import matplotlib.pyplot as plt

plt.rcParams.update({'figure.max_open_warning': 0})

from sklearn import preprocessing

from tqdm import tqdm

import seaborn as sns

sns.set_style('whitegrid')

import os

from sklearn.model_selection import train_test_split

from nltk.corpus import stopwords

from nltk.stem import PorterStemmer

from bs4 import BeautifulSoup

import re

import scipy

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.preprocessing import LabelBinarizer

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Input, Dropout, Dense, concatenate, GRU, Embedding, Flatten, Activation

from keras.optimizers import Adam

from keras.models import Model

from keras import backend as K



from sklearn.metrics import mean_absolute_error

from sklearn.metrics import mean_squared_error

import math

import joblib
print("Reading Data")



train = pd.read_csv('../input/mercari-price-suggestion-challenge/train.tsv', sep='\t')

test = pd.read_csv('../input/mercari-price-suggestion-challenge/test_stg2.tsv', sep='\t')



print("Shape of train data: ",train.shape)

print("Shape of test data: ",test.shape)
y_train = np.log1p(train["price"])
NUM_BRANDS = 2500

NAME_MIN_DF = 10

MAX_FEAT_DESCP = 50000
print('No of duplicates in train: {}'.format(sum(train.duplicated())))

print('No of duplicates in test : {}'.format(sum(test.duplicated())))
train.isnull().any()
print('We have {} NaN/Null values in train'.format(train.isnull().values.sum()))

print('We have {} NaN/Null values in test'.format(test.isnull().values.sum()))
train["category_name"] = train["category_name"].fillna("Other").astype("category")

train["brand_name"] = train["brand_name"].fillna("unknown")



test["category_name"] = test["category_name"].fillna("Other").astype("category")

test["brand_name"] = test["brand_name"].fillna("unknown")



top_brands = train["brand_name"].value_counts().index[:NUM_BRANDS]

train.loc[~train["brand_name"].isin(top_brands), "brand_name"] = "Other"

test.loc[~test["brand_name"].isin(top_brands), "brand_name"] = "Other"



train["item_description"] = train["item_description"].fillna("None")

train["brand_name"] = train["brand_name"].astype("category")



test["item_description"] = test["item_description"].fillna("None")

test["brand_name"] = test["brand_name"].astype("category")
train_cond_id = Counter(list(train['item_condition_id']))

test_cond_id = Counter(list(test['item_condition_id']))



fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,4))



ax1.bar(train_cond_id.keys(), train_cond_id.values(), width=0.2, align='edge', label='Train')

ax1.set_xticks([1,2,3,4,5])

ax1.set_xlabel('item_condition_id')

ax1.legend()



ax2.bar(test_cond_id.keys(), test_cond_id.values(), width=-0.2, align='edge', label='Test')

ax2.set_xticks([1,2,3,4,5])

ax2.set_xlabel('item_condition_id')

ax2.legend()



fig.show()
print(train['category_name'].describe())

category_nam = Counter(list(train['category_name']))
print("Top 15 category in train data: ")

category_nam.most_common(15)
print(test['category_name'].describe())

category_nam = Counter(list(test['category_name']))
print("Top 15 category in test data: ")

category_nam.most_common(15)
print(train['brand_name'].describe())

brand_nam = Counter(list(train['brand_name']))

print("Top 15 brands in train data: ")

brand_nam.most_common(15)
print(test['brand_name'].describe())

brand_nam = Counter(list(test['brand_name']))

print("Top 15 brands in test data: ")

brand_nam.most_common(15)
train.price.describe()
fig, ax = plt.subplots( figsize = (10, 5))

ax.hist(train.price, bins = 100, color = "blue")

ax.set_title("\n \n  Histogram ", fontsize = 15)

ax.set_xlabel(" Price", fontsize = 10)

plt.title("Distribution of Price")

plt.show()
train_ship = Counter(list(train['shipping']))

test_ship = Counter(list(test['shipping']))



fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,4))



ax1.bar(train_ship.keys(), train_ship.values(), width=0.1, align='edge', label='Train')

ax1.set_xticks([0,1])

ax1.set_xlabel('shipping')

ax1.legend()



ax2.bar(test_ship.keys(), test_ship.values(), width=-0.1, align='edge', label='Test')

ax2.set_xticks([1,0])

ax2.set_xlabel('shipping')

ax2.legend()



fig.show()
print("Encodings")

count_nm = CountVectorizer(min_df=NAME_MIN_DF)

train_name_vec = count_nm.fit_transform(train["name"])

test_name_vec = count_nm.transform(test["name"])

print("Shape of train Name feature: ",train_name_vec.shape)

print("Shape of test Name feature: ",test_name_vec.shape)
print("Descp encoders")

count_desc = TfidfVectorizer(max_features = MAX_FEAT_DESCP, 

                              ngram_range = (1,3),

                              stop_words = "english")

train_desc_vec = count_desc.fit_transform(train["item_description"])

test_desc_vec = count_desc.transform(test["item_description"])

print("Shape of train Name feature: ",train_desc_vec.shape)

print("Shape of test Name feature: ",test_desc_vec.shape)
print("Category Encoders")

unique_categories = pd.Series("/".join(train["category_name"].unique().astype("str")).split("/")).unique()

count_category = CountVectorizer()

encoder_cat_train = count_category.fit_transform(train["category_name"])

encoder_cat_test= count_category.transform(test["category_name"])
print(encoder_cat_train.shape)

print(encoder_cat_test.shape)
from sklearn.preprocessing import LabelBinarizer



print("Brand encoders")

vect_brand = LabelBinarizer(sparse_output=True)



encoder_brnd_train = vect_brand.fit_transform(train["brand_name"])

encoder_brnd_test= vect_brand.transform(test["brand_name"])
print(encoder_brnd_train.shape)

print(encoder_brnd_test.shape)
X_train = scipy.sparse.hstack((

                         train_desc_vec,

                         encoder_brnd_train,

                         encoder_cat_train,

                         train_name_vec,

                         np.array(train['item_condition_id']).reshape(-1,1),

                         np.array(train['shipping']).reshape(-1,1)

                        )).tocsr()

print(X_train.shape)
X_test = scipy.sparse.hstack((

                         test_desc_vec,

                         encoder_brnd_test,

                         encoder_cat_test,

                         test_name_vec,

                         np.array(test['item_condition_id']).reshape(-1,1),

                         np.array(test['shipping']).reshape(-1,1)

)).tocsr()

print(X_test.shape)
from xgboost import XGBRegressor

# from sklearn.model_selection import GridSearchCV



# params = { 

#           'gamma':[i/10.0 for i in range(3,8,2)],  

#           'max_depth': [4,8,16]}



# xgb = XGBRegressor() 



# grid = GridSearchCV(estimator=xgb, param_grid=params, n_jobs=-1, cv=2, verbose=3)

# grid.fit(X_train, y_train)

# print("Best estimator : ", grid.best_estimator_)

# print("Best Score : ", grid.best_score_)
xgb = XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,

             colsample_bytree=1, gamma=0.7, learning_rate=0.1, max_delta_step=0,

             max_depth=16, min_child_weight=1, missing=None, n_estimators=100,

             n_jobs=-1, random_state=0, reg_alpha=0, reg_lambda=1, scale_pos_weight=1, 

             seed=None, silent=True, subsample=1)



print("Fitting Model 1")

xgb.fit(X_train, y_train)
y_pred = xgb.predict(X_test)    
from sklearn.linear_model import RidgeCV



model = RidgeCV(fit_intercept=True, alphas=[5.0], normalize=False, cv = 2, scoring='neg_mean_squared_error')





print("Fitting Model")

model.fit(X_train, y_train)
preds = model.predict(X_test)
from sklearn.preprocessing import LabelBinarizer, LabelEncoder

from keras.preprocessing.text import Tokenizer
full_df = pd.concat([train, test])
print("Processing categorical data...")

le = LabelEncoder()



le.fit(full_df.category_name)

full_df.category_name = le.transform(full_df.category_name)



le.fit(full_df.brand_name)

full_df.brand_name = le.transform(full_df.brand_name)



del le
print("Transforming text data to sequences...")

raw_text = np.hstack([full_df.item_description.str.lower(), full_df.name.str.lower()])



print("   Fitting tokenizer...")

tok_raw = Tokenizer()

tok_raw.fit_on_texts(raw_text)



print("   Transforming text to sequences...")

full_df['seq_item_description'] = tok_raw.texts_to_sequences(full_df.item_description.str.lower())

full_df['seq_name'] = tok_raw.texts_to_sequences(full_df.name.str.lower())



del tok_raw
# Define constants to use when define RNN model

MAX_NAME_SEQ = 10

MAX_ITEM_DESC_SEQ = 75

MAX_TEXT = np.max([

    np.max(full_df.seq_name.max()),

    np.max(full_df.seq_item_description.max()),

]) + 4

MAX_CATEGORY = np.max(full_df.category_name.max()) + 1

MAX_BRAND = np.max(full_df.brand_name.max()) + 1

MAX_CONDITION = np.max(full_df.item_condition_id.max()) + 1
def get_keras_data(df):

    X = {

        'name': pad_sequences(df.seq_name, maxlen=MAX_NAME_SEQ),

        'item_desc': pad_sequences(df.seq_item_description, maxlen=MAX_ITEM_DESC_SEQ),

        'brand_name': np.array(df.brand_name),

        'category_name': np.array(df.category_name),

        'item_condition': np.array(df.item_condition_id),

        'num_vars': np.array(df[["shipping"]]),

    }

    return X

# Calculate number of train/dev/test examples.

n_trains = train.shape[0]

n_tests = test.shape[0]



train = full_df[:n_trains]

test = full_df[n_trains:]



X_train = get_keras_data(train)

X_test = get_keras_data(test)
from keras import optimizers

def new_rnn_model(lr=0.001, decay=0.0):    

    # Inputs

    name = Input(shape=[X_train["name"].shape[1]], name="name")

    item_desc = Input(shape=[X_train["item_desc"].shape[1]], name="item_desc")

    brand_name = Input(shape=[1], name="brand_name")

    category_name = Input(shape=[1], name="category_name")

    item_condition = Input(shape=[1], name="item_condition")

    num_vars = Input(shape=[X_train["num_vars"].shape[1]], name="num_vars")



    # Embeddings layers

    emb_name = Embedding(MAX_TEXT, 20)(name)

    emb_item_desc = Embedding(MAX_TEXT, 60)(item_desc)

    emb_brand_name = Embedding(MAX_BRAND, 10)(brand_name)

    emb_category_name = Embedding(MAX_CATEGORY, 10)(category_name)



    # rnn layers

    rnn_layer1 = GRU(16) (emb_item_desc)

    rnn_layer2 = GRU(8) (emb_name)



    # main layers

    main_l = concatenate([

        Flatten() (emb_brand_name),

        Flatten() (emb_category_name),

        item_condition,

        rnn_layer1,

        rnn_layer2,

        num_vars,

    ])



    main_l = Dense(256)(main_l)

    main_l = Activation('elu')(main_l)



    main_l = Dense(128)(main_l)

    main_l = Activation('elu')(main_l)



    main_l = Dense(64)(main_l)

    main_l = Activation('elu')(main_l)



    # the output layer.

    output = Dense(1, activation="linear") (main_l)



    model = Model([name, item_desc, brand_name , category_name, item_condition, num_vars], output)



    optimizer = optimizers.Adam(lr=lr, decay=decay)

    model.compile(loss="mse", optimizer=optimizer)



    return model



model = new_rnn_model()

model.summary()

del model
# Set hyper parameters for the model.

BATCH_SIZE = 1024

epochs = 2



# Calculate learning rate decay.

exp_decay = lambda init, fin, steps: (init/fin)**(1/(steps-1)) - 1

steps = int(n_trains / BATCH_SIZE) * epochs

lr_init, lr_fin = 0.007, 0.0005

lr_decay = exp_decay(lr_init, lr_fin, steps)



rnn_model = new_rnn_model(lr=lr_init, decay=lr_decay)



print("Fitting RNN model to training examples...")

rnn_model.fit(

        X_train, y_train, epochs=epochs, batch_size=BATCH_SIZE, verbose=2)
preds = rnn_model.predict(X_test, batch_size=BATCH_SIZE)
test["price"] = np.expm1(preds)

test["test_id"] = pd.to_numeric(test["test_id"], downcast='integer')

test[["test_id", "price"]].to_csv("submission.csv", index = False)
res = pd.read_csv("submission.csv")

res.head()
type(res["test_id"][0])
from prettytable import PrettyTable

table = PrettyTable()

table.title = "Comparison of Models"

table.field_names = [ "Model"," RMLSE "]

table.add_row(["XGBRegressor","0.52"])

table.add_row(["RidgeCV Regressor","0.46"])

table.add_row(["RNN Model","0.43"])

print(table)