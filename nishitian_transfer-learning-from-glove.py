import re
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dropout, Dense, concatenate, GRU, Embedding, Flatten, Activation, Bidirectional
from keras.optimizers import Adam, SGD
from keras.models import Model
from keras import backend as K

def rmsle(Y, Y_pred):
    # Y and Y_red have already been in log scale.
    assert Y.shape == Y_pred.shape
    return np.sqrt(np.mean(np.square(Y_pred - Y )))

train_df = pd.read_table('../input/mercari-price-suggestion-challenge/train.tsv')
test_df = pd.read_table('../input/mercari-price-suggestion-challenge/test.tsv')
# print(train_df.shape, test_df.shape)

def split_cat(text):
    try:
        return text.split("/")
    except:
        return ("No Label", "No Label", "No Label")


def fill_missing_values(df):
    df.category_name.fillna(value="Other", inplace=True)
    df.general_cat.fillna(value="Other", inplace=True)
    df.subcat_1.fillna(value="Other", inplace=True)
    df.subcat_2.fillna(value="Other", inplace=True)
    df.brand_name.fillna(value="missing", inplace=True)
    df.item_description.fillna(value="None", inplace=True)
    return df

train_df['general_cat'], train_df['subcat_1'], train_df['subcat_2'] = \
    zip(*train_df['category_name'].apply(lambda x: split_cat(x)))

test_df['general_cat'], test_df['subcat_1'], test_df['subcat_2'] = \
    zip(*test_df['category_name'].apply(lambda x: split_cat(x)))

# train_df.drop('category_name', axis=1, inplace=True)
# test_df = test_df.drop('category_name', axis=1, inplace=True)

train_df = fill_missing_values(train_df)
test_df = fill_missing_values(test_df)
submission: pd.DataFrame = test_df[['test_id']]
test_df = test_df.drop("test_id",axis=1)

# Scale target variable to log.
train_df["target"] = np.log1p(train_df.price)

# Split training examples into train/dev examples.
train_df, dev_df = train_test_split(train_df, random_state=347, train_size=0.99)

Y_train = train_df.target.values.reshape(-11, 1)
Y_dev = dev_df.target.values.reshape(-1, 1)

# Calculate number of train/dev/test examples.
n_trains = train_df.shape[0]
n_devs = dev_df.shape[0]
n_tests = test_df.shape[0]
# print("Training on", n_trains, "examples")
# print("Validating on", n_devs, "examples")
# print("Testing on", n_tests, "examples")

full_df = pd.concat([train_df, dev_df, test_df])
full_df.head()
def preprocess(text):
    
    text = text.lower()
    
    bad_char_re = r"[^a-z1-9',.!;:/|$%&+-=()]"
    text = re.sub(bad_char_re," ",text)
    
    text = re.sub( r'([1-9])([a-z])', r'\1 \2', text)
    text = re.sub( r'([a-z])([1-9])', r'\1 \2', text)
    text = re.sub( r'([1-9])\'([1-9])', r' ', text)
    text = text.replace("' ", " ").replace("’ ", " ")\
        .replace("'re", " are").replace("’re", " are")\
        .replace("'ve", " have").replace("’ve", " have")\
        .replace("n't"," not").replace("n’t"," not")\
        .replace("'ll"," will").replace("’ll"," will")\
        .replace("it's","it is").replace("it’s","it is")\
        .replace("'d'"," had").replace("’d"," had")\
        .replace("'s"," 's").replace("’s"," 's")

    return text
full_df.item_description = [preprocess(a) for a in full_df.item_description.values]
full_df.general_cat = [preprocess(a) for a in full_df.general_cat.values]
full_df.subcat_1 = [preprocess(a) for a in full_df.subcat_1.values]
full_df.subcat_2 = [preprocess(a) for a in full_df.subcat_2.values]
full_df.name = [preprocess(a) for a in full_df.name.values]
full_df
embeddings_index_42b = dict()
f_42B = open('../input/glove-stanford/glove.42B.300d.txt')
for line in f_42B:
	values = line.split()
	word = values[0]
	coefs = np.asarray(values[1:], dtype='float32')
	embeddings_index_42b[word] = coefs
f_42B.close()
print('Loaded %s word vectors.' % len(embeddings_index_42b))
%%time
t = Tokenizer()
t.fit_on_texts(full_df.item_description)
full_df['seq_item_description'] = t.texts_to_sequences(full_df.item_description)
item_description_vocab_size = len(t.word_index) + 1
print("description vocab_size: ",item_description_vocab_size)
item_description_embedding_matrix = np.zeros((item_description_vocab_size, 300))
item_description_undefined_count = 0
item_description_defined_count = 0
for word, i in t.word_index.items():
	embedding_vector = embeddings_index_42b.get(word)
	if embedding_vector is not None:
		item_description_embedding_matrix[i] = embedding_vector
		item_description_defined_count += 1
	else:
		item_description_embedding_matrix[i] = np.random.rand(1,300)*6
		item_description_undefined_count += 1
print("Defined words:",item_description_defined_count,"Undefined words:",item_description_undefined_count)

t.fit_on_texts(full_df.name)
full_df['seq_name'] = t.texts_to_sequences(full_df.name)
name_vocab_size = len(t.word_index) + 1
print("name vocab_size: ",name_vocab_size)
name_embedding_matrix = np.zeros((name_vocab_size, 300))
name_undefined_count = 0
name_defined_count = 0
for word, i in t.word_index.items():
	embedding_vector = embeddings_index_42b.get(word)
	if embedding_vector is not None:
		name_embedding_matrix[i] = embedding_vector
		name_defined_count += 1
	else:
		name_embedding_matrix[i] = np.random.rand(1,300)*6
		name_undefined_count += 1
print("Defined words:",name_defined_count,"Undefined words:",name_undefined_count)

t.fit_on_texts(full_df.general_cat)
full_df['seq_general_cat'] = t.texts_to_sequences(full_df.general_cat)
general_cat_vocab_size = len(t.word_index) + 1
print("general_cat vocab_size: ",general_cat_vocab_size)
general_cat_embedding_matrix = np.zeros((general_cat_vocab_size, 300))
general_cat_undefined_count = 0
general_cat_defined_count = 0
for word, i in t.word_index.items():
	embedding_vector = embeddings_index_42b.get(word)
	if embedding_vector is not None:
		general_cat_embedding_matrix[i] = embedding_vector
		general_cat_defined_count += 1
	else:
		general_cat_embedding_matrix[i] = np.random.rand(1,300)*6
		general_cat_undefined_count += 1
print("Defined words:",general_cat_defined_count,"Undefined words:",general_cat_undefined_count)        

t.fit_on_texts(full_df.subcat_1)
full_df['seq_subcat_1'] = t.texts_to_sequences(full_df.subcat_1)
subcat_1_vocab_size = len(t.word_index) + 1
print("subcat_1 vocab_size: ",subcat_1_vocab_size)
subcat_1_embedding_matrix = np.zeros((subcat_1_vocab_size, 300))
subcat_1_undefined_count = 0
subcat_1_defined_count = 0
for word, i in t.word_index.items():
	embedding_vector = embeddings_index_42b.get(word)
	if embedding_vector is not None:
		subcat_1_embedding_matrix[i] = embedding_vector
		subcat_1_defined_count += 1
	else:
		subcat_1_embedding_matrix[i] = np.random.rand(1,300)*6
		subcat_1_undefined_count += 1
print("Defined words:",subcat_1_defined_count,"Undefined words:",subcat_1_undefined_count)    

t.fit_on_texts(full_df.subcat_2)
full_df['seq_subcat_2'] = t.texts_to_sequences(full_df.subcat_2)
subcat_2_vocab_size = len(t.word_index) + 1
print("subcat_2 vocab_size: ",subcat_2_vocab_size)
subcat_2_embedding_matrix = np.zeros((subcat_2_vocab_size, 300))
subcat_2_undefined_count = 0
subcat_2_defined_count = 0
for word, i in t.word_index.items():
	embedding_vector = embeddings_index_42b.get(word)
	if embedding_vector is not None:
		subcat_2_embedding_matrix[i] = embedding_vector
		subcat_2_defined_count += 1
	else:
		subcat_2_embedding_matrix[i] = np.random.rand(1,300)*6
		subcat_2_undefined_count += 1
print("Defined words:",subcat_2_defined_count,"Undefined words:",subcat_2_undefined_count)    
max_seq_item_description_len = max([len(a) for a in full_df.seq_item_description])
max_seq_item_description_len = min(30, max_seq_item_description_len) #save memory
full_df.seq_item_description = [a[:max_seq_item_description_len] for a in full_df.seq_item_description]

max_seq_name_len = max([len(a) for a in full_df.seq_name])
max_seq_name_len = min(6, max_seq_name_len) #save memory
full_df.seq_name = [a[:max_seq_name_len] for a in full_df.seq_name]

max_seq_general_cat_len = max([len(a) for a in full_df.seq_general_cat])
max_seq_general_cat_len = min(6, max_seq_general_cat_len) #save memory
full_df.seq_general_cat = [a[:max_seq_general_cat_len] for a in full_df.seq_general_cat]

max_seq_subcat_1_len = max([len(a) for a in full_df.seq_subcat_1])
max_seq_subcat_1_len = min(5, max_seq_subcat_1_len) #save memory
full_df.seq_subcat_1 = [a[:max_seq_subcat_1_len] for a in full_df.seq_subcat_1]

max_seq_subcat_2_len = max([len(a) for a in full_df.seq_subcat_2])
max_seq_subcat_2_len = min(4, max_seq_subcat_2_len) #save memory
full_df.seq_subcat_2 = [a[:max_seq_subcat_2_len] for a in full_df.seq_subcat_2]
le = LabelEncoder()
le.fit(full_df.brand_name)
full_df.brand_name = le.transform(full_df.brand_name)
del le
MAX_BRAND = np.max(full_df.brand_name.max()) + 1
import gc
del embeddings_index_42b
del f_42B
del train_df
del dev_df
del test_df
gc.collect()
full_df = full_df.drop("item_description",axis=1)
full_df = full_df.drop("general_cat",axis=1)
full_df = full_df.drop("subcat_1",axis=1)
full_df = full_df.drop("subcat_2",axis=1)
full_df = full_df.drop("name",axis=1)
full_df = full_df.drop("category_name",axis=1)
full_df.head()
def get_keras_data(df):
    X = {
        'name': pad_sequences(df.seq_name, maxlen=max_seq_name_len,padding='post', truncating='post'),
        'item_desc': pad_sequences(df.seq_item_description, maxlen=max_seq_item_description_len,padding='post', truncating='post'),
        'brand_name': np.array(df.brand_name),
        # 'category_name': np.array(df.category_name),
        'general_cat': pad_sequences(df.seq_general_cat, maxlen=max_seq_general_cat_len,padding='post', truncating='post'),
        'subcat_1': pad_sequences(df.seq_subcat_1, maxlen=max_seq_subcat_1_len,padding='post', truncating='post'),
        'subcat_2': pad_sequences(df.seq_subcat_2, maxlen=max_seq_subcat_2_len,padding='post', truncating='post'),
        'item_condition': np.array(df.item_condition_id),
        'num_vars': np.array(df[["shipping"]]),
    }
    return X
train = full_df[:n_trains]
dev = full_df[n_trains:n_trains+n_devs]
test = full_df[n_trains+n_devs:]

X_train = get_keras_data(train)
X_dev = get_keras_data(dev)
X_test = get_keras_data(test)
del full_df
del train
del dev
del t
gc.collect()
def new_rnn_model(lr=0.01, decay=1e-6):    
    # Inputs
    name = Input(shape=[X_train["name"].shape[1]], name="name")
    item_desc = Input(shape=[X_train["item_desc"].shape[1]], name="item_desc")
    general_cat = Input(shape=[X_train["general_cat"].shape[1]], name="general_cat")
    subcat_1 = Input(shape=[X_train["subcat_1"].shape[1]], name="subcat_1")
    subcat_2 = Input(shape=[X_train["subcat_2"].shape[1]], name="subcat_2")
    brand_name = Input(shape=[1], name="brand_name")
    item_condition = Input(shape=[1], name="item_condition")
    num_vars = Input(shape=[X_train["num_vars"].shape[1]], name="num_vars")

    # Embeddings layers
    emb_name = Embedding(name_embedding_matrix.shape[0], 
                              name_embedding_matrix.shape[1],
                              weights=[name_embedding_matrix], 
                              input_length=max_seq_name_len, 
                              trainable=False)(name)
    emb_item_desc = Embedding(item_description_embedding_matrix.shape[0], 
                              item_description_embedding_matrix.shape[1],
                              weights=[item_description_embedding_matrix], 
                              input_length=max_seq_item_description_len, 
                              trainable=False)(item_desc)
    emb_general_cat = Embedding(general_cat_embedding_matrix.shape[0], 
                              general_cat_embedding_matrix.shape[1],
                              weights=[general_cat_embedding_matrix], 
                              input_length=max_seq_general_cat_len, 
                              trainable=False)(general_cat)
    emb_subcat_1 = Embedding(subcat_1_embedding_matrix.shape[0], 
                              subcat_1_embedding_matrix.shape[1],
                              weights=[subcat_1_embedding_matrix], 
                              input_length=max_seq_subcat_1_len, 
                              trainable=False)(subcat_1)
    emb_subcat_2 = Embedding(subcat_2_embedding_matrix.shape[0], 
                              subcat_2_embedding_matrix.shape[1],
                              weights=[subcat_2_embedding_matrix], 
                              input_length=max_seq_subcat_2_len, 
                              trainable=False)(subcat_2)
    emb_brand_name = Embedding(MAX_BRAND, 10)(brand_name)
    

    # rnn layers
    rnn_layer1 = Bidirectional(GRU(8, return_sequences=True)) (emb_item_desc)
    rnn_layer1 = GRU(8) (rnn_layer1)
    
    # main layers
    main_l = concatenate([
        Flatten() (emb_brand_name),
        Flatten() (emb_name),
        Flatten() (emb_general_cat),
        Flatten() (emb_subcat_1),
        Flatten() (emb_subcat_2),
        item_condition,
        rnn_layer1,
        num_vars
    ])
    
    main_l = Dense(64)(main_l)
    main_l = Activation('elu')(main_l)
    main_l = Dense(64)(main_l)
    main_l = Activation('elu')(main_l)

    # the output layer.
    output = Dense(1, activation="elu") (main_l) 

    model = Model([name, item_desc, brand_name , general_cat, subcat_1, subcat_2, item_condition, num_vars], output)

    # SGD momentum leads to wrong direction and increasing loss
    # Use tanh activation instead of "elu" or "relu" to avoid exploding loss to NaN
    # optimizer = SGD(lr=lr,momentum=0.9,decay=decay,nesterov=True)  
    
    optimizer = Adam(lr=lr,decay=decay)
    model.compile(loss="mse", optimizer=optimizer)

    return model
# Set hyper parameters for the model.
batch_sizes = [1024,512]
epochs = 1
# Calculate learning rate decay.
exp_decay = lambda init, fin, steps: (init/fin)**(1/(steps-1)) - 1
steps = 0
for batch_size in batch_sizes:
    steps += int(n_trains / batch_size) * epochs
lr_init, lr_fin = 0.01, 0.0005
lr_decay = exp_decay(lr_init, lr_fin, steps)
# X_train_small = {}
# for i in X_train:
#     X_train_small[i] = X_train[i][:1000]
# Y_train_small = Y_train[:1000]
# rnn_model = new_rnn_model(lr=lr_init, decay=lr_decay)
# rnn_model.fit(
#         X_train_small, Y_train_small, epochs=1, batch_size=256,
# #         validation_data=(X_dev, Y_dev), 
#     verbose=1
# )
# rnn_preds = rnn_model.predict(X_test_small, batch_size=1024, verbose=1)
# rnn_preds = np.expm1(rnn_preds)
# rnn_preds
# del rnn_model
# gc.collect()
rnn_model = new_rnn_model(lr=lr_init, decay=lr_decay)
for batch_size in batch_sizes:
    rnn_model.fit(
            X_train, Y_train, epochs=epochs, batch_size=batch_size,
            validation_data=(X_dev, Y_dev), verbose=1,
    )
rnn_preds = rnn_model.predict(X_test, batch_size=1024, verbose=1)
rnn_preds = np.expm1(rnn_preds)
submission.loc[:, 'price'] = rnn_preds
submission.loc[submission['price'] < 1.0, 'price'] = 1.0
submission.head()
submission.to_csv("submission_glove_transfer.csv", index=False)