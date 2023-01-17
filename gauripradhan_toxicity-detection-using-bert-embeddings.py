# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

import os

import gc

import re

import numpy as np 

import pandas as pd 

from tqdm import tqdm, trange

import pickle

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import train_test_split

# pytorch bert imports

from pytorch_pretrained_bert import BertTokenizer

from pytorch_pretrained_bert.modeling import BertModel

# keras imports

from keras.utils import np_utils

from keras.preprocessing import text, sequence

from keras.layers import CuDNNLSTM, Activation, Dense, Dropout, Input, Embedding, concatenate, Bidirectional

from keras.optimizers import Adam

from keras.models import Sequential, Model

from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler

from keras.layers import SpatialDropout1D, Dropout, add, GlobalMaxPooling1D, GlobalAveragePooling1D

from keras.losses import binary_crossentropy

from keras import backend as K

import keras.layers as L

from keras.engine.topology import Layer

from keras import initializers, regularizers, constraints, optimizers, layers
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
BERT_PRETRAINED_DIR = '../input/bert-base-uncased-model/'

INPUT_DIR = '../input/jigsaw-unintended-bias-in-toxicity-classification/'

BERT_VOCAB_DIR = '../input/bert-base-uncased-vocab-file/vocab.txt'

MAX_LENGTH = 250
# Getting the bert encoded training and test data

train_data = pd.read_csv(INPUT_DIR + 'train.csv')

test_data = pd.read_csv(INPUT_DIR + 'test.csv')
# Feature Engineering for the training data

regex = re.compile('[@_!#$%^&*()<>?/\|}{~:]')

train_data['capitals'] = train_data['comment_text'].apply(lambda x: sum(1 for c in x if c.isupper()))

train_data['exclamation_points'] = train_data['comment_text'].apply(lambda x: len(regex.findall(x)))

train_data['total_length'] = train_data['comment_text'].apply(len)



# Feature Engineering for the test data

test_data['capitals'] = test_data['comment_text'].apply(lambda x: sum(1 for c in x if c.isupper()))

test_data['exclamation_points'] = test_data['comment_text'].apply(lambda x: len(regex.findall(x)))

test_data['total_length'] = test_data['comment_text'].apply(len)
new_features = ['capitals','exclamation_points','total_length']

identity_columns = ['male','female','homosexual_gay_or_lesbian','christian','jewish','muslim',

                    'black','white','psychiatric_or_mental_illness']
# Customizing the weights

y_ids= (train_data[identity_columns] >= 0.5).astype(int).values

# Overall

weights = np.ones((len(train_data),)) / 4

# Subgroup

weights += (train_data[identity_columns].fillna(0).values>=0.5).sum(axis=1).astype(bool).astype(np.int) / 4

# Background Positive, Subgroup Negative

weights += (( (train_data['target'].values>=0.5).astype(bool).astype(np.int) +

   (train_data[identity_columns].fillna(0).values<0.5).sum(axis=1).astype(bool).astype(np.int) ) > 1 ).astype(bool).astype(np.int) / 4

# Background Negative, Subgroup Positive

weights += (( (train_data['target'].values<0.5).astype(bool).astype(np.int) +

   (train_data[identity_columns].fillna(0).values>=0.5).sum(axis=1).astype(bool).astype(np.int) ) > 1 ).astype(bool).astype(np.int) / 4

loss_weight = 1.0 / weights.mean()



y_train = np.vstack([(train_data['target'].values>=0.5).astype(np.int),weights]).T

y_aux_train = train_data[['target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']].values

# Conversion of continuous target columns to categorical

for column in identity_columns + ['target']:

    train_data[column]= np.where(train_data[column] >= 0.5, True, False)
def nlp_preprocessing(text):

    filter_char = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\r“”’\'∞θ÷α•à−β∅³π‘₹´°£€\×™√²—'

    text = text.lower()

    text = text.replace(filter_char,'')

    text = text.replace('[^a-zA-Z0-9 ]', '')

    return text
train_data['comment_text'] = train_data['comment_text'].apply(nlp_preprocessing)

test_data['comment_text'] = test_data['comment_text'].apply(nlp_preprocessing)
# Initialising BERT tokenizer

tokenizer = BertTokenizer(vocab_file=BERT_VOCAB_DIR)

def tokenization(row):

    row = tokenizer.tokenize(row)

    row = tokenizer.convert_tokens_to_ids(row)

    return row
train_data['comment_text'] = train_data['comment_text'].apply(tokenization)

test_data['comment_text'] = test_data['comment_text'].apply(tokenization)
def string_ids(doc):

    doc = [str(i) for i in doc]

    return ' '.join(doc)

train_data['comment_text'] = train_data['comment_text'].apply(string_ids)

test_data['comment_text'] = test_data['comment_text'].apply(string_ids)
x_train = np.zeros((train_data.shape[0],MAX_LENGTH),dtype=np.int)



for i,ids in tqdm(enumerate(list(train_data['comment_text']))):

    input_ids = [int(i) for i in ids.split()[:MAX_LENGTH]]

    inp_len = len(input_ids)

    x_train[i,:inp_len] = np.array(input_ids)

    

x_test = np.zeros((test_data.shape[0],MAX_LENGTH),dtype=np.int)



for i,ids in tqdm(enumerate(list(test_data['comment_text']))):



    input_ids = [int(i) for i in ids.split()[:MAX_LENGTH]]

    inp_len = len(input_ids)

    x_test[i,:inp_len] = np.array(input_ids)

    

with open('temporary.pickle', mode='wb') as f:

    pickle.dump(x_test, f) # use temporary file to reduce memory



# Removing extra variables to free up the memory

del x_test

del test_data

del train_data



gc.collect()
def custom_loss_func(y_true, y_preds):

    loss = binary_crossentropy(K.reshape(y_true[:,0],(-1,1)), y_preds) * y_true[:,1]

    return loss
def get_bert_embed_matrix():

    bert = BertModel.from_pretrained(BERT_PRETRAINED_DIR)

    bert_embeddings = list(bert.children())[0]

    bert_word_embeddings = list(bert_embeddings.children())[0]

    mat = bert_word_embeddings.weight.data.numpy()

    return mat
embedding_matrix = get_bert_embed_matrix()
def build_model(embedding_matrix, num_aux_targets, loss_weight):

    '''

    credits go to: https://www.kaggle.com/thousandvoices/simple-lstm/

    '''

    words = Input(shape=(MAX_LENGTH,))

    x = Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=False)(words)

    x = SpatialDropout1D(0.5)(x)

    x = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True))(x)

    x = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True))(x)



    hidden = concatenate([GlobalMaxPooling1D()(x),GlobalAveragePooling1D()(x),])

    hidden = add([hidden, Dense(HIDDEN_UNITS, activation='relu')(hidden)])

    hidden = add([hidden, Dense(HIDDEN_UNITS, activation='relu')(hidden)])

    result = Dense(1, activation='sigmoid')(hidden)

    aux_result = Dense(num_aux_targets, activation='sigmoid')(hidden)

    

    model = Model(inputs=words, outputs=[result, aux_result])

    model.compile(loss=[custom_loss_func,'binary_crossentropy'], loss_weights=[loss_weight, 1.0],

                  optimizer=Adam(lr = 0.001))



    return model
tr_idx, val_idx = train_test_split(list(range(len(x_train))) ,test_size = 0.05, random_state = 42)
epochs = 5

LSTM_UNITS = 128

HIDDEN_UNITS = 4 * LSTM_UNITS

model_predictions = []

model_val_preds = []

weights = []



# Model Training and Prediction Phase

model = build_model(embedding_matrix, y_aux_train.shape[-1],loss_weight)

for epoch in range(epochs):

    model.fit(x_train[tr_idx],[y_train[tr_idx], y_aux_train[tr_idx]],

              validation_data = (x_train[val_idx],[y_train[val_idx], y_aux_train[val_idx]]),

              batch_size=512,

              epochs=1,

              verbose=1,

              callbacks=[LearningRateScheduler(lambda epoch: 1e-3 * (0.6 ** epoch))])

    with open('temporary.pickle', mode='rb') as f:

        x_test = pickle.load(f) 

    model_predictions.append(model.predict(x_test, batch_size=2048)[0].flatten())

    model_val_preds.append(model.predict(x_train[val_idx], batch_size=2048)[0].flatten())

    del x_test

    gc.collect()

    weights.append(2 ** epoch)

del model

gc.collect()
val_preds = np.average(model_val_preds, weights = weights, axis = 0)
""" Following section is drawn from a set of functions used on https://www.kaggle.com/christofhenkel/bert-embeddings-lstm/ """



from sklearn.metrics import roc_auc_score



def get_s_auc(y_true,y_pred,y_identity):

    mask = y_identity==1

    try:

        s_auc = roc_auc_score(y_true[mask],y_pred[mask])

    except:

        s_auc = 1

    return s_auc



def get_bspn_auc(y_true,y_pred,y_identity):

    mask = (y_identity==1) & (y_true==1) | (y_identity==0) & (y_true==0)

    try:

        bspn_auc = roc_auc_score(y_true[mask],y_pred[mask])

    except:

        bspn_auc = 1

    return bspn_auc



def get_bpsn_auc(y_true,y_pred,y_identity):

    mask = (y_identity==1) & (y_true==0) | (y_identity==0) & (y_true==1)

    try:

        bpsn_auc = roc_auc_score(y_true[mask],y_pred[mask])

    except:

        bpsn_auc = 1

    return bpsn_auc



def get_total_auc(y_true,y_pred,y_identities):

    N = y_identities.shape[1]

    

    saucs = np.array([get_s_auc(y_true,y_pred,y_identities[:,i]) for i in range(N)])

    bpsns = np.array([get_bpsn_auc(y_true,y_pred,y_identities[:,i]) for i in range(N)])

    bspns = np.array([get_bspn_auc(y_true,y_pred,y_identities[:,i]) for i in range(N)])



    M_s_auc = np.power(np.mean(np.power(saucs, -5)),1/-5)

    M_bpsns_auc = np.power(np.mean(np.power(bpsns, -5)),1/-5)

    M_bspns_auc = np.power(np.mean(np.power(bspns, -5)),1/-5)

    r_auc = roc_auc_score(y_true,y_pred)

    

    total_auc = M_s_auc + M_bpsns_auc + M_bspns_auc + r_auc

    total_auc/= 4



    return total_auc



get_total_auc(y_train[val_idx][:,0],val_preds,y_ids[val_idx])
# Calculate average predictions for the model

predictions = np.average(model_predictions, weights=weights, axis=0)



df_submission = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv')

df_submission.drop(['comment_text'],axis = 1, inplace = True)

df_submission['prediction'] = predictions

df_submission.to_csv('submission.csv', index=False)