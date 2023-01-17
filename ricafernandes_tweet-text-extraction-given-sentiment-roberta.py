import math
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, Conv1D, Flatten, LeakyReLU, Activation
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import StratifiedKFold
import transformers
from transformers import RobertaConfig, TFRobertaModel
import tokenizers

print('TF version: ', tf.__version__)
train = pd.read_csv('../input/tweet-sentiment-extraction/train.csv').fillna('')
print(train.info())
# Maximum length
lengths = train['text'].apply(lambda x: len(x)).tolist()
max(lengths)
class config:
    MAX_LEN = 141
    PAD_ID = 1
    PATH = '../input/tf-roberta/'
    tokenizer = tokenizers.ByteLevelBPETokenizer(
        vocab_file = PATH+'vocab-roberta-base.json',
        merges_file = PATH+'merges-roberta-base.txt',
        lowercase = True,
        add_prefix_space = True
    )
    sentiment_id = {'positive': 1313, 'negative': 2430, 'neutral': 7974}
    n_splits = 5
    seed = 42
    epochs = 3
    tf.random.set_seed(seed)
    np.random.seed(seed)
    label_smoothing = 0.1
    batch_size = 32
ct = train.shape[0]
input_ids = np.ones((ct, config.MAX_LEN), dtype='int32')
attention_mask = np.zeros((ct, config.MAX_LEN), dtype='int32')
token_type_ids = np.zeros((ct, config.MAX_LEN), dtype='int32')
start_tokens = np.zeros((ct, config.MAX_LEN), dtype='int32')
end_tokens = np.zeros((ct, config.MAX_LEN), dtype='int32')

for k in range(train.shape[0]):
    # Selected text masking
    text1 = " " + " ".join(train.loc[k, 'text'].split())
    text2 = " ".join(train.loc[k, 'selected_text'].split())
    
    selected_idx = text1.find(text2)
    is_selected = np.zeros((len(text1)))
    is_selected[selected_idx:selected_idx+len(text2)] = 1
    if text1[selected_idx-1] == " ":
        is_selected[selected_idx-1] = 1
        
    enc = config.tokenizer.encode(text1)
    
    # IDs start and end offsets (A.K.A.: indexes)
    offsets = []
    idx = 0
    for t in enc.ids:
        w = config.tokenizer.decode([t])
        offsets.append((idx, idx+len(w)))
        idx += len(w)
        
    # START and END tokens
    toks = []
    for i, (a, b) in enumerate(offsets):
        verification_sum = np.sum(is_selected[a:b])
        if verification_sum > 0:
            toks.append(i)
            
    sentiment_tok = config.sentiment_id[train.loc[k, 'sentiment']]
    input_ids[k, :len(enc.ids)+3] = [0, sentiment_tok] + enc.ids + [2]
    attention_mask[k, :len(enc.ids)+3] = 1
    if len(toks) > 0:
        start_tokens[k, toks[0]+2] = 1
        end_tokens[k, toks[-1]+2]  = 1
test = pd.read_csv('../input/tweet-sentiment-extraction/test.csv').fillna('')

ct = test.shape[0]
input_ids_test = np.ones((ct, config.MAX_LEN), dtype='int32')
attention_mask_test = np.zeros((ct, config.MAX_LEN), dtype='int32')
token_type_ids_test = np.zeros((ct, config.MAX_LEN), dtype='int32')

for k in range(ct):
    
    # Input IDs
    text1 = " " + " ".join(test.loc[k, 'text'].split())
    enc = config.tokenizer.encode(text1)
    
    sentiment_tok = config.sentiment_id[test.loc[k, 'sentiment']]
    input_ids_test[k, :len(enc.ids)+5] = [0] + enc.ids + [2, 2] + [sentiment_tok] + [2]
    attention_mask_test[k, :len(enc.ids)+5] = 1

test.info()
print(test.shape)
import pickle

def save_weights(model, dst_fn):
    weights = model.get_weights()
    with open(dst_fn, 'wb') as f:
        pickle.dump(weights, f)
        
def load_weights(model, weight_fn):
    with open(weight_fn, 'rb') as f:
        weights = pickle.load(f)
        
    model.set_weights(weights)
    return model

def loss_fn(y_true, y_pred):
    ll = tf.shape(y_pred)[1]
    y_true = y_true[:, :ll]
    loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=False, label_smoothing=config.label_smoothing)
    loss = tf.reduce_mean(loss)
    return loss

# from https://www.kaggle.com/cdeotte/tensorflow-roberta-0-705/data?#Load-Libraries,-Data,-Tokenizer
'''def build_model():
    ids = tf.keras.layers.Input((config.MAX_LEN,), dtype=tf.int32)
    att = tf.keras.layers.Input((config.MAX_LEN,), dtype=tf.int32)
    tok = tf.keras.layers.Input((config.MAX_LEN,), dtype=tf.int32)

    roberta_config = RobertaConfig.from_pretrained(config.PATH+'config-roberta-base.json')
    bert_model = TFRobertaModel.from_pretrained(config.PATH+
            'pretrained-roberta-base.h5',config=roberta_config)
    x = bert_model(ids,attention_mask=att,token_type_ids=tok)

    x1 = tf.keras.layers.Conv1D(1,1)(x[0])
    print(x1.shape)
    x1 = tf.keras.layers.Flatten()(x1)
    print(x1.shape)
    x1 = tf.keras.layers.Activation('softmax')(x1)
    print(x1.shape)
    
    x2 = tf.keras.layers.Conv1D(1,1)(x[0])
    x2 = tf.keras.layers.Flatten()(x2)
    x2 = tf.keras.layers.Activation('softmax')(x2)

    model = tf.keras.models.Model(inputs=[ids, att, tok], outputs=[x1,x2])
    optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    return model
'''

# from https://www.kaggle.com/khoongweihao/tse2020-roberta-cnn-random-seed-distribution?scriptVersionId=34603010
def build_model():
    ids = Input((config.MAX_LEN,), dtype=tf.int32)
    att = Input((config.MAX_LEN,), dtype=tf.int32)
    tok = Input((config.MAX_LEN,), dtype=tf.int32)
    padding = tf.cast(tf.equal(ids, config.PAD_ID), tf.int32)
    
    lens = config.MAX_LEN - tf.reduce_sum(padding, -1)
    max_len = tf.reduce_max(lens)
    ids_ = ids[:, :max_len]
    att_ = att[:, :max_len]
    tok_ = tok[:, :max_len]
    
    roberta_config = RobertaConfig.from_pretrained(config.PATH+'config-roberta-base.json')
    bert_model = TFRobertaModel.from_pretrained(config.PATH+'pretrained-roberta-base.h5', config=roberta_config)
    
    x = bert_model(ids_, attention_mask=att_, token_type_ids=tok_) #for non-padded model: (ids, attention_mask=att, token_type_ids=tok)
    #print(len(x))
    #x = tf.convert_to_tensor(x[0])
    #print(x.shape)
    #print(type(x))
    #print(type(x[0]))
    
    x1 = Dropout(0.15)(x[0])
    #print(x1.shape)
    x1 = Conv1D(768, 2, padding='same')(x1)
    #print(x1.shape)
    x1 = LeakyReLU()(x1)
    #print(x1.shape)
    x1 = Conv1D(64, 2, padding='same')(x1)
    #print(x1.shape)
    x1 = Dense(1)(x1)
    #print(x1.shape)
    x1 = Flatten()(x1)
    #print(x1.shape)
    x1 = Activation('softmax')(x1)
    #print(x1.shape)
    
    x2 = Dropout(0.15)(x[0])
    x2 = Conv1D(768, 2, padding='same')(x2)
    x2 = LeakyReLU()(x2)
    x2 = Conv1D(64, 2, padding='same')(x2)
    x2 = Dense(1)(x2)
    x2 = Flatten()(x2)
    x2 = Activation('softmax')(x2)
    
    model = Model(inputs=[ids, att, tok], outputs=[x1, x2])
    optimizer = Adam(learning_rate=3e-5)
    model.compile(loss=loss_fn,
                  optimizer=optimizer)
  
    x1_padded = tf.pad(x1, [[0, 0], [0, config.MAX_LEN - max_len]], constant_values=0.)
    x2_padded = tf.pad(x2, [[0, 0], [0, config.MAX_LEN - max_len]], constant_values=0.)
    
    padded_model = tf.keras.models.Model(inputs=[ids, att, tok], outputs=[x1_padded, x2_padded])
    
    return model, padded_model
def jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    if(len(a)==0) & (len(b)==0):
        return 0.5
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))
%%time
jac = []
VER = 'v0'
DISPLAY = 1

oof_start = np.zeros((input_ids.shape[0], config.MAX_LEN))
oof_end = np.zeros((input_ids.shape[0], config.MAX_LEN))

preds_start_train = np.zeros((input_ids.shape[0], config.MAX_LEN))
preds_end_train = np.zeros((input_ids.shape[0], config.MAX_LEN))
preds_start = np.zeros((input_ids_test.shape[0], config.MAX_LEN))
preds_end = np.zeros((input_ids_test.shape[0], config.MAX_LEN))

skf = StratifiedKFold(n_splits=config.n_splits, shuffle=True, random_state=config.seed)
for fold, (idxText, idxSentValue) in enumerate(skf.split(input_ids, train.sentiment.values)):
    print('\n')
    print('Fold', (fold+1))
    print('\n')

    K.clear_session()
    model, padded_model = build_model()

    sv = tf.keras.callbacks.ModelCheckpoint(
        '%s-roberta-%i.h5'%(VER,fold), monitor='val_loss', verbose=1, save_best_only=True,
        save_weights_only=True, mode='auto', save_freq='epoch')

    inputText = [input_ids[idxText,], attention_mask[idxText,], token_type_ids[idxText,]]
    targetText = [start_tokens[idxText,], end_tokens[idxText,]]

    inputSentValue = [input_ids[idxSentValue,], attention_mask[idxSentValue,], token_type_ids[idxSentValue,]]
    targetSentValue = [start_tokens[idxSentValue,], end_tokens[idxSentValue,]]

    # Sorting validation data
    shuffleSentValue = np.int32(sorted(range(len(inputSentValue[0])), key=lambda k: (inputSentValue[0][k] == config.PAD_ID).sum(), reverse=True))
    inputSentValue = [arr[shuffleSentValue] for arr in inputSentValue]
    targetSentValue = [arr[shuffleSentValue] for arr in targetSentValue]

    weight_fn = '%s-roberta-%i.h5'%(VER,fold)
    
    for epoch in range(1, config.epochs + 1):
        print('\n')
        print('Preparing data.')
        print('\n')
        # add random numbers in order to avoid having the same order in each epoch
        shuffleText = np.int32(sorted(range(len(inputText[0])), key=lambda k: (inputText[0][k] == config.PAD_ID).sum() + np.random.randint(-3, 3), reverse=True))
        
        # shuffle in batches, otherwise short batches will always come in the beginning of each epoch
        num_batches = math.ceil(len(shuffleText) / config.batch_size)
        batch_idxs = np.random.permutation(num_batches)
        shuffleText_ = []
        for batch_idx in batch_idxs:
            shuffleText_.append(shuffleText[batch_idx * config.batch_size: (batch_idx + 1) * config.batch_size])
        shuffleText = np.concatenate(shuffleText_)
        
        # reorder the input data
        inputText = [arr[shuffleText] for arr in inputText]
        targetText = [arr[shuffleText] for arr in targetText]
        
        print('\n')
        print('Fitting the model')
        print('\n')
        #preds = padded_model.predict([input_ids_test,attention_mask_test,token_type_ids_t],verbose=DISPLAY)
        model.fit(inputText, targetText, epochs=config.epochs, initial_epoch=epoch - 1, 
                  batch_size=config.batch_size, verbose=DISPLAY, callbacks=[], 
                  validation_data=(inputSentValue, targetSentValue), shuffle=False) #don't shuffle in fit
        save_weights(model, weight_fn)
        
    print('\n')
    print('Loading model.')
    print('\n')
    #model.load_weights('%s-roberta-%i.h5'%(VER, fold))
    load_weights(model, weight_fn)

    print('\n')
    print('Predicting OOF.')
    print('\n')
    oof_start[idxSentValue,], oof_end[idxSentValue,] = padded_model.predict([input_ids[idxSentValue,], attention_mask[idxSentValue,], token_type_ids[idxSentValue,]], 
                                                                            verbose=DISPLAY)
    #oof_start[idxSentValue,], oof_end[idxSentValue,] = model.predict([input_ids[idxSentValue,], attention_mask[idxSentValue,], token_type_ids[idxSentValue,]], 
                                                                     #verbose=DISPLAY)
    
    #print('\n')
    #print('Predicting all Train for Outlier analysis.')
    #print('\n')
    #preds_train = padded_model.predict([input_ids, attention_mask, token_type_ids], verbose=DISPLAY)
    #preds_start_train += preds_train[0] / skf.n_splits
    #preds_end_train += preds_train[1] / skf.n_splits

    print('\n')
    print('Predicting test data.')
    print('\n')
    preds = padded_model.predict([input_ids_test, attention_mask_test, token_type_ids_test], verbose=DISPLAY)
    #preds = model.predict([input_ids_test, attention_mask_test, token_type_ids_test], verbose=DISPLAY)
    preds_start += preds[0] / skf.n_splits
    preds_end += preds[1] / skf.n_splits

    # display fold jaccard
    all = []
    for k in idxSentValue:
        a = np.argmax(oof_start[k,])
        b = np.argmax(oof_end[k,])

        if a > b:
            selected_text = train.loc[k, 'text']
        else:
            text1 = " " + " ".join(train.loc[k, 'text'].split())
            enc = config.tokenizer.encode(text1)
            selected_text = config.tokenizer.decode(enc.ids[a-2:b-1])
        all.append(jaccard(selected_text, train.loc[k, 'selected_text']))
    jac_score = np.mean(all)
    jac.append(jac_score)

    print('\n')
    print('>>>>> FOLD', (fold+1), ": \n\tJaccard = ", jac_score)
    print('\n')
print('Overall 5Fold Cross-Validation Jaccard score:', jac_score)
out_dir = '../output/model/'

all = []
for k in range(input_ids_test.shape[0]):
    a = np.argmax(preds_start[k, ])
    b = np.argmax(preds_end[k, ])
    
    if a > b:
        st = test.loc[k, 'text']
    else:
        text1 = " " + " ".join(test.loc[k, 'text'].split())
        enc = config.tokenizer.encode(text1)
        st = config.tokenizer.decode(enc.ids[a-2:b-1])
    
    all.append(st)
print(test.shape)
print(len(all))
test['selected_text'] = all
test[['textID', 'selected_text']].to_csv('submission.csv', index=False)
pd.set_option('max_colwidth', 60)
test[['textID', 'selected_text']].sample(25)
