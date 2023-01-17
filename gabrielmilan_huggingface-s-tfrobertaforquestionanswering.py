EPOCHS = 3

BATCH_SIZE = 32

LABEL_SMOOTHING = 0.1
import pandas as pd, numpy as np

import tensorflow as tf

import tensorflow.keras.backend as K

from sklearn.model_selection import StratifiedKFold

from transformers import *

import tokenizers

import math

import matplotlib.pyplot as plt

from lightgbm import LGBMRegressor
MAX_LEN = 96

PATH = '../input/tf-roberta/'

tokenizer = tokenizers.ByteLevelBPETokenizer(

    vocab_file=PATH+'vocab-roberta-base.json', 

    merges_file=PATH+'merges-roberta-base.txt', 

    lowercase=True,

    add_prefix_space=True

)

PAD_ID = 1

SEED = 777

tf.random.set_seed(SEED)

np.random.seed(SEED)

sentiment_id = {'positive': 1313, 'negative': 2430, 'neutral': 7974}

train = pd.read_csv('../input/tweet-sentiment-extraction/train.csv').fillna('')
ct = train.shape[0]

input_ids = np.ones((ct,MAX_LEN),dtype='int32')

attention_mask = np.zeros((ct,MAX_LEN),dtype='int32')

token_type_ids = np.zeros((ct,MAX_LEN),dtype='int32')

start_tokens = np.zeros((ct,MAX_LEN),dtype='int32')

end_tokens = np.zeros((ct,MAX_LEN),dtype='int32')



for k in range(train.shape[0]):

    

    # FIND OVERLAP

    text1 = " "+" ".join(train.loc[k,'text'].split())

    text2 = " ".join(train.loc[k,'selected_text'].split())

    idx = text1.find(text2)

    chars = np.zeros((len(text1)))

    chars[idx:idx+len(text2)]=1

    if text1[idx-1]==' ': chars[idx-1] = 1 

    enc = tokenizer.encode(text1) 

        

    # ID_OFFSETS

    offsets = []; idx=0

    for t in enc.ids:

        w = tokenizer.decode([t])

        offsets.append((idx,idx+len(w)))

        idx += len(w)

    

    # START END TOKENS

    toks = []

    for i,(a,b) in enumerate(offsets):

        sm = np.sum(chars[a:b])

        if sm>0: toks.append(i) 

        

    s_tok = sentiment_id[train.loc[k,'sentiment']]

    input_ids[k,:len(enc.ids)+5] = [0] + enc.ids + [2,2] + [s_tok] + [2]

    attention_mask[k,:len(enc.ids)+5] = 1

    if len(toks)>0:

        start_tokens[k,toks[0]+1] = 1

        end_tokens[k,toks[-1]+1] = 1
test = pd.read_csv('../input/tweet-sentiment-extraction/test.csv').fillna('')



ct = test.shape[0]

input_ids_t = np.ones((ct,MAX_LEN),dtype='int32')

attention_mask_t = np.zeros((ct,MAX_LEN),dtype='int32')

token_type_ids_t = np.zeros((ct,MAX_LEN),dtype='int32')



for k in range(test.shape[0]):

        

    # INPUT_IDS

    text1 = " "+" ".join(test.loc[k,'text'].split())

    enc = tokenizer.encode(text1)                

    s_tok = sentiment_id[test.loc[k,'sentiment']]

    input_ids_t[k,:len(enc.ids)+5] = [0] + enc.ids + [2,2] + [s_tok] + [2]

    attention_mask_t[k,:len(enc.ids)+5] = 1
import pickle



def scheduler(epoch, lr):

    if (epoch < 2):

        new_lr = lr

    else:

        new_lr = lr * 0.25

    print ("Setting learning rate to {:10.3E}".format(new_lr))

    return new_lr



def save_weights(model, dst_fn):

    weights = model.get_weights()

    with open(dst_fn, 'wb') as f:

        pickle.dump(weights, f)





def load_weights(model, weight_fn):

    with open(weight_fn, 'rb') as f:

        weights = pickle.load(f)

    model.set_weights(weights)

    return model



def build_model():

    ids = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)

    att = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)

    tok = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)

    padding = tf.cast(tf.equal(ids, PAD_ID), tf.int32)



    lens = MAX_LEN - tf.reduce_sum(padding, -1)

    max_len = tf.reduce_max(lens)

    ids_ = ids[:, :max_len]

    att_ = att[:, :max_len]

    tok_ = tok[:, :max_len]



    config = RobertaConfig.from_pretrained(PATH+'config-roberta-base.json')

    model  = TFRobertaForQuestionAnswering.from_pretrained(PATH+'pretrained-roberta-base.h5',config=config)

    

    x = model(ids_,attention_mask=att_,token_type_ids=tok_)

        

    def loss_fn(y_true, y_pred):

        ll = tf.shape(y_pred)[1]

        y_true = y_true[:, :ll]

        loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred,

            from_logits=False, label_smoothing=LABEL_SMOOTHING)

        loss = tf.reduce_mean(loss)

        return loss



    model = tf.keras.models.Model(inputs=[ids, att, tok], outputs=[x[0], x[1]])

    optimizer = tf.keras.optimizers.Nadam(learning_rate=3e-5)

    model.compile(loss=loss_fn, optimizer=optimizer)



    x1_padded = tf.pad(x[0], [[0, 0], [0, MAX_LEN - max_len]], constant_values=0.)

    x2_padded = tf.pad(x[1], [[0, 0], [0, MAX_LEN - max_len]], constant_values=0.)



    padded_model = tf.keras.models.Model(inputs=[ids, att, tok], outputs=[x1_padded,x2_padded])

    return model, padded_model



model, padded_model = build_model()

model.summary()

def jaccard(str1, str2): 

    a = set(str1.lower().split()) 

    b = set(str2.lower().split())

    if (len(a)==0) & (len(b)==0): return 0.5

    c = a.intersection(b)

    return float(len(c)) / (len(a) + len(b) - len(c))
jac = []; DISPLAY=1 # USE display=1 FOR INTERACTIVE

oof_start = np.zeros((input_ids.shape[0],MAX_LEN))

oof_end = np.zeros((input_ids.shape[0],MAX_LEN))

preds_start = np.zeros((input_ids_t.shape[0],MAX_LEN))

preds_end = np.zeros((input_ids_t.shape[0],MAX_LEN))



skf = StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED)



for fold,(idxT,idxV) in enumerate(skf.split(input_ids,train.sentiment.values)):



    print('#'*25)

    print('### FOLD %i'%(fold+1))

    print('#'*25)

    

    K.clear_session()

    model, padded_model = build_model()

    

    reduce_lr = tf.keras.callbacks.LearningRateScheduler(scheduler)

    

    inpT = [input_ids[idxT,], attention_mask[idxT,], token_type_ids[idxT,]]

    targetT = [start_tokens[idxT,], end_tokens[idxT,]]

    inpV = [input_ids[idxV,],attention_mask[idxV,],token_type_ids[idxV,]]

    targetV = [start_tokens[idxV,], end_tokens[idxV,]]

    # sort the validation data

    shuffleV = np.int32(sorted(range(len(inpV[0])), key=lambda k: (inpV[0][k] == PAD_ID).sum(), reverse=True))

    inpV = [arr[shuffleV] for arr in inpV]

    targetV = [arr[shuffleV] for arr in targetV]

    weight_fn = 'sb-model-fold_%i.h5'%(fold)

    for epoch in range(1, EPOCHS + 1):

        # sort and shuffle: We add random numbers to not have the same order in each epoch

        shuffleT = np.int32(sorted(range(len(inpT[0])), key=lambda k: (inpT[0][k] == PAD_ID).sum() + np.random.randint(-3, 3), reverse=True))

        # shuffle in batches, otherwise short batches will always come in the beginning of each epoch

        num_batches = math.ceil(len(shuffleT) / BATCH_SIZE)

        batch_inds = np.random.permutation(num_batches)

        shuffleT_ = []

        for batch_ind in batch_inds:

            shuffleT_.append(shuffleT[batch_ind * BATCH_SIZE: (batch_ind + 1) * BATCH_SIZE])

        shuffleT = np.concatenate(shuffleT_)

        # reorder the input data

        inpT = [arr[shuffleT] for arr in inpT]

        targetT = [arr[shuffleT] for arr in targetT]

        history = model.fit(

            inpT, targetT, 

            epochs=epoch,

            initial_epoch=epoch - 1,

            batch_size=BATCH_SIZE,

            verbose=DISPLAY,

            callbacks=[reduce_lr],

            validation_data=(inpV, targetV),

            shuffle=False)  # don't shuffle in `fit`

        save_weights(model, weight_fn)

                            

    print('Loading model...')

    load_weights(model, weight_fn)



    print('Predicting OOF...')

    oof_start[idxV,],oof_end[idxV,] = padded_model.predict([input_ids[idxV,],attention_mask[idxV,],token_type_ids[idxV,]],verbose=DISPLAY)

            

    print('Predicting Test...')

    preds = padded_model.predict([input_ids_t,attention_mask_t,token_type_ids_t],verbose=DISPLAY)

    preds_start += preds[0]/skf.n_splits

    preds_end += preds[1]/skf.n_splits

    

    # DISPLAY FOLD JACCARD

    all = []

    for k in idxV:

        if (train.loc[k,'sentiment'] == 'neutral'):

            st = train.loc[k,'text']

            all.append(jaccard(st,train.loc[k,'selected_text']))

        else:

            a = np.argmax(oof_start[k,])

            b = np.argmax(oof_end[k,])

            if a>b:

                st = train.loc[k,'text']

            else:

                text1 = " "+" ".join(train.loc[k,'text'].split())

                enc = tokenizer.encode(text1)

                st = tokenizer.decode(enc.ids[a-1:b])

            all.append(jaccard(st,train.loc[k,'selected_text']))

    jac.append(np.mean(all))

    print('>>>> FOLD %i Jaccard ='%(fold+1),np.mean(all))

    print()
all = []

for k in range(input_ids_t.shape[0]):

    if (test.loc[k,'sentiment'] == 'neutral'):

        st = test.loc[k,'text']

    else:

        a = np.argmax(oof_start[k,])

        b = np.argmax(oof_end[k,])

        if a>b:

            st = train.loc[k,'text']

        else:

            text1 = " "+" ".join(test.loc[k,'text'].split())

            enc = tokenizer.encode(text1)

            st = tokenizer.decode(enc.ids[a-1:b])

    all.append(st)
test['selected_text'] = all

test[['textID','selected_text']].to_csv('submission.csv',index=False)

pd.set_option('max_colwidth', 60)

test.sample(25)